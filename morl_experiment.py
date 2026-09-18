"""Training/evaluation orchestration for the MORL mode; no work on import."""
import csv
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from env import MiniSystem, propulsion_energy
from morl_td3 import ALGORITHM_VERSION, Agent, preference_vector, sample_preference

ROOT = Path(__file__).resolve().parent
META_FILE = 'morl_metadata.json'


def seed_everything(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        raise ValueError('No rows to write')
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_metadata(run_path):
    meta = json.loads((Path(run_path) / META_FILE).read_text(encoding='utf-8'))
    if meta.get('algorithm') != 'morl_td3' or meta.get('version') != ALGORITHM_VERSION:
        raise ValueError('Unsupported MORL metadata version')
    return meta


def observations(system, rng, observation_noise_std):
    """Noise is applied identically on reset and subsequent observations.

    Only CSI is corrupted; positions and remaining-time observations are exact.
    The two independent critics retain their original local observations.
    """
    comm = np.asarray(system.observe(), dtype=np.float32)
    channel_dim = 2 * (system.user_num + system.attacker_num) * system.UAV.ant_num
    if observation_noise_std:
        comm[:channel_dim] += rng.normal(0, observation_noise_std, channel_dim).astype(np.float32)
    uav = np.array([*system.UAV.coordinate, 1.0 - system.elapsed_steps / system.step_num], dtype=np.float32)
    return comm, uav


def step_joint(system, comm_action, uav_action):
    split = 2 * system.UAV.ant_num * system.user_num
    return system.step(action_0=uav_action[0], action_1=uav_action[1],
                       G=comm_action[:split], Phi=comm_action[split:])


def record_preference(system, raw_reward, preference, scales):
    normalized = np.asarray(raw_reward) / np.asarray(scales)
    utility = float(np.dot(normalized, preference))
    for key, value in (('preference', preference.tolist()),
                       ('normalized_reward', normalized.tolist()), ('scalar_utility', [utility])):
        system.data_manager.store_data(value, key)
    return normalized, utility


def make_system(env_config, output_dir, input_dir):
    output = Path(output_dir).resolve()
    return MiniSystem(**env_config, project_name=output.name, store_path=str(output.parent),
                      data_dir=str(Path(input_dir).resolve()), render_enabled=False)


def train_morl(episodes=300, seeds=None, preference=None, output_dir=None,
               step_num=100, slot_duration_s=0.1, rate_ref=10.0, energy_ref=None,
               observation_noise_std=6e-8, device='cpu', threads=1,
               hidden_sizes=None, batch_size=64):
    """Train a pair of agents and return the actual, collision-free run folder."""
    seeds = [0] if seeds is None else list(seeds)
    if len(seeds) not in (1, 2) or any(s < 0 or s >= 2**32 - 100 for s in seeds):
        raise ValueError('Supply one or two nonnegative 32-bit seeds')
    if episodes < 1 or threads < 1 or step_num < 1:
        raise ValueError('Episodes, step count and threads must be positive')
    if observation_noise_std < 0 or not math.isfinite(observation_noise_std):
        raise ValueError('Observation noise must be finite and nonnegative')
    if preference is not None:
        preference_vector(preference)
    scales = [float(rate_ref), float(energy_ref if energy_ref is not None else
                                   propulsion_energy([0, 0], slot_duration_s))]
    if not np.all(np.isfinite(scales)) or min(scales) <= 0:
        raise ValueError('Reward scales must be positive and finite')
    seed_everything(seeds[0], threads)
    label = 'sampled' if preference is None else f'fixed_{preference:g}'
    if output_dir is None:
        output_dir = ROOT / 'data' / 'storage' / 'morl' / f'{label}_seed_{seeds[0]}'
    env_config = dict(user_num=2, attacker_num=1, RIS_ant_num=4, UAV_ant_num=4,
                      if_dir_link=1, if_with_RIS=True, if_move_users=True, if_movements=True,
                      reverse_x_y=[False, False], if_UAV_pos_state=True, reward_design='morl',
                      step_num=step_num, slot_duration_s=slot_duration_s)
    system = make_system(env_config, output_dir, ROOT / 'data')
    output = Path(system.data_manager.store_path)
    (output / 'inputs').mkdir()
    shutil.copyfile(ROOT / 'data' / 'init_location.xlsx', output / 'inputs' / 'init_location.xlsx')
    sizes = hidden_sizes or ([800, 600, 512, 256], [400, 300, 256, 128])
    agent_common = dict(reward_scales=scales, batch_size=batch_size,
                        max_size=max(batch_size, episodes * step_num), device=device)
    agents = [Agent(obs_dim=system.get_system_state_dim(), n_actions=system.get_system_action_dim()-2,
                    hidden_sizes=sizes[0], seed=seeds[0], **agent_common),
              Agent(obs_dim=4, n_actions=2, hidden_sizes=sizes[1], seed=seeds[-1], **agent_common)]
    initial_actors = [[p.detach().clone() for p in a.actor.parameters()] for a in agents]
    metadata = dict(algorithm='morl_td3', version=ALGORITHM_VERSION,
                    environment=env_config, input_locations='inputs/init_location.xlsx',
                    reward_scales=scales, reward_names=['sum_secrecy_rate', 'negative_propulsion_energy_j'],
                    units=['bits/s/Hz', 'J'], seeds=seeds, training_episodes=episodes,
                    preference=preference, preference_sampling=dict(energy_endpoint=0.1, rate_endpoint=0.1, uniform=0.8),
                    observation_noise_std=observation_noise_std, action_noise_start=[0.1, 0.5],
                    device=str(device), threads=threads, agents=[a.config for a in agents],
                    model_files=['communication.pt', 'trajectory.pt'], completed_episodes=0,
                    status='training')
    write_json(output / META_FILE, metadata)
    # Preference, observation and exploration/replay RNGs have separate streams.
    pref_rng = np.random.default_rng(np.random.SeedSequence([seeds[0], 10]))
    obs_rng = np.random.default_rng(np.random.SeedSequence([seeds[0], 20]))
    summaries = []
    print(f'MORL run: {output}', flush=True)
    for episode in range(episodes):
        system.reset()
        w = sample_preference(pref_rng, preference)
        state = observations(system, obs_rng, observation_noise_std)
        rate_total = energy_total = utility_total = 0.0
        boundary_count = power_count = 0
        losses = [[], []]
        for t in range(step_num):
            factor = (1 - episode / episodes) ** 2
            actions = [a.choose_action(s, w, noise_std=start * factor)
                       for a, s, start in zip(agents, state, (0.1, 0.5))]
            _, reward, done, info = step_joint(system, *actions)
            new_state = observations(system, obs_rng, observation_noise_std)
            _, utility = record_preference(system, reward, w, scales)
            for index, agent in enumerate(agents):
                agent.remember(state[index], actions[index], reward, new_state[index], done, w)
                result = agent.learn()
                if result is not None:
                    losses[index].append(result)
            state = new_state
            rate_total += info['sum_secrecy_rate']
            energy_total += info['propulsion_energy_j']
            utility_total += utility
            boundary_count += int(info['boundary_projected'])
            power_count += int(info['power_projected'])
            if done != (t == step_num - 1):
                raise RuntimeError('MORL episode horizon does not match training configuration')
        summary = dict(episode=episode, rate_weight=float(w[0]), energy_weight=float(w[1]),
                       mean_secrecy_rate=rate_total / step_num, total_energy_j=energy_total,
                       scalar_return=utility_total, boundary_projections=boundary_count,
                       power_projections=power_count, steps=step_num)
        for index, history in enumerate(losses):
            for key in ('critic_loss', 'actor_loss'):
                values = [x[key] for x in history if x[key] is not None]
                summary[f'agent_{index+1}_{key}'] = float(np.mean(values)) if values else None
        summaries.append(summary)
        system.data_manager.save_file(episode)
        write_csv(output / 'training_summary.csv', summaries)
        metadata['completed_episodes'] = episode + 1
        if (episode + 1) % 10 == 0 or episode == episodes - 1:
            for agent, name in zip(agents, metadata['model_files']):
                agent.save(output / name)
            write_json(output / META_FILE, metadata)
        print(f'ep={episode+1}/{episodes} lambda={w[0]:.3f} SSR={summary["mean_secrecy_rate"]:.4f} '
              f'energy={energy_total:.2f} J', flush=True)
    metadata['status'] = 'complete'
    metadata['actor_changed'] = [any(not torch.equal(before, after) for before, after in
                                  zip(initial, a.actor.parameters())) for initial, a in zip(initial_actors, agents)]
    metadata['learning_updates'] = [a.learn_step_cntr for a in agents]
    write_json(output / META_FILE, metadata)
    return output


def nondominated_mask(rates, energies):
    """Maximise rate and minimise energy; keep ties as separate observations."""
    rates, energies = np.asarray(rates), np.asarray(energies)
    if not (np.all(np.isfinite(rates)) and np.all(np.isfinite(energies))):
        raise ValueError('Nonfinite Pareto metrics')
    return np.array([not np.any((rates >= r) & (energies <= e) & ((rates > r) | (energies < e)))
                     for r, e in zip(rates, energies)])


def evaluation_summary(rows):
    summaries = []
    for weight in sorted({r['rate_weight'] for r in rows}):
        group = [r for r in rows if r['rate_weight'] == weight]
        summary = dict(rate_weight=weight, evaluations=len(group))
        for key in ('mean_secrecy_rate', 'total_energy_j', 'scalar_return',
                    'boundary_projections', 'power_projections'):
            values = [float(r[key]) for r in group]
            summary[key] = float(np.mean(values))
            summary[key + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summaries.append(summary)
    front = nondominated_mask([s['mean_secrecy_rate'] for s in summaries], [s['total_energy_j'] for s in summaries])
    for s, flag in zip(summaries, front):
        s['nondominated'] = bool(flag)
    return summaries


def evaluate_morl(run_path, preferences=None, eval_seeds=None, output_dir=None, device='cpu', threads=1):
    run = Path(run_path).resolve()
    meta = load_metadata(run)
    if meta.get('completed_episodes', 0) < 1:
        raise ValueError('No completed checkpoint available')
    seeds = [1000, 1001, 1002] if eval_seeds is None else list(eval_seeds)
    if not seeds or any(s < 0 or s >= 2**32 for s in seeds):
        raise ValueError('Evaluation seeds must be nonnegative 32-bit integers')
    if len(set(seeds)) != len(seeds):
        raise ValueError('Evaluation seeds must be distinct')
    if preferences is None:
        preferences = [meta['preference']] if meta['preference'] is not None else np.linspace(0, 1, 21)
    weights = sorted({float(preference_vector(w)[0]) for w in preferences})
    if not weights:
        raise ValueError('At least one preference is required')
    seed_everything(seeds[0], threads)
    agents = [Agent.load(run / name, device=device) for name in meta['model_files']]
    for agent, config in zip(agents, meta['agents']):
        if agent.config != config:
            raise ValueError('Checkpoint and run metadata disagree')
    system = make_system(meta['environment'], output_dir or run / 'evaluation',
                         (run / meta['input_locations']).parent)
    output = Path(system.data_manager.store_path)
    rows, steps = [], []
    scales = meta['reward_scales']
    episode = 0
    for weight in weights:
        w = preference_vector(weight)
        for seed in seeds:
            # Common random numbers across preferences and across compared models.
            seed_everything(seed, threads)
            obs_rng = np.random.default_rng(np.random.SeedSequence([seed, 20]))
            system.reset()
            state = observations(system, obs_rng, meta['observation_noise_std'])
            total_rate = total_energy = total_utility = 0.0
            boundary_count = power_count = 0
            for t in range(system.step_num):
                actions = [a.choose_action(s, w) for a, s in zip(agents, state)]
                old_position = system.UAV.coordinate.copy()
                _, reward, done, info = step_joint(system, *actions)
                normalized, utility = record_preference(system, reward, w, scales)
                state = observations(system, obs_rng, meta['observation_noise_std'])
                total_rate += info['sum_secrecy_rate']
                total_energy += info['propulsion_energy_j']
                total_utility += utility
                boundary_count += int(info['boundary_projected'])
                power_count += int(info['power_projected'])
                steps.append(dict(rate_weight=weight, eval_seed=seed, step=t,
                                  x_start=float(old_position[0]), y_start=float(old_position[1]),
                                  x=float(system.UAV.coordinate[0]), y=float(system.UAV.coordinate[1]),
                                  z=float(system.UAV.coordinate[2]), **info,
                                  normalized_rate=float(normalized[0]), normalized_energy=float(normalized[1]),
                                  scalar_utility=utility))
                if done != (t == system.step_num - 1):
                    raise RuntimeError('Evaluation horizon mismatch')
            rows.append(dict(training_seed=meta['seeds'][0], rate_weight=weight, energy_weight=float(w[1]),
                             eval_seed=seed, mean_secrecy_rate=total_rate/system.step_num,
                             total_energy_j=total_energy, scalar_return=total_utility,
                             boundary_projections=boundary_count, power_projections=power_count,
                             steps=system.step_num))
            system.data_manager.save_file(episode)
            episode += 1
        print(f'eval lambda={weight:.2f}: SSR={rows[-1]["mean_secrecy_rate"]:.4f}, '
              f'energy={rows[-1]["total_energy_j"]:.2f} J', flush=True)
    write_csv(output / 'evaluation_episodes.csv', rows)
    write_csv(output / 'evaluation_steps.csv', steps)
    summaries = evaluation_summary(rows)
    write_csv(output / 'evaluation_summary.csv', summaries)
    write_json(output / 'evaluation_config.json', dict(model_path=str(run), model_metadata=meta,
               preferences=weights, eval_seeds=seeds, action_exploration=False,
               observation_noise_std=meta['observation_noise_std']))
    from morl_plot import plot_evaluation
    plot_evaluation(output)
    return output
