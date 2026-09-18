# Deep Reinforcement Learning for Secrecy Energy-Efficient UAV Communication with Reconfigurable Intelligent Surfaces

## Preference-conditioned MORL extension

The new `morl_td3` mode retains two independent agents: one controls UAV
beamforming/RIS phases, and the other controls the UAV trajectory. Both receive
the same preference `[lambda, 1-lambda]` and the same vector reward
`[sum_secrecy_rate, -propulsion_energy_j]`. Each agent has two vector critics,
each with two outputs. The original `td3`/`ddpg` SSR/SEE commands and checkpoints
retain their legacy behaviour.

### Train and evaluate

Run these commands from the repository directory with the project's Python
environment (Python 3.10, NumPy < 2, PyTorch). On this Windows installation it is
available through `conda activate Twin-TD3`.

```powershell
# Sample preferences once per episode: 10% lambda=0, 10% lambda=1, 80% uniform.
python main_train.py --drl morl_td3 --reward morl --ep-num 300 --seeds 0

# Fixed-preference baselines use exactly the same environment and learner.
python main_train.py --drl morl_td3 --reward morl --ep-num 300 --seeds 0 --preference 0
python main_train.py --drl morl_td3 --reward morl --ep-num 300 --seeds 0 --preference 0.5
python main_train.py --drl morl_td3 --reward morl --ep-num 300 --seeds 0 --preference 1

# Evaluate a sampled model at 21 preferences with common seeds 1000,1001,1002.
# Use the actual run folder printed by training; repeated names receive suffixes.
python run_simulation.py --path data/storage/morl/sampled_seed_0

# Evaluate a selected grid or regenerate plots without running the model.
python run_simulation.py --path data/storage/morl/sampled_seed_0 --preferences 0 0.5 1 --eval-seeds 1000 1001 1002
python load_and_plot.py --path data/storage/morl/sampled_seed_0
```

`--seeds` accepts one seed for both networks/environment or two network seeds
(the first also seeds the environment and preference/observation streams).
MORL defaults to seed 0, CPU and one PyTorch thread; use `--device cuda:0` and/or
`--threads` to change execution. `--trained-uav` is rejected for MORL because
legacy network input/output dimensions and reward semantics differ. The two
agents train together; the trajectory policy is never frozen in this mode.

Optional MORL training arguments are `--output-dir`, `--step-num` (100),
`--slot-duration` (0.1 seconds), `--rate-ref` (10 bits/s/Hz), `--energy-ref`
(one slot of hovering energy), and `--observation-noise-std` (6e-8). Normalization
scales are fixed during training and restored from metadata at evaluation.
Checkpoints are saved every ten episodes and at completion. They contain all
online/target networks and optimizer states; this first version supports model
loading for evaluation, **not exact interrupted-training resumption** (replay
contents/RNG progress are not saved).

### Physical and learning semantics

- MORL capacities use `log2(1+SINR)` in bits/s/Hz; secrecy is summed over users
  after subtracting the strongest eavesdropper and taking the positive part.
  There is no positive per-user secrecy-rate threshold in this version.
- The environment records actual displacement, computes speed as distance / slot
  duration and propulsion energy as power × duration in joules. Aircraft
  parameters are inherited from the original repository; communication,
  amplifier, circuit and RIS power are not included in this energy objective.
- Commands are clipped to `[-1,1]`. Each horizontal axis permits 0.25 m per
  step (maximum diagonal speed is approximately 3.536 m/s at the default slot
  duration). Positions are projected to the flight box and beamforming power
  is projected to its existing limit; RIS coefficients remain unit magnitude.
- Each episode lasts exactly 100 steps by default (10 seconds). Boundary requests
  do not terminate the task early. Energy uses the projected, actual movement.
  Both observations include remaining-time fraction, and gamma is 1. Terminal
  Bellman targets contain immediate rewards only.
- Replay stores raw vector rewards and the original preference. Critic targets
  select the **whole vector** from the target critic with the lower weighted
  value. Actor loss is `-mean(sum(w * Q1(s, actor(s,w), w)))`.
- Target action noise is independent per sample/action, standard deviation 0.2,
  clipped at ±0.5; target actions are bounded. Actor/target updates occur every
  two critic updates. No preference relabelling or centralized critic is used.
- CSI noise is applied on both reset and subsequent observations, using a
  separate random stream. Coordinates and time are exact. Action exploration
  is disabled during evaluation, while the saved CSI observation-noise model
  remains active. The inherited channel and user trajectories are otherwise
  deterministic; evaluation seeds do not create different physical layouts.

The MORL physics fixes are opt-in. Legacy SSR/SEE retain their original log10
and energy code paths for reproducibility. Published legacy numbers must not be
treated as directly comparable with these corrected MORL metrics. The new
sampled/fixed-preference comparison uses the same corrected environment.

### Outputs and experiments

Each run contains `morl_metadata.json`, a snapshot of the initial coordinates in
`inputs/`, `communication.pt`, `trajectory.pt`, per-episode `.mat` logs and
`training_summary.csv`. MAT logs include named fields `reward`, `preference`,
`normalized_reward`, `scalar_utility`, `sum_secrecy_rate`, `propulsion_energy_j`,
`speed_mps`, and projection flags, along with the existing radio/trajectory data.

Evaluation creates a separate `evaluation` folder (suffix added if needed) with
`evaluation_config.json`, `evaluation_steps.csv`, `evaluation_episodes.csv`,
`evaluation_summary.csv`, per-episode MAT logs, `pareto.png`, and
`trajectories.png`. The energy totals and plots read named environment metrics;
they do not recompute a different energy model. The default grid for a fixed
baseline is its training preference only. Non-dominance maximizes rate and
minimizes energy, using per-preference means; this is an empirical set, not a
claim of complete or statistically certified Pareto coverage.

```powershell
# Three training seeds × (one sampled model + three fixed baselines).
# A new output directory is required. Each model receives the same step budget.
python morl_benchmark.py --ep-num 300 --seeds 0 1 2 --output-dir data/storage/morl/benchmark

# Short operational check; results do not establish algorithm performance.
python morl_benchmark.py --ep-num 3 --seeds 0 1 2 --preferences 0 0.5 1 --eval-seeds 1000 --output-dir data/storage/morl/smoke_comparison

# Compare existing evaluation folders; include at least three training seeds
# per method for a research comparison. All settings/budgets must match.
python morl_plot.py --paths RUN_A/evaluation RUN_B/evaluation RUN_C/evaluation --output-dir data/storage/morl/comparison

python -W ignore::PendingDeprecationWarning -m unittest discover -s tests -v
```

Comparison first averages repeated evaluations within each trained model, then
computes standard deviations across training seeds. It emits per-model CSV,
`comparison_summary.csv`, `matched_utility.csv` (sampled minus fixed utility at
matched preferences/seeds), `comparison.png`, and `comparison_report.md` with
preference-sensitivity diagnostics. Small rate/energy spans require examining
conflict strength, scaling and training coverage; successful execution alone
does not demonstrate MORL superiority. A complete sweep costs more interactions
than any one baseline; report both per-model and total training budgets.

Implementation: `env.py` contains the MORL environment path; `morl_td3.py` the
networks/replay/update rules; `morl_experiment.py` training and evaluation;
`morl_plot.py` plotting/comparison; `morl_benchmark.py` the seed experiment.