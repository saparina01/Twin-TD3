"""Preference-conditioned TD3 with independent, vector-valued twin critics.

Each agent owns its own replay and critics. Stored preferences are never
relabelled: the other agent's behaviour also depends on that preference.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ALGORITHM_VERSION = 1


def preference_vector(rate_weight):
    weight = float(rate_weight)
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('preference must lie in [0, 1]')
    return np.array([weight, 1.0 - weight], dtype=np.float32)


def validate_preference(preference):
    w = np.asarray(preference, dtype=np.float32)
    if (w.shape != (2,) or not np.all(np.isfinite(w)) or np.any(w < 0)
            or not np.isclose(w.sum(), 1.0, atol=1e-6)):
        raise ValueError('preference must be a nonnegative length-2 vector summing to one')
    return w


def sample_preference(rng, fixed=None):
    if fixed is not None:
        return preference_vector(fixed)
    selector = rng.random()
    return preference_vector(0 if selector < 0.1 else 1 if selector < 0.2 else rng.random())


class ReplayBuffer:
    def __init__(self, max_size, obs_dim, n_actions, seed=0):
        if max_size < 1:
            raise ValueError('Replay capacity must be positive')
        self.mem_size, self.mem_cntr = int(max_size), 0
        self.rng = np.random.default_rng(seed)
        self.state_memory = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.new_state_memory = np.zeros_like(self.state_memory)
        self.action_memory = np.zeros((max_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros((max_size, 2), dtype=np.float32)
        self.preference_memory = np.zeros((max_size, 2), dtype=np.float32)
        self.terminal_memory = np.zeros(max_size, dtype=np.float32)

    def store_transition(self, state, action, reward, new_state, done, preference):
        w = validate_preference(preference)
        fields = ((self.state_memory, state), (self.action_memory, action),
                  (self.reward_memory, reward), (self.new_state_memory, new_state))
        for storage, value in fields:
            if np.shape(value) != storage.shape[1:] or not np.all(np.isfinite(value)):
                raise ValueError('Invalid transition shape or nonfinite value')
        if np.any(np.abs(action) > 1):
            raise ValueError('Replay actions must be the bounded commands sent to the environment')
        index = self.mem_cntr % self.mem_size
        for storage, value in fields:
            storage[index] = value
        self.preference_memory[index] = w
        self.terminal_memory[index] = bool(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        indices = self.rng.choice(min(self.mem_size, self.mem_cntr), batch_size, replace=False)
        return tuple(a[indices] for a in (self.state_memory, self.action_memory,
                     self.reward_memory, self.new_state_memory, self.terminal_memory,
                     self.preference_memory))


def hidden_stack(input_dim, hidden_sizes, final_relu=True):
    layers = []
    for i, width in enumerate(hidden_sizes):
        layer = nn.Linear(input_dim, width)
        bound = 1.0 / np.sqrt(width)
        nn.init.uniform_(layer.weight, -bound, bound)
        nn.init.uniform_(layer.bias, -bound, bound)
        layers.extend([layer, nn.LayerNorm(width)])
        if final_relu or i < len(hidden_sizes) - 1:
            layers.append(nn.ReLU())
        input_dim = width
    return nn.Sequential(*layers)


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_sizes):
        super().__init__()
        self.trunk = hidden_stack(obs_dim + 2, hidden_sizes)
        self.mu = nn.Linear(hidden_sizes[-1], n_actions)
        nn.init.uniform_(self.mu.weight, -0.003, 0.003)
        nn.init.uniform_(self.mu.bias, -0.003, 0.003)

    def forward(self, state, preference):
        return torch.tanh(self.mu(self.trunk(torch.cat((state, preference), dim=-1))))


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_sizes):
        super().__init__()
        self.trunk = hidden_stack(obs_dim + 2, hidden_sizes, final_relu=False)
        self.action_value = nn.Linear(n_actions, hidden_sizes[-1])
        self.q = nn.Linear(hidden_sizes[-1], 2)
        nn.init.uniform_(self.q.weight, -0.003, 0.003)
        nn.init.uniform_(self.q.bias, -0.003, 0.003)

    def forward(self, state, action, preference):
        value = self.trunk(torch.cat((state, preference), dim=-1))
        return self.q(F.relu(value + F.relu(self.action_value(action))))


def conservative_vector(q1, q2, preference):
    """Select a whole vector by scalarized value, not a componentwise minimum."""
    choose_first = ((q1 * preference).sum(-1, keepdim=True)
                    <= (q2 * preference).sum(-1, keepdim=True))
    return torch.where(choose_first, q1, q2)


def bellman_target(reward, done, q1, q2, preference, gamma=1.0):
    bootstrap = conservative_vector(q1, q2, preference)
    return reward + gamma * (1.0 - done.reshape(-1, 1)) * bootstrap


def actor_objective(q, preference):
    return -(q * preference).sum(dim=-1).mean()


class Agent:
    def __init__(self, obs_dim, n_actions, hidden_sizes, reward_scales,
                 alpha=1e-4, beta=1e-3, tau=0.001, gamma=1.0,
                 batch_size=64, max_size=30000, update_actor_interval=2,
                 policy_noise=0.2, noise_clip=0.5, seed=0, device='cpu'):
        scales = np.asarray(reward_scales, dtype=np.float32)
        if scales.shape != (2,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError('Two positive finite reward scales are required')
        if batch_size <= 0 or max_size < batch_size or update_actor_interval <= 0:
            raise ValueError('Invalid batch size, replay capacity or update interval')
        if len(hidden_sizes) != 4 or min(hidden_sizes) < 2:
            raise ValueError('Four hidden layer sizes of at least two are required')
        self.config = dict(obs_dim=obs_dim, n_actions=n_actions, hidden_sizes=list(hidden_sizes),
                           reward_scales=scales.tolist(), alpha=alpha, beta=beta, tau=tau,
                           gamma=gamma, batch_size=batch_size, max_size=max_size,
                           update_actor_interval=update_actor_interval, policy_noise=policy_noise,
                           noise_clip=noise_clip, seed=seed)
        self.device = torch.device(device)
        self.reward_scales = torch.as_tensor(scales, device=self.device)
        self.gamma, self.tau = gamma, tau
        self.batch_size, self.update_actor_interval = batch_size, update_actor_interval
        self.policy_noise, self.noise_clip = policy_noise, noise_clip
        self.learn_step_cntr = 0
        self.rng = np.random.default_rng(seed)
        self.memory = ReplayBuffer(max_size, obs_dim, n_actions, seed=seed + 1)
        # Isolate each agent's initialization from the other agent and caller.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.actor = ActorNetwork(obs_dim, n_actions, hidden_sizes).to(self.device)
            self.critic_1 = CriticNetwork(obs_dim, n_actions, hidden_sizes).to(self.device)
            self.critic_2 = CriticNetwork(obs_dim, n_actions, hidden_sizes).to(self.device)
        self.target_actor = deepcopy(self.actor).requires_grad_(False)
        self.target_critic_1 = deepcopy(self.critic_1).requires_grad_(False)
        self.target_critic_2 = deepcopy(self.critic_2).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=beta)

    def choose_action(self, observation, preference, noise_std=0.0):
        w = validate_preference(preference)
        with torch.no_grad():
            state = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            pref = torch.as_tensor(w, device=self.device)
            action = self.actor(state, pref).cpu().numpy()
        if noise_std:
            action = action + self.rng.normal(0, noise_std, size=action.shape)
        return np.clip(action, -1, 1).astype(np.float32)

    def remember(self, state, action, reward, new_state, done, preference):
        self.memory.store_transition(state, action, reward, new_state, done, preference)

    def learn(self):
        if min(self.memory.mem_cntr, self.memory.mem_size) < self.batch_size:
            return None
        state, action, reward, new_state, done, w = (
            torch.as_tensor(a, device=self.device) for a in self.memory.sample_buffer(self.batch_size))
        reward = reward / self.reward_scales
        with torch.no_grad():
            target_action = self.target_actor(new_state, w)
            # A private RNG keeps reproducibility independent of the other agent.
            noise = torch.as_tensor(self.rng.normal(0, self.policy_noise, target_action.shape),
                                    dtype=torch.float32, device=self.device).clamp(-self.noise_clip, self.noise_clip)
            target_action = (target_action + noise).clamp(-1, 1)
            target = bellman_target(reward, done,
                self.target_critic_1(new_state, target_action, w),
                self.target_critic_2(new_state, target_action, w), w, self.gamma)
        loss = (F.mse_loss(self.critic_1(state, action, w), target)
                + F.mse_loss(self.critic_2(state, action, w), target))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite MORL critic loss')
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        self.learn_step_cntr += 1
        metrics = dict(critic_loss=float(loss.detach()), actor_loss=None)
        if self.learn_step_cntr % self.update_actor_interval:
            return metrics
        self.critic_1.requires_grad_(False)
        actor_loss = actor_objective(self.critic_1(state, self.actor(state, w), w), w)
        if not torch.isfinite(actor_loss):
            raise FloatingPointError('Nonfinite MORL actor loss')
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_1.requires_grad_(True)
        self.update_network_parameters()
        metrics['actor_loss'] = float(actor_loss.detach())
        return metrics

    @torch.no_grad()
    def update_network_parameters(self):
        for online, target in ((self.actor, self.target_actor),
                               (self.critic_1, self.target_critic_1), (self.critic_2, self.target_critic_2)):
            for source, dest in zip(online.parameters(), target.parameters()):
                dest.mul_(1 - self.tau).add_(source, alpha=self.tau)

    def save(self, filename):
        payload = dict(algorithm='morl_td3', version=ALGORITHM_VERSION, config=self.config,
                       learn_step_cntr=self.learn_step_cntr)
        for name in ('actor', 'critic_1', 'critic_2', 'target_actor', 'target_critic_1', 'target_critic_2'):
            payload[name] = getattr(self, name).state_dict()
        payload['actor_optimizer'] = self.actor_optimizer.state_dict()
        payload['critic_optimizer'] = self.critic_optimizer.state_dict()
        torch.save(payload, Path(filename))

    @classmethod
    def load(cls, filename, device='cpu'):
        # weights_only is available in recent torch; support the repo's torch 1.12 as well.
        try:
            payload = torch.load(filename, map_location=device, weights_only=True)
        except TypeError:
            payload = torch.load(filename, map_location=device)
        if payload.get('algorithm') != 'morl_td3' or payload.get('version') != ALGORITHM_VERSION:
            raise ValueError('Unsupported MORL checkpoint')
        agent = cls(**payload['config'], device=device)
        for name in ('actor', 'critic_1', 'critic_2', 'target_actor', 'target_critic_1', 'target_critic_2'):
            getattr(agent, name).load_state_dict(payload[name])
        agent.actor_optimizer.load_state_dict(payload['actor_optimizer'])
        agent.critic_optimizer.load_state_dict(payload['critic_optimizer'])
        agent.learn_step_cntr = payload['learn_step_cntr']
        return agent
