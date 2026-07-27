import torch
import torch.nn as nn
import os
import numpy as np
from dataclasses import dataclass

try:
    from .shared_arch import shared
except ImportError:
    from shared_arch import shared


class GRUMLPAgent(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dim=64,
        gru_layers=1,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_dim = int(hidden_dim)
        self.gru_layers = int(gru_layers)
        self.history_input_dim = self.obs_dim + self.act_dim + 1

        self.gru = nn.GRU(
            input_size=self.history_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.gru_layers,
            batch_first=True,
        )

        self.policy_input_dim = self.obs_dim + self.hidden_dim
        self.fc = shared(input_dim=self.policy_input_dim)

        self.fc_mean = None
        self.fc_logstd = None
        self.reset_heads()

    def reset_heads(self):
        # 假设 shared(...) 的输出维度是 256
        self.fc_mean = nn.Linear(256, self.act_dim)
        self.fc_logstd = nn.Linear(256, self.act_dim)

    def initial_hidden(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device

        return torch.zeros(
            self.gru_layers,
            batch_size,
            self.hidden_dim,
            device=device,
        )

    def encode(self, history, batch_size=None, h=None):
        if history is None or history.size(1) == 0:
            if batch_size is None:
                batch_size = h.size(1) if h is not None else 1
            h = self.initial_hidden(batch_size, device=next(self.parameters()).device)
            return h[-1], h

        # history: [B, history_len, obs_dim + act_dim + 1]
        # h: [gru_layers, B, hidden_dim]
        _, h = self.gru(history, h)

        # 取最后一层 GRU 的最后隐状态
        # shape: [B, hidden_dim]
        return h[-1], h

    def forward(self, x, history=None, h=None, return_hidden=False):
        """
        x:
            [batch_size, obs_dim]

        history:
            [batch_size, history_len, obs_dim + act_dim + 1]
            containing previous (obs, action, reward) tuples.

        h:
            [gru_layers, batch_size, hidden_dim]

        return:
            默认返回:
                mean, log_std

            如果 return_hidden=True:
                mean, log_std, h
        """

        last_hidden, h = self.encode(history, batch_size=x.size(0), h=h)

        policy_input = torch.cat([x, last_hidden], dim=-1)
        x = self.fc(policy_input)

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)

        if return_hidden:
            return mean, log_std, h

        return mean, log_std

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, f"{dirname}/model.pt")

    @staticmethod
    def load(dirname, map_location=None, reset_heads=False):
        model = torch.load(f"{dirname}/model.pt", map_location=map_location)
        if reset_heads:
            model.reset_heads()
        return model


class GRUNoContextAgent(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dim=64,
        gru_layers=1,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_dim = int(hidden_dim)
        self.gru_layers = int(gru_layers)
        self.history_input_dim = self.obs_dim + self.act_dim + 1

        self.gru = nn.GRU(
            input_size=self.history_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.gru_layers,
            batch_first=True,
        )

        self.policy_input_dim = self.obs_dim
        self.fc = shared(input_dim=self.policy_input_dim)
        self.fc_mean = None
        self.fc_logstd = None
        self.reset_heads()

    def reset_heads(self):
        self.fc_mean = nn.Linear(256, self.act_dim)
        self.fc_logstd = nn.Linear(256, self.act_dim)

    def initial_hidden(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device

        return torch.zeros(
            self.gru_layers,
            batch_size,
            self.hidden_dim,
            device=device,
        )

    def encode(self, history, batch_size=None, h=None):
        if history is None or history.size(1) == 0:
            if batch_size is None:
                batch_size = h.size(1) if h is not None else 1
            h = self.initial_hidden(batch_size, device=next(self.parameters()).device)
            return h[-1], h

        _, h = self.gru(history, h)
        return h[-1], h

    def forward(self, x, history=None, h=None, return_hidden=False):
        _, h = self.encode(history, batch_size=x.size(0), h=h)
        x = self.fc(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)

        if return_hidden:
            return mean, log_std, h

        return mean, log_std

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, f"{dirname}/model.pt")

    @staticmethod
    def load(dirname, map_location=None, reset_heads=False):
        model = torch.load(f"{dirname}/model.pt", map_location=map_location)
        if reset_heads:
            model.reset_heads()
        return model


class StateGRUContextAgent(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dim=64,
        gru_layers=1,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_dim = int(hidden_dim)
        self.gru_layers = int(gru_layers)

        self.gru = nn.GRU(
            input_size=self.obs_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.gru_layers,
            batch_first=True,
        )

        self.policy_input_dim = self.obs_dim + self.hidden_dim
        self.fc = shared(input_dim=self.policy_input_dim)
        self.fc_mean = None
        self.fc_logstd = None
        self.reset_heads()

    def reset_heads(self):
        self.fc_mean = nn.Linear(256, self.act_dim)
        self.fc_logstd = nn.Linear(256, self.act_dim)

    def initial_hidden(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device

        return torch.zeros(
            self.gru_layers,
            batch_size,
            self.hidden_dim,
            device=device,
        )

    def encode(self, state_sequence, batch_size=None, h=None):
        if state_sequence is None or state_sequence.size(1) == 0:
            if batch_size is None:
                batch_size = h.size(1) if h is not None else 1
            h = self.initial_hidden(batch_size, device=next(self.parameters()).device)
            return h[-1], h

        _, h = self.gru(state_sequence, h)
        return h[-1], h

    def forward(self, x, state_sequence=None, h=None, return_hidden=False):
        last_hidden, h = self.encode(state_sequence, batch_size=x.size(0), h=h)
        policy_input = torch.cat([x, last_hidden], dim=-1)
        x = self.fc(policy_input)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)

        if return_hidden:
            return mean, log_std, h

        return mean, log_std

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, f"{dirname}/model.pt")

    @staticmethod
    def load(dirname, map_location=None, reset_heads=False):
        model = torch.load(f"{dirname}/model.pt", map_location=map_location)
        if reset_heads:
            model.reset_heads()
        return model


class RecurrentSoftQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64, gru_layers=1, gru=None):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_dim = int(hidden_dim)
        self.gru_layers = int(gru_layers)
        self.history_input_dim = self.obs_dim + self.act_dim + 1

        self.gru = gru if gru is not None else nn.GRU(
            input_size=self.history_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.gru_layers,
            batch_first=True,
        )
        self.fc = shared(self.obs_dim + self.act_dim + self.hidden_dim)
        self.fc_out = nn.Linear(256, 1)

    def encode(self, history, batch_size):
        if history is None or history.size(1) == 0:
            return torch.zeros(
                batch_size,
                self.hidden_dim,
                dtype=next(self.parameters()).dtype,
                device=next(self.parameters()).device,
            )

        _, h = self.gru(history)
        return h[-1]

    def forward(self, x, a, history=None):
        z = self.encode(history, batch_size=x.size(0))
        q_input = torch.cat([x, a, z], dim=-1)
        x = self.fc(q_input)
        return self.fc_out(x)


@dataclass
class GRUReplayBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    histories: torch.Tensor
    next_histories: torch.Tensor


@dataclass
class StateGRUReplayBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    state_sequences: torch.Tensor
    next_state_sequences: torch.Tensor


def _stored_replay_size(replay_buffer):
    return replay_buffer.buffer_size if replay_buffer.full else replay_buffer.pos


def _as_torch(array, device):
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _transition_history_features(replay_buffer, indices, env_idx):
    obs = replay_buffer.observations[indices, env_idx].reshape(len(indices), -1)
    actions = replay_buffer.actions[indices, env_idx].reshape(len(indices), -1)
    rewards = replay_buffer.rewards[indices, env_idx].reshape(len(indices), 1)
    return np.concatenate([obs, actions, rewards], axis=-1)


def _logical_to_physical(logical_idx, oldest_idx, capacity, full):
    if full:
        return (oldest_idx + logical_idx) % capacity
    return logical_idx


def _padded_transition_history_features(
    replay_buffer,
    logical_indices,
    env_idx,
    oldest_idx,
    history_dim,
):
    history = np.zeros((len(logical_indices), history_dim), dtype=np.float32)
    capacity = replay_buffer.buffer_size
    full = replay_buffer.full

    valid_logical_indices = [idx for idx in logical_indices if idx >= 0]
    latest_done_logical = None
    for logical_idx in valid_logical_indices:
        physical_idx = _logical_to_physical(logical_idx, oldest_idx, capacity, full)
        if replay_buffer.dones[physical_idx, env_idx]:
            latest_done_logical = logical_idx

    for pos, logical_idx in enumerate(logical_indices):
        if logical_idx < 0:
            continue
        if latest_done_logical is not None and logical_idx <= latest_done_logical:
            continue

        physical_idx = _logical_to_physical(logical_idx, oldest_idx, capacity, full)
        history[pos] = _transition_history_features(
            replay_buffer,
            np.asarray([physical_idx]),
            env_idx,
        )[0]

    return history


def _padded_state_sequence(
    replay_buffer,
    logical_obs_indices,
    env_idx,
    oldest_idx,
    obs_dim,
    next_logical_idx=None,
):
    sequence = np.zeros((len(logical_obs_indices), obs_dim), dtype=np.float32)
    capacity = replay_buffer.buffer_size
    full = replay_buffer.full

    valid_obs_indices = [idx for idx in logical_obs_indices if idx >= 0]
    latest_done_logical = None
    if len(valid_obs_indices) > 0:
        end_logical = max(valid_obs_indices)
        start_logical = min(valid_obs_indices)
        for logical_idx in range(start_logical, end_logical):
            physical_idx = _logical_to_physical(logical_idx, oldest_idx, capacity, full)
            if replay_buffer.dones[physical_idx, env_idx]:
                latest_done_logical = logical_idx

    for pos, logical_idx in enumerate(logical_obs_indices):
        if logical_idx < 0:
            continue
        if latest_done_logical is not None and logical_idx <= latest_done_logical:
            continue

        if next_logical_idx is not None and logical_idx == next_logical_idx:
            prev_physical_idx = _logical_to_physical(
                logical_idx - 1,
                oldest_idx,
                capacity,
                full,
            )
            sequence[pos] = replay_buffer.next_observations[
                prev_physical_idx,
                env_idx,
            ].reshape(-1)
        else:
            physical_idx = _logical_to_physical(logical_idx, oldest_idx, capacity, full)
            sequence[pos] = replay_buffer.observations[
                physical_idx,
                env_idx,
            ].reshape(-1)

    return sequence


def sample_gru_replay_batch(replay_buffer, batch_size, history_len, device):
    """
    Sample transitions with previous (obs, action, reward) histories.

    Each sampled transition i returns:
      histories      = [(s, a, r)_{i-K}, ..., (s, a, r)_{i-1}]
      next_histories = [(s, a, r)_{i-K+1}, ..., (s, a, r)_i]

    Missing history at the beginning of training or before the current episode
    is left as all-zero padding. The terminal flag at i is allowed, because SAC
    masks the bootstrap target.
    """

    size = _stored_replay_size(replay_buffer)
    history_len = int(history_len)
    batch_size = int(batch_size)

    if history_len < 1:
        raise ValueError(f"history_len must be >= 1, got {history_len}")
    if size < 1:
        raise ValueError("Need at least one transition for GRU sampling.")

    capacity = replay_buffer.buffer_size
    n_envs = replay_buffer.n_envs
    oldest_idx = replay_buffer.pos if replay_buffer.full else 0
    history_dim = (
        int(np.prod(replay_buffer.observations.shape[2:]))
        + int(np.prod(replay_buffer.actions.shape[2:]))
        + 1
    )

    histories = []
    next_histories = []
    observations = []
    next_observations = []
    actions = []
    rewards = []
    dones = []

    while len(observations) < batch_size:
        logical_end_idx = np.random.randint(0, size)
        end_idx = _logical_to_physical(
            logical_end_idx,
            oldest_idx,
            capacity,
            replay_buffer.full,
        )

        env_idx = np.random.randint(0, n_envs)
        history_indices = [
            logical_end_idx - history_len + offset
            for offset in range(history_len)
        ]
        next_history_indices = [
            logical_end_idx - history_len + 1 + offset
            for offset in range(history_len)
        ]

        histories.append(
            _padded_transition_history_features(
                replay_buffer,
                history_indices,
                env_idx,
                oldest_idx,
                history_dim,
            )
        )
        next_histories.append(
            _padded_transition_history_features(
                replay_buffer,
                next_history_indices,
                env_idx,
                oldest_idx,
                history_dim,
            )
        )
        observations.append(replay_buffer.observations[end_idx, env_idx])
        next_observations.append(replay_buffer.next_observations[end_idx, env_idx])
        actions.append(replay_buffer.actions[end_idx, env_idx])
        rewards.append(replay_buffer.rewards[end_idx, env_idx])
        dones.append(replay_buffer.dones[end_idx, env_idx])

    return GRUReplayBatch(
        observations=_as_torch(np.stack(observations, axis=0), device),
        actions=_as_torch(np.stack(actions, axis=0), device),
        next_observations=_as_torch(np.stack(next_observations, axis=0), device),
        dones=_as_torch(np.asarray(dones, dtype=np.float32).reshape(-1, 1), device),
        rewards=_as_torch(np.asarray(rewards, dtype=np.float32).reshape(-1, 1), device),
        histories=_as_torch(np.stack(histories, axis=0), device),
        next_histories=_as_torch(np.stack(next_histories, axis=0), device),
    )


def sample_state_gru_replay_batch(replay_buffer, batch_size, sequence_len, device):
    """
    Sample SAC transitions plus fixed-length state windows:
      state_sequences      = [x_{t-K+1}, ..., x_t]
      next_state_sequences = [x_{t-K+2}, ..., x_t, x_{t+1}]

    Missing states before the buffer start or current episode are all-zero
    padded on the left.
    """

    size = _stored_replay_size(replay_buffer)
    sequence_len = int(sequence_len)
    batch_size = int(batch_size)

    if sequence_len < 1:
        raise ValueError(f"sequence_len must be >= 1, got {sequence_len}")
    if size < 1:
        raise ValueError("Need at least one transition for state-GRU sampling.")

    capacity = replay_buffer.buffer_size
    n_envs = replay_buffer.n_envs
    oldest_idx = replay_buffer.pos if replay_buffer.full else 0
    obs_dim = int(np.prod(replay_buffer.observations.shape[2:]))

    state_sequences = []
    next_state_sequences = []
    observations = []
    next_observations = []
    actions = []
    rewards = []
    dones = []

    while len(observations) < batch_size:
        logical_end_idx = np.random.randint(0, size)
        end_idx = _logical_to_physical(
            logical_end_idx,
            oldest_idx,
            capacity,
            replay_buffer.full,
        )
        env_idx = np.random.randint(0, n_envs)

        state_indices = [
            logical_end_idx - sequence_len + 1 + offset
            for offset in range(sequence_len)
        ]
        next_state_indices = [
            logical_end_idx - sequence_len + 2 + offset
            for offset in range(sequence_len)
        ]

        state_sequences.append(
            _padded_state_sequence(
                replay_buffer=replay_buffer,
                logical_obs_indices=state_indices,
                env_idx=env_idx,
                oldest_idx=oldest_idx,
                obs_dim=obs_dim,
            )
        )
        next_state_sequences.append(
            _padded_state_sequence(
                replay_buffer=replay_buffer,
                logical_obs_indices=next_state_indices,
                env_idx=env_idx,
                oldest_idx=oldest_idx,
                obs_dim=obs_dim,
                next_logical_idx=logical_end_idx + 1,
            )
        )
        observations.append(replay_buffer.observations[end_idx, env_idx])
        next_observations.append(replay_buffer.next_observations[end_idx, env_idx])
        actions.append(replay_buffer.actions[end_idx, env_idx])
        rewards.append(replay_buffer.rewards[end_idx, env_idx])
        dones.append(replay_buffer.dones[end_idx, env_idx])

    return StateGRUReplayBatch(
        observations=_as_torch(np.stack(observations, axis=0), device),
        actions=_as_torch(np.stack(actions, axis=0), device),
        next_observations=_as_torch(np.stack(next_observations, axis=0), device),
        dones=_as_torch(np.asarray(dones, dtype=np.float32).reshape(-1, 1), device),
        rewards=_as_torch(np.asarray(rewards, dtype=np.float32).reshape(-1, 1), device),
        state_sequences=_as_torch(np.stack(state_sequences, axis=0), device),
        next_state_sequences=_as_torch(np.stack(next_state_sequences, axis=0), device),
    )


if __name__ == "__main__":
    agent = GRUMLPAgent(obs_dim=4, act_dim=4)

    x = torch.randn(8, 4)
    history = torch.randn(8, 20, 4 + 4 + 1)
    mean, log_std, h = agent(x, history=history, return_hidden=True)
    print(mean.shape)      # torch.Size([8, 4])
    print(log_std.shape)   # torch.Size([8, 4])
    print(h.shape)         # torch.Size([1, 8, 64])

    critic = RecurrentSoftQNetwork(obs_dim=4, act_dim=4)
    q = critic(x, torch.randn(8, 4), history=history)
    print(q.shape)         # torch.Size([8, 1])

    dirname = "runs/test_gru_mlp"
    agent.save(dirname)
