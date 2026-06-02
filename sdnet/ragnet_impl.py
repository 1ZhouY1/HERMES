import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
import numpy as np
from typing import Union, Optional
from dataclasses import dataclass

class Identity(nn.Module):
    def __call__(self, args, **kwargs):
        return args

@dataclass
class TrajectoryBatch:
    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
    ):
        self.states = states
        self.actions = actions
        self.rewards = rewards

class LongTermMemory:
    """
    Long-term memory module for previous tasks. It stores multiple trajectories with l timesteps.
    """

    def __init__(
        self, 
        capacity: int,
        traj_len: int,
        obs_dim: int,
        action_dim: int,
        reward_dim: int,
        device: Union[torch.device, str],
        dtype: torch.dtype = torch.float32,
    ):
        self.capacity = capacity
        self.traj_len = traj_len
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.device = device
        self.dtype = dtype

        # Initialize memory buffers for observations, actions, rewards, and done flags
        self.states = torch.zeros((capacity, traj_len, obs_dim), device=device, dtype=dtype)
        self.actions = torch.zeros((capacity, traj_len, action_dim), device=device, dtype=dtype)
        self.rewards = torch.zeros((capacity, traj_len, reward_dim), device=device, dtype=dtype)
        self.dones = torch.zeros((capacity, traj_len), device=device, dtype=torch.bool)

        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def clear(self) -> None:
        self.ptr = 0
        self.size = 0
    
    def add_trajectory(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
    ):
        """
        Add one complete fixed-length trajectory.
        Args:
            states: (traj_len, obs_dim)
            actions: (traj_len, action_dim)
            rewards: (traj_len, reward_dim)
        """

        assert states.shape == (self.traj_len, self.obs_dim), \
            f"states should be {(self.traj_len, self.obs_dim)}, but got {states.shape}"
        assert actions.shape == (self.traj_len, self.action_dim), \
            f"actions should be {(self.traj_len, self.action_dim)}, but got {actions.shape}"
        assert rewards.shape == (self.traj_len, self.reward_dim), \
            f"rewards should be {(self.traj_len, self.reward_dim)}, but got {rewards.shape}"

        states = torch.as_tensor(states, device=self.device, dtype=self.dtype)
        actions = torch.as_tensor(actions, device=self.device, dtype=self.dtype)
        rewards = torch.as_tensor(rewards, device=self.device, dtype=self.dtype)

        self.states[self.ptr].copy_(states)
        self.actions[self.ptr].copy_(actions)
        self.rewards[self.ptr].copy_(rewards)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_trajectories(
        self,
        n_traj: int,
        device: Optional[Union[torch.device, str]] = None,
    ) -> TrajectoryBatch:
        device = torch.device(device)
        """
        Sample n_traj trajectories from memory.
        """
        if n_traj > self.size:
            n_traj = self.size
        indices = torch.randperm(self.size, device=self.device)[:n_traj]

        batch_states = self.states[indices].to(device)
        batch_actions = self.actions[indices].to(device)
        batch_rewards = self.rewards[indices].to(device)

        return TrajectoryBatch(
            states=batch_states,
            actions=batch_actions,
            rewards=batch_rewards,
        )
    def get_trajectory(self, index: int) -> TrajectoryBatch:
        """
        Get one trajectory by index.
        """
        if index >= self.size:
            raise IndexError(f"Index {index} out of range for memory size {self.size}")
        
        traj_states = self.states[index]
        traj_actions = self.actions[index]
        traj_rewards = self.rewards[index]

        return TrajectoryBatch(
            states=traj_states,
            actions=traj_actions,
            rewards=traj_rewards,
        )

class Encoder(nn.Module):
    """
    Encoder module for the current state xt.
    Input: 
        x: (batch_size, obs_dim) or (batch_size, seq_len, obs_dim)
    
    Returns:
        h_n: the hidden state, h_n = Encoder(x)
    """
    def __init__(self, obs_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(obs_dim, hidden_dim, num_layers, batch_first=True)
    def forward(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x[:, None, :]
        _, h_n = self.gru(x)
        return h_n[-1]
    

class Summarizer(nn.Module):
    """
    Two-head RNN summarizer.

    For the i-th memory trajectory:
        forward summarizer forward_h[i, j] = f_forward(x_1, ..., x_j)
        backward summarizer backward_b[i, j] = f_backward(x_l, ..., x_j)
    """

    def __init__(self, obs_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forward_rnn = nn.RNN(obs_dim, hidden_dim, num_layers, batch_first=True)
        self.backward_rnn = nn.RNN(obs_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, memory_obs: torch.Tensor):
        """
        memory_obs: (n_traj, l, obs_dim) or (batch_size, n_traj, l, obs_dim)
        """
        added_batch_dim = False
        if memory_obs.dim() == 3:
            memory_obs = memory_obs[None, ...]
            added_batch_dim = True
        assert memory_obs.dim() == 4, f"memory_obs should be (batch_size, n_traj, l, obs_dim) or (n_traj, l, obs_dim), but got {memory_obs.shape}"

        B, N, L, D = memory_obs.shape
        flat_obs = memory_obs.reshape(B * N, L, D)

        forward_h, _ = self.forward_rnn(flat_obs)
        reverse_obs = torch.flip(flat_obs, dims=[1])
        reverse_b, _ = self.backward_rnn(reverse_obs)
        backward_b = torch.flip(reverse_b, dims=[1])

        forward_h = forward_h.reshape(B, N, L, self.hidden_dim)
        backward_b = backward_b.reshape(B, N, L, self.hidden_dim)


        if added_batch_dim:
            forward_h = forward_h[0]
            backward_b = backward_b[0]
        
        return forward_h, backward_b

class RetrieveAttentionHead(nn.Module):
    """
    Retrieval attention head.
    """

    def __init__(
        self, 
        state_dim: int,
        summary_dim: int,
        context_dim: int,
        att_dim: int,
        proj_bias: bool = True,
        att_heads_init: object = Identity()
    ):
        super().__init__()
        self.state_dim = state_dim
        self.summary_dim = summary_dim
        self.context_dim = context_dim
        self.att_dim = att_dim
        self.att_temp = math.sqrt(self.att_dim)

        self.wq = att_heads_init(nn.Linear(state_dim, att_dim, bias=proj_bias))
        self.wk = att_heads_init(nn.Linear(summary_dim, att_dim, bias=proj_bias))
        self.wv = att_heads_init(nn.Linear(summary_dim, context_dim, bias=proj_bias))

        
