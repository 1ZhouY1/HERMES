import math
import os
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shared_arch import shared


class PairMemory:
    """
    FIFO memory for state-action pairs.

    It stores:
        states:  (capacity, obs_dim)
        actions: (capacity, action_dim)
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.dtype = dtype

        self.states = torch.zeros(
            self.capacity,
            self.obs_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.actions = torch.zeros(
            self.capacity,
            self.action_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add_pair(self, state, action, reward=None) -> None:
        state = torch.as_tensor(state, dtype=self.dtype, device=self.device).reshape(-1)
        action = torch.as_tensor(action, dtype=self.dtype, device=self.device).reshape(-1)

        assert state.shape == (self.obs_dim,), (
            f"state should be {(self.obs_dim,)}, got {tuple(state.shape)}"
        )
        assert action.shape == (self.action_dim,), (
            f"action should be {(self.action_dim,)}, got {tuple(action.shape)}"
        )

        self.states[self.ptr].copy_(state)
        self.actions[self.ptr].copy_(action)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def current(self, device: Optional[Union[str, torch.device]] = None) -> Dict[str, torch.Tensor]:
        assert self.size > 0, "PairMemory is empty."
        out_device = self.device if device is None else torch.device(device)

        if self.size < self.capacity:
            states = self.states[: self.size]
            actions = self.actions[: self.size]
        else:
            states = torch.cat([self.states[self.ptr :], self.states[: self.ptr]], dim=0)
            actions = torch.cat([self.actions[self.ptr :], self.actions[: self.ptr]], dim=0)

        return {
            "states": states.to(out_device),
            "actions": actions.to(out_device),
        }

    def save(self, dirname: str, filename: str = "pair_memory.pt") -> None:
        os.makedirs(dirname, exist_ok=True)
        torch.save(
            {
                "capacity": self.capacity,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "dtype": self.dtype,
                "ptr": self.ptr,
                "size": self.size,
                "states": self.states.detach().cpu(),
                "actions": self.actions.detach().cpu(),
            },
            os.path.join(dirname, filename),
        )


class RewardReplacingPairMemory(PairMemory):
    """
    Pair memory that keeps better-reward actions for similar states.

    When memory is full:
        1. compute distance between the new state and all stored states
        2. keep the nearest `replace_neighbors` candidates
        3. replace the lowest-reward candidate only if its reward is lower
           than the new transition reward
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        replace_neighbors: int = 50,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            capacity=capacity,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            dtype=dtype,
        )
        self.replace_neighbors = int(replace_neighbors)
        self.rewards = torch.zeros(
            self.capacity,
            1,
            device=self.device,
            dtype=self.dtype,
        )
        self.num_replacements = 0
        self.num_skips = 0

    def add_pair(self, state, action, reward=None) -> None:
        assert reward is not None, "RewardReplacingPairMemory requires reward."

        state = torch.as_tensor(state, dtype=self.dtype, device=self.device).reshape(-1)
        action = torch.as_tensor(action, dtype=self.dtype, device=self.device).reshape(-1)
        reward = torch.as_tensor(reward, dtype=self.dtype, device=self.device).reshape(-1)

        assert state.shape == (self.obs_dim,), (
            f"state should be {(self.obs_dim,)}, got {tuple(state.shape)}"
        )
        assert action.shape == (self.action_dim,), (
            f"action should be {(self.action_dim,)}, got {tuple(action.shape)}"
        )
        assert reward.numel() == 1, f"reward should have one value, got {tuple(reward.shape)}"

        if self.size < self.capacity:
            self.states[self.ptr].copy_(state)
            self.actions[self.ptr].copy_(action)
            self.rewards[self.ptr].copy_(reward.reshape(1))
            self.ptr = (self.ptr + 1) % self.capacity
            self.size += 1
            return

        distances = torch.norm(self.states - state.unsqueeze(0), dim=-1)
        k = min(max(self.replace_neighbors, 1), self.size)
        neighbor_indices = torch.topk(distances, k=k, largest=False).indices

        neighbor_rewards = self.rewards[neighbor_indices, 0]
        lower_reward_mask = neighbor_rewards < reward.item()
        if not bool(lower_reward_mask.any()):
            self.num_skips += 1
            return

        lower_reward_indices = neighbor_indices[lower_reward_mask]
        lower_rewards = self.rewards[lower_reward_indices, 0]
        replace_index = lower_reward_indices[torch.argmin(lower_rewards)].item()

        self.states[replace_index].copy_(state)
        self.actions[replace_index].copy_(action)
        self.rewards[replace_index].copy_(reward.reshape(1))
        self.num_replacements += 1

    def current(self, device: Optional[Union[str, torch.device]] = None) -> Dict[str, torch.Tensor]:
        assert self.size > 0, "RewardReplacingPairMemory is empty."
        out_device = self.device if device is None else torch.device(device)
        return {
            "states": self.states[: self.size].to(out_device),
            "actions": self.actions[: self.size].to(out_device),
            "rewards": self.rewards[: self.size].to(out_device),
        }

    def save(self, dirname: str, filename: str = "reward_replacing_pair_memory.pt") -> None:
        os.makedirs(dirname, exist_ok=True)
        torch.save(
            {
                "capacity": self.capacity,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "replace_neighbors": self.replace_neighbors,
                "dtype": self.dtype,
                "ptr": self.ptr,
                "size": self.size,
                "num_replacements": self.num_replacements,
                "num_skips": self.num_skips,
                "states": self.states.detach().cpu(),
                "actions": self.actions.detach().cpu(),
                "rewards": self.rewards.detach().cpu(),
            },
            os.path.join(dirname, filename),
        )


class RARLAgent(nn.Module):
    """
    Pair-memory RARL actor for SAC.

    For current state st:
        Q  = Wq(st)
        K  = Wk(memory_states)
        V  = Wv(memory_actions)
        zt = softmax(Q K^T / sqrt(att_dim)) V
        mean, log_std = Policy(Hstack(st, zt))
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        memory: Optional[PairMemory] = None,
        att_dim: int = 128,
        value_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.att_dim = int(att_dim)
        self.value_dim = int(value_dim)
        self.memory = memory

        self.q_proj = nn.Linear(self.obs_dim, self.att_dim)
        self.k_proj = nn.Linear(self.obs_dim, self.att_dim)
        self.v_proj = nn.Linear(self.act_dim, self.value_dim)

        self.policy = shared(self.obs_dim + self.value_dim)
        self.fc_mean = nn.Linear(256, self.act_dim)
        self.fc_logstd = nn.Linear(256, self.act_dim)

    def retrieve_context(
        self,
        obs: torch.Tensor,
        memory_batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        single_input = obs.dim() == 1
        if single_input:
            obs = obs.unsqueeze(0)

        if memory_batch is None:
            assert self.memory is not None, (
                "memory_batch is None and this RARLAgent has no attached PairMemory."
            )
            memory_batch = self.memory.current(device=obs.device)

        memory_states = memory_batch["states"].to(obs.device)
        memory_actions = memory_batch["actions"].to(obs.device)

        q = self.q_proj(obs)
        k = self.k_proj(memory_states)
        v = self.v_proj(memory_actions)

        attn_logits = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(self.att_dim)
        attn_weights = F.softmax(attn_logits, dim=-1)
        z_t = torch.matmul(attn_weights, v)

        return z_t.squeeze(0) if single_input else z_t

    def forward(
        self,
        x: torch.Tensor,
        memory_batch: Optional[Dict[str, torch.Tensor]] = None,
        **unused_kwargs,
    ):
        single_input = x.dim() == 1
        x_in = x.unsqueeze(0) if single_input else x

        z_t = self.retrieve_context(x_in, memory_batch=memory_batch)
        policy_input = torch.cat([x_in, z_t], dim=-1)
        hidden = self.policy(policy_input)
        mean = self.fc_mean(hidden)
        log_std = self.fc_logstd(hidden)

        if single_input:
            mean = mean.squeeze(0)
            log_std = log_std.squeeze(0)
        return mean, log_std

    def save(self, dirname: str, save_memory: bool = False) -> None:
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, os.path.join(dirname, "model.pt"))
        if save_memory and self.memory is not None:
            self.memory.save(dirname)
