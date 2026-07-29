import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HybridRARLTrajectoryBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "states": self.states,
            "obs": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
        }


class ReservoirTrajectoryMemory:
    """
    Per-task trajectory memory for HybridRARL.

    It stores at most `capacity` fixed-length trajectories. Every complete
    trajectory seen by this memory has equal probability of being retained,
    because replacement follows standard reservoir sampling.
    """

    def __init__(
        self,
        capacity: int = 100,
        traj_len: int = 20,
        obs_dim: int = 0,
        action_dim: int = 0,
        reward_dim: int = 1,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        assert capacity > 0
        assert traj_len > 0
        assert obs_dim > 0
        assert action_dim > 0
        assert reward_dim > 0

        self.capacity = int(capacity)
        self.traj_len = int(traj_len)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.reward_dim = int(reward_dim)
        self.device = torch.device(device)
        self.dtype = dtype

        self.states = torch.zeros(
            self.capacity,
            self.traj_len,
            self.obs_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.actions = torch.zeros(
            self.capacity,
            self.traj_len,
            self.action_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.rewards = torch.zeros(
            self.capacity,
            self.traj_len,
            self.reward_dim,
            dtype=self.dtype,
            device=self.device,
        )

        self.size = 0
        self.num_seen = 0
        self.num_replacements = 0

    def __len__(self) -> int:
        return self.size

    def clear(self) -> None:
        self.size = 0
        self.num_seen = 0
        self.num_replacements = 0

    def add_trajectory(
        self,
        states: Union[torch.Tensor, List],
        actions: Union[torch.Tensor, List],
        rewards: Optional[Union[torch.Tensor, List]] = None,
    ) -> int:
        states = torch.as_tensor(states, dtype=self.dtype, device=self.device)
        actions = torch.as_tensor(actions, dtype=self.dtype, device=self.device)

        if rewards is None:
            rewards = torch.zeros(
                self.traj_len,
                self.reward_dim,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            rewards = torch.as_tensor(rewards, dtype=self.dtype, device=self.device)
            if rewards.dim() == 1:
                rewards = rewards[:, None]

        assert states.shape == (self.traj_len, self.obs_dim), (
            f"states should be {(self.traj_len, self.obs_dim)}, got {tuple(states.shape)}"
        )
        assert actions.shape == (self.traj_len, self.action_dim), (
            f"actions should be {(self.traj_len, self.action_dim)}, got {tuple(actions.shape)}"
        )
        assert rewards.shape == (self.traj_len, self.reward_dim), (
            f"rewards should be {(self.traj_len, self.reward_dim)}, got {tuple(rewards.shape)}"
        )

        self.num_seen += 1
        if self.size < self.capacity:
            index = self.size
            self.size += 1
        else:
            index = random.randrange(self.num_seen)
            if index >= self.capacity:
                return -1
            self.num_replacements += 1

        self.states[index].copy_(states)
        self.actions[index].copy_(actions)
        self.rewards[index].copy_(rewards)
        return index

    def sample_trajectories(
        self,
        n_traj: int,
        device: Optional[Union[str, torch.device]] = None,
        replace: bool = False,
    ) -> HybridRARLTrajectoryBatch:
        assert self.size > 0, "ReservoirTrajectoryMemory is empty."
        n_traj = int(n_traj)
        assert n_traj > 0

        out_device = self.device if device is None else torch.device(device)
        if replace:
            indices = torch.randint(0, self.size, size=(n_traj,), device=self.device)
        else:
            n_traj = min(n_traj, self.size)
            indices = torch.randperm(self.size, device=self.device)[:n_traj]

        return HybridRARLTrajectoryBatch(
            states=self.states[indices].to(out_device),
            actions=self.actions[indices].to(out_device),
            rewards=self.rewards[indices].to(out_device),
        )

    def current(self, device: Optional[Union[str, torch.device]] = None) -> Dict[str, torch.Tensor]:
        assert self.size > 0, "ReservoirTrajectoryMemory is empty."
        out_device = self.device if device is None else torch.device(device)
        states = self.states[: self.size].to(out_device)
        actions = self.actions[: self.size].to(out_device)
        rewards = self.rewards[: self.size].to(out_device)
        return {
            "states": states,
            "obs": states,
            "actions": actions,
            "rewards": rewards,
        }

    def save(self, dirname: str, filename: str = "hybridrarl_memory.pt") -> None:
        os.makedirs(dirname, exist_ok=True)
        torch.save(
            {
                "capacity": self.capacity,
                "traj_len": self.traj_len,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "reward_dim": self.reward_dim,
                "dtype": self.dtype,
                "size": self.size,
                "num_seen": self.num_seen,
                "num_replacements": self.num_replacements,
                "states": self.states.detach().cpu(),
                "actions": self.actions.detach().cpu(),
                "rewards": self.rewards.detach().cpu(),
            },
            os.path.join(dirname, filename),
        )

    @staticmethod
    def load(
        dirname: Union[str, Path],
        filename: str = "hybridrarl_memory.pt",
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> "ReservoirTrajectoryMemory":
        data = torch.load(Path(dirname) / filename, map_location=map_location)
        memory = ReservoirTrajectoryMemory(
            capacity=data["capacity"],
            traj_len=data["traj_len"],
            obs_dim=data["obs_dim"],
            action_dim=data["action_dim"],
            reward_dim=data["reward_dim"],
            device=map_location or "cpu",
            dtype=data.get("dtype", torch.float32),
        )
        memory.size = int(data["size"])
        memory.num_seen = int(data.get("num_seen", memory.size))
        memory.num_replacements = int(data.get("num_replacements", 0))
        memory.states.copy_(data["states"].to(memory.device))
        memory.actions.copy_(data["actions"].to(memory.device))
        memory.rewards.copy_(data["rewards"].to(memory.device))
        return memory


class StateActionTrajectoryCache:
    """
    Small online cache. Once it has `traj_len` state-action pairs, it writes one
    complete trajectory into memory and clears itself.
    """

    def __init__(self, traj_len: int = 20):
        self.traj_len = int(traj_len)
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.states)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

    def append(
        self,
        state: Union[torch.Tensor, List],
        action: Union[torch.Tensor, List],
        reward: Optional[Union[torch.Tensor, List]] = None,
        memory: Optional[ReservoirTrajectoryMemory] = None,
    ) -> bool:
        self.states.append(torch.as_tensor(state, dtype=torch.float32).detach().cpu())
        self.actions.append(torch.as_tensor(action, dtype=torch.float32).detach().cpu())
        if reward is None:
            reward = torch.zeros(1, dtype=torch.float32)
        self.rewards.append(torch.as_tensor(reward, dtype=torch.float32).reshape(-1).detach().cpu())

        if len(self.states) < self.traj_len:
            return False

        if memory is not None:
            memory.add_trajectory(
                states=torch.stack(self.states, dim=0),
                actions=torch.stack(self.actions, dim=0),
                rewards=torch.stack(self.rewards, dim=0),
            )
        self.clear()
        return True


def build_policy_trunk(input_dim: int, hidden_dim: int, num_layers: int = 4) -> nn.Sequential:
    assert num_layers >= 2
    layers: List[nn.Module] = []
    in_dim = int(input_dim)
    for _ in range(int(num_layers)):
        layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
        in_dim = int(hidden_dim)
    return nn.Sequential(*layers)


class HybridRARLAttentionHead(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        traj_hidden_dim: int,
        att_dim: int,
        value_dim: int,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.traj_hidden_dim = int(traj_hidden_dim)
        self.att_dim = int(att_dim)
        self.value_dim = int(value_dim)

        self.q_proj = nn.Linear(self.obs_dim, self.att_dim)
        self.k_proj = nn.Linear(self.traj_hidden_dim, self.att_dim)
        self.v_proj = nn.Linear(self.traj_hidden_dim, self.value_dim)

    def forward(self, st: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert st.dim() == 2, f"st should be (B,obs_dim), got {tuple(st.shape)}"
        assert h.dim() == 3, f"h should be (B,R,H), got {tuple(h.shape)}"

        q = self.q_proj(st).unsqueeze(1)
        k = self.k_proj(h)
        v = self.v_proj(h)
        logits = torch.matmul(q, k.transpose(1, 2)).squeeze(1)
        logits = logits / math.sqrt(self.att_dim)
        weights = F.softmax(logits, dim=-1)
        z_t = torch.matmul(weights.unsqueeze(1), v).squeeze(1)
        return z_t, weights, logits


class HybridRARLAgent(nn.Module):
    """
    HybridRARL actor used by SAC.

    Retrieval path:
        1. coarse retrieval compares current state st with the first state of
           each stored trajectory and keeps top-R trajectories
        2. a GRU encodes the full state-action sequence of each selected
           trajectory; its last hidden state is H
        3. Q(st), K(H), V(H) attention produces retrieved information z_t
        4. policy([st, z_t]) outputs SAC mean and logstd

    For transfer to a new task, call `copy_transfer_modules_from`. It copies the
    trajectory GRU and the first two policy Linear layers; the attention head,
    output heads, remaining policy layers, and memory stay task-specific.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        memory: Optional[ReservoirTrajectoryMemory] = None,
        memory_capacity: int = 100,
        traj_len: int = 20,
        top_r: int = 5,
        traj_hidden_dim: int = 128,
        att_dim: int = 128,
        value_dim: int = 128,
        policy_hidden_dim: int = 256,
        policy_layers: int = 4,
        coarse_metric: str = "cosine",
    ):
        super().__init__()
        assert coarse_metric in ["cosine", "l2"]

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.memory_capacity = int(memory_capacity)
        self.traj_len = int(traj_len)
        self.top_r = int(top_r)
        self.traj_hidden_dim = int(traj_hidden_dim)
        self.att_dim = int(att_dim)
        self.value_dim = int(value_dim)
        self.policy_hidden_dim = int(policy_hidden_dim)
        self.policy_layers = int(policy_layers)
        self.coarse_metric = coarse_metric
        self.reused_source = None

        self.memory = memory
        self.trajectory_rnn = nn.GRU(
            input_size=self.obs_dim + self.act_dim,
            hidden_size=self.traj_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.attention_head = HybridRARLAttentionHead(
            obs_dim=self.obs_dim,
            traj_hidden_dim=self.traj_hidden_dim,
            att_dim=self.att_dim,
            value_dim=self.value_dim,
        )
        self.policy = build_policy_trunk(
            input_dim=self.obs_dim + self.value_dim,
            hidden_dim=self.policy_hidden_dim,
            num_layers=self.policy_layers,
        )
        self.fc_mean = nn.Linear(self.policy_hidden_dim, self.act_dim)
        self.fc_logstd = nn.Linear(self.policy_hidden_dim, self.act_dim)

    def _memory_from_forward_arg(
        self,
        x: torch.Tensor,
        memory_batch: Optional[Union[HybridRARLTrajectoryBatch, Dict[str, torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if memory_batch is None:
            assert self.memory is not None, (
                "memory_batch is None and this HybridRARLAgent has no attached memory."
            )
            memory_batch = self.memory.current(device=x.device)

        if isinstance(memory_batch, HybridRARLTrajectoryBatch):
            states, actions = memory_batch.states, memory_batch.actions
        else:
            states = memory_batch.get("states", memory_batch.get("obs", None))
            actions = memory_batch.get("actions", None)

        assert states is not None and actions is not None
        return states.to(x.device), actions.to(x.device)

    def coarse_retrieve(
        self,
        st: torch.Tensor,
        memory_states: torch.Tensor,
        top_r: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        assert st.dim() == 2
        assert memory_states.dim() == 3
        first_states = memory_states[:, 0, :]
        r = min(int(top_r or self.top_r), first_states.shape[0])

        if self.coarse_metric == "cosine":
            st_norm = F.normalize(st, dim=-1)
            key_norm = F.normalize(first_states, dim=-1)
            scores = torch.matmul(st_norm, key_norm.transpose(0, 1))
            top_scores, top_indices = torch.topk(scores, k=r, dim=-1)
            distances = 1.0 - top_scores
        else:
            distances_all = torch.cdist(st, first_states)
            neg_distances, top_indices = torch.topk(-distances_all, k=r, dim=-1)
            distances = -neg_distances
            top_scores = neg_distances

        return {
            "top_indices": top_indices,
            "top_scores": top_scores,
            "top_distances": distances,
        }

    def _gather_selected_trajectories(
        self,
        memory_states: torch.Tensor,
        memory_actions: torch.Tensor,
        top_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, r = top_indices.shape
        _, length, obs_dim = memory_states.shape
        act_dim = memory_actions.shape[-1]

        states = memory_states.unsqueeze(0).expand(bsz, -1, -1, -1)
        actions = memory_actions.unsqueeze(0).expand(bsz, -1, -1, -1)
        state_index = top_indices[:, :, None, None].expand(bsz, r, length, obs_dim)
        action_index = top_indices[:, :, None, None].expand(bsz, r, length, act_dim)
        return (
            torch.gather(states, dim=1, index=state_index),
            torch.gather(actions, dim=1, index=action_index),
        )

    def encode_trajectories(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        assert states.dim() == 4
        assert actions.dim() == 4
        bsz, r, length, _ = states.shape
        seq = torch.cat([states, actions], dim=-1)
        flat_seq = seq.reshape(bsz * r, length, self.obs_dim + self.act_dim)
        _, h_n = self.trajectory_rnn(flat_seq)
        return h_n[-1].reshape(bsz, r, self.traj_hidden_dim)

    def retrieve_context(
        self,
        st: torch.Tensor,
        memory_batch: Optional[Union[HybridRARLTrajectoryBatch, Dict[str, torch.Tensor]]] = None,
        return_info: bool = False,
    ):
        single_input = st.dim() == 1
        st_in = st.unsqueeze(0) if single_input else st

        memory_states, memory_actions = self._memory_from_forward_arg(st_in, memory_batch)
        coarse_info = self.coarse_retrieve(st_in, memory_states, top_r=self.top_r)
        top_states, top_actions = self._gather_selected_trajectories(
            memory_states=memory_states,
            memory_actions=memory_actions,
            top_indices=coarse_info["top_indices"],
        )
        h = self.encode_trajectories(top_states, top_actions)
        z_t, attn_weights, attn_logits = self.attention_head(st_in, h)

        if single_input:
            z_out = z_t.squeeze(0)
        else:
            z_out = z_t

        if not return_info:
            return z_out

        info = {
            "zt": z_t,
            "coarse_top_indices": coarse_info["top_indices"],
            "coarse_top_scores": coarse_info["top_scores"],
            "coarse_top_distances": coarse_info["top_distances"],
            "top_states": top_states,
            "top_actions": top_actions,
            "trajectory_h": h,
            "attention_weights": attn_weights,
            "attention_logits": attn_logits,
        }
        return z_out, info

    def encode_state(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(
        self,
        x: torch.Tensor,
        memory_batch: Optional[Union[HybridRARLTrajectoryBatch, Dict[str, torch.Tensor]]] = None,
        return_info: bool = False,
        **unused_kwargs,
    ):
        single_input = x.dim() == 1
        x_in = x.unsqueeze(0) if single_input else x
        if return_info:
            z_t, info = self.retrieve_context(x_in, memory_batch=memory_batch, return_info=True)
        else:
            z_t = self.retrieve_context(x_in, memory_batch=memory_batch)
            info = None

        hidden = self.policy(torch.cat([x_in, z_t], dim=-1))
        mean = self.fc_mean(hidden)
        logstd = self.fc_logstd(hidden)

        if single_input:
            mean = mean.squeeze(0)
            logstd = logstd.squeeze(0)

        if return_info:
            return mean, logstd, info
        return mean, logstd

    def _policy_linear_layers(self) -> List[nn.Linear]:
        return [layer for layer in self.policy if isinstance(layer, nn.Linear)]

    def freeze_reused_modules(self) -> None:
        for param in self.trajectory_rnn.parameters():
            param.requires_grad_(False)
        for layer in self._policy_linear_layers()[:2]:
            for param in layer.parameters():
                param.requires_grad_(False)

    def copy_transfer_modules_from(self, source: nn.Module, freeze: bool = True) -> None:
        if not isinstance(source, HybridRARLAgent):
            raise TypeError(
                "HybridRARL transfer expects a previous HybridRARLAgent, "
                f"got {type(source)}"
            )

        self.trajectory_rnn.load_state_dict(source.trajectory_rnn.state_dict())

        source_layers = source._policy_linear_layers()
        target_layers = self._policy_linear_layers()
        for source_layer, target_layer in zip(source_layers[:2], target_layers[:2]):
            with torch.no_grad():
                target_layer.weight.copy_(source_layer.weight)
                target_layer.bias.copy_(source_layer.bias)

        self.reused_source = getattr(source, "reused_source", None)
        if freeze:
            self.freeze_reused_modules()

    def save(self, dirname: str, save_memory: bool = True) -> None:
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, os.path.join(dirname, "model.pt"))
        torch.save(
            {
                "trajectory_rnn": self.trajectory_rnn.state_dict(),
                "attention_head": self.attention_head.state_dict(),
                "policy": self.policy.state_dict(),
                "fc_mean": self.fc_mean.state_dict(),
                "fc_logstd": self.fc_logstd.state_dict(),
                "meta": {
                    "obs_dim": self.obs_dim,
                    "act_dim": self.act_dim,
                    "memory_capacity": self.memory_capacity,
                    "traj_len": self.traj_len,
                    "top_r": self.top_r,
                    "traj_hidden_dim": self.traj_hidden_dim,
                    "att_dim": self.att_dim,
                    "value_dim": self.value_dim,
                    "policy_hidden_dim": self.policy_hidden_dim,
                    "policy_layers": self.policy_layers,
                    "coarse_metric": self.coarse_metric,
                    "reused_source": self.reused_source,
                },
            },
            os.path.join(dirname, "hybridrarl_modules.pt"),
        )
        if save_memory and self.memory is not None:
            self.memory.save(dirname)
