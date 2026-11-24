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


class ReservoirStateMemory:
    """
    Per-task state/action memory for the HybridRARL random-state ablation.

    It stores at most `capacity` individual state/action pairs.  Every pair seen
    by this memory has equal probability of being retained, because replacement
    follows standard reservoir sampling.
    """

    def __init__(
        self,
        capacity: int = 2000,
        obs_dim: int = 0,
        action_dim: int = 0,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        assert capacity > 0
        assert obs_dim > 0
        assert action_dim > 0

        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.dtype = dtype

        self.states = torch.zeros(
            self.capacity,
            self.obs_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.actions = torch.zeros(
            self.capacity,
            self.action_dim,
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

    def add_pair(
        self,
        state: Union[torch.Tensor, List],
        action: Union[torch.Tensor, List],
    ) -> int:
        state = torch.as_tensor(state, dtype=self.dtype, device=self.device).reshape(-1)
        action = torch.as_tensor(action, dtype=self.dtype, device=self.device).reshape(-1)

        assert state.shape == (self.obs_dim,), (
            f"state should be {(self.obs_dim,)}, got {tuple(state.shape)}"
        )
        assert action.shape == (self.action_dim,), (
            f"action should be {(self.action_dim,)}, got {tuple(action.shape)}"
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

        self.states[index].copy_(state)
        self.actions[index].copy_(action)
        return index

    def sample_states(
        self,
        n_states: int,
        device: Optional[Union[str, torch.device]] = None,
        replace: bool = False,
    ) -> Dict[str, torch.Tensor]:
        assert self.size > 0, "ReservoirStateMemory is empty."
        n_states = int(n_states)
        assert n_states > 0

        out_device = self.device if device is None else torch.device(device)
        if replace:
            indices = torch.randint(0, self.size, size=(n_states,), device=self.device)
        else:
            n_states = min(n_states, self.size)
            indices = torch.randperm(self.size, device=self.device)[:n_states]

        states = self.states[indices].to(out_device)
        actions = self.actions[indices].to(out_device)
        return {
            "states": states,
            "obs": states,
            "actions": actions,
        }

    def current(self, device: Optional[Union[str, torch.device]] = None) -> Dict[str, torch.Tensor]:
        assert self.size > 0, "ReservoirStateMemory is empty."
        out_device = self.device if device is None else torch.device(device)
        states = self.states[: self.size].to(out_device)
        actions = self.actions[: self.size].to(out_device)
        return {
            "states": states,
            "obs": states,
            "actions": actions,
        }

    def save(self, dirname: str, filename: str = "hybridrarl_state_memory.pt") -> None:
        os.makedirs(dirname, exist_ok=True)
        torch.save(
            {
                "capacity": self.capacity,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "dtype": self.dtype,
                "size": self.size,
                "num_seen": self.num_seen,
                "num_replacements": self.num_replacements,
                "states": self.states.detach().cpu(),
                "actions": self.actions.detach().cpu(),
            },
            os.path.join(dirname, filename),
        )

    @staticmethod
    def load(
        dirname: Union[str, Path],
        filename: str = "hybridrarl_state_memory.pt",
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> "ReservoirStateMemory":
        data = torch.load(Path(dirname) / filename, map_location=map_location)
        memory = ReservoirStateMemory(
            capacity=data["capacity"],
            obs_dim=data["obs_dim"],
            action_dim=data["action_dim"],
            device=map_location or "cpu",
            dtype=data.get("dtype", torch.float32),
        )
        memory.size = int(data["size"])
        memory.num_seen = int(data.get("num_seen", memory.size))
        memory.num_replacements = int(data.get("num_replacements", 0))
        memory.states.copy_(data["states"].to(memory.device))
        memory.actions.copy_(data["actions"].to(memory.device))
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
        key_value_dim: Optional[int] = None,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.traj_hidden_dim = int(traj_hidden_dim)
        self.key_value_dim = int(key_value_dim or traj_hidden_dim)
        self.att_dim = int(att_dim)
        self.value_dim = int(value_dim)

        self.q_proj = nn.Linear(self.obs_dim, self.att_dim)
        self.k_proj = nn.Linear(self.key_value_dim, self.att_dim)
        self.v_proj = nn.Linear(self.key_value_dim, self.value_dim)

    def forward(self, st: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert st.dim() == 2, f"st should be (B,obs_dim), got {tuple(st.shape)}"
        assert h.dim() == 3, f"h should be (B,R,H), got {tuple(h.shape)}"
        key_value_dim = int(getattr(self, "key_value_dim", self.traj_hidden_dim))
        assert h.size(-1) == key_value_dim, (
            f"h last dim should be {key_value_dim}, got {h.size(-1)}"
        )

        q = self.q_proj(st).unsqueeze(1)
        k = self.k_proj(h)
        v = self.v_proj(h)
        logits = torch.matmul(q, k.transpose(1, 2)).squeeze(1)
        logits = logits / math.sqrt(self.att_dim)
        weights = F.softmax(logits, dim=-1)
        z_t = torch.matmul(weights.unsqueeze(1), v).squeeze(1)
        return z_t, weights, logits


class PrevUnitMeanAttentionHead(nn.Module):
    """
    Attention over previous task modules.

    Query: current state st
    Key:   previous-module means Phi
    Value: previous-module means Phi
    """

    def __init__(self, obs_dim: int, act_dim: int, att_dim: int):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.att_dim = int(att_dim)
        self.q_proj = nn.Linear(self.obs_dim, self.att_dim)
        self.k_proj = nn.Linear(self.act_dim, self.att_dim)

    def forward(self, st: torch.Tensor, phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert st.dim() == 2, f"st should be (B,obs_dim), got {tuple(st.shape)}"
        assert phi.dim() == 3, f"phi should be (B,P,act_dim), got {tuple(phi.shape)}"
        assert phi.size(-1) == self.act_dim

        q = self.q_proj(st).unsqueeze(1)
        k = self.k_proj(phi)
        logits = torch.matmul(q, k.transpose(1, 2)).squeeze(1)
        logits = logits / math.sqrt(self.att_dim)
        weights = F.softmax(logits, dim=-1)
        mu_o = torch.matmul(weights.unsqueeze(1), phi).squeeze(1)
        return mu_o, weights, logits


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
        random_coarse_retrieval: bool = False,
        random_state_retrieval: bool = False,
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
        self.random_coarse_retrieval = bool(random_coarse_retrieval)
        self.random_state_retrieval = bool(random_state_retrieval)
        self.reused_source = None

        self.memory = memory
        self.trajectory_rnn = nn.GRU(
            input_size=self.obs_dim + self.act_dim,
            hidden_size=self.traj_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.register_buffer("trajectory_feature_mean", torch.zeros(self.obs_dim + self.act_dim))
        self.register_buffer("trajectory_feature_std", torch.ones(self.obs_dim + self.act_dim))
        self.pretrained_encoder_info = None
        self.attention_head = HybridRARLAttentionHead(
            obs_dim=self.obs_dim,
            traj_hidden_dim=self.traj_hidden_dim,
            att_dim=self.att_dim,
            value_dim=self.value_dim,
            key_value_dim=(
                self.obs_dim if self.random_state_retrieval else self.traj_hidden_dim
            ),
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
            if isinstance(self.memory, dict):
                memory_batch = self.memory
            else:
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
            distances_all = 1.0 - scores
            if getattr(self, "random_coarse_retrieval", False):
                random_scores = torch.rand(
                    st.shape[0],
                    first_states.shape[0],
                    device=st.device,
                )
                _, top_indices = torch.topk(random_scores, k=r, dim=-1)
                top_scores = torch.gather(scores, dim=-1, index=top_indices)
                distances = torch.gather(distances_all, dim=-1, index=top_indices)
            else:
                top_scores, top_indices = torch.topk(scores, k=r, dim=-1)
                distances = 1.0 - top_scores
        else:
            distances_all = torch.cdist(st, first_states)
            scores = -distances_all
            if getattr(self, "random_coarse_retrieval", False):
                random_scores = torch.rand(
                    st.shape[0],
                    first_states.shape[0],
                    device=st.device,
                )
                _, top_indices = torch.topk(random_scores, k=r, dim=-1)
                distances = torch.gather(distances_all, dim=-1, index=top_indices)
                top_scores = torch.gather(scores, dim=-1, index=top_indices)
            else:
                neg_distances, top_indices = torch.topk(-distances_all, k=r, dim=-1)
                distances = -neg_distances
                top_scores = neg_distances

        return {
            "top_indices": top_indices,
            "top_scores": top_scores,
            "top_distances": distances,
            "random_coarse_retrieval": torch.full(
                (st.shape[0],),
                float(getattr(self, "random_coarse_retrieval", False)),
                dtype=st.dtype,
                device=st.device,
            ),
        }

    def random_state_retrieve(
        self,
        st: torch.Tensor,
        memory_states: torch.Tensor,
        memory_actions: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Randomly select flat state positions from replay/state memory.

        This ablation bypasses trajectory-level coarse retrieval and the GRU
        trajectory summarizer.  The selected state vectors are fed directly to
        the attention head as keys/values.
        """
        assert st.dim() == 2
        assert memory_states.dim() in (2, 3)
        flat_states = memory_states.reshape(-1, self.obs_dim)
        total_states = flat_states.shape[0]
        k = min(int(top_k or self.top_r), total_states)
        random_scores = torch.rand(
            st.shape[0],
            total_states,
            device=st.device,
        )
        _, state_indices = torch.topk(random_scores, k=k, dim=-1)

        state_bank = flat_states.unsqueeze(0).expand(st.shape[0], -1, -1)
        selected_states = torch.gather(
            state_bank,
            dim=1,
            index=state_indices[:, :, None].expand(st.shape[0], k, self.obs_dim),
        )

        selected_actions = None
        if memory_actions is not None:
            assert memory_actions.dim() in (2, 3)
            flat_actions = memory_actions.reshape(-1, self.act_dim)
            action_bank = flat_actions.unsqueeze(0).expand(st.shape[0], -1, -1)
            selected_actions = torch.gather(
                action_bank,
                dim=1,
                index=state_indices[:, :, None].expand(st.shape[0], k, self.act_dim),
            )

        if self.coarse_metric == "cosine":
            st_norm = F.normalize(st, dim=-1).unsqueeze(1)
            selected_norm = F.normalize(selected_states, dim=-1)
            selected_scores = torch.sum(st_norm * selected_norm, dim=-1)
            selected_distances = 1.0 - selected_scores
        else:
            selected_distances = torch.norm(
                st.unsqueeze(1) - selected_states,
                p=2,
                dim=-1,
            )
            selected_scores = -selected_distances

        return {
            "state_indices": state_indices,
            "selected_states": selected_states,
            "selected_actions": selected_actions,
            "selected_scores": selected_scores,
            "selected_distances": selected_distances,
            "random_state_retrieval": torch.ones(
                st.shape[0],
                dtype=st.dtype,
                device=st.device,
            ),
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
        if getattr(self, "pretrained_encoder_info", None) is not None:
            flat_seq = (flat_seq - self.trajectory_feature_mean) / self.trajectory_feature_std
        _, h_n = self.trajectory_rnn(flat_seq)
        return h_n[-1].reshape(bsz, r, self.traj_hidden_dim)

    def load_pretrained_encoder(self, checkpoint_path: Union[str, Path],
                                domain: str, task_id: int, mode: str) -> None:
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint["config"]
        expected = {"domain": domain, "input_dim": self.obs_dim + self.act_dim,
                    "hidden_dim": self.traj_hidden_dim, "trajectory_steps": self.traj_len,
                    "feature_order": "state_action"}
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"{checkpoint_path}: {key}={config.get(key)!r}; expected {value!r}")
        trained_tasks = [int(item) for item in config["task_ids"]]
        if task_id not in trained_tasks or (mode == "per-task" and trained_tasks != [task_id]):
            raise ValueError(f"{checkpoint_path}: task IDs {trained_tasks} do not match "
                             f"{mode} encoder for task {task_id}")
        if mode not in ("shared", "per-task"):
            raise ValueError(f"Unsupported pretrained encoder mode: {mode}")
        mean = torch.as_tensor(checkpoint["feature_mean"], dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(checkpoint["feature_std"], dtype=torch.float32).reshape(-1)
        if (mean.shape != self.trajectory_feature_mean.shape or
                std.shape != self.trajectory_feature_std.shape or
                not torch.isfinite(mean).all() or not torch.isfinite(std).all() or
                not torch.all(std > 0)):
            raise ValueError(f"{checkpoint_path}: invalid trajectory feature normalization")
        self.trajectory_rnn.load_state_dict(checkpoint["encoder_state_dict"])
        self.trajectory_feature_mean.copy_(mean.to(self.trajectory_feature_mean.device))
        self.trajectory_feature_std.copy_(std.to(self.trajectory_feature_std.device))
        for parameter in self.trajectory_rnn.parameters():
            parameter.requires_grad_(False)
        self.trajectory_rnn.eval()
        self.pretrained_encoder_info = {"path": str(checkpoint_path), "mode": mode,
                                        "domain": domain, "task_id": int(task_id)}

    def retrieve_context(
        self,
        st: torch.Tensor,
        memory_batch: Optional[Union[HybridRARLTrajectoryBatch, Dict[str, torch.Tensor]]] = None,
        return_info: bool = False,
    ):
        single_input = st.dim() == 1
        st_in = st.unsqueeze(0) if single_input else st

        memory_states, memory_actions = self._memory_from_forward_arg(st_in, memory_batch)
        if getattr(self, "random_state_retrieval", False):
            state_info = self.random_state_retrieve(
                st_in,
                memory_states,
                memory_actions=memory_actions,
                top_k=self.top_r,
            )
            top_states = state_info["selected_states"]
            top_actions = state_info["selected_actions"]
            h = top_states
            z_t, attn_weights, attn_logits = self.attention_head(st_in, h)
            coarse_info = {
                "top_indices": state_info["state_indices"],
                "top_scores": state_info["selected_scores"],
                "top_distances": state_info["selected_distances"],
            }
        else:
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
            "random_state_retrieval": torch.full(
                (st_in.shape[0],),
                float(getattr(self, "random_state_retrieval", False)),
                dtype=st_in.dtype,
                device=st_in.device,
            ),
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

        if getattr(self, "pretrained_encoder_info", None) is None:
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
                "trajectory_feature_mean": self.trajectory_feature_mean.detach().cpu(),
                "trajectory_feature_std": self.trajectory_feature_std.detach().cpu(),
                "attention_head": self.attention_head.state_dict(),
                "policy": self.policy.state_dict(),
                "fc_mean": self.fc_mean.state_dict(),
                "fc_logstd": self.fc_logstd.state_dict(),
                "meta": {
                    "obs_dim": self.obs_dim,
                    "act_dim": self.act_dim,
                    "memory_capacity": self.memory_capacity,
                    "state_memory_capacity": (
                        getattr(self.memory, "capacity", None)
                        if getattr(self, "random_state_retrieval", False)
                        else None
                    ),
                    "traj_len": self.traj_len,
                    "top_r": self.top_r,
                    "traj_hidden_dim": self.traj_hidden_dim,
                    "att_dim": self.att_dim,
                    "value_dim": self.value_dim,
                    "policy_hidden_dim": self.policy_hidden_dim,
                    "policy_layers": self.policy_layers,
                    "coarse_metric": self.coarse_metric,
                    "random_coarse_retrieval": getattr(self, "random_coarse_retrieval", False),
                    "random_state_retrieval": getattr(self, "random_state_retrieval", False),
                    "reused_source": self.reused_source,
                    "pretrained_encoder_info": self.pretrained_encoder_info,
                },
            },
            os.path.join(dirname, "hybridrarl_modules.pt"),
        )
        if (
            save_memory
            and self.memory is not None
            and hasattr(self.memory, "save")
        ):
            self.memory.save(dirname)


class HybridRARLPrevUnitsAgent(HybridRARLAgent):
    """
    HybridRARL with similar-task previous units.

    It keeps HybridRARL retrieval:
        z_t = Retrieve(st, current_task_memory)

    and adds previous-unit transfer:
        Phi = vstack([mu_1(st), ..., mu_P(st)])
        mu_o = Attention(Q(st), K(Phi), V=Phi)
        mu_p, sigma_p = Policy([st, z_t, mu_o])
        final mu = mu_o + mu_p
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        memory: Optional[ReservoirTrajectoryMemory] = None,
        prev_units: Optional[List[nn.Module]] = None,
        prev_unit_paths: Optional[List[Union[str, Path]]] = None,
        memory_capacity: int = 100,
        traj_len: int = 20,
        top_r: int = 5,
        traj_hidden_dim: int = 128,
        att_dim: int = 128,
        value_dim: int = 128,
        prev_att_dim: int = 128,
        policy_hidden_dim: int = 256,
        policy_layers: int = 4,
        coarse_metric: str = "cosine",
        random_coarse_retrieval: bool = False,
        random_state_retrieval: bool = False,
    ):
        super().__init__(
            obs_dim=obs_dim,
            act_dim=act_dim,
            memory=memory,
            memory_capacity=memory_capacity,
            traj_len=traj_len,
            top_r=top_r,
            traj_hidden_dim=traj_hidden_dim,
            att_dim=att_dim,
            value_dim=value_dim,
            policy_hidden_dim=policy_hidden_dim,
            policy_layers=policy_layers,
            coarse_metric=coarse_metric,
            random_coarse_retrieval=random_coarse_retrieval,
            random_state_retrieval=random_state_retrieval,
        )
        self.prev_att_dim = int(prev_att_dim)
        self.prev_unit_paths = [str(path) for path in (prev_unit_paths or [])]
        self.prev_unit_task_ids: List[int] = []

        self.prev_unit_attention = PrevUnitMeanAttentionHead(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            att_dim=self.prev_att_dim,
        )
        self.policy = build_policy_trunk(
            input_dim=self.obs_dim + self.value_dim + self.act_dim,
            hidden_dim=self.policy_hidden_dim,
            num_layers=self.policy_layers,
        )
        self.fc_mean = nn.Linear(self.policy_hidden_dim, self.act_dim)
        self.fc_logstd = nn.Linear(self.policy_hidden_dim, self.act_dim)
        self._prev_units: List[nn.Module] = []
        self.set_prev_units(prev_units or [])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_prev_units"] = []
        return state

    def set_prev_units(self, prev_units: List[nn.Module]) -> None:
        self._prev_units = []
        for unit in prev_units:
            unit.eval()
            for param in unit.parameters():
                param.requires_grad_(False)
            self._prev_units.append(unit)

    @property
    def prev_units(self) -> List[nn.Module]:
        return self._prev_units

    @staticmethod
    def _first_model_output(out) -> torch.Tensor:
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def previous_mean_context(
        self,
        st: torch.Tensor,
        return_info: bool = False,
    ):
        single_input = st.dim() == 1
        st_in = st.unsqueeze(0) if single_input else st
        assert st_in.dim() == 2

        if len(self._prev_units) == 0:
            mu_o = torch.zeros(st_in.size(0), self.act_dim, dtype=st_in.dtype, device=st_in.device)
            if single_input:
                mu_o = mu_o.squeeze(0)
            if return_info:
                return mu_o, {
                    "phi": torch.zeros(st_in.size(0), 0, self.act_dim, dtype=st_in.dtype, device=st_in.device),
                    "prev_attention_weights": torch.zeros(st_in.size(0), 0, dtype=st_in.dtype, device=st_in.device),
                    "prev_attention_logits": torch.zeros(st_in.size(0), 0, dtype=st_in.dtype, device=st_in.device),
                }
            return mu_o

        means = []
        with torch.no_grad():
            for unit in self._prev_units:
                out = unit(st_in)
                mean = self._first_model_output(out)
                means.append(mean.detach())

        phi = torch.stack(means, dim=1)
        mu_o, weights, logits = self.prev_unit_attention(st_in, phi)
        mu_out = mu_o.squeeze(0) if single_input else mu_o

        if not return_info:
            return mu_out

        return mu_out, {
            "phi": phi,
            "prev_attention_weights": weights,
            "prev_attention_logits": logits,
            "mu_o": mu_o,
        }

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
            z_t, retrieval_info = self.retrieve_context(x_in, memory_batch=memory_batch, return_info=True)
            mu_o, prev_info = self.previous_mean_context(x_in, return_info=True)
        else:
            z_t = self.retrieve_context(x_in, memory_batch=memory_batch)
            mu_o = self.previous_mean_context(x_in)
            retrieval_info = None
            prev_info = None

        hidden = self.policy(torch.cat([x_in, z_t, mu_o], dim=-1))
        mu_p = self.fc_mean(hidden)
        logstd = self.fc_logstd(hidden)
        mean = mu_o + mu_p

        if single_input:
            mean = mean.squeeze(0)
            logstd = logstd.squeeze(0)

        if not return_info:
            return mean, logstd

        info = {
            "zt": z_t,
            "mu_o": mu_o,
            "mu_p": mu_p,
            "prev_unit_paths": self.prev_unit_paths,
            "prev_unit_task_ids": self.prev_unit_task_ids,
        }
        if retrieval_info is not None:
            info.update(retrieval_info)
        if prev_info is not None:
            info.update(prev_info)
        return mean, logstd, info

    def save(self, dirname: str, save_memory: bool = True) -> None:
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, os.path.join(dirname, "model.pt"))
        torch.save(
            {
                "trajectory_rnn": self.trajectory_rnn.state_dict(),
                "trajectory_feature_mean": self.trajectory_feature_mean.detach().cpu(),
                "trajectory_feature_std": self.trajectory_feature_std.detach().cpu(),
                "attention_head": self.attention_head.state_dict(),
                "prev_unit_attention": self.prev_unit_attention.state_dict(),
                "policy": self.policy.state_dict(),
                "fc_mean": self.fc_mean.state_dict(),
                "fc_logstd": self.fc_logstd.state_dict(),
                "meta": {
                    "obs_dim": self.obs_dim,
                    "act_dim": self.act_dim,
                    "memory_capacity": self.memory_capacity,
                    "state_memory_capacity": (
                        getattr(self.memory, "capacity", None)
                        if getattr(self, "random_state_retrieval", False)
                        else None
                    ),
                    "traj_len": self.traj_len,
                    "top_r": self.top_r,
                    "traj_hidden_dim": self.traj_hidden_dim,
                    "att_dim": self.att_dim,
                    "value_dim": self.value_dim,
                    "prev_att_dim": self.prev_att_dim,
                    "policy_hidden_dim": self.policy_hidden_dim,
                    "policy_layers": self.policy_layers,
                    "coarse_metric": self.coarse_metric,
                    "random_coarse_retrieval": getattr(self, "random_coarse_retrieval", False),
                    "random_state_retrieval": getattr(self, "random_state_retrieval", False),
                    "prev_unit_paths": self.prev_unit_paths,
                    "prev_unit_task_ids": self.prev_unit_task_ids,
                    "pretrained_encoder_info": self.pretrained_encoder_info,
                },
            },
            os.path.join(dirname, "hybridrarl_prev_units_modules.pt"),
        )
        if (
            save_memory
            and self.memory is not None
            and hasattr(self.memory, "save")
        ):
            self.memory.save(dirname)


class NoRetrievalAgent(nn.Module):
    """
    Ablation actor without trajectory retrieval.

    It removes trajectory memory, coarse retrieval, RNN encoding, and trajectory
    attention. The policy is a plain MLP over the current state:

        mu, sigma = Policy(st)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        policy_hidden_dim: int = 256,
        policy_layers: int = 4,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.policy_hidden_dim = int(policy_hidden_dim)
        self.policy_layers = int(policy_layers)
        self.memory = None

        self.policy = build_policy_trunk(
            input_dim=self.obs_dim,
            hidden_dim=self.policy_hidden_dim,
            num_layers=self.policy_layers,
        )
        self.fc_mean = nn.Linear(self.policy_hidden_dim, self.act_dim)
        self.fc_logstd = nn.Linear(self.policy_hidden_dim, self.act_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_info: bool = False,
        **unused_kwargs,
    ):
        single_input = x.dim() == 1
        x_in = x.unsqueeze(0) if single_input else x
        hidden = self.policy(x_in)
        mean = self.fc_mean(hidden)
        logstd = self.fc_logstd(hidden)

        if single_input:
            mean = mean.squeeze(0)
            logstd = logstd.squeeze(0)

        if not return_info:
            return mean, logstd
        return mean, logstd, {
            "ablation": "no_retrieval",
            "mu_p": mean,
            "mu_o": torch.zeros_like(mean),
        }

    def save(self, dirname: str, save_memory: bool = True) -> None:
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, os.path.join(dirname, "model.pt"))
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "fc_mean": self.fc_mean.state_dict(),
                "fc_logstd": self.fc_logstd.state_dict(),
                "meta": {
                    "obs_dim": self.obs_dim,
                    "act_dim": self.act_dim,
                    "policy_hidden_dim": self.policy_hidden_dim,
                    "policy_layers": self.policy_layers,
                    "ablation": "no_retrieval",
                },
            },
            os.path.join(dirname, "no_retrieval_modules.pt"),
        )


class NoRetrievalPrevUnitsAgent(NoRetrievalAgent):
    """
    Ablation actor without trajectory retrieval but with selective modular reuse.

        Phi = vstack([mu_1(st), ..., mu_P(st)])
        mu_o = Attention(Q(st), K(Phi), V=Phi)
        mu_p, sigma_p = Policy([st, mu_o])
        final mu = mu_o + mu_p
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        prev_units: Optional[List[nn.Module]] = None,
        prev_unit_paths: Optional[List[Union[str, Path]]] = None,
        prev_att_dim: int = 128,
        policy_hidden_dim: int = 256,
        policy_layers: int = 4,
    ):
        super().__init__(
            obs_dim=obs_dim,
            act_dim=act_dim,
            policy_hidden_dim=policy_hidden_dim,
            policy_layers=policy_layers,
        )
        self.prev_att_dim = int(prev_att_dim)
        self.prev_unit_paths = [str(path) for path in (prev_unit_paths or [])]
        self.prev_unit_task_ids: List[int] = []
        self.prev_unit_attention = PrevUnitMeanAttentionHead(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            att_dim=self.prev_att_dim,
        )
        self.policy = build_policy_trunk(
            input_dim=self.obs_dim + self.act_dim,
            hidden_dim=self.policy_hidden_dim,
            num_layers=self.policy_layers,
        )
        self.fc_mean = nn.Linear(self.policy_hidden_dim, self.act_dim)
        self.fc_logstd = nn.Linear(self.policy_hidden_dim, self.act_dim)
        self._prev_units: List[nn.Module] = []
        self.set_prev_units(prev_units or [])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_prev_units"] = []
        return state

    def set_prev_units(self, prev_units: List[nn.Module]) -> None:
        self._prev_units = []
        for unit in prev_units:
            unit.eval()
            for param in unit.parameters():
                param.requires_grad_(False)
            self._prev_units.append(unit)

    @property
    def prev_units(self) -> List[nn.Module]:
        return self._prev_units

    @staticmethod
    def _first_model_output(out) -> torch.Tensor:
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def previous_mean_context(
        self,
        st: torch.Tensor,
        return_info: bool = False,
    ):
        single_input = st.dim() == 1
        st_in = st.unsqueeze(0) if single_input else st
        assert st_in.dim() == 2

        if len(self._prev_units) == 0:
            mu_o = torch.zeros(st_in.size(0), self.act_dim, dtype=st_in.dtype, device=st_in.device)
            if single_input:
                mu_o = mu_o.squeeze(0)
            if return_info:
                return mu_o, {
                    "phi": torch.zeros(st_in.size(0), 0, self.act_dim, dtype=st_in.dtype, device=st_in.device),
                    "prev_attention_weights": torch.zeros(st_in.size(0), 0, dtype=st_in.dtype, device=st_in.device),
                    "prev_attention_logits": torch.zeros(st_in.size(0), 0, dtype=st_in.dtype, device=st_in.device),
                    "mu_o": mu_o,
                }
            return mu_o

        means = []
        with torch.no_grad():
            for unit in self._prev_units:
                out = unit(st_in)
                mean = self._first_model_output(out)
                means.append(mean.detach())

        phi = torch.stack(means, dim=1)
        mu_o, weights, logits = self.prev_unit_attention(st_in, phi)
        mu_out = mu_o.squeeze(0) if single_input else mu_o

        if not return_info:
            return mu_out

        return mu_out, {
            "phi": phi,
            "prev_attention_weights": weights,
            "prev_attention_logits": logits,
            "mu_o": mu_o,
        }

    def forward(
        self,
        x: torch.Tensor,
        return_info: bool = False,
        **unused_kwargs,
    ):
        single_input = x.dim() == 1
        x_in = x.unsqueeze(0) if single_input else x

        if return_info:
            mu_o, prev_info = self.previous_mean_context(x_in, return_info=True)
        else:
            mu_o = self.previous_mean_context(x_in)
            prev_info = None

        hidden = self.policy(torch.cat([x_in, mu_o], dim=-1))
        mu_p = self.fc_mean(hidden)
        logstd = self.fc_logstd(hidden)
        mean = mu_o + mu_p

        if single_input:
            mean = mean.squeeze(0)
            logstd = logstd.squeeze(0)

        if not return_info:
            return mean, logstd

        info = {
            "ablation": "no_retrieval",
            "mu_o": mu_o,
            "mu_p": mu_p,
            "prev_unit_paths": self.prev_unit_paths,
            "prev_unit_task_ids": self.prev_unit_task_ids,
        }
        if prev_info is not None:
            info.update(prev_info)
        return mean, logstd, info

    def save(self, dirname: str, save_memory: bool = True) -> None:
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, os.path.join(dirname, "model.pt"))
        torch.save(
            {
                "prev_unit_attention": self.prev_unit_attention.state_dict(),
                "policy": self.policy.state_dict(),
                "fc_mean": self.fc_mean.state_dict(),
                "fc_logstd": self.fc_logstd.state_dict(),
                "meta": {
                    "obs_dim": self.obs_dim,
                    "act_dim": self.act_dim,
                    "prev_att_dim": self.prev_att_dim,
                    "policy_hidden_dim": self.policy_hidden_dim,
                    "policy_layers": self.policy_layers,
                    "prev_unit_paths": self.prev_unit_paths,
                    "prev_unit_task_ids": self.prev_unit_task_ids,
                    "ablation": "no_retrieval",
                },
            },
            os.path.join(dirname, "no_retrieval_prev_units_modules.pt"),
        )
