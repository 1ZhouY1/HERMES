
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic utilities
# ============================================================

def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    n_layers: int = 3,
    activation: nn.Module = nn.ReLU,
) -> nn.Sequential:
    assert n_layers >= 1
    layers: List[nn.Module] = []
    in_dim = int(input_dim)

    for _ in range(n_layers - 1):
        layers += [
            nn.Linear(in_dim, hidden_dim),
            activation(),
        ]
        in_dim = hidden_dim

    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


@dataclass
class TrajectoryBatch:
    """
    states:  (N, L, obs_dim)
    actions: (N, L, action_dim)
    rewards: (N, L, reward_dim)
    """
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


# ============================================================
# Long-term trajectory memory
# ============================================================

class LongTermMemory:
    """
    Store fixed-length trajectories directly.

    Replacement:
        FIFO / ring-buffer.
    """

    def __init__(
        self,
        capacity: int,
        traj_len: int,
        obs_dim: int,
        action_dim: int,
        reward_dim: int = 1,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.capacity = int(capacity)
        self.traj_len = int(traj_len)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.reward_dim = int(reward_dim)
        self.device = torch.device(device)
        self.dtype = dtype

        assert self.capacity > 0
        assert self.traj_len > 0
        assert self.obs_dim > 0
        assert self.action_dim > 0
        assert self.reward_dim > 0

        self.states = torch.zeros(
            self.capacity,
            self.traj_len,
            self.obs_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.actions = torch.zeros(
            self.capacity,
            self.traj_len,
            self.action_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.rewards = torch.zeros(
            self.capacity,
            self.traj_len,
            self.reward_dim,
            device=self.device,
            dtype=self.dtype,
        )

        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def clear(self) -> None:
        self.ptr = 0
        self.size = 0

    def add_trajectory(
        self,
        states: Union[torch.Tensor, List],
        actions: Union[torch.Tensor, List],
        rewards: Optional[Union[torch.Tensor, List]] = None,
    ) -> None:
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

        self.states[self.ptr].copy_(states)
        self.actions[self.ptr].copy_(actions)
        self.rewards[self.ptr].copy_(rewards)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_trajectories(
        self,
        n_traj: int,
        device: Optional[Union[str, torch.device]] = None,
        replace: bool = False,
    ) -> TrajectoryBatch:
        assert self.size > 0, "LongTermMemory is empty."
        n_traj = int(n_traj)
        assert n_traj > 0

        if not replace:
            assert n_traj <= self.size, (
                f"n_traj={n_traj} > memory size={self.size}. "
                f"Set replace=True or reduce n_traj."
            )

        out_device = self.device if device is None else torch.device(device)

        if replace:
            idx = torch.randint(
                low=0,
                high=self.size,
                size=(n_traj,),
                device=self.device,
            )
        else:
            idx = torch.randperm(self.size, device=self.device)[:n_traj]

        return TrajectoryBatch(
            states=self.states[idx].to(out_device),
            actions=self.actions[idx].to(out_device),
            rewards=self.rewards[idx].to(out_device),
        )

    def save(self, dirname: str, filename: str = "long_term_memory.pt") -> None:
        os.makedirs(dirname, exist_ok=True)
        torch.save(
            {
                "capacity": self.capacity,
                "traj_len": self.traj_len,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "reward_dim": self.reward_dim,
                "dtype": self.dtype,
                "ptr": self.ptr,
                "size": self.size,
                "states": self.states.detach().cpu(),
                "actions": self.actions.detach().cpu(),
                "rewards": self.rewards.detach().cpu(),
            },
            os.path.join(dirname, filename),
        )

    @staticmethod
    def load(
        dirname: str,
        filename: str = "long_term_memory.pt",
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> "LongTermMemory":
        data = torch.load(os.path.join(dirname, filename), map_location=map_location)
        memory = LongTermMemory(
            capacity=data["capacity"],
            traj_len=data["traj_len"],
            obs_dim=data["obs_dim"],
            action_dim=data["action_dim"],
            reward_dim=data["reward_dim"],
            device=map_location or "cpu",
            dtype=data.get("dtype", torch.float32),
        )
        memory.ptr = data["ptr"]
        memory.size = data["size"]
        memory.states.copy_(data["states"].to(memory.device))
        memory.actions.copy_(data["actions"].to(memory.device))
        memory.rewards.copy_(data["rewards"].to(memory.device))
        return memory


# ============================================================
# Encoder
# ============================================================

class GRUStateEncoder(nn.Module):
    """
    s_t = Encoder(x_t)

    Accept:
        x: (obs_dim,)
        x: (B, obs_dim)
        x: (B, T, obs_dim)

    Return:
        s: (state_dim,) or (B, state_dim)
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        num_layers: int = 1,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.num_layers = int(num_layers)

        self.gru = nn.GRU(
            input_size=self.obs_dim,
            hidden_size=self.state_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single_input = False

        if x.dim() == 1:
            x = x.view(1, 1, -1)
            single_input = True
        elif x.dim() == 2:
            x = x[:, None, :]
        elif x.dim() == 3:
            pass
        else:
            raise ValueError(
                f"x should be (obs_dim,), (B,obs_dim), or (B,T,obs_dim), got {tuple(x.shape)}"
            )

        assert x.size(-1) == self.obs_dim, (
            f"last dim should be obs_dim={self.obs_dim}, got {x.size(-1)}"
        )

        _, h_n = self.gru(x)
        s = h_n[-1]

        if single_input:
            s = s.squeeze(0)
        return s


# ============================================================
# Trajectory summarizer for coarse retrieval
# ============================================================

class TrajectorySummarizer(nn.Module):
    """
    Summarize each trajectory into one trajectory-level semantic vector.

    Input:
        memory_state_embeddings: (N,L,state_dim) or (B,N,L,state_dim)

    Output:
        traj_summary: (N,summary_dim) or (B,N,summary_dim)
    """

    def __init__(
        self,
        state_dim: int,
        summary_dim: int,
        num_layers: int = 1,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.summary_dim = int(summary_dim)

        self.gru = nn.GRU(
            input_size=self.state_dim,
            hidden_size=self.summary_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, memory_state_embeddings: torch.Tensor) -> torch.Tensor:
        added_batch = False

        if memory_state_embeddings.dim() == 3:
            memory_state_embeddings = memory_state_embeddings[None, ...]
            added_batch = True

        assert memory_state_embeddings.dim() == 4, (
            "memory_state_embeddings should be (N,L,D) or (B,N,L,D), "
            f"got {tuple(memory_state_embeddings.shape)}"
        )

        B, N, L, D = memory_state_embeddings.shape
        assert D == self.state_dim

        flat = memory_state_embeddings.reshape(B * N, L, D)
        _, h_n = self.gru(flat)
        traj_summary = h_n[-1].reshape(B, N, self.summary_dim)

        if added_batch:
            traj_summary = traj_summary[0]
        return traj_summary


# ============================================================
# Coarse semantic trajectory retriever
# ============================================================

class SemanticTrajectoryRetriever(nn.Module):
    """
    Coarse retrieval:
        query = s_t
        key   = trajectory_summaries

    Return top-r trajectory indices.
    """

    def __init__(
        self,
        state_dim: int,
        summary_dim: int,
        att_dim: int = 128,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.summary_dim = int(summary_dim)
        self.att_dim = int(att_dim)
        self.temp = math.sqrt(self.att_dim)

        self.wq = nn.Linear(self.state_dim, self.att_dim)
        self.wk = nn.Linear(self.summary_dim, self.att_dim)

    def forward(
        self,
        s: torch.Tensor,
        traj_summary: torch.Tensor,
        top_r: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            s: (B,state_dim)
            traj_summary: (N,summary_dim) or (B,N,summary_dim)

        Returns:
            top_traj_indices: (B,R)
        """
        assert s.dim() == 2
        B = s.size(0)

        if traj_summary.dim() == 2:
            traj_summary = traj_summary[None, ...].expand(B, -1, -1)

        assert traj_summary.dim() == 3
        assert traj_summary.size(0) == B

        N = traj_summary.size(1)
        R = min(int(top_r), N)

        q = self.wq(s)                     # (B,A)
        k = self.wk(traj_summary)          # (B,N,A)

        logits = torch.matmul(q[:, None, :], k.transpose(1, 2)).squeeze(1)
        logits = logits / self.temp        # (B,N)

        top_scores, top_indices = torch.topk(logits, k=R, dim=-1)
        top_alpha = F.softmax(top_scores, dim=-1)

        return {
            "traj_logits": logits,
            "top_traj_indices": top_indices,
            "top_traj_scores": top_scores,
            "top_traj_alpha": top_alpha,
        }


# ============================================================
# Fine exact state retriever
# ============================================================

class ExactStateRetriever(nn.Module):
    """
    Fine retrieval inside top-r trajectories:
        query = x_t
        key   = memory states inside top-r trajectories
        value = corresponding actions

    Output:
        top-k retrieved candidate actions.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        att_dim: int = 64,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.att_dim = int(att_dim)
        self.temp = math.sqrt(self.att_dim)

        self.state_norm = nn.LayerNorm(self.obs_dim) if use_layer_norm else nn.Identity()
        self.wq = nn.Linear(self.obs_dim, self.att_dim)
        self.wk = nn.Linear(self.obs_dim, self.att_dim)

    def forward(
        self,
        x: torch.Tensor,
        candidate_states: torch.Tensor,
        candidate_actions: torch.Tensor,
        top_k: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B,obs_dim)
            candidate_states:  (B,R,L,obs_dim)
            candidate_actions: (B,R,L,action_dim)

        Returns:
            exact_actions: (B,K,action_dim)
        """
        assert x.dim() == 2
        assert candidate_states.dim() == 4
        assert candidate_actions.dim() == 4

        B, R, L, D = candidate_states.shape
        assert D == self.obs_dim
        assert candidate_actions.shape[:3] == (B, R, L)
        assert candidate_actions.size(-1) == self.action_dim

        total = R * L
        K = min(int(top_k), total)

        flat_states = candidate_states.reshape(B, total, self.obs_dim)
        flat_actions = candidate_actions.reshape(B, total, self.action_dim)

        q = self.wq(self.state_norm(x))             # (B,A)
        k = self.wk(self.state_norm(flat_states))   # (B,total,A)

        logits = torch.matmul(q[:, None, :], k.transpose(1, 2)).squeeze(1)
        logits = logits / self.temp                 # (B,total)

        top_scores, top_indices = torch.topk(logits, k=K, dim=-1)
        top_alpha = F.softmax(top_scores, dim=-1)

        gather_s_idx = top_indices[..., None].expand(-1, -1, self.obs_dim)
        gather_a_idx = top_indices[..., None].expand(-1, -1, self.action_dim)

        exact_states = torch.gather(flat_states, dim=1, index=gather_s_idx)
        exact_actions = torch.gather(flat_actions, dim=1, index=gather_a_idx)

        exact_weighted_action = torch.sum(
            top_alpha[..., None] * exact_actions,
            dim=1,
        )

        return {
            "exact_logits": logits,
            "exact_indices": top_indices,
            "exact_scores": top_scores,
            "exact_alpha": top_alpha,
            "exact_states": exact_states,
            "exact_actions": exact_actions,
            "exact_weighted_action": exact_weighted_action,
        }


# ============================================================
# Candidate action attention
# ============================================================

class CandidateActionAttention(nn.Module):
    """
    Attention over retrieved action candidates.

    query = s_t
    key/value = top-k retrieved candidate actions
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        att_dim: int = 64,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.att_dim = int(att_dim)
        self.temp = math.sqrt(self.att_dim)

        self.wq = nn.Linear(self.state_dim, self.att_dim)
        self.wk = nn.Linear(self.action_dim, self.att_dim)
        self.wv = nn.Linear(self.action_dim, self.action_dim)

    def forward(
        self,
        s: torch.Tensor,
        candidate_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            s: (B,state_dim)
            candidate_actions: (B,K,action_dim)

        Returns:
            a_ret: (B,action_dim)
            alpha: (B,K)
            logits: (B,K)
        """
        assert s.dim() == 2
        assert candidate_actions.dim() == 3
        assert candidate_actions.size(-1) == self.action_dim

        q = self.wq(s)                         # (B,A)
        k = self.wk(candidate_actions)         # (B,K,A)
        v = self.wv(candidate_actions)         # (B,K,action_dim)

        logits = torch.matmul(q[:, None, :], k.transpose(1, 2)).squeeze(1)
        logits = logits / self.temp

        alpha = F.softmax(logits, dim=-1)
        a_ret = torch.sum(alpha[..., None] * v, dim=1)

        return a_ret, alpha, logits


# ============================================================
# Hybrid RAGRL actor mean core
# ============================================================

class HybridRAGRLActorCore(nn.Module):
    """
    Implements:

        s_t = Encoder(x_t)

        top-r trajectories = SemanticTrajRetrieve(s_t, trajectory_summaries)

        top-k states/actions = ExactStateRetrieve(x_t, states inside top-r trajectories)

        a_ret = CandidateActionAttention(
            query=s_t,
            candidates=top-k retrieved actions
        )

        a1 = PolicyRetrieve(concat(s_t, a_ret))
        a2 = PolicyBase(s_t)

        mean = a2 + sigmoid(gate) * a1

    Here a1 is treated as retrieval residual correction.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        state_dim: int = 128,
        traj_summary_dim: int = 256,
        hidden_dim: int = 256,
        coarse_att_dim: int = 128,
        exact_att_dim: int = 128,
        candidate_att_dim: int = 256,
        top_r: int = 8,
        top_k_exact: int = 16,
        gate_init: float = -4.0,
        use_layer_norm_exact: bool = True,
        fusion_mode: str = "residual",
    ):
        super().__init__()
        assert fusion_mode in ["residual", "mixture"], (
            "fusion_mode should be 'residual' or 'mixture'"
        )

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.traj_summary_dim = int(traj_summary_dim)
        self.hidden_dim = int(hidden_dim)
        self.top_r = int(top_r)
        self.top_k_exact = int(top_k_exact)
        self.fusion_mode = fusion_mode

        self.encoder = GRUStateEncoder(
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
        )

        self.traj_summarizer = TrajectorySummarizer(
            state_dim=self.state_dim,
            summary_dim=self.traj_summary_dim,
        )

        self.coarse_retriever = SemanticTrajectoryRetriever(
            state_dim=self.state_dim,
            summary_dim=self.traj_summary_dim,
            att_dim=coarse_att_dim,
        )

        self.exact_retriever = ExactStateRetriever(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            att_dim=exact_att_dim,
            use_layer_norm=use_layer_norm_exact,
        )

        self.candidate_action_attention = CandidateActionAttention(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            att_dim=candidate_att_dim,
        )

        self.base_policy = build_mlp(
            input_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.action_dim,
            n_layers=3,
        )

        self.retrieval_policy = build_mlp(
            input_dim=self.state_dim + self.action_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.action_dim,
            n_layers=3,
        )

        # Learnable scalar gate. Updated by actor loss.
        # sigmoid(-4) ~= 0.018, so model starts near base_policy.
        self.retrieval_gate = nn.Parameter(torch.tensor(float(gate_init)))

    @staticmethod
    def _unpack_memory_batch(
        memory_batch: Union[TrajectoryBatch, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(memory_batch, TrajectoryBatch):
            return memory_batch.states, memory_batch.actions

        assert isinstance(memory_batch, dict), (
            f"memory_batch should be TrajectoryBatch or dict, got {type(memory_batch)}"
        )

        states = memory_batch.get("obs", memory_batch.get("states", None))
        actions = memory_batch.get("actions", None)

        assert states is not None, "memory_batch must contain 'obs' or 'states'."
        assert actions is not None, "memory_batch must contain 'actions'."
        return states, actions

    def encode_state(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def _encode_memory_states(self, memory_states: torch.Tensor) -> torch.Tensor:
        """
        Encode every memory state independently.

        Args:
            memory_states: (N,L,obs_dim) or (B,N,L,obs_dim)

        Returns:
            memory_embeddings: (N,L,state_dim) or (B,N,L,state_dim)
        """
        added_batch = False

        if memory_states.dim() == 3:
            memory_states = memory_states[None, ...]
            added_batch = True

        assert memory_states.dim() == 4, (
            f"memory_states should be (N,L,D) or (B,N,L,D), got {tuple(memory_states.shape)}"
        )

        B, N, L, D = memory_states.shape
        assert D == self.obs_dim

        flat = memory_states.reshape(B * N * L, D)
        flat_emb = self.encoder(flat)
        memory_embeddings = flat_emb.reshape(B, N, L, self.state_dim)

        if added_batch:
            memory_embeddings = memory_embeddings[0]
        return memory_embeddings

    @staticmethod
    def _expand_memory_for_batch(t: torch.Tensor, batch_size: int) -> torch.Tensor:
        if t.dim() == 3:
            return t[None, ...].expand(batch_size, -1, -1, -1)

        assert t.dim() == 4, f"memory tensor should be (N,L,D) or (B,N,L,D), got {tuple(t.shape)}"
        assert t.size(0) == batch_size, (
            f"memory batch dim {t.size(0)} does not match current batch size {batch_size}"
        )
        return t

    def _gather_top_r_trajectories(
        self,
        memory_states: torch.Tensor,
        memory_actions: torch.Tensor,
        top_traj_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            memory_states: (N,L,obs_dim) or (B,N,L,obs_dim)
            memory_actions: (N,L,action_dim) or (B,N,L,action_dim)
            top_traj_indices: (B,R)

        Returns:
            top_r_states:  (B,R,L,obs_dim)
            top_r_actions: (B,R,L,action_dim)
        """
        B, R = top_traj_indices.shape

        memory_states = self._expand_memory_for_batch(memory_states, B)
        memory_actions = self._expand_memory_for_batch(memory_actions, B)

        _, N, L, D_state = memory_states.shape
        D_action = memory_actions.size(-1)

        gather_s_idx = top_traj_indices[:, :, None, None].expand(B, R, L, D_state)
        gather_a_idx = top_traj_indices[:, :, None, None].expand(B, R, L, D_action)

        top_r_states = torch.gather(memory_states, dim=1, index=gather_s_idx)
        top_r_actions = torch.gather(memory_actions, dim=1, index=gather_a_idx)

        return top_r_states, top_r_actions

    def forward(
        self,
        x: torch.Tensor,
        memory_batch: Union[TrajectoryBatch, Dict[str, torch.Tensor]],
        return_info: bool = False,
    ):
        single_input = False

        if x.dim() == 1:
            x = x[None, :]
            single_input = True

        assert x.dim() == 2, f"x should be (B,obs_dim), got {tuple(x.shape)}"

        memory_states, memory_actions = self._unpack_memory_batch(memory_batch)

        # 1. Current-state encoding.
        s = self.encoder(x)  # (B,state_dim)

        # 2. Trajectory summaries from encoded memory states.
        memory_embeddings = self._encode_memory_states(memory_states)
        trajectory_summaries = self.traj_summarizer(memory_embeddings)

        # 3. Coarse semantic top-r trajectory retrieval.
        coarse_info = self.coarse_retriever(
            s=s,
            traj_summary=trajectory_summaries,
            top_r=self.top_r,
        )
        top_traj_indices = coarse_info["top_traj_indices"]

        # 4. Gather top-r trajectories.
        top_r_states, top_r_actions = self._gather_top_r_trajectories(
            memory_states=memory_states,
            memory_actions=memory_actions,
            top_traj_indices=top_traj_indices,
        )

        # 5. Fine exact retrieval inside top-r trajectories.
        exact_info = self.exact_retriever(
            x=x,
            candidate_states=top_r_states,
            candidate_actions=top_r_actions,
            top_k=self.top_k_exact,
        )
        exact_actions = exact_info["exact_actions"]  # (B,K,action_dim)

        # 6. Candidate action attention.
        a_ret, candidate_alpha, candidate_logits = self.candidate_action_attention(
            s=s,
            candidate_actions=exact_actions,
        )

        # 7. Policy branches.
        a2 = self.base_policy(s)
        a1 = self.retrieval_policy(torch.cat([s, a_ret], dim=-1))

        # 8. Gated fusion.
        gate = torch.sigmoid(self.retrieval_gate)

        # Hybrid Encoder method
        # mean = a2
        # Hybrid Context method
        # mean = a1

        if self.fusion_mode == "residual":
            mean = a2 + gate * a1
        elif self.fusion_mode == "mixture":
            mean = (1.0 - gate) * a2 + gate * a1
        else:
            raise RuntimeError(f"Unknown fusion_mode={self.fusion_mode}")

        # mean = a2 + a1
        # mean = F.softmax(logits, dim=1)

        mean_out = mean.squeeze(0) if single_input else mean

        if not return_info:
            return mean_out

        info = {
            "encoded_state": s,
            "memory_embeddings": memory_embeddings,
            "trajectory_summaries": trajectory_summaries,
            "top_traj_indices": top_traj_indices,
            "top_traj_scores": coarse_info["top_traj_scores"],
            "top_traj_alpha": coarse_info["top_traj_alpha"],
            "traj_logits": coarse_info["traj_logits"],
            "top_r_states": top_r_states,
            "top_r_actions": top_r_actions,
            "exact_actions": exact_actions,
            "exact_states": exact_info["exact_states"],
            "exact_indices": exact_info["exact_indices"],
            "exact_alpha": exact_info["exact_alpha"],
            "exact_logits": exact_info["exact_logits"],
            "exact_weighted_action": exact_info["exact_weighted_action"],
            "a_ret": a_ret,
            "candidate_alpha": candidate_alpha,
            "candidate_logits": candidate_logits,
            "a1_retrieval_delta": a1,
            "a2_base_mean": a2,
            "retrieval_gate": gate,
            "fusion_mode": self.fusion_mode,
            "mean": mean,
        }

        return mean_out, info


# ============================================================
# SAC-compatible agent wrapper
# ============================================================

class HybridRAGRLAgent(nn.Module):
    """
    SAC-compatible actor model.

    This class returns:
        mean, logstd

    Therefore it can be used inside your existing SAC Actor wrapper:

        mean, logstd = self.model(obs, memory_batch=memory_batch)

    Critic suggestion:
        Use agent.encode_state(obs) as critic state input:
            Q(s_t, a_t)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        memory: Optional[LongTermMemory] = None,
        n_traj: int = 64,
        state_dim: int = 128,
        traj_summary_dim: int = 128,
        hidden_dim: int = 256,
        coarse_att_dim: int = 128,
        exact_att_dim: int = 64,
        candidate_att_dim: int = 64,
        top_r: int = 8,
        top_k_exact: int = 16,
        gate_init: float = -4.0,
        fusion_mode: str = "residual",
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.n_traj = int(n_traj)
        self.memory = memory
        self.state_dim = int(state_dim)
        self.fusion_mode = fusion_mode

        self.net_mean = HybridRAGRLActorCore(
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            state_dim=state_dim,
            traj_summary_dim=traj_summary_dim,
            hidden_dim=hidden_dim,
            coarse_att_dim=coarse_att_dim,
            exact_att_dim=exact_att_dim,
            candidate_att_dim=candidate_att_dim,
            top_r=top_r,
            top_k_exact=top_k_exact,
            gate_init=gate_init,
            fusion_mode=fusion_mode,
        )

        # logstd branch uses encoded state s_t.
        self.net_logstd = build_mlp(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=act_dim,
            n_layers=3,
        )

    def _get_memory_batch(
        self,
        x: torch.Tensor,
        memory_batch: Optional[Union[TrajectoryBatch, Dict[str, torch.Tensor]]],
    ) -> Union[TrajectoryBatch, Dict[str, torch.Tensor]]:
        if memory_batch is not None:
            return memory_batch

        assert self.memory is not None, (
            "memory_batch is None and this HybridRAGRLAgent has no attached LongTermMemory."
        )

        return self.memory.sample_trajectories(
            n_traj=self.n_traj,
            device=x.device,
        )

    def encode_state(self, x: torch.Tensor) -> torch.Tensor:
        return self.net_mean.encode_state(x)

    def forward(
        self,
        x: torch.Tensor,
        memory_batch: Optional[Union[TrajectoryBatch, Dict[str, torch.Tensor]]] = None,
        return_info: bool = False,
    ):
        single_input = x.dim() == 1
        x_for_memory = x[None, :] if single_input else x

        memory_batch = self._get_memory_batch(x_for_memory, memory_batch)

        if return_info:
            mean, info = self.net_mean(
                x,
                memory_batch=memory_batch,
                return_info=True,
            )
            s = info["encoded_state"]
        else:
            mean = self.net_mean(
                x,
                memory_batch=memory_batch,
                return_info=False,
            )
            s = self.encode_state(x_for_memory)

        logstd = self.net_logstd(s)

        if single_input:
            logstd = logstd.squeeze(0)

        if return_info:
            return mean, logstd, info

        return mean, logstd

    def save(self, dirname: str, save_memory: bool = False) -> None:
        os.makedirs(dirname, exist_ok=True)
        torch.save(
            {
                "net_mean": self.net_mean,
                "net_logstd": self.net_logstd,
                "meta": {
                    "obs_dim": self.obs_dim,
                    "act_dim": self.act_dim,
                    "n_traj": self.n_traj,
                    "state_dim": self.state_dim,
                    "fusion_mode": self.fusion_mode,
                },
            },
            os.path.join(dirname, "model.pt"),
        )

        if save_memory and self.memory is not None:
            self.memory.save(dirname)


# ============================================================
# Optional critic Q(s_t, a_t)
# ============================================================

class HybridSoftQNetwork(nn.Module):
    """
    Optional critic:
        Q(s_t, a_t)

    Here s_t is encoded state from:
        s_t = agent.encode_state(obs)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.net = build_mlp(
            input_dim=self.state_dim + self.action_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            n_layers=3,
        )

    def forward(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        assert s.dim() == 2, f"s should be (B,state_dim), got {tuple(s.shape)}"
        assert a.dim() == 2, f"a should be (B,action_dim), got {tuple(a.shape)}"
        x = torch.cat([s, a], dim=-1)
        return self.net(x)


# ============================================================
# Minimal shape test
# ============================================================

if __name__ == "__main__":
    obs_dim = 39
    act_dim = 4
    traj_len = 50
    batch_size = 8
    n_traj = 64

    memory = LongTermMemory(
        capacity=128,
        traj_len=traj_len,
        obs_dim=obs_dim,
        action_dim=act_dim,
        device="cpu",
    )

    for _ in range(100):
        memory.add_trajectory(
            states=torch.randn(traj_len, obs_dim),
            actions=torch.randn(traj_len, act_dim).clamp(-1.0, 1.0),
            rewards=torch.randn(traj_len, 1),
        )

    memory_batch = memory.sample_trajectories(n_traj=n_traj)

    agent = HybridRAGRLAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        memory=memory,
        n_traj=n_traj,
        state_dim=128,
        traj_summary_dim=128,
        hidden_dim=256,
        top_r=8,
        top_k_exact=16,
    )

    x = torch.randn(batch_size, obs_dim)

    mean, logstd, info = agent(
        x,
        memory_batch=memory_batch,
        return_info=True,
    )

    print("mean:", mean.shape)
    print("logstd:", logstd.shape)
    print("encoded_state:", info["encoded_state"].shape)
    print("top_traj_indices:", info["top_traj_indices"].shape)
    print("top_r_states:", info["top_r_states"].shape)
    print("exact_actions:", info["exact_actions"].shape)
    print("a_ret:", info["a_ret"].shape)
    print("gate:", info["retrieval_gate"].item())

    q_net = HybridSoftQNetwork(state_dim=128, action_dim=act_dim)
    s = agent.encode_state(x)
    q = q_net(s, mean)
    print("q:", q.shape)
