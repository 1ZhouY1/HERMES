
"""
rarl_sac_impl.py

SAC version of Retrieval-Augmented Reinforcement Learning (RARL-style).

This file implements the architecture only:

    1. Encode current observation:
        s_t = Encoder(x_t)

    2. Encode retrieval trajectories:
        e_i,j = Encoder(x_i,j)

    3. Forward/backward trajectory summarization:
        h_i,j = ForwardSummarizer(e_i,1 ... e_i,j)
        b_i,j = BackwardSummarizer(e_i,L ... e_i,j)

    4. RARL-style retrieval process with multiple memory slots:
        slots m_t^k query h_i,j
        retrieved bottleneck variables z_t^k are aggregated into u_t
        s_tilde = LayerNorm(s_t + Proj(u_t))

    5. SAC actor:
        mean, log_std = ActorHead(s_tilde)

    6. SAC critic:
        Q(s_tilde, action)

The file does NOT implement the whole SAC training loop. Use:
    - RARLSACAgent as the actor model inside your existing Actor wrapper.
    - RARLSACCritic as qf1/qf2/qf1_target/qf2_target.
"""

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utils
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
        layers += [nn.Linear(in_dim, hidden_dim), activation()]
        in_dim = hidden_dim

    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    rewards: (N,L,1) or (B,N,L,1)
    """
    added_batch = False
    if rewards.dim() == 3:
        rewards = rewards[None, ...]
        added_batch = True

    assert rewards.dim() == 4
    B, N, L, R = rewards.shape

    out = torch.zeros_like(rewards)
    running = torch.zeros(B, N, R, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(L)):
        running = rewards[:, :, t, :] + gamma * running
        out[:, :, t, :] = running

    return out[0] if added_batch else out


@dataclass
class TrajectoryBatch:
    states: torch.Tensor   # (N,L,obs_dim)
    actions: torch.Tensor  # (N,L,action_dim)
    rewards: torch.Tensor  # (N,L,reward_dim)

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "states": self.states,
            "obs": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
        }


# ============================================================
# Long-term memory
# ============================================================

class LongTermMemory:
    """
    Fixed-length trajectory memory.
    FIFO/ring-buffer replacement.

    It stores complete trajectories:
        states:  (capacity, traj_len, obs_dim)
        actions: (capacity, traj_len, action_dim)
        rewards: (capacity, traj_len, reward_dim)
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

        self.states = torch.zeros(
            self.capacity, self.traj_len, self.obs_dim,
            device=self.device, dtype=self.dtype,
        )
        self.actions = torch.zeros(
            self.capacity, self.traj_len, self.action_dim,
            device=self.device, dtype=self.dtype,
        )
        self.rewards = torch.zeros(
            self.capacity, self.traj_len, self.reward_dim,
            device=self.device, dtype=self.dtype,
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
                self.traj_len, self.reward_dim,
                dtype=self.dtype, device=self.device,
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
        if not replace:
            n_traj = min(n_traj, self.size)

        out_device = self.device if device is None else torch.device(device)

        if replace:
            idx = torch.randint(0, self.size, size=(n_traj,), device=self.device)
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


# ============================================================
# Encoder and summarizer
# ============================================================

class GRUStateEncoder(nn.Module):
    """
    f_enc.

    Accept:
        x: (obs_dim,)
        x: (B,obs_dim)
        x: (B,T,obs_dim)

    Return:
        s: (state_dim,) or (B,state_dim)
    """

    def __init__(self, obs_dim: int, state_dim: int, num_layers: int = 1):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)

        self.gru = nn.GRU(
            input_size=self.obs_dim,
            hidden_size=self.state_dim,
            num_layers=num_layers,
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
            raise ValueError(f"bad x shape: {tuple(x.shape)}")

        assert x.size(-1) == self.obs_dim
        _, h_n = self.gru(x)
        s = h_n[-1]
        return s.squeeze(0) if single_input else s


class ForwardBackwardSummarizer(nn.Module):
    """
    Forward/backward summarizer over encoded retrieval trajectories.

    Input:
        memory_embeddings: (N,L,state_dim) or (B,N,L,state_dim)

    Return:
        forward_h:  (N,L,summary_dim) or (B,N,L,summary_dim)
        backward_b: (N,L,summary_dim) or (B,N,L,summary_dim)
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

        self.forward_gru = nn.GRU(
            input_size=self.state_dim,
            hidden_size=self.summary_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.backward_gru = nn.GRU(
            input_size=self.state_dim,
            hidden_size=self.summary_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, memory_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        added_batch = False

        if memory_embeddings.dim() == 3:
            memory_embeddings = memory_embeddings[None, ...]
            added_batch = True

        assert memory_embeddings.dim() == 4, (
            f"memory_embeddings should be (N,L,D) or (B,N,L,D), got {tuple(memory_embeddings.shape)}"
        )

        B, N, L, D = memory_embeddings.shape
        assert D == self.state_dim

        flat = memory_embeddings.reshape(B * N, L, D)

        forward_h, _ = self.forward_gru(flat)

        rev = torch.flip(flat, dims=[1])
        rev_h, _ = self.backward_gru(rev)
        backward_b = torch.flip(rev_h, dims=[1])

        forward_h = forward_h.reshape(B, N, L, self.summary_dim)
        backward_b = backward_b.reshape(B, N, L, self.summary_dim)

        if added_batch:
            forward_h = forward_h[0]
            backward_b = backward_b[0]

        return forward_h, backward_b


# ============================================================
# RARL-style retrieval process
# ============================================================

class RARLRetrievalProcess(nn.Module):
    """
    Multiple-slot RARL retrieval process.

    Slots:
        m_t^k = GRUCell(s_t, m_{t-1}^k)

    Each slot retrieves over forward summaries h_i,j.
    Retrieved info is produced from backward summaries b_i,j through
    an information bottleneck.

    Output:
        s_tilde = LayerNorm(s_t + Proj(u_t))
    """

    def __init__(
        self,
        state_dim: int,
        summary_dim: int,
        slot_dim: int = 128,
        z_dim: int = 128,
        num_slots: int = 4,
        att_dim: int = 128,
        top_k: int = 8,
        use_information_bottleneck: bool = True,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.summary_dim = int(summary_dim)
        self.slot_dim = int(slot_dim)
        self.z_dim = int(z_dim)
        self.num_slots = int(num_slots)
        self.att_dim = int(att_dim)
        self.top_k = int(top_k)
        self.use_information_bottleneck = bool(use_information_bottleneck)

        self.initial_slots = nn.Parameter(torch.zeros(self.num_slots, self.slot_dim))
        nn.init.normal_(self.initial_slots, mean=0.0, std=0.02)

        self.slot_update = nn.GRUCell(
            input_size=self.state_dim,
            hidden_size=self.slot_dim,
        )

        self.wq = nn.Linear(self.slot_dim, self.att_dim)
        self.wk = nn.Linear(self.summary_dim, self.att_dim)

        if self.use_information_bottleneck:
            self.value_mu = nn.Linear(self.summary_dim, self.z_dim)
            self.value_logvar = nn.Linear(self.summary_dim, self.z_dim)
        else:
            self.value = nn.Linear(self.summary_dim, self.z_dim)

        self.agent_query = nn.Linear(self.state_dim, self.att_dim)
        self.z_key = nn.Linear(self.z_dim, self.att_dim)
        self.z_value = nn.Linear(self.z_dim, self.z_dim)
        self.u_proj = nn.Linear(self.z_dim, self.state_dim)
        self.state_norm = nn.LayerNorm(self.state_dim)

        self.temp = math.sqrt(self.att_dim)

    @staticmethod
    def _expand_for_batch(t: torch.Tensor, B: int) -> torch.Tensor:
        if t.dim() == 3:
            return t[None, ...].expand(B, -1, -1, -1)
        assert t.dim() == 4
        assert t.size(0) == B
        return t

    def _update_slots(
        self,
        s: torch.Tensor,
        prev_slots: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B = s.size(0)

        if prev_slots is None:
            prev_slots = self.initial_slots[None, :, :].expand(B, -1, -1)

        assert prev_slots.shape == (B, self.num_slots, self.slot_dim)

        s_rep = s[:, None, :].expand(B, self.num_slots, self.state_dim)

        flat_s = s_rep.reshape(B * self.num_slots, self.state_dim)
        flat_prev = prev_slots.reshape(B * self.num_slots, self.slot_dim)

        flat_new = self.slot_update(flat_s, flat_prev)
        return flat_new.reshape(B, self.num_slots, self.slot_dim)

    def forward(
        self,
        s: torch.Tensor,
        forward_h: torch.Tensor,
        backward_b: torch.Tensor,
        prev_slots: Optional[torch.Tensor] = None,
        deterministic_z: bool = False,
    ) -> Dict[str, torch.Tensor]:
        assert s.dim() == 2
        B = s.size(0)

        forward_h = self._expand_for_batch(forward_h, B)
        backward_b = self._expand_for_batch(backward_b, B)

        assert forward_h.shape[:3] == backward_b.shape[:3]

        B, N, L, H = forward_h.shape
        total = N * L
        K = min(self.top_k, total)

        slots = self._update_slots(s, prev_slots)  # (B,S,slot_dim)

        q = self.wq(slots)  # (B,S,A)
        k = self.wk(forward_h).reshape(B, total, self.att_dim)

        logits = torch.matmul(q, k.transpose(1, 2)) / self.temp  # (B,S,total)

        top_scores, top_indices = torch.topk(logits, k=K, dim=-1)
        top_alpha = F.softmax(top_scores, dim=-1)

        flat_b = backward_b.reshape(B, total, H)
        expanded_b = flat_b[:, None, :, :].expand(B, self.num_slots, total, H)
        gather_idx = top_indices[..., None].expand(B, self.num_slots, K, H)

        selected_b = torch.gather(expanded_b, dim=2, index=gather_idx)  # (B,S,K,H)

        if self.use_information_bottleneck:
            selected_mu = self.value_mu(selected_b)
            selected_logvar = self.value_logvar(selected_b)
            selected_logvar = torch.clamp(selected_logvar, min=-10.0, max=5.0)

            z_mu = torch.sum(top_alpha[..., None] * selected_mu, dim=2)
            z_logvar = torch.sum(top_alpha[..., None] * selected_logvar, dim=2)

            if deterministic_z:
                z = z_mu
            else:
                eps = torch.randn_like(z_mu)
                z = z_mu + eps * torch.exp(0.5 * z_logvar)

            kl = -0.5 * torch.sum(
                1.0 + z_logvar - z_mu.pow(2) - z_logvar.exp(),
                dim=-1,
            )
            kl_loss = kl.mean()
        else:
            selected_v = self.value(selected_b)
            z = torch.sum(top_alpha[..., None] * selected_v, dim=2)
            z_mu = z
            z_logvar = torch.zeros_like(z)
            kl_loss = torch.zeros((), device=s.device, dtype=s.dtype)

        # Aggregate slots into u_t.
        agent_q = self.agent_query(s)       # (B,A)
        z_k = self.z_key(z)                 # (B,S,A)
        z_v = self.z_value(z)               # (B,S,z_dim)

        slot_logits = torch.matmul(
            agent_q[:, None, :],
            z_k.transpose(1, 2),
        ).squeeze(1) / self.temp

        slot_alpha = F.softmax(slot_logits, dim=-1)

        u = torch.sum(slot_alpha[..., None] * z_v, dim=1)  # (B,z_dim)
        u_state = self.u_proj(u)                           # (B,state_dim)
        s_tilde = self.state_norm(s + u_state)

        return {
            "slots": slots,
            "z": z,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "kl_loss": kl_loss,
            "u": u,
            "u_state": u_state,
            "s_tilde": s_tilde,
            "retrieval_logits": logits,
            "top_scores": top_scores,
            "top_indices": top_indices,
            "top_alpha": top_alpha,
            "slot_logits": slot_logits,
            "slot_alpha": slot_alpha,
            "selected_backward": selected_b,
        }


# ============================================================
# SAC Actor: pi(a | s_tilde)
# ============================================================

class RARLSACAgent(nn.Module):
    """
    SAC-compatible actor model.

    forward(obs, memory_batch) -> mean, logstd

    It can be used inside your existing SAC Actor wrapper:
        mean, logstd = self.model(obs, memory_batch=memory_batch)

    It also provides:
        get_s_tilde(...)
        compute_aux_loss(...)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        memory: Optional[LongTermMemory] = None,
        n_traj: int = 64,
        state_dim: int = 128,
        summary_dim: int = 128,
        slot_dim: int = 128,
        z_dim: int = 128,
        num_slots: int = 4,
        att_dim: int = 128,
        top_k: int = 8,
        hidden_dim: int = 256,
        use_information_bottleneck: bool = True,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.n_traj = int(n_traj)
        self.state_dim = int(state_dim)
        self.summary_dim = int(summary_dim)
        self.memory = memory

        self.encoder = GRUStateEncoder(
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
        )

        self.summarizer = ForwardBackwardSummarizer(
            state_dim=self.state_dim,
            summary_dim=self.summary_dim,
        )

        self.retrieval_process = RARLRetrievalProcess(
            state_dim=self.state_dim,
            summary_dim=self.summary_dim,
            slot_dim=slot_dim,
            z_dim=z_dim,
            num_slots=num_slots,
            att_dim=att_dim,
            top_k=top_k,
            use_information_bottleneck=use_information_bottleneck,
        )

        self.mean_head = build_mlp(
            input_dim=self.state_dim,
            hidden_dim=hidden_dim,
            output_dim=self.act_dim,
            n_layers=3,
        )

        self.logstd_head = build_mlp(
            input_dim=self.state_dim,
            hidden_dim=hidden_dim,
            output_dim=self.act_dim,
            n_layers=3,
        )

        aux_in = 2 * self.summary_dim
        self.aux_reward_head = build_mlp(aux_in, hidden_dim, 1, n_layers=2)
        self.aux_return_head = build_mlp(aux_in, hidden_dim, 1, n_layers=2)
        self.aux_action_head = build_mlp(aux_in, hidden_dim, self.act_dim, n_layers=2)

    @staticmethod
    def _unpack_memory_batch(
        memory_batch: Union[TrajectoryBatch, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(memory_batch, TrajectoryBatch):
            return memory_batch.states, memory_batch.actions, memory_batch.rewards

        assert isinstance(memory_batch, dict)
        states = memory_batch.get("obs", memory_batch.get("states", None))
        actions = memory_batch.get("actions", None)
        rewards = memory_batch.get("rewards", None)

        assert states is not None
        assert actions is not None
        assert rewards is not None
        return states, actions, rewards

    def _get_memory_batch(
        self,
        x: torch.Tensor,
        memory_batch: Optional[Union[TrajectoryBatch, Dict[str, torch.Tensor]]],
    ) -> Union[TrajectoryBatch, Dict[str, torch.Tensor]]:
        if memory_batch is not None:
            return memory_batch

        assert self.memory is not None, (
            "memory_batch is None and this RARLSACAgent has no attached LongTermMemory."
        )
        return self.memory.sample_trajectories(
            n_traj=self.n_traj,
            device=x.device,
        )

    def encode_memory_states(self, memory_states: torch.Tensor) -> torch.Tensor:
        """
        Encode retrieval states independently.

        memory_states: (N,L,obs_dim) or (B,N,L,obs_dim)
        """
        added_batch = False

        if memory_states.dim() == 3:
            memory_states = memory_states[None, ...]
            added_batch = True

        assert memory_states.dim() == 4
        B, N, L, D = memory_states.shape
        assert D == self.obs_dim

        flat = memory_states.reshape(B * N * L, D)
        flat_emb = self.encoder(flat)
        memory_emb = flat_emb.reshape(B, N, L, self.state_dim)

        if added_batch:
            memory_emb = memory_emb[0]
        return memory_emb

    def compute_features(
        self,
        obs: torch.Tensor,
        memory_batch: Optional[Union[TrajectoryBatch, Dict[str, torch.Tensor]]] = None,
        prev_slots: Optional[torch.Tensor] = None,
        deterministic_z: bool = False,
        return_info: bool = False,
    ):
        """
        Compute s_tilde for actor or critic.

        Returns:
            s_tilde
            info if return_info=True
        """
        single_input = False

        if obs.dim() == 1:
            obs = obs[None, :]
            single_input = True

        memory_batch = self._get_memory_batch(obs, memory_batch)
        memory_states, memory_actions, memory_rewards = self._unpack_memory_batch(memory_batch)

        s = self.encoder(obs)

        memory_emb = self.encode_memory_states(memory_states)
        forward_h, backward_b = self.summarizer(memory_emb)

        ret_info = self.retrieval_process(
            s=s,
            forward_h=forward_h,
            backward_b=backward_b,
            prev_slots=prev_slots,
            deterministic_z=deterministic_z,
        )

        s_tilde = ret_info["s_tilde"]

        if single_input:
            s_tilde_out = s_tilde.squeeze(0)
        else:
            s_tilde_out = s_tilde

        if not return_info:
            return s_tilde_out

        info = {
            **ret_info,
            "encoded_state": s,
            "memory_embeddings": memory_emb,
            "forward_h": forward_h,
            "backward_b": backward_b,
        }

        return s_tilde_out, info

    def forward(
        self,
        x: torch.Tensor,
        memory_batch: Optional[Union[TrajectoryBatch, Dict[str, torch.Tensor]]] = None,
        prev_slots: Optional[torch.Tensor] = None,
        deterministic_z: bool = False,
        return_info: bool = False,
        **unused_kwargs,
    ):
        """
        SAC actor interface:
            mean, logstd = model(x, memory_batch=memory_batch)

        Extra writer/global_step kwargs from existing code are accepted by **unused_kwargs.
        """
        if return_info:
            s_tilde, info = self.compute_features(
                x,
                memory_batch=memory_batch,
                prev_slots=prev_slots,
                deterministic_z=deterministic_z,
                return_info=True,
            )
        else:
            s_tilde = self.compute_features(
                x,
                memory_batch=memory_batch,
                prev_slots=prev_slots,
                deterministic_z=deterministic_z,
                return_info=False,
            )
            info = None

        single_input = s_tilde.dim() == 1
        s_in = s_tilde[None, :] if single_input else s_tilde

        mean = self.mean_head(s_in)
        logstd = self.logstd_head(s_in)

        if single_input:
            mean = mean.squeeze(0)
            logstd = logstd.squeeze(0)

        if return_info:
            return mean, logstd, info
        return mean, logstd

    def compute_aux_loss(
        self,
        memory_batch: Union[TrajectoryBatch, Dict[str, torch.Tensor]],
        gamma: float = 0.99,
        reward_coef: float = 1.0,
        return_coef: float = 1.0,
        action_coef: float = 0.1,
        normalize_targets: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Auxiliary summarizer loss:
            reward prediction
            return-to-go prediction
            action prediction
        """
        memory_states, memory_actions, memory_rewards = self._unpack_memory_batch(memory_batch)

        memory_emb = self.encode_memory_states(memory_states)
        forward_h, backward_b = self.summarizer(memory_emb)

        if forward_h.dim() == 3:
            forward_h = forward_h[None, ...]
            backward_b = backward_b[None, ...]
            memory_actions = memory_actions[None, ...]
            memory_rewards = memory_rewards[None, ...]

        z_aux = torch.cat([forward_h, backward_b], dim=-1)

        pred_reward = self.aux_reward_head(z_aux)
        pred_return = self.aux_return_head(z_aux)
        pred_action = self.aux_action_head(z_aux)

        returns = discounted_returns(memory_rewards, gamma=gamma)

        reward_target = memory_rewards
        return_target = returns

        if normalize_targets:
            reward_target = (reward_target - reward_target.mean()) / (reward_target.std() + 1e-6)
            return_target = (return_target - return_target.mean()) / (return_target.std() + 1e-6)

        loss_reward = F.smooth_l1_loss(pred_reward, reward_target)
        loss_return = F.smooth_l1_loss(pred_return, return_target)
        loss_action = F.smooth_l1_loss(pred_action, memory_actions)

        aux_loss = (
            reward_coef * loss_reward
            + return_coef * loss_return
            + action_coef * loss_action
        )

        info = {
            "aux_loss": aux_loss.detach(),
            "loss_reward": loss_reward.detach(),
            "loss_return": loss_return.detach(),
            "loss_action": loss_action.detach(),
        }

        return aux_loss, info

    def save(self, dirname: str, save_memory: bool = False) -> None:
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, os.path.join(dirname, "rarl_sac_agent.pt"))
        if save_memory and self.memory is not None:
            self.memory.save(dirname)


# ============================================================
# SAC critic: Q(s_tilde, a)
# ============================================================

class RARLSACCritic(nn.Module):
    """
    Critic for SAC version of RARL:
        Q(s_tilde, action)
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)

        self.q = build_mlp(
            input_dim=self.state_dim + self.act_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            n_layers=3,
        )

    def forward(self, s_tilde: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        assert s_tilde.dim() == 2, f"s_tilde should be (B,D), got {tuple(s_tilde.shape)}"
        assert action.dim() == 2, f"action should be (B,A), got {tuple(action.shape)}"
        return self.q(torch.cat([s_tilde, action], dim=-1))


# ============================================================
# Minimal shape test
# ============================================================

if __name__ == "__main__":
    obs_dim = 39
    act_dim = 4
    traj_len = 50
    n_traj = 32
    batch_size = 8

    memory = LongTermMemory(
        capacity=128,
        traj_len=traj_len,
        obs_dim=obs_dim,
        action_dim=act_dim,
        reward_dim=1,
        device="cpu",
    )

    for _ in range(64):
        memory.add_trajectory(
            states=torch.randn(traj_len, obs_dim),
            actions=torch.randn(traj_len, act_dim),
            rewards=torch.randn(traj_len, 1),
        )

    memory_batch = memory.sample_trajectories(n_traj=n_traj)

    agent = RARLSACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        memory=memory,
        n_traj=n_traj,
        state_dim=128,
        summary_dim=128,
        slot_dim=128,
        z_dim=128,
        num_slots=4,
        top_k=8,
    )

    x = torch.randn(batch_size, obs_dim)

    mean, logstd, info = agent(
        x,
        memory_batch=memory_batch,
        return_info=True,
        deterministic_z=True,
    )

    print("mean:", mean.shape)
    print("logstd:", logstd.shape)
    print("s_tilde:", info["s_tilde"].shape)
    print("slots:", info["slots"].shape)
    print("z:", info["z"].shape)
    print("top_indices:", info["top_indices"].shape)
    print("kl_loss:", info["kl_loss"].item())

    critic = RARLSACCritic(state_dim=128, act_dim=act_dim)
    q = critic(info["s_tilde"], mean)
    print("q:", q.shape)

    aux_loss, aux_info = agent.compute_aux_loss(memory_batch)
    print("aux_loss:", aux_loss.item())
