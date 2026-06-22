import os
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Small utilities, kept close to the CompoNet style.
# -----------------------------------------------------------------------------


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    n_layers: int = 3,
    activation: nn.Module = nn.ReLU,
) -> nn.Sequential:
    """Default MLP used for policy mean / logstd heads."""
    assert n_layers >= 1, "n_layers must be >= 1"

    layers: List[nn.Module] = []
    in_dim = input_dim
    for _ in range(n_layers - 1):
        layers += [nn.Linear(in_dim, hidden_dim), activation()]
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


def net(input_dim: int, output_dim: int, hidden_dim: int = 256) -> nn.Sequential:
    """Same role as `net()` in the CompoNet agent code."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class Identity:
    """Pickle-safe callable identity initializer, matching the CompoNet style."""

    def __call__(self, arg, **kwargs):
        return arg


@dataclass
class TrajectoryBatch:
    """
    A batch of fixed-length trajectories from the long-term memory.

    states:  (n_traj, traj_len, obs_dim)
    actions: (n_traj, traj_len, action_dim)
    rewards: (n_traj, traj_len, reward_dim)
    """

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        """Return aliases accepted by RetrievalAugmentedNet.forward()."""
        return {
            "states": self.states,
            "obs": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
        }


# -----------------------------------------------------------------------------
# Long-term trajectory memory.
# -----------------------------------------------------------------------------


class LongTermMemory:
    """
    Long-term memory for previous-task trajectories.

    The memory directly stores complete fixed-length trajectories:
        states:  (capacity, traj_len, obs_dim)
        actions: (capacity, traj_len, action_dim)
        rewards: (capacity, traj_len, reward_dim)

    It does not store individual transitions and does not split / pad / crop during sampling.
    """

    def __init__(
        self,
        capacity: int,
        traj_len: int,
        obs_dim: int,
        action_dim: int,
        reward_dim: int = 1,
        device: Union[torch.device, str] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        assert capacity > 0, "capacity must be positive"
        assert traj_len > 0, "traj_len must be positive"
        assert obs_dim > 0, "obs_dim must be positive"
        assert action_dim > 0, "action_dim must be positive"
        assert reward_dim > 0, "reward_dim must be positive"

        self.capacity = capacity
        self.traj_len = traj_len
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.device = torch.device(device)
        self.dtype = dtype

        self.states = torch.zeros(capacity, traj_len, obs_dim, device=self.device, dtype=dtype)
        self.actions = torch.zeros(capacity, traj_len, action_dim, device=self.device, dtype=dtype)
        self.rewards = torch.zeros(capacity, traj_len, reward_dim, device=self.device, dtype=dtype)

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
        """
        Add one complete trajectory.

        Args:
            states:  (traj_len, obs_dim)
            actions: (traj_len, action_dim)
            rewards: (traj_len,), (traj_len, 1), or (traj_len, reward_dim)
        """
        states = torch.as_tensor(states, device=self.device, dtype=self.dtype)
        actions = torch.as_tensor(actions, device=self.device, dtype=self.dtype)

        if rewards is None:
            rewards = torch.zeros(self.traj_len, self.reward_dim, device=self.device, dtype=self.dtype)
        else:
            rewards = torch.as_tensor(rewards, device=self.device, dtype=self.dtype)
            if rewards.dim() == 1:
                rewards = rewards[:, None]

        assert states.shape == (self.traj_len, self.obs_dim), \
            f"states should be {(self.traj_len, self.obs_dim)}, got {tuple(states.shape)}"
        assert actions.shape == (self.traj_len, self.action_dim), \
            f"actions should be {(self.traj_len, self.action_dim)}, got {tuple(actions.shape)}"
        assert rewards.shape == (self.traj_len, self.reward_dim), \
            f"rewards should be {(self.traj_len, self.reward_dim)}, got {tuple(rewards.shape)}"

        self.states[self.ptr].copy_(states)
        self.actions[self.ptr].copy_(actions)
        self.rewards[self.ptr].copy_(rewards)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_trajectories(
        self,
        n_traj: int,
        device: Optional[Union[torch.device, str]] = None,
        replace: bool = False,
    ) -> TrajectoryBatch:
        """
        Sample complete trajectories.

        Returns:
            TrajectoryBatch with shapes:
                states:  (n_traj, traj_len, obs_dim)
                actions: (n_traj, traj_len, action_dim)
                rewards: (n_traj, traj_len, reward_dim)
        """
        assert self.size > 0, "LongTermMemory is empty."
        assert n_traj > 0, "n_traj must be positive."

        if not replace:
            assert n_traj <= self.size, \
                f"n_traj={n_traj} is larger than memory size={self.size}. Set replace=True if needed."

        out_device = self.device if device is None else torch.device(device)

        if replace:
            indices = torch.randint(0, self.size, size=(n_traj,), device=self.device)
        else:
            indices = torch.randperm(self.size, device=self.device)[:n_traj]

        return TrajectoryBatch(
            states=self.states[indices].to(out_device),
            actions=self.actions[indices].to(out_device),
            rewards=self.rewards[indices].to(out_device),
        )

    def get_trajectory(
        self,
        index: int,
        device: Optional[Union[torch.device, str]] = None,
    ) -> TrajectoryBatch:
        assert 0 <= index < self.size, f"index should be in [0, {self.size}), got {index}"
        out_device = self.device if device is None else torch.device(device)
        return TrajectoryBatch(
            states=self.states[index].to(out_device),
            actions=self.actions[index].to(out_device),
            rewards=self.rewards[index].to(out_device),
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


# -----------------------------------------------------------------------------
# Encoder and trajectory summarizer.
# -----------------------------------------------------------------------------


class Encoder(nn.Module):
    """
    f_enc: encode current observation x_t into s_t.

    Accepted inputs:
        x: (obs_dim,)
        x: (B, obs_dim)
        x: (B, T, obs_dim)

    Returns:
        s_t: (hidden_dim,) for a single unbatched input, otherwise (B, hidden_dim)
    """

    def __init__(self, obs_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(obs_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single_input = False
        if x.dim() == 1:
            x = x.view(1, 1, -1)
            single_input = True
        elif x.dim() == 2:
            x = x[:, None, :]
        elif x.dim() != 3:
            raise ValueError(f"x should be (obs_dim,), (B,obs_dim), or (B,T,obs_dim), got {tuple(x.shape)}")

        assert x.size(-1) == self.obs_dim, f"last dim should be obs_dim={self.obs_dim}, got {x.size(-1)}"
        _, h_n = self.gru(x)
        out = h_n[-1]
        return out.squeeze(0) if single_input else out


class Summarizer(nn.Module):
    """
    Two-head trajectory summarizer.

    For a trajectory i and step j:
        forward_h[i, j]  = f_forward(x_1^i, ..., x_j^i)
        backward_b[i, j] = f_backward(x_l^i, ..., x_j^i)

    The returned tensors keep one hidden state per trajectory position, because retrieval uses
    Key_{i,j}=Linear(forward_h[i,j]) and Value_{i,j}=Linear(backward_b[i,j]).
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        rnn_type: str = "rnn",
    ):
        super().__init__()
        assert rnn_type in ["rnn", "gru"], "rnn_type should be 'rnn' or 'gru'"
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        rnn_cls = nn.GRU if rnn_type == "gru" else nn.RNN
        self.forward_rnn = rnn_cls(obs_dim, hidden_dim, num_layers, batch_first=True)
        self.backward_rnn = rnn_cls(obs_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, memory_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            memory_obs: (N, L, obs_dim) or (B, N, L, obs_dim)

        Returns:
            forward_h:  (N,L,H) or (B,N,L,H)
            backward_b: (N,L,H) or (B,N,L,H)
        """
        added_batch_dim = False
        if memory_obs.dim() == 3:
            memory_obs = memory_obs[None, ...]
            added_batch_dim = True

        assert memory_obs.dim() == 4, \
            f"memory_obs should be (N,L,D) or (B,N,L,D), got {tuple(memory_obs.shape)}"
        B, N, L, D = memory_obs.shape
        assert D == self.obs_dim, f"last dim should be obs_dim={self.obs_dim}, got {D}"

        # Only merge the independent trajectory dimensions. The sequence dimension L is kept.
        flat_obs = memory_obs.reshape(B * N, L, D)

        forward_h, _ = self.forward_rnn(flat_obs)         # (B*N, L, H)
        reverse_obs = torch.flip(flat_obs, dims=[1])      # x_l, ..., x_1
        reverse_b, _ = self.backward_rnn(reverse_obs)     # hidden states in reversed order
        backward_b = torch.flip(reverse_b, dims=[1])      # align back to original j index

        forward_h = forward_h.reshape(B, N, L, self.hidden_dim)
        backward_b = backward_b.reshape(B, N, L, self.hidden_dim)

        if added_batch_dim:
            forward_h = forward_h[0]
            backward_b = backward_b[0]
        return forward_h, backward_b


# -----------------------------------------------------------------------------
# Attention heads.
# -----------------------------------------------------------------------------


class RetrieveAttentionHead(nn.Module):
    """
    Retrieval attention head.

    It computes QK^T over all sampled trajectory positions (N*L), selects top-k positions,
    and aggregates Value=Linear(backward_summarizer_hidden) into context.
    """

    def __init__(
        self,
        state_dim: int,
        summary_dim: int,
        context_dim: int,
        att_dim: int,
        proj_bias: bool = True,
        att_heads_init: object = Identity(),
    ):
        super().__init__()
        self.state_dim = state_dim
        self.summary_dim = summary_dim
        self.context_dim = context_dim
        self.att_dim = att_dim
        self.att_temp = math.sqrt(att_dim)

        self.wq = att_heads_init(nn.Linear(state_dim, att_dim, bias=proj_bias))
        self.wk = att_heads_init(nn.Linear(summary_dim, att_dim, bias=proj_bias))
        self.wv = att_heads_init(nn.Linear(summary_dim, context_dim, bias=proj_bias))

    @staticmethod
    def _expand_memory_for_batch(t: torch.Tensor, batch_size: int) -> torch.Tensor:
        if t.dim() == 3:
            return t[None, ...].expand(batch_size, -1, -1, -1)
        assert t.dim() == 4, f"memory tensor should be (N,L,D) or (B,N,L,D), got {tuple(t.shape)}"
        assert t.size(0) == batch_size, \
            f"memory batch dim {t.size(0)} does not match current batch size {batch_size}"
        return t

    @staticmethod
    def _expand_mask_for_batch(mask: torch.Tensor, batch_size: int) -> torch.Tensor:
        if mask.dim() == 2:
            return mask[None, ...].expand(batch_size, -1, -1)
        assert mask.dim() == 3, f"mask should be (N,L) or (B,N,L), got {tuple(mask.shape)}"
        assert mask.size(0) == batch_size, \
            f"mask batch dim {mask.size(0)} does not match current batch size {batch_size}"
        return mask

    def forward(
        self,
        s: torch.Tensor,
        forward_h: torch.Tensor,
        backward_b: torch.Tensor,
        memory_obs: torch.Tensor,
        memory_actions: torch.Tensor,
        top_k: int = 10,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert s.dim() == 2, f"s should be (B,state_dim), got {tuple(s.shape)}"
        B = s.size(0)

        if forward_h.dim() == 3:
            forward_h = forward_h[None, ...].expand(B, -1, -1, -1)
        if backward_b.dim() == 3:
            backward_b = backward_b[None, ...].expand(B, -1, -1, -1)

        memory_obs = self._expand_memory_for_batch(memory_obs, B)
        memory_actions = self._expand_memory_for_batch(memory_actions, B)

        assert forward_h.shape[:3] == backward_b.shape[:3] == memory_obs.shape[:3] == memory_actions.shape[:3], \
            "forward_h, backward_b, memory_obs, and memory_actions must share (B,N,L)."

        B, N, L, _ = forward_h.shape
        total_memory_size = N * L
        top_k = min(top_k, total_memory_size)

        # Standard batched QK^T attention over the flattened memory dimension.
        query = self.wq(s)                                      # (B, A)
        keys = self.wk(forward_h).reshape(B, total_memory_size, self.att_dim)  # (B, N*L, A)
        flat_logits = torch.matmul(
            query[:, None, :],                                  # (B, 1, A)
            keys.transpose(1, 2),                                # (B, A, N*L)
        ).squeeze(1) / self.att_temp                             # (B, N*L)

        if memory_mask is not None:
            memory_mask = self._expand_mask_for_batch(memory_mask.bool(), B)
            flat_mask = memory_mask.reshape(B, total_memory_size)
            flat_logits = flat_logits.masked_fill(~flat_mask, torch.finfo(flat_logits.dtype).min)

        logits = flat_logits.reshape(B, N, L)                    # (B, N, L)

        top_scores, top_indices = torch.topk(flat_logits, k=top_k, dim=-1)  # (B, K)
        top_alpha = F.softmax(top_scores, dim=-1)                         # (B, K)

        traj_indices = top_indices // L                           # (B, K)
        step_indices = top_indices % L                            # (B, K)

        flat_backward = backward_b.reshape(B, total_memory_size, -1)
        flat_obs = memory_obs.reshape(B, total_memory_size, -1)
        flat_actions = memory_actions.reshape(B, total_memory_size, -1)

        gather_b_idx = top_indices[..., None].expand(-1, -1, flat_backward.size(-1))
        gather_s_idx = top_indices[..., None].expand(-1, -1, flat_obs.size(-1))
        gather_a_idx = top_indices[..., None].expand(-1, -1, flat_actions.size(-1))

        selected_backward = torch.gather(flat_backward, dim=1, index=gather_b_idx)  # (B,K,H)
        selected_obs = torch.gather(flat_obs, dim=1, index=gather_s_idx)            # (B,K,obs_dim)
        selected_actions = torch.gather(flat_actions, dim=1, index=gather_a_idx)    # (B,K,action_dim)

        values = self.wv(selected_backward)                         # (B,K,context_dim)
        context = torch.sum(top_alpha[..., None] * values, dim=1)    # (B,context_dim)

        info = {
            "retrieve_logits": logits,
            "flat_retrieve_logits": flat_logits,
            "top_scores": top_scores,
            "top_alpha": top_alpha,
            "top_indices": top_indices,
            "traj_indices": traj_indices,
            "step_indices": step_indices,
            "selected_obs": selected_obs,
            "selected_actions": selected_actions,
            "selected_backward": selected_backward,
            "retrieve_values": values,
        }
        return context, info


class ActionAttentionHead(nn.Module):
    """
    Action attention head.

    Query: current encoded state s_t
    Key:   selected retrieved states phi_s
    Value: selected retrieved actions phi_a
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        action_dim: int,
        att_dim: Optional[int] = None,
        proj_bias: bool = True,
        att_heads_init: object = Identity(),
    ):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.att_dim = state_dim if att_dim is None else att_dim
        self.att_temp = math.sqrt(self.att_dim)

        self.wq = att_heads_init(nn.Linear(state_dim, self.att_dim, bias=proj_bias))
        self.wk = att_heads_init(nn.Linear(obs_dim, self.att_dim, bias=proj_bias))

    def forward(self, s: torch.Tensor, phi_s: torch.Tensor, phi_a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert s.dim() == 2, f"s should be (B,state_dim), got {tuple(s.shape)}"
        assert phi_s.dim() == 3, f"phi_s should be (B,K,obs_dim), got {tuple(phi_s.shape)}"
        assert phi_a.dim() == 3, f"phi_a should be (B,K,action_dim), got {tuple(phi_a.shape)}"
        assert phi_s.shape[:2] == phi_a.shape[:2]

        query = self.wq(s)                                      # (B,A)
        keys = self.wk(phi_s)                                   # (B,K,A)
        logits = torch.matmul(query[:, None, :], keys.transpose(1, 2)).squeeze(1) / self.att_temp
        att = F.softmax(logits, dim=-1)                         # (B,K)
        a_att = torch.sum(att[..., None] * phi_a, dim=1)         # (B,action_dim)
        return a_att, att


# -----------------------------------------------------------------------------
# Core architecture.
# -----------------------------------------------------------------------------


class RetrievalAugmentedNet(nn.Module):
    """
    Single-network RAGNet architecture.

    It only implements the forward architecture:
        s_t = f_enc(x_t)
        sampled trajectories -> forward/backward summarizers
        retrieve attention -> context
        policy_net([s_t, context]) + action attention -> final output

    It does not implement SAC losses, critic networks, target networks, replay updates,
    entropy terms, or optimizers.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        state_dim: int = 128,
        summary_dim: int = 256,
        context_dim: int = 256,
        hidden_dim: int = 128,
        top_k: int = 10,
        encoder: Optional[nn.Module] = None,
        summarizer: Optional[nn.Module] = None,
        policy_net: Optional[nn.Module] = None,
        final_norm: str = "identity",
        feature_fusion: str = "concat",
        proj_bias: bool = True,
        att_heads_init: object = Identity(),
    ):
        super().__init__()
        assert final_norm in ["identity", "tanh", "softmax"], \
            "final_norm should be one of: identity, tanh, softmax"
        assert feature_fusion in ["encoded", "concat", "add"], \
            "feature_fusion should be one of: encoded, concat, add"

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.summary_dim = summary_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.final_norm = final_norm
        self.feature_fusion = feature_fusion

        self.encoder = encoder if encoder is not None else Encoder(obs_dim, state_dim)
        self.summarizer = summarizer if summarizer is not None else Summarizer(obs_dim, summary_dim)

        self.retrieve_head = RetrieveAttentionHead(
            state_dim=state_dim,
            summary_dim=summary_dim,
            context_dim=context_dim,
            att_dim=context_dim,
            proj_bias=proj_bias,
            att_heads_init=att_heads_init,
        )

        if feature_fusion == "concat":
            policy_input_dim = state_dim + context_dim
            self.context_proj = None
        elif feature_fusion == "add":
            policy_input_dim = state_dim
            self.context_proj = nn.Linear(context_dim, state_dim, bias=proj_bias)
        else:  # "encoded"
            policy_input_dim = state_dim
            self.context_proj = None

        self.policy_input_dim = policy_input_dim
        self.policy_net = policy_net if policy_net is not None else build_mlp(
            input_dim=policy_input_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            n_layers=3,
        )

        self.action_head = ActionAttentionHead(
            state_dim=state_dim,
            obs_dim=obs_dim,
            action_dim=action_dim,
            att_dim=context_dim,
            proj_bias=proj_bias,
            att_heads_init=att_heads_init,
        )

    def _normalize_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        if self.final_norm == "identity":
            return raw_action
        if self.final_norm == "tanh":
            return torch.tanh(raw_action)
        if self.final_norm == "softmax":
            return F.softmax(raw_action, dim=-1)
        raise RuntimeError(f"Unknown final_norm={self.final_norm}")

    def _build_policy_input(self, s: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Build the feature actually fed to policy_net.

        feature_fusion="encoded": policy_input = s
        feature_fusion="concat":  policy_input = [s, context]
        feature_fusion="add":     policy_input = s + Linear(context)

        This returned tensor is also the recommended feature for a context-aware critic.
        """
        if self.feature_fusion == "encoded":
            return s
        if self.feature_fusion == "concat":
            return torch.cat([s, context], dim=-1)
        if self.feature_fusion == "add":
            assert self.context_proj is not None
            return s + self.context_proj(context)
        raise RuntimeError(f"Unknown feature_fusion={self.feature_fusion}")

    @staticmethod
    def _unpack_memory_batch(memory_batch: Union[TrajectoryBatch, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(memory_batch, TrajectoryBatch):
            return memory_batch.states, memory_batch.actions, None

        assert isinstance(memory_batch, dict), \
            f"memory_batch should be TrajectoryBatch or dict, got {type(memory_batch)}"
        memory_obs = memory_batch.get("obs", memory_batch.get("states", None))
        memory_actions = memory_batch.get("actions", None)
        memory_mask = memory_batch.get("mask", None)
        assert memory_obs is not None, "memory_batch must contain key 'obs' or 'states'."
        assert memory_actions is not None, "memory_batch must contain key 'actions'."
        return memory_obs, memory_actions, memory_mask

    def sample_memory(
        self,
        memory: LongTermMemory,
        n_traj: Optional[int] = None,
        device: Optional[Union[torch.device, str]] = None,
        replace: bool = False,
    ) -> TrajectoryBatch:
        """Convenience wrapper around LongTermMemory.sample_trajectories()."""
        return memory.sample_trajectories(
            n_traj=n_traj or self.top_k,
            device=device,
            replace=replace,
        )

    def forward(
        self,
        x: torch.Tensor,
        memory_batch: Union[TrajectoryBatch, Dict[str, torch.Tensor]],
        return_atts: bool = False,
        ret_encoder_out: bool = False,
        ret_context: bool = False,
        ret_selected: bool = False,
        ret_policy_out: bool = False,
        ret_policy_input: bool = False,
        ret_action_att: bool = False,
        return_info: bool = False,
    ):
        """
        Args:
            x: (obs_dim,), (B, obs_dim), or (B, T_cur, obs_dim)
            memory_batch: TrajectoryBatch or dict with keys:
                obs/states: (N,L,obs_dim) or (B,N,L,obs_dim)
                actions:    (N,L,action_dim) or (B,N,L,action_dim)
                mask:       optional, (N,L) or (B,N,L)

        Default return:
            action-like output: (B, action_dim), or (action_dim,) if x is unbatched.
        """
        single_input = False
        if x.dim() == 1:
            x = x[None, :]
            single_input = True

        memory_obs, memory_actions, memory_mask = self._unpack_memory_batch(memory_batch)

        s = self.encoder(x)                                            # (B,state_dim)
        assert s.dim() == 2, f"encoder should return (B,state_dim), got {tuple(s.shape)}"

        forward_h, backward_b = self.summarizer(memory_obs)             # (N,L,H) or (B,N,L,H)

        context, retrieve_info = self.retrieve_head(
            s=s,
            forward_h=forward_h,
            backward_b=backward_b,
            memory_obs=memory_obs,
            memory_actions=memory_actions,
            top_k=self.top_k,
            memory_mask=memory_mask,
        )
        policy_in = self._build_policy_input(s, context)                # (B, policy_input_dim)
        policy_out = self.policy_net(policy_in)                         # (B,action_dim)

        action_att, action_alpha = self.action_head(
            s=s,
            phi_s=retrieve_info["selected_obs"],
            phi_a=retrieve_info["selected_actions"],
        )

        raw_action = policy_out
        # raw_action = policy_out + action_att
        action = self._normalize_action(raw_action)

        info = {
            **retrieve_info,
            "encoded_state": s,
            "context": context,
            "policy_input": policy_in,
            "critic_feature": policy_in,
            "policy_out": policy_out,
            "action_att": action_att,
            "action_alpha": action_alpha,
            "raw_action": raw_action,
            "action": action,
        }

        if single_input:
            action_out = action.squeeze(0)
            if return_info:
                # Keep info batched, because top-k indices naturally carry a batch dimension.
                return action_out, info
        else:
            action_out = action
            if return_info:
                return action_out, info

        ret_vals: List[Any] = [action_out]
        if ret_encoder_out:
            ret_vals.append(s)
        if ret_context:
            ret_vals.append(context)
        if return_atts:
            ret_vals += [retrieve_info["top_alpha"], action_alpha]
        if ret_selected:
            ret_vals.append(
                {
                    "top_indices": retrieve_info["top_indices"],
                    "traj_indices": retrieve_info["traj_indices"],
                    "step_indices": retrieve_info["step_indices"],
                    "selected_obs": retrieve_info["selected_obs"],
                    "selected_actions": retrieve_info["selected_actions"],
                    "selected_backward": retrieve_info["selected_backward"],
                }
            )
        if ret_policy_out:
            ret_vals.append(policy_out)
        if ret_policy_input:
            ret_vals.append(policy_in)
        if ret_action_att:
            ret_vals.append(action_att)

        return ret_vals if len(ret_vals) > 1 else action_out

    def save(self, dirname: str, filename: str = "net_mean.pt") -> None:
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, os.path.join(dirname, filename))

    @staticmethod
    def load(dirname: str, filename: str = "net_mean.pt", map_location=None) -> "RetrievalAugmentedNet":
        return torch.load(os.path.join(dirname, filename), map_location=map_location)


# -----------------------------------------------------------------------------
# Agent wrapper, analogous to CompoNetAgent.
# -----------------------------------------------------------------------------


class RAGNetAgent(nn.Module):
    """
    Agent wrapper analogous to CompoNetAgent.

    It provides:
        net_mean:   RetrievalAugmentedNet architecture
        net_logstd: independent MLP log-std head
        forward():  returns (mean, logstd)
        save/load:  same usage pattern as the CompoNet wrapper

    This still does not implement SAC updates. It only exposes the actor architecture outputs
    that an external SAC implementation can consume.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        memory: Optional[LongTermMemory] = None,
        n_traj: int = 32,
        top_k: int = 8,
        state_dim: int = 128,
        summary_dim: int = 128,
        context_dim: int = 128,
        hidden_dim: int = 128,
        final_norm: str = "identity",
        feature_fusion: str = "concat",
        map_location=None,
    ):
        super().__init__()
        del map_location  # kept for signature symmetry with CompoNetAgent

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_traj = n_traj
        self.memory = memory
        self.feature_fusion = feature_fusion

        self.net_logstd = net(obs_dim, act_dim, hidden_dim=hidden_dim)
        self.net_mean = RetrievalAugmentedNet(
            obs_dim=obs_dim,
            action_dim=act_dim,
            state_dim=state_dim,
            summary_dim=summary_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            top_k=top_k,
            final_norm=final_norm,
            feature_fusion=feature_fusion,
        )

    def _get_memory_batch(
        self,
        x: torch.Tensor,
        memory_batch: Optional[Union[TrajectoryBatch, Dict[str, torch.Tensor]]],
    ) -> Union[TrajectoryBatch, Dict[str, torch.Tensor]]:
        if memory_batch is not None:
            return memory_batch
        assert self.memory is not None, \
            "memory_batch is None and this RAGNetAgent has no attached LongTermMemory."
        device = x.device if isinstance(x, torch.Tensor) else None
        return self.net_mean.sample_memory(self.memory, n_traj=self.n_traj, device=device)

    @staticmethod
    def _log_attention(writer, tag_prefix: str, alpha: torch.Tensor, global_step: int) -> None:
        # alpha: (B,K). Log mean attention value per retrieved slot.
        mean_alpha = alpha.detach().mean(dim=0)
        for i, v in enumerate(mean_alpha):
            writer.add_scalar(f"charts/{tag_prefix}_{i}", v.item(), global_step)

    def forward(
        self,
        x: torch.Tensor,
        memory_batch: Optional[Union[TrajectoryBatch, Dict[str, torch.Tensor]]] = None,
        writer=None,
        global_step: Optional[int] = None,
        return_info: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Return:
            mean, logstd

        This mirrors CompoNetAgent.forward(), but RAGNet's mean additionally requires a
        sampled long-term memory batch.
        """
        single_input = x.dim() == 1
        x_for_logstd = x[None, :] if single_input else x
        memory_batch = self._get_memory_batch(x_for_logstd, memory_batch)

        need_info = return_info or (writer is not None and global_step is not None)
        if not need_info:
            mean = self.net_mean(x, memory_batch)
            info = None
        else:
            mean, info = self.net_mean(x, memory_batch, return_info=True)

        if writer is not None and global_step is not None:
            self._log_attention(writer, "retrieve_att", info["top_alpha"], global_step)
            self._log_attention(writer, "action_att", info["action_alpha"], global_step)

            mean_for_metric = mean[None, :] if mean.dim() == 1 else mean
            with torch.no_grad():
                dist_out_policy = (mean_for_metric - info["policy_out"]).abs().mean().item()
                dist_out_action_att = (mean_for_metric - info["action_att"]).abs().mean().item()
                dist_policy_action_att = (info["policy_out"] - info["action_att"]).abs().mean().item()
                context_norm = info["context"].norm(dim=-1).mean().item()

                writer.add_scalar("charts/dist_out_policy", dist_out_policy, global_step)
                writer.add_scalar("charts/dist_out_action_att", dist_out_action_att, global_step)
                writer.add_scalar("charts/dist_policy_action_att", dist_policy_action_att, global_step)
                writer.add_scalar("charts/context_norm", context_norm, global_step)

        logstd = self.net_logstd(x_for_logstd)
        if single_input:
            logstd = logstd.squeeze(0)
        if return_info:
            assert info is not None
            return mean, logstd, info
        return mean, logstd

    def get_policy_features(
        self,
        x: torch.Tensor,
        memory_batch: Optional[Union[TrajectoryBatch, Dict[str, torch.Tensor]]] = None,
        detach: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Return the exact feature produced by RAGNet before the final policy head.

        This is useful when the critic should be Q(policy_feature, action),
        where policy_feature may be encoded_state, concat(encoded_state, context),
        or encoded_state + projected_context depending on feature_fusion.

        Returns:
            policy_input: (B, policy_input_dim)
            info: dict containing encoded_state, context, policy_input, etc.
        """
        single_input = x.dim() == 1
        x_for_memory = x[None, :] if single_input else x
        memory_batch = self._get_memory_batch(x_for_memory, memory_batch)
        _, info = self.net_mean(x, memory_batch, return_info=True)
        feature = info["policy_input"]
        if detach:
            feature = feature.detach()
            info = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in info.items()}
        return feature, info

    def save(self, dirname: str, save_memory: bool = False) -> None:
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.net_mean, os.path.join(dirname, "net_mean.pt"))
        torch.save(self.net_logstd, os.path.join(dirname, "net_logstd.pt"))
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "n_traj": self.n_traj,
                "feature_fusion": self.feature_fusion,
            },
            os.path.join(dirname, "ragnet_agent_meta.pt"),
        )
        if save_memory and self.memory is not None:
            self.memory.save(dirname)

    @staticmethod
    def load(
        dirname: str,
        obs_dim: int,
        act_dim: int,
        memory: Optional[LongTermMemory] = None,
        n_traj: int = 32,
        top_k: int = 8,
        state_dim: int = 128,
        summary_dim: int = 128,
        context_dim: int = 128,
        hidden_dim: int = 256,
        final_norm: str = "identity",
        feature_fusion: str = "concat",
        map_location=None,
    ) -> "RAGNetAgent":
        model = RAGNetAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            memory=memory,
            n_traj=n_traj,
            top_k=top_k,
            state_dim=state_dim,
            summary_dim=summary_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            final_norm=final_norm,
            feature_fusion=feature_fusion,
            map_location=map_location,
        )
        model.net_mean = torch.load(os.path.join(dirname, "net_mean.pt"), map_location=map_location)
        model.net_logstd = torch.load(os.path.join(dirname, "net_logstd.pt"), map_location=map_location)
        return model


if __name__ == "__main__":
    # Minimal tests, similar to the CompoNet example style.
    obs_dim = 39
    act_dim = 4
    batch_size = 3
    n_traj = 32
    traj_len = 50
    top_k = 10

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

    model = RetrievalAugmentedNet(
        obs_dim=obs_dim,
        action_dim=act_dim,
        state_dim=128,
        summary_dim=256,
        context_dim=128,
        hidden_dim=256,
        top_k=top_k,
        final_norm="identity",
    )

    x = torch.randn(batch_size, obs_dim)
    out = model(
        x,
        memory_batch,
        ret_encoder_out=True,
        ret_context=True,
        return_atts=True,
        ret_selected=True,
        ret_policy_out=True,
        ret_action_att=True,
    )
    action, s, context, retrieve_alpha, action_alpha, selected_info, policy_out, action_att = out

    print(model)
    print(f"Input x:             {x.shape}")
    print(f"Action output:       {action.shape}")
    print(f"Encoded state s_t:   {s.shape}")
    print(f"Context:             {context.shape}")
    print(f"Retrieve attention:  {retrieve_alpha.shape}")
    print(f"Action attention:    {action_alpha.shape}")
    print(f"Selected obs:        {selected_info['selected_obs'].shape}")
    print(f"Selected actions:    {selected_info['selected_actions'].shape}")
    print(f"Policy out:          {policy_out.shape}")
    print(f"Action att:          {action_att.shape}")

    agent = RAGNetAgent(obs_dim=obs_dim, act_dim=act_dim, memory=memory, n_traj=n_traj, top_k=top_k)
    mean, logstd = agent(x)
    print(f"Agent mean:          {mean.shape}")
    print(f"Agent logstd:        {logstd.shape}")

    x_single = torch.randn(obs_dim)
    mean_single, logstd_single = agent(x_single, memory_batch=memory_batch)
    print(f"Single mean:         {mean_single.shape}")
    print(f"Single logstd:       {logstd_single.shape}")
