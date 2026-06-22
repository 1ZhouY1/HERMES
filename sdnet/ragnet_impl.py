import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
import numpy as np
from typing import Union, Optional, Tuple, Dict, List
from dataclasses import dataclass

def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, n_layers: int = 2,
              activation: nn.Module = nn.ReLU) -> nn.Sequential:
    """Small helper for the default policy network."""
    assert n_layers >= 1
    layers: List[nn.Module] = []
    in_dim = input_dim
    for _ in range(n_layers - 1):
        layers += [nn.Linear(in_dim, hidden_dim), activation()]
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)

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

    @staticmethod
    def _expand_memory_for_batch(t: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Convert memory tensor from (N,L,*) to (B,N,L,*), or keep (B,N,L,*)."""
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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = s.size(0)

        if forward_h.dim() == 3:
            forward_h = forward_h[None, ...].expand(B, -1, -1, -1)
        if backward_b.dim() == 3:
            backward_b = backward_b[None, ...].expand(B, -1, -1, -1)
        memory_obs = self._expand_memory_for_batch(memory_obs, B)
        memory_actions = self._expand_memory_for_batch(memory_actions, B)

        assert forward_h.shape[:3] == backward_b.shape[:3] == memory_obs.shape[:3] == memory_actions.shape[:3]
        B, N, L, _ = forward_h.shape
        total_memory_size = N * L
        top_k = min(top_k, total_memory_size)

        query = self.wq(s) # (B, hidden_dim)
        keys = self.wk(forward_h) # (B, n_traj, L, hidden_dim)
        logits = (query[:, None, None, :] * keys).sum(dim=-1) / self.att_temp # (B, n_traj, L)

        flat_logits = logits.reshape(B, total_memory_size)        # (B, N*L)
        top_scores, top_indices = torch.topk(flat_logits, k=top_k, dim=-1)
        top_alpha = F.softmax(top_scores, dim=-1)                 # (B, K)

        traj_indices = top_indices // L                           # Get topk trajectory indices
        step_indices = top_indices % L                            # Get topk timesteps indices within trajectories

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
            "top_scores": top_scores,
            "top_alpha": top_alpha,
            "top_indices": top_indices,
            "traj_indices": traj_indices,
            "step_indices": step_indices,
            "selected_obs": selected_obs,
            "selected_actions": selected_actions,
            "selected_backward": selected_backward,
        }
        return context, info


class ActionAttentionHead(nn.Module):
    """
    Action attention head.

    phi_s = selected states from retrieval head
    phi_a = corresponding actions from retrieval head

    Query comes from current encoded state s_t; Keys come from phi_s; Values are phi_a.
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
        self.att_dim = att_dim
        self.att_temp = math.sqrt(self.att_dim)

        self.wq = att_heads_init(nn.Linear(state_dim, self.att_dim, bias=proj_bias))
        self.wk = att_heads_init(nn.Linear(obs_dim, self.att_dim, bias=proj_bias))

    def forward(self, s: torch.Tensor, phi_s: torch.Tensor, phi_a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert phi_s.dim() == 3, f"phi_s should be (B,K,obs_dim), got {tuple(phi_s.shape)}"
        assert phi_a.dim() == 3, f"phi_a should be (B,K,action_dim), got {tuple(phi_a.shape)}"
        assert phi_s.shape[:2] == phi_a.shape[:2]

        query = self.wq(s)                                      # (B,A)
        keys = self.wk(phi_s)                                   # (B,K,A)
        logits = torch.sum(query[:, None, :] * keys, dim=-1) / self.att_temp
        att = F.softmax(logits, dim=-1)                         # (B,K)
        a_att = torch.sum(att[..., None] * phi_a, dim=1)         # (B,action_dim)
        return a_att, att


class RetrievalAugmentedNet(nn.Module):
    """
    Implementation of the proposed single-network architecture.

    Components:
        1. Long-term trajectory buffer is provided externally or via sample_memory().
        2. f_enc: GRU state encoder.
        3. forward summarizer: GRU from x_1 to x_t.
        4. backward summarizer: GRU from x_l to x_t.
        5. retrieve attention head: top-k retrieval + context aggregation.
        6. policy_net: maps [s_t, context] to policy action/logit vector.
        7. action attention head: attends over retrieved state-action pairs.

    This module only implements the architecture forward pass. It does not implement SAC loss,
    critic, replay update, entropy term, target network, or optimizer logic.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        state_dim: int = 128,
        summary_dim: int = 128,
        context_dim: int = 128,
        hidden_dim: int = 256,
        top_k: int = 8,
        encoder: Optional[nn.Module] = None,
        summarizer: Optional[nn.Module] = None,
        policy_net: Optional[nn.Module] = None,
        final_norm: str = "softmax",
        proj_bias: bool = True,
        att_heads_init: object = Identity(),
    ):
        super().__init__()
        assert final_norm in ["identity", "tanh", "softmax"], \
            "final_norm should be one of: identity, tanh, softmax"

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.summary_dim = summary_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.final_norm = final_norm

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

        self.policy_net = policy_net if policy_net is not None else build_mlp(
            input_dim=state_dim + context_dim,
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

    def sample_memory(
        self,
        buffer: LongTermMemory,
        n_traj: Optional[int] = None,
        max_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        replace: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Convenience wrapper for sampling trajectories from LongTermTrajectoryBuffer."""
        return buffer.sample_trajectories(
            n_traj=n_traj or self.top_k,
            max_len=max_len,
            device=device,
            replace=replace,
        )

    def forward(
        self,
        x: torch.Tensor,
        memory_batch: Dict[str, torch.Tensor],
        return_atts: bool = False,
        ret_encoder_out: bool = False,
        ret_context: bool = False,
        ret_selected: bool = False,
        ret_policy_out: bool = False,
    ):
        """
        Forward pass.

        Args:
            x:
                Current observation/state history.
                Shape: (B, obs_dim) or (B, T_cur, obs_dim).
            memory_batch:
                Dict containing at least:
                    obs:     (N,L,obs_dim) or (B,N,L,obs_dim)
                    actions: (N,L,action_dim) or (B,N,L,action_dim)
                    mask:    (N,L) or (B,N,L), optional
            return_atts:
                Return retrieve/action attention weights.
            ret_encoder_out:
                Return encoded current state s_t.
            ret_context:
                Return retrieved context.
            ret_selected:
                Return selected trajectory/state/action indices and tensors.
            ret_policy_out:
                Return raw policy_net output before adding action attention.

        Returns:
            By default:
                action: (B, action_dim)

            Optional values are appended in this order:
                s_t, context, retrieve_alpha, action_alpha, selected_info, policy_out
        """
        memory_obs = memory_batch["obs"]
        memory_actions = memory_batch["actions"]
        memory_mask = memory_batch.get("mask", None)

        s = self.encoder(x)                                      # (B,state_dim)
        forward_h, backward_b = self.summarizer(memory_obs)       # (N,L,H) or (B,N,L,H)

        context, retrieve_info = self.retrieve_head(
            s=s,
            forward_h=forward_h,
            backward_b=backward_b,
            memory_obs=memory_obs,
            memory_actions=memory_actions,
            top_k=self.top_k,
        )

        policy_in = torch.hstack([s, context])                    # (B,state_dim+context_dim)
        policy_out = self.policy_net(policy_in)                   # (B,action_dim)

        a_att, action_alpha = self.action_head(
            s=s,
            phi_s=retrieve_info["selected_obs"],
            phi_a=retrieve_info["selected_actions"],
        )

        raw_action = policy_out + a_att
        action = self._normalize_action(raw_action)

        ret_vals = [action]
        if ret_encoder_out:
            ret_vals.append(s)
        if ret_context:
            ret_vals.append(context)
        if return_atts:
            ret_vals += [retrieve_info["top_alpha"], action_alpha]
        if ret_selected:
            selected_info = {
                "top_indices": retrieve_info["top_indices"],
                "traj_indices": retrieve_info["traj_indices"],
                "step_indices": retrieve_info["step_indices"],
                "selected_obs": retrieve_info["selected_obs"],
                "selected_actions": retrieve_info["selected_actions"],
            }
            ret_vals.append(selected_info)
        if ret_policy_out:
            ret_vals.append(policy_out)

        if len(ret_vals) == 1:
            return action
        return ret_vals
class RAGNetAgent(nn.Module):
    """
    SAC-compatible wrapper for RetrievalAugmentedNet.

    It follows the same interface as SimpleAgent / CompoNetAgent:
        forward(x, memory_batch=...) -> mean, log_std

    RetrievalAugmentedNet is used as net_mean.
    A separate MLP is used as net_logstd.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        state_dim: int = 128,
        summary_dim: int = 256,
        context_dim: int = 256,
        hidden_dim: int = 256,
        top_k: int = 10,
    ):
        super().__init__()

        self.net_mean = RetrievalAugmentedNet(
            obs_dim=obs_dim,
            action_dim=act_dim,
            state_dim=state_dim,
            summary_dim=summary_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            top_k=top_k,
            final_norm="identity",
        )

        self.net_logstd = build_mlp(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=act_dim,
            n_layers=3,
        )

    def forward(
        self,
        x: torch.Tensor,
        memory_batch: Dict[str, torch.Tensor],
        writer=None,
        global_step=None,
    ):
        mean = self.net_mean(
            x,
            memory_batch=memory_batch,
        )

        log_std = self.net_logstd(x)

        return mean, log_std

    def save(self, dirname):
        import os
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.net_mean, f"{dirname}/net_mean.pt")
        torch.save(self.net_logstd, f"{dirname}/net_logstd.pt")

if __name__ == "__main__":
    # Minimal shape test, similar to the CompoNet example usage.
    obs_dim = 39
    action_dim = 4
    batch_size = 3
    n_traj = 100
    traj_len = 50
    top_k = 10

    model = RetrievalAugmentedNet(
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=128,
        summary_dim=256,
        context_dim=256,
        hidden_dim=256,
        top_k=top_k,
        final_norm="softmax",       # use "softmax" if you want the exact normalization in the idea text
    )

    x = torch.randn(batch_size, obs_dim)
    memory_batch = {
        "obs": torch.randn(n_traj, traj_len, obs_dim),
        "actions": torch.randn(n_traj, traj_len, action_dim),
        "rewards": torch.randn(n_traj, traj_len, 1),
        "mask": torch.ones(n_traj, traj_len, dtype=torch.bool),
    }

    out = model(
        x,
        memory_batch,
        ret_encoder_out=True,
        ret_context=True,
        return_atts=True,
        ret_selected=True,
        ret_policy_out=True,
    )

    action, s, context, retrieve_alpha, action_alpha, selected_info, policy_out = out
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
