import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rarl_pair_impl import PairMemory
from .shared_arch import shared


def cosine_position_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    if length <= 0:
        return torch.zeros(0, dim, device=device)

    positions = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    frequencies = torch.arange(dim, dtype=torch.float32, device=device).unsqueeze(0)
    frequencies = torch.pow(10000.0, -2.0 * torch.floor(frequencies / 2.0) / max(dim, 1))
    angles = positions * frequencies
    encoding = torch.cos(angles)
    encoding[:, 1::2] = torch.sin(angles[:, 1::2])
    return encoding


def first_output(out: Union[torch.Tensor, tuple, list]) -> torch.Tensor:
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


class AdaptiveRARLAgent(nn.Module):
    """
    RARL actor with frozen previous units and optional policy-prefix reuse.

    For current state st:
        1. current memory retrieves zt from state-action PairMemory
        2. each frozen previous unit maps st to an action-mean vector
        3. previous means are stacked as Phi
        4. Q(st), K(pos(Phi)), V(Phi) attention produces mu_o
        5. policy([st, zt, mu_o]) produces mu_p and sigma_p
        6. final SAC mean is mu = mu_p + mu_o

    The previous units are intentionally not serialized into model.pt. Their
    paths are stored, and training code can attach them again when needed.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        memory: Optional[PairMemory] = None,
        prev_units: Optional[Iterable[nn.Module]] = None,
        prev_unit_paths: Optional[Iterable[Union[str, Path]]] = None,
        att_dim: int = 128,
        value_dim: int = 128,
        prev_att_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.att_dim = int(att_dim)
        self.value_dim = int(value_dim)
        self.prev_att_dim = int(prev_att_dim)
        self.memory = memory
        self.prev_unit_paths = [str(path) for path in (prev_unit_paths or [])]
        self.reused_policy_source = None

        self.q_proj = nn.Linear(self.obs_dim, self.att_dim)
        self.k_proj = nn.Linear(self.obs_dim, self.att_dim)
        self.v_proj = nn.Linear(self.act_dim, self.value_dim)

        self.prev_q_proj = nn.Linear(self.obs_dim, self.prev_att_dim)
        self.prev_k_proj = nn.Linear(self.act_dim, self.prev_att_dim)

        self.policy = shared(self.obs_dim + self.value_dim + self.act_dim)
        self.fc_mean = nn.Linear(256, self.act_dim)
        self.fc_logstd = nn.Linear(256, self.act_dim)

        self._prev_units: List[nn.Module] = []
        self.set_prev_units(prev_units or [])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_prev_units"] = []
        return state

    def set_prev_units(self, prev_units: Iterable[nn.Module]) -> None:
        self._prev_units = []
        for unit in prev_units:
            unit.eval()
            for param in unit.parameters():
                param.requires_grad_(False)
            self._prev_units.append(unit)

    @property
    def prev_units(self) -> List[nn.Module]:
        return self._prev_units

    def freeze_reused_policy_prefix(self) -> None:
        for layer_index in [0, 2]:
            layer = self.policy[layer_index]
            if isinstance(layer, nn.Linear):
                for param in layer.parameters():
                    param.requires_grad_(False)

    def copy_policy_prefix_from(self, source_model: nn.Module, freeze: bool = True) -> None:
        source_policy = getattr(source_model, "policy", None)
        if source_policy is None:
            source_policy = getattr(source_model, "fc", None)
        if source_policy is None:
            raise TypeError(f"Cannot find policy/fc backbone in {type(source_model)}")

        for layer_index in [0, 2]:
            source_layer = source_policy[layer_index]
            target_layer = self.policy[layer_index]
            if not isinstance(source_layer, nn.Linear) or not isinstance(target_layer, nn.Linear):
                continue

            with torch.no_grad():
                target_layer.weight.zero_()
                target_layer.bias.copy_(source_layer.bias)
                out_dim = min(target_layer.weight.shape[0], source_layer.weight.shape[0])
                in_dim = min(target_layer.weight.shape[1], source_layer.weight.shape[1])
                target_layer.weight[:out_dim, :in_dim].copy_(source_layer.weight[:out_dim, :in_dim])

        if freeze:
            self.freeze_reused_policy_prefix()

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
                "memory_batch is None and this AdaptiveRARLAgent has no attached PairMemory."
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

    def previous_mean_context(self, obs: torch.Tensor) -> torch.Tensor:
        single_input = obs.dim() == 1
        if single_input:
            obs = obs.unsqueeze(0)

        if len(self._prev_units) == 0:
            zeros = torch.zeros(obs.shape[0], self.act_dim, dtype=obs.dtype, device=obs.device)
            return zeros.squeeze(0) if single_input else zeros

        means = []
        with torch.no_grad():
            for unit in self._prev_units:
                out = unit(obs)
                mean = first_output(out)
                means.append(mean.detach())

        phi = torch.stack(means, dim=1)
        phi_for_key = phi + cosine_position_encoding(
            length=phi.shape[1],
            dim=phi.shape[2],
            device=phi.device,
        ).unsqueeze(0)

        q = self.prev_q_proj(obs).unsqueeze(1)
        k = self.prev_k_proj(phi_for_key)
        attn_logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.prev_att_dim)
        attn_weights = F.softmax(attn_logits, dim=-1)
        mu_o = torch.matmul(attn_weights, phi).squeeze(1)

        return mu_o.squeeze(0) if single_input else mu_o

    def forward(
        self,
        x: torch.Tensor,
        memory_batch: Optional[Dict[str, torch.Tensor]] = None,
        **unused_kwargs,
    ):
        single_input = x.dim() == 1
        x_in = x.unsqueeze(0) if single_input else x

        z_t = self.retrieve_context(x_in, memory_batch=memory_batch)
        mu_o = self.previous_mean_context(x_in)
        policy_input = torch.cat([x_in, z_t, mu_o], dim=-1)
        hidden = self.policy(policy_input)
        mu_p = self.fc_mean(hidden)
        log_std = self.fc_logstd(hidden)
        mean = mu_p + mu_o

        if single_input:
            mean = mean.squeeze(0)
            log_std = log_std.squeeze(0)
        return mean, log_std

    def save(self, dirname: str, save_memory: bool = False) -> None:
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, os.path.join(dirname, "model.pt"))
        if save_memory and self.memory is not None:
            self.memory.save(dirname)
