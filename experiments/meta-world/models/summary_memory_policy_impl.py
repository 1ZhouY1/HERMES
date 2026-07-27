import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass
class SummaryMemoryWriteInfo:
    action: str
    index: int
    similarity: float
    previous_return: float


class SummaryMemory:
    """
    Bounded memory of trajectory summaries.

    Each item stores one trajectory-level summary vector H and the return of
    the trajectory that produced it. Similar high-return summaries can replace
    lower-return entries, so memory size stays bounded.
    """

    def __init__(
        self,
        capacity: int,
        summary_dim: int,
        similarity_threshold: float = 0.95,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.capacity = int(capacity)
        self.summary_dim = int(summary_dim)
        self.similarity_threshold = float(similarity_threshold)
        self.device = torch.device(device)
        self.dtype = dtype

        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if self.summary_dim <= 0:
            raise ValueError(f"summary_dim must be positive, got {summary_dim}")

        self.summaries = torch.zeros(
            self.capacity,
            self.summary_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.returns = torch.full(
            (self.capacity,),
            -float("inf"),
            device=self.device,
            dtype=self.dtype,
        )
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return int(self.size)

    def clear(self) -> None:
        self.ptr = 0
        self.size = 0
        self.summaries.zero_()
        self.returns.fill_(-float("inf"))

    def current_summaries(self, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        out_device = self.device if device is None else torch.device(device)
        return self.summaries[: self.size].to(out_device)

    def current_returns(self, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        out_device = self.device if device is None else torch.device(device)
        return self.returns[: self.size].to(out_device)

    def add_summary(
        self,
        summary: torch.Tensor,
        trajectory_return: Union[float, torch.Tensor],
    ) -> SummaryMemoryWriteInfo:
        summary = torch.as_tensor(summary, dtype=self.dtype, device=self.device).reshape(-1)
        if summary.numel() != self.summary_dim:
            raise ValueError(
                f"summary should have dim={self.summary_dim}, got shape={tuple(summary.shape)}"
            )

        trajectory_return = float(torch.as_tensor(trajectory_return).item())

        if self.size == 0:
            self.summaries[0].copy_(summary)
            self.returns[0] = trajectory_return
            self.size = 1
            self.ptr = 1 % self.capacity
            return SummaryMemoryWriteInfo("add", 0, 0.0, -float("inf"))

        active = self.summaries[: self.size]
        sims = F.cosine_similarity(summary[None, :], active, dim=-1)
        max_sim, max_idx = torch.max(sims, dim=0)
        similar = sims >= self.similarity_threshold

        if bool(similar.any()):
            similar_indices = torch.nonzero(similar, as_tuple=False).flatten()
            similar_returns = self.returns[similar_indices]
            lowest_pos = int(torch.argmin(similar_returns).item())
            replace_idx = int(similar_indices[lowest_pos].item())
            previous_return = float(self.returns[replace_idx].item())

            if trajectory_return > previous_return:
                self.summaries[replace_idx].copy_(summary)
                self.returns[replace_idx] = trajectory_return
                return SummaryMemoryWriteInfo(
                    "replace_similar",
                    replace_idx,
                    float(max_sim.item()),
                    previous_return,
                )

            return SummaryMemoryWriteInfo(
                "skip_similar",
                int(max_idx.item()),
                float(max_sim.item()),
                previous_return,
            )

        if self.size < self.capacity:
            idx = self.size
            self.summaries[idx].copy_(summary)
            self.returns[idx] = trajectory_return
            self.size += 1
            self.ptr = self.size % self.capacity
            return SummaryMemoryWriteInfo("add", idx, float(max_sim.item()), -float("inf"))

        replace_idx = int(torch.argmin(self.returns).item())
        previous_return = float(self.returns[replace_idx].item())
        if trajectory_return > previous_return:
            self.summaries[replace_idx].copy_(summary)
            self.returns[replace_idx] = trajectory_return
            return SummaryMemoryWriteInfo(
                "replace_low_return",
                replace_idx,
                float(max_sim.item()),
                previous_return,
            )

        return SummaryMemoryWriteInfo(
            "skip_full",
            replace_idx,
            float(max_sim.item()),
            previous_return,
        )

    def save(self, dirname: str, filename: str = "summary_memory.pt") -> None:
        os.makedirs(dirname, exist_ok=True)
        torch.save(
            {
                "capacity": self.capacity,
                "summary_dim": self.summary_dim,
                "similarity_threshold": self.similarity_threshold,
                "dtype": self.dtype,
                "ptr": self.ptr,
                "size": self.size,
                "summaries": self.summaries.detach().cpu(),
                "returns": self.returns.detach().cpu(),
            },
            os.path.join(dirname, filename),
        )

    @staticmethod
    def load(
        dirname: str,
        filename: str = "summary_memory.pt",
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> "SummaryMemory":
        data = torch.load(os.path.join(dirname, filename), map_location=map_location)
        memory = SummaryMemory(
            capacity=data["capacity"],
            summary_dim=data["summary_dim"],
            similarity_threshold=data.get("similarity_threshold", 0.95),
            device=map_location or "cpu",
            dtype=data.get("dtype", torch.float32),
        )
        memory.ptr = data["ptr"]
        memory.size = data["size"]
        memory.summaries.copy_(data["summaries"].to(memory.device))
        memory.returns.copy_(data["returns"].to(memory.device))
        return memory


class TrajectorySummaryEncoder(nn.Module):
    """
    GRU summary encoder.

    A single observation is treated as a length-1 sequence. A partial or full
    trajectory can be passed as (B,T,obs_dim), and the final hidden state is H.
    """

    def __init__(self, obs_dim: int, summary_dim: int, num_layers: int = 1):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.summary_dim = int(summary_dim)
        self.num_layers = int(num_layers)
        self.gru = nn.GRU(
            input_size=self.obs_dim,
            hidden_size=self.summary_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )

    def _prepare_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
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

        if x.size(-1) != self.obs_dim:
            raise ValueError(f"last dim should be obs_dim={self.obs_dim}, got {x.size(-1)}")
        return x, single_input

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x, single_input = self._prepare_input(x)
        outputs, _ = self.gru(x)
        return outputs.squeeze(0) if single_input else outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, single_input = self._prepare_input(x)
        _, h_n = self.gru(x)
        summary = h_n[-1]
        return summary.squeeze(0) if single_input else summary


class ActionPredictionSummaryModel(nn.Module):
    """
    Pretraining wrapper for Summary:

        H_t = Summary(x_0, ..., x_t)
        a_hat_t = ActionHead(H_t)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        summary_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.summary_dim = int(summary_dim)
        self.hidden_dim = int(hidden_dim)

        self.summary = TrajectorySummaryEncoder(
            obs_dim=self.obs_dim,
            summary_dim=self.summary_dim,
        )
        self.action_head = nn.Sequential(
            nn.Linear(self.summary_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    def forward(self, obs_sequences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.summary.forward_sequence(obs_sequences)
        action_pred = self.action_head(hidden)
        return action_pred, hidden


class SummaryMemoryAttention(nn.Module):
    def __init__(self, summary_dim: int, context_dim: int, att_dim: int = 128):
        super().__init__()
        self.summary_dim = int(summary_dim)
        self.context_dim = int(context_dim)
        self.att_dim = int(att_dim)
        self.temp = math.sqrt(self.att_dim)

        self.wq = nn.Linear(self.summary_dim, self.att_dim)
        self.wk = nn.Linear(self.summary_dim, self.att_dim)
        self.wv = nn.Linear(self.summary_dim, self.context_dim)

    def forward(
        self,
        summary: torch.Tensor,
        memory: Optional[SummaryMemory],
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        if summary.dim() == 1:
            summary = summary[None, :]

        batch_size = summary.size(0)
        if memory is None or len(memory) == 0:
            context = summary.new_zeros(batch_size, self.context_dim)
            return context, {
                "attention_logits": None,
                "attention_alpha": None,
                "memory_returns": None,
            }

        memory_summaries = memory.current_summaries(device=summary.device)
        memory_returns = memory.current_returns(device=summary.device)

        q = self.wq(summary)
        k = self.wk(memory_summaries)
        v = self.wv(memory_summaries)

        logits = torch.matmul(q, k.transpose(0, 1)) / self.temp
        alpha = F.softmax(logits, dim=-1)
        context = torch.matmul(alpha, v)

        return context, {
            "attention_logits": logits,
            "attention_alpha": alpha,
            "memory_returns": memory_returns,
        }


class SummaryMemoryAgent(nn.Module):
    """
    SAC-compatible policy:

        H_t = Summary(x_{t-window:t})
        context = Attention(H_t, SummaryMemory)
        mean, logstd = Policy([x_t, context])
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        memory: Optional[SummaryMemory] = None,
        summary_dim: int = 128,
        context_dim: int = 128,
        hidden_dim: int = 256,
        att_dim: int = 128,
        traj_len: int = 50,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.summary_dim = int(summary_dim)
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.traj_len = int(traj_len)
        self.memory = memory
        self.register_buffer("obs_mean", torch.zeros(self.obs_dim))
        self.register_buffer("obs_std", torch.ones(self.obs_dim))
        self.use_obs_normalization = False

        self.summary = TrajectorySummaryEncoder(
            obs_dim=self.obs_dim,
            summary_dim=self.summary_dim,
        )
        self.memory_attention = SummaryMemoryAttention(
            summary_dim=self.summary_dim,
            context_dim=self.context_dim,
            att_dim=att_dim,
        )
        self.policy_backbone = build_mlp(
            input_dim=self.obs_dim + self.context_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        )
        self.fc_mean = nn.Linear(self.hidden_dim, self.act_dim)
        self.fc_logstd = nn.Linear(self.hidden_dim, self.act_dim)

    def set_obs_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        mean = torch.as_tensor(mean, dtype=self.obs_mean.dtype, device=self.obs_mean.device).reshape(-1)
        std = torch.as_tensor(std, dtype=self.obs_std.dtype, device=self.obs_std.device).reshape(-1)
        if mean.numel() != self.obs_dim or std.numel() != self.obs_dim:
            raise ValueError(
                f"normalization should have dim={self.obs_dim}, got mean={tuple(mean.shape)}, std={tuple(std.shape)}"
            )
        self.obs_mean.copy_(mean)
        self.obs_std.copy_(std.clamp_min(1e-6))
        self.use_obs_normalization = True

    def _summary_input(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_obs_normalization:
            return x
        return (x - self.obs_mean.to(x.device)) / self.obs_std.to(x.device)

    def load_pretrained_summary(
        self,
        checkpoint_path: Union[str, os.PathLike],
        freeze: bool = True,
        load_normalization: bool = True,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> Dict:
        checkpoint_path = Path(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location=map_location or "cpu")

        state_dict = payload.get("summary_state_dict")
        if state_dict is None:
            raise KeyError(
                f"{checkpoint_path} should contain 'summary_state_dict'. "
                "Use train_summary_action.py to create the checkpoint."
            )

        self.summary.load_state_dict(state_dict)

        if load_normalization and "obs_mean" in payload and "obs_std" in payload:
            self.set_obs_normalization(payload["obs_mean"], payload["obs_std"])

        if freeze:
            for param in self.summary.parameters():
                param.requires_grad_(False)
            self.summary.eval()

        return payload.get("meta", {})

    def summarize(self, x: torch.Tensor) -> torch.Tensor:
        return self.summary(self._summary_input(x))

    def summarize_trajectory(self, states: torch.Tensor) -> torch.Tensor:
        if states.dim() == 2:
            states = states[None, :, :]
        return self.summary(self._summary_input(states))

    def forward(
        self,
        x: torch.Tensor,
        summary_sequence: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ):
        single_input = x.dim() == 1
        if single_input:
            x = x[None, :]

        if summary_sequence is None:
            h_t = self.summary(self._summary_input(x))
        else:
            h_t = self.summary(self._summary_input(summary_sequence))

        context, attention_info = self.memory_attention(h_t, self.memory)
        policy_input = torch.cat([x, context], dim=-1)
        hidden = self.policy_backbone(policy_input)
        mean = self.fc_mean(hidden)
        logstd = self.fc_logstd(hidden)

        if single_input:
            mean = mean.squeeze(0)
            logstd = logstd.squeeze(0)

        if not return_info:
            return mean, logstd

        info = {
            "summary": h_t,
            "context": context,
            **attention_info,
        }
        return mean, logstd, info

    def save(self, dirname: str, save_memory: bool = False) -> None:
        os.makedirs(dirname, exist_ok=True)
        torch.save(
            {
                "model": self,
                "meta": {
                    "obs_dim": self.obs_dim,
                    "act_dim": self.act_dim,
                    "summary_dim": self.summary_dim,
                    "context_dim": self.context_dim,
                    "hidden_dim": self.hidden_dim,
                    "traj_len": self.traj_len,
                    "use_obs_normalization": self.use_obs_normalization,
                },
                "obs_mean": self.obs_mean.detach().cpu(),
                "obs_std": self.obs_std.detach().cpu(),
            },
            os.path.join(dirname, "model.pt"),
        )
        if save_memory and self.memory is not None:
            self.memory.save(dirname)
