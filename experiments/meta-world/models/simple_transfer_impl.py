import copy
import os
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn


def safe_torch_load(path: Union[str, os.PathLike], map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_simple_agent(dirname: Union[str, os.PathLike], map_location=None) -> nn.Module:
    dirname = Path(dirname)
    model_path = dirname / "model.pt" if dirname.is_dir() else dirname
    model = safe_torch_load(model_path, map_location=map_location)
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected a saved SimpleAgent module in {model_path}, got {type(model)}")
    if not all(hasattr(model, name) for name in ["fc", "fc_mean", "fc_logstd", "obs_dim", "act_dim"]):
        raise TypeError(f"{model_path} does not look like a SimpleAgent checkpoint.")
    return model


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


class FrozenBackboneReplaceHeadAgent(nn.Module):
    """
    Scheme 1:
        Load a SimpleAgent, freeze its feature extractor, and replace the final
        policy heads. Only the new heads are trainable.

    In this codebase SimpleAgent is:
        fc:        Linear-ReLU-Linear-ReLU
        fc_mean:   final mean layer
        fc_logstd: final logstd layer

    So "freeze the first layers and replace the last layer" is implemented as:
        frozen fc + new fc_mean/fc_logstd.
    """

    def __init__(self, simple_agent: nn.Module):
        super().__init__()
        self.obs_dim = int(simple_agent.obs_dim)
        self.act_dim = int(simple_agent.act_dim)

        self.fc = copy.deepcopy(simple_agent.fc)
        freeze_module(self.fc)

        self.fc_mean = nn.Linear(256, self.act_dim)
        self.fc_logstd = nn.Linear(256, self.act_dim)

    @classmethod
    def load_from_simple(
        cls,
        dirname: Union[str, os.PathLike],
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> "FrozenBackboneReplaceHeadAgent":
        simple_agent = load_simple_agent(dirname, map_location=map_location)
        return cls(simple_agent)

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            features = self.fc(x)
        mean = self.fc_mean(features)
        logstd = self.fc_logstd(features)
        return mean, logstd

    def save(self, dirname: str) -> None:
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, os.path.join(dirname, "model.pt"))


class ResidualAdapterMLP(nn.Module):
    """Three Linear-layer adapter: state -> hidden -> hidden -> action residual."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrozenSimpleResidualAdapterAgent(nn.Module):
    """
    Scheme 2:
        Load and freeze the whole SimpleAgent. The current state is fed to both
        the frozen SimpleAgent and a trainable three-layer Adapter. The final
        pre-tanh action mean is:

            mean = frozen_simple_mean(x) + adapter_delta(x)

        The logstd branch is kept from the frozen SimpleAgent.
    """

    def __init__(
        self,
        simple_agent: nn.Module,
        adapter_hidden_dim: int = 256,
    ):
        super().__init__()
        self.obs_dim = int(simple_agent.obs_dim)
        self.act_dim = int(simple_agent.act_dim)

        self.base_policy = copy.deepcopy(simple_agent)
        freeze_module(self.base_policy)

        self.adapter = ResidualAdapterMLP(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_dim=adapter_hidden_dim,
        )

    @classmethod
    def load_from_simple(
        cls,
        dirname: Union[str, os.PathLike],
        map_location: Optional[Union[str, torch.device]] = None,
        adapter_hidden_dim: int = 256,
    ) -> "FrozenSimpleResidualAdapterAgent":
        simple_agent = load_simple_agent(dirname, map_location=map_location)
        return cls(
            simple_agent=simple_agent,
            adapter_hidden_dim=adapter_hidden_dim,
        )

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            base_mean, base_logstd = self.base_policy(x)
        adapter_delta = self.adapter(x)
        mean = base_mean + adapter_delta
        return mean, base_logstd

    def save(self, dirname: str) -> None:
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, os.path.join(dirname, "model.pt"))
