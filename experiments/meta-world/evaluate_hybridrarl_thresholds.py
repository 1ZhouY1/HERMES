#!/usr/bin/env python3
"""
Standalone evaluator for Meta-World HybridRARL threshold ablations.

This script does not call experiments/test_rl_model.py.  It independently:

1. resolves each threshold-specific HybridRARL prev-units checkpoint;
2. falls back to the base HybridRARL checkpoint when that threshold checkpoint
   is absent, meaning no reusable module was selected;
3. reconstructs HybridRARL / HybridRARLPrevUnits actors from either model.pt or
   module checkpoints;
4. evaluates return and success rate on Meta-World;
5. writes per-task records and per-threshold summary files.

Default threshold mapping:
  0.72 -> csv_threshold_0p72
  0.80 -> csv_threshold_transfer
  0.85 -> csv_threshold_0p85
  0.90 -> csv_threshold_0p90
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


LOG_STD_MAX = 2
LOG_STD_MIN = -20
DEFAULT_THRESHOLDS = ("0.72", "0.80", "0.85", "0.90")


class LoadedHybridDictAgent(nn.Module):
    """Compatibility wrapper for older dict-style Hybrid/RAG checkpoints."""

    def __init__(self, net_mean: nn.Module, net_logstd: Optional[nn.Module], memory, n_traj: int):
        super().__init__()
        self.net_mean = net_mean
        self.net_logstd = net_logstd
        self.memory = memory
        self.n_traj = int(n_traj)

    def forward(self, x, memory_batch=None, return_info: bool = False, **unused_kwargs):
        if memory_batch is None:
            assert self.memory is not None, "Hybrid model requires attached memory or memory_batch."
            memory_batch = self.memory.sample_trajectories(n_traj=self.n_traj, device=x.device)

        if return_info:
            mean, info = self.net_mean(x, memory_batch=memory_batch, return_info=True)
            encoded_state = info["encoded_state"]
        else:
            mean = self.net_mean(x, memory_batch=memory_batch, return_info=False)
            encoded_state = self.net_mean.encode_state(x)

        logstd = torch.zeros_like(mean) if self.net_logstd is None else self.net_logstd(encoded_state)
        if return_info:
            return mean, logstd, info
        return mean, logstd


def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def exp_dir() -> Path:
    return Path(__file__).resolve().parent


def import_meta_world_modules():
    directory = exp_dir()
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    tasks_module = importlib.import_module("tasks")
    models_module = importlib.import_module("models")
    return tasks_module, models_module


def default_task_ids(task_names: Sequence[str]) -> List[int]:
    return list(range(min(10, len(task_names))))


def parse_threshold(raw: str) -> str:
    return f"{float(raw):.2f}"


def threshold_token(threshold: str) -> str:
    return parse_threshold(threshold).replace(".", "p")


def threshold_exp_name(threshold: str) -> str:
    threshold = parse_threshold(threshold)
    if threshold == "0.80":
        return "csv_threshold_transfer"
    return f"csv_threshold_{threshold_token(threshold)}"


def resolve_path(path: Path, base_dir: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (base_dir / path).resolve()


def model_dir_name(task_id: int, model_name: str, exp_name: str, seed: int) -> str:
    return f"task_{int(task_id)}__{model_name}__{exp_name}__{int(seed)}"


def model_name_from_dir(dirname: str) -> str:
    parts = Path(dirname).name.split("__")
    if len(parts) >= 4:
        return parts[1]
    return ""


def normalized_model_name(model_name: str) -> str:
    return str(model_name).strip().lower().replace("_", "-")


def has_saved_actor(model_dir: Path) -> bool:
    return (
        (model_dir / "model.pt").exists()
        or (model_dir / "hybridrarl_modules.pt").exists()
        or (model_dir / "hybridrarl_prev_units_modules.pt").exists()
    )


def candidate_prev_units_dir(
    *,
    prev_root: Path,
    task_id: int,
    threshold: str,
    train_seed: int,
    prev_model_name: str,
) -> Path:
    return prev_root / model_dir_name(
        task_id=task_id,
        model_name=prev_model_name,
        exp_name=threshold_exp_name(threshold),
        seed=train_seed,
    )


def candidate_base_dir(
    *,
    model_root: Path,
    task_id: int,
    train_seed: int,
    base_model_name: str,
    base_exp_name: str,
) -> Path:
    return model_root / model_dir_name(
        task_id=task_id,
        model_name=base_model_name,
        exp_name=base_exp_name,
        seed=train_seed,
    )


def resolve_checkpoint_for_threshold(
    *,
    prev_root: Path,
    model_root: Path,
    task_id: int,
    threshold: str,
    train_seed: int,
    prev_model_name: str,
    base_model_name: str,
    base_exp_name: str,
) -> Dict[str, object]:
    prev_dir = candidate_prev_units_dir(
        prev_root=prev_root,
        task_id=task_id,
        threshold=threshold,
        train_seed=train_seed,
        prev_model_name=prev_model_name,
    )
    if prev_dir.exists() and has_saved_actor(prev_dir):
        return {
            "model_dir": prev_dir.resolve(),
            "source": "threshold_prev_units",
            "used_reusable_modules": True,
            "threshold_model_dir": str(prev_dir.resolve()),
            "fallback_reason": "",
        }

    base_dir = candidate_base_dir(
        model_root=model_root,
        task_id=task_id,
        train_seed=train_seed,
        base_model_name=base_model_name,
        base_exp_name=base_exp_name,
    )
    if base_dir.exists() and has_saved_actor(base_dir):
        reason = "threshold_checkpoint_missing"
        if prev_dir.exists():
            reason = "threshold_checkpoint_has_no_saved_actor"
        return {
            "model_dir": base_dir.resolve(),
            "source": "base_hybridrarl_no_reuse",
            "used_reusable_modules": False,
            "threshold_model_dir": str(prev_dir.resolve()),
            "fallback_reason": reason,
        }

    return {
        "model_dir": None,
        "source": "missing",
        "used_reusable_modules": False,
        "threshold_model_dir": str(prev_dir.resolve()),
        "fallback_reason": "threshold_and_base_checkpoints_missing",
        "base_model_dir": str(base_dir.resolve()),
    }


def ordered_unique_paths(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for path in paths:
        path = Path(path).expanduser().resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        out.append(path)
    return out


def relocate_saved_model_dir(saved_path: Path, search_roots: Sequence[Path]) -> Path:
    saved_path = Path(saved_path).expanduser()
    if has_saved_actor(saved_path):
        return saved_path.resolve()

    for root in search_roots:
        root = Path(root).expanduser()
        for candidate in (root / saved_path.name, root.parent / saved_path.name):
            if has_saved_actor(candidate):
                return candidate.resolve()
        for pattern in ("model.pt", "hybridrarl_modules.pt", "hybridrarl_prev_units_modules.pt"):
            for model_path in root.rglob(pattern):
                if model_path.parent.name == saved_path.name and has_saved_actor(model_path.parent):
                    return model_path.parent.resolve()

    return saved_path.resolve()


def load_long_term_memory(model_dir: Path, models_module, map_location):
    memory_path = model_dir / "long_term_memory.pt"
    memory_cls = getattr(models_module, "LongTermMemory", None)
    if not memory_path.exists() or memory_cls is None:
        return None
    return memory_cls.load(model_dir, map_location=map_location)


def attach_hybridrarl_memory(model: nn.Module, model_dir: Path, models_module, map_location) -> None:
    if not hasattr(model, "memory"):
        return

    memory_path = model_dir / "hybridrarl_memory.pt"
    if memory_path.exists():
        memory_cls = getattr(models_module, "ReservoirTrajectoryMemory", None)
        if memory_cls is not None:
            model.memory = memory_cls.load(model_dir, map_location=map_location)
            return

    state_memory_path = model_dir / "hybridrarl_state_memory.pt"
    if state_memory_path.exists():
        state_memory_cls = getattr(models_module, "ReservoirStateMemory", None)
        if state_memory_cls is not None:
            model.memory = state_memory_cls.load(model_dir, map_location=map_location)
            return

    replay_state_path = model_dir / "hybridrarl_replay_state_memory.pt"
    if replay_state_path.exists():
        payload = safe_torch_load(replay_state_path, map_location=map_location)
        if isinstance(payload, dict) and "states" in payload and "actions" in payload:
            model.memory = {
                "states": payload["states"].to(map_location),
                "obs": payload.get("obs", payload["states"]).to(map_location),
                "actions": payload["actions"].to(map_location),
            }


def load_hybridrarl_modules(
    model_dir: Path,
    models_module,
    map_location,
    search_roots: Sequence[Path],
    loading_stack: set,
) -> nn.Module:
    prev_modules_path = model_dir / "hybridrarl_prev_units_modules.pt"
    base_modules_path = model_dir / "hybridrarl_modules.pt"
    modules_path = prev_modules_path if prev_modules_path.exists() else base_modules_path
    if not modules_path.exists():
        raise FileNotFoundError(
            f"No HybridRARL module checkpoint found in {model_dir}."
        )

    checkpoint = safe_torch_load(modules_path, map_location=map_location)
    if not isinstance(checkpoint, dict) or "meta" not in checkpoint:
        raise TypeError(f"Invalid HybridRARL module checkpoint: {modules_path}")

    meta = checkpoint["meta"]
    is_prev_units = modules_path.name == "hybridrarl_prev_units_modules.pt"
    actor_cls_name = "HybridRARLPrevUnitsAgent" if is_prev_units else "HybridRARLAgent"
    actor_cls = getattr(models_module, actor_cls_name, None)
    memory_cls = getattr(models_module, "ReservoirTrajectoryMemory", None)
    state_memory_cls = getattr(models_module, "ReservoirStateMemory", None)
    if actor_cls is None or memory_cls is None:
        raise ImportError(
            f"`models` does not expose {actor_cls_name} and ReservoirTrajectoryMemory."
        )

    memory = None
    if (model_dir / "hybridrarl_memory.pt").exists():
        memory = memory_cls.load(model_dir, map_location=map_location)
    elif (model_dir / "hybridrarl_state_memory.pt").exists() and state_memory_cls is not None:
        memory = state_memory_cls.load(model_dir, map_location=map_location)
    elif (model_dir / "hybridrarl_replay_state_memory.pt").exists():
        payload = safe_torch_load(model_dir / "hybridrarl_replay_state_memory.pt", map_location=map_location)
        if isinstance(payload, dict) and "states" in payload and "actions" in payload:
            memory = {
                "states": payload["states"].to(map_location),
                "obs": payload.get("obs", payload["states"]).to(map_location),
                "actions": payload["actions"].to(map_location),
            }

    common_kwargs = {
        key: meta[key]
        for key in (
            "obs_dim",
            "act_dim",
            "memory_capacity",
            "traj_len",
            "top_r",
            "traj_hidden_dim",
            "att_dim",
            "value_dim",
            "policy_hidden_dim",
            "policy_layers",
            "coarse_metric",
            "random_coarse_retrieval",
            "random_state_retrieval",
        )
        if key in meta
    }
    common_kwargs["memory"] = memory

    if "random_coarse_retrieval" not in common_kwargs and (
        normalized_model_name(model_name_from_dir(model_dir.name)) == "hybridrarl-random-retrieval"
    ):
        common_kwargs["random_coarse_retrieval"] = True
    if "random_state_retrieval" not in common_kwargs and (
        normalized_model_name(model_name_from_dir(model_dir.name)) == "hybridrarl-random-state-retrieval"
    ):
        common_kwargs["random_state_retrieval"] = True

    if is_prev_units:
        prev_paths = [
            relocate_saved_model_dir(Path(path), search_roots)
            for path in meta.get("prev_unit_paths", [])
        ]
        prev_units = [
            load_policy_unit(
                path,
                models_module=models_module,
                map_location=map_location,
                search_roots=search_roots,
                loading_stack=loading_stack,
            )
            for path in prev_paths
        ]
        common_kwargs["prev_units"] = prev_units
        common_kwargs["prev_unit_paths"] = prev_paths
        if "prev_att_dim" in meta:
            common_kwargs["prev_att_dim"] = meta["prev_att_dim"]

    model = actor_cls(**common_kwargs)
    for name in (
        "trajectory_rnn",
        "attention_head",
        "prev_unit_attention",
        "policy",
        "fc_mean",
        "fc_logstd",
    ):
        state_dict = checkpoint.get(name)
        if state_dict is not None and hasattr(model, name):
            getattr(model, name).load_state_dict(state_dict)

    if is_prev_units and hasattr(model, "prev_unit_task_ids"):
        model.prev_unit_task_ids = [int(value) for value in meta.get("prev_unit_task_ids", [])]
    return model


def load_policy_unit(
    model_dir: Path,
    models_module,
    map_location,
    search_roots: Sequence[Path],
    loading_stack: Optional[set] = None,
) -> nn.Module:
    loading_stack = set() if loading_stack is None else loading_stack
    model_dir = relocate_saved_model_dir(model_dir, search_roots)
    if model_dir in loading_stack:
        raise RuntimeError(f"Recursive prev-unit cycle detected at {model_dir}")

    loading_stack.add(model_dir)
    model_path = model_dir / "model.pt"
    if model_path.exists():
        model = safe_torch_load(model_path, map_location=map_location)
    elif (model_dir / "hybridrarl_modules.pt").exists() or (model_dir / "hybridrarl_prev_units_modules.pt").exists():
        model = load_hybridrarl_modules(
            model_dir=model_dir,
            models_module=models_module,
            map_location=map_location,
            search_roots=search_roots,
            loading_stack=loading_stack,
        )
    else:
        available = sorted(path.name for path in model_dir.iterdir()) if model_dir.exists() else []
        raise FileNotFoundError(
            f"No actor checkpoint found in {model_dir}. Available files: {available}"
        )

    if isinstance(model, dict) and "net_mean" in model:
        memory = load_long_term_memory(model_dir, models_module=models_module, map_location=map_location)
        n_traj = int(model.get("meta", {}).get("n_traj", 64))
        model = LoadedHybridDictAgent(
            net_mean=model["net_mean"],
            net_logstd=model.get("net_logstd", None),
            memory=memory,
            n_traj=n_traj,
        )

    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module in {model_path}, got {type(model)}")

    attach_hybridrarl_memory(model, model_dir, models_module=models_module, map_location=map_location)

    if hasattr(model, "set_prev_units") and hasattr(model, "prev_unit_paths"):
        nested_paths = [
            relocate_saved_model_dir(Path(path), search_roots)
            for path in getattr(model, "prev_unit_paths", [])
        ]
        nested_units = [
            load_policy_unit(
                path,
                models_module=models_module,
                map_location=map_location,
                search_roots=search_roots,
                loading_stack=loading_stack,
            )
            for path in nested_paths
        ]
        model.prev_unit_paths = [str(path) for path in nested_paths]
        model.set_prev_units(nested_units)

    model = model.to(map_location).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    loading_stack.remove(model_dir)
    return model


def reset_env(env, seed: Optional[int]):
    try:
        out = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        if seed is not None and hasattr(env, "seed"):
            env.seed(seed)
        out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated or truncated), info
    obs, reward, done, info = out
    return obs, float(reward), bool(done), info


def infer_episode_limit(env, fallback: int) -> int:
    spec = getattr(env, "spec", None)
    max_steps = getattr(spec, "max_episode_steps", None) if spec is not None else None
    if max_steps is not None:
        return int(max_steps)
    max_path_length = getattr(env, "max_path_length", None)
    if max_path_length is not None:
        return int(max_path_length)
    return int(fallback)


def first_two_outputs(out) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(out, (tuple, list)):
        mean = out[0]
        logstd = out[1] if len(out) > 1 and torch.is_tensor(out[1]) else None
        return mean, logstd
    return out, None


def model_kwargs(model: nn.Module, device) -> Dict[str, Dict[str, torch.Tensor]]:
    memory = getattr(model, "memory", None)
    if isinstance(memory, dict) and "states" in memory and "actions" in memory:
        return {
            "memory_batch": {
                "states": memory["states"].to(device),
                "obs": memory.get("obs", memory["states"]).to(device),
                "actions": memory["actions"].to(device),
            }
        }
    if memory is not None and hasattr(memory, "__len__") and len(memory) > 0 and hasattr(memory, "current"):
        return {"memory_batch": memory.current(device=device)}
    return {}


@torch.no_grad()
def select_action(
    model: nn.Module,
    obs: np.ndarray,
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
    device,
    deterministic: bool,
) -> np.ndarray:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).reshape(1, -1)
    mean, raw_logstd = first_two_outputs(model(obs_t, **model_kwargs(model, device)))

    if deterministic or raw_logstd is None:
        squashed = torch.tanh(mean)
    else:
        logstd = torch.tanh(raw_logstd)
        logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (logstd + 1)
        normal = torch.distributions.Normal(mean, logstd.exp())
        squashed = torch.tanh(normal.sample())

    action = squashed * action_scale + action_bias
    return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


def evaluate_one_task(
    *,
    tasks_module,
    models_module,
    task_id: int,
    eval_task_id: Optional[int],
    model_dir: Path,
    model_root: Path,
    train_seed: int,
    test_seed: int,
    episodes: int,
    max_steps: int,
    fallback_max_steps: int,
    deterministic: bool,
    device,
) -> Dict:
    env_task_id = int(task_id if eval_task_id is None else eval_task_id)
    env = tasks_module.get_task(env_task_id)
    model_dir = Path(model_dir).resolve()

    search_roots = ordered_unique_paths(
        [
            model_dir.parent,
            Path(model_root).resolve(),
            exp_dir() / "agents",
            exp_dir() / "agents" / "hybridrarl_prev_units",
        ]
    )

    load_error = None
    try:
        model = load_policy_unit(
            model_dir,
            models_module=models_module,
            map_location=device,
            search_roots=search_roots,
        )
    except (FileNotFoundError, RuntimeError, EOFError, OSError, pickle.UnpicklingError) as exc:
        load_error = exc
        raise RuntimeError(f"Failed to load model from {model_dir}") from load_error

    action_scale = torch.as_tensor(
        (env.action_space.high - env.action_space.low) / 2.0,
        dtype=torch.float32,
        device=device,
    )
    action_bias = torch.as_tensor(
        (env.action_space.high + env.action_space.low) / 2.0,
        dtype=torch.float32,
        device=device,
    )
    episode_limit = int(max_steps) if int(max_steps) > 0 else infer_episode_limit(env, fallback_max_steps)

    returns: List[float] = []
    successes: List[float] = []
    lengths: List[int] = []
    task_name = tasks_module.get_task_name(env_task_id)
    checkpoint_model_name = model_name_from_dir(model_dir.name)

    print(f"\n*** Eval task {task_id} model on env task {env_task_id}: {task_name}")
    print(f"*** Checkpoint model: {checkpoint_model_name or 'unknown'}")
    print(f"*** Model: {model_dir}")

    for ep in range(int(episodes)):
        episode_seed = int(test_seed) + ep
        obs = reset_env(env, seed=episode_seed)
        ep_return = 0.0
        ep_success = 0.0
        step = 0

        for step in range(1, episode_limit + 1):
            action = select_action(
                model=model,
                obs=np.asarray(obs, dtype=np.float32),
                action_scale=action_scale,
                action_bias=action_bias,
                device=device,
                deterministic=bool(deterministic),
            )
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, info = step_env(env, action)
            ep_return += float(reward)
            ep_success = max(ep_success, float(info.get("success", 0.0)))
            if done:
                break

        returns.append(float(ep_return))
        successes.append(float(ep_success))
        lengths.append(int(step))
        print(
            f"episode={ep + 1:03d}, seed={episode_seed}, "
            f"return={ep_return:.4f}, success={ep_success:.4f}, length={step}"
        )

    env.close()
    return {
        "benchmark": "meta-world",
        "model_name": "hybridrarl-threshold",
        "checkpoint_model_name": checkpoint_model_name,
        "model_task_id": int(task_id),
        "checkpoint_task_id": int(task_id),
        "eval_task_id": int(env_task_id),
        "task_name": task_name,
        "train_seed": int(train_seed),
        "test_seed": int(test_seed),
        "episodes": int(episodes),
        "deterministic": bool(deterministic),
        "model_dir": str(model_dir),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_success": float(np.mean(successes)),
        "std_success": float(np.std(successes)),
        "mean_length": float(np.mean(lengths)),
        "returns": returns,
        "successes": successes,
        "lengths": lengths,
    }


def overall_success_summary(records: List[Dict]) -> Dict[str, float]:
    task_mean_successes = np.asarray([float(r["mean_success"]) for r in records], dtype=np.float64)
    task_mean_returns = np.asarray([float(r["mean_return"]) for r in records], dtype=np.float64)
    episode_successes = np.asarray(
        [float(v) for r in records for v in r.get("successes", [])],
        dtype=np.float64,
    )
    episode_returns = np.asarray(
        [float(v) for r in records for v in r.get("returns", [])],
        dtype=np.float64,
    )
    if task_mean_successes.size == 0:
        return {
            "num_tasks": 0,
            "num_episodes": 0,
            "mean_return_across_tasks": 0.0,
            "var_return_across_tasks": 0.0,
            "mean_return_across_episodes": 0.0,
            "var_return_across_episodes": 0.0,
            "mean_success_across_tasks": 0.0,
            "var_success_across_tasks": 0.0,
            "mean_success_across_episodes": 0.0,
            "var_success_across_episodes": 0.0,
        }
    return {
        "num_tasks": int(task_mean_successes.size),
        "num_episodes": int(episode_successes.size),
        "mean_return_across_tasks": float(np.mean(task_mean_returns)),
        "var_return_across_tasks": float(np.var(task_mean_returns)),
        "mean_return_across_episodes": float(np.mean(episode_returns)) if episode_returns.size else 0.0,
        "var_return_across_episodes": float(np.var(episode_returns)) if episode_returns.size else 0.0,
        "mean_success_across_tasks": float(np.mean(task_mean_successes)),
        "var_success_across_tasks": float(np.var(task_mean_successes)),
        "mean_success_across_episodes": float(np.mean(episode_successes)) if episode_successes.size else 0.0,
        "var_success_across_episodes": float(np.var(episode_successes)) if episode_successes.size else 0.0,
    }


def exception_chain_message(exc: BaseException) -> str:
    messages = []
    current: Optional[BaseException] = exc
    while current is not None:
        messages.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " | caused by: ".join(messages)


def is_model_load_failure(exc: BaseException) -> bool:
    message = str(exc)
    return (
        isinstance(exc, RuntimeError)
        and message.startswith("Failed to load model from ")
    )


def sorted_records_for_csv(records: Iterable[Dict]) -> List[Dict]:
    return sorted(records, key=lambda row: (float(row["threshold_value"]), int(row["model_task_id"])))


def write_records(records: List[Dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "threshold",
        "threshold_value",
        "threshold_exp_name",
        "model_task_id",
        "eval_task_id",
        "task_name",
        "checkpoint_model_name",
        "checkpoint_task_id",
        "selected_source",
        "used_reusable_modules",
        "fallback_reason",
        "model_dir",
        "threshold_model_dir",
        "train_seed",
        "test_seed",
        "episodes",
        "deterministic",
        "mean_return",
        "std_return",
        "mean_success",
        "std_success",
        "mean_length",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted_records_for_csv(records):
            writer.writerow({key: record.get(key, "") for key in fieldnames})

    with output_csv.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(sorted_records_for_csv(records), f, indent=2, sort_keys=True)


def write_summary(summaries: List[Dict], output_csv: Path) -> None:
    fieldnames = [
        "threshold",
        "threshold_value",
        "threshold_exp_name",
        "num_tasks",
        "num_threshold_prev_units",
        "num_base_fallback",
        "num_missing_skipped",
        "num_episodes",
        "mean_return_across_tasks",
        "var_return_across_tasks",
        "mean_return_across_episodes",
        "var_return_across_episodes",
        "mean_success_across_tasks",
        "var_success_across_tasks",
        "mean_success_across_episodes",
        "var_success_across_episodes",
    ]
    summary_csv = output_csv.with_name(output_csv.stem + "__summary.csv")
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in sorted(summaries, key=lambda row: float(row["threshold_value"])):
            writer.writerow({key: summary.get(key, "") for key in fieldnames})

    summary_json = output_csv.with_name(output_csv.stem + "__summary.json")
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            sorted(summaries, key=lambda row: float(row["threshold_value"])),
            f,
            indent=2,
            sort_keys=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Meta-World evaluator for HybridRARL threshold prev-units "
            "checkpoints with base-HybridRARL fallback."
        )
    )
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--test-seed", type=int, required=True)
    parser.add_argument("--episodes", "--episode", dest="episodes", type=int, required=True)
    parser.add_argument("--tasks", type=int, nargs="*", default=None)
    parser.add_argument("--thresholds", nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--model-root", type=Path, default=Path("agents"))
    parser.add_argument("--prev-root", type=Path, default=Path("agents/hybridrarl_prev_units"))
    parser.add_argument("--base-model-name", default="hybridrarl")
    parser.add_argument("--base-exp-name", default="SAC_hybridRAGRL")
    parser.add_argument("--prev-model-name", default="hybridrarl-prev-units")
    parser.add_argument("--eval-task-id", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--fallback-max-steps", type=int, default=500)
    parser.add_argument("--deterministic", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--cuda", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/threshold_eval"))
    parser.add_argument("--file-prefix", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = exp_dir()
    tasks_module, models_module = import_meta_world_modules()
    task_names = getattr(tasks_module, "tasks", [])
    task_ids = args.tasks if args.tasks is not None else default_task_ids(task_names)

    model_root = resolve_path(args.model_root, base_dir)
    prev_root = resolve_path(args.prev_root, base_dir)
    output_dir = resolve_path(args.output_dir, base_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    set_seed(args.test_seed)

    prefix = args.file_prefix or f"hybridrarl_thresholds__train{args.train_seed}__test{args.test_seed}"
    output_csv = output_dir / f"{prefix}.csv"

    print("*** Standalone evaluator: does not use experiments/test_rl_model.py")
    print("*** Benchmark: meta-world")
    print(f"*** Thresholds: {[parse_threshold(t) for t in args.thresholds]}")
    print(f"*** Tasks: {[int(t) for t in task_ids]}")
    print(f"*** Train seed: {args.train_seed}")
    print(f"*** Test seed: {args.test_seed}")
    print(f"*** Episodes: {args.episodes}")
    print(f"*** Device: {device}")
    print(f"*** Prev-units root: {prev_root}")
    print(f"*** Base model root: {model_root}")

    all_records: List[Dict] = []
    summaries: List[Dict] = []
    skipped_records: List[Dict] = []

    for raw_threshold in args.thresholds:
        threshold = parse_threshold(raw_threshold)
        exp_name = threshold_exp_name(threshold)
        threshold_records: List[Dict] = []
        print(f"\n=== Threshold {threshold} ({exp_name}) ===")

        for raw_task_id in task_ids:
            task_id = int(raw_task_id)
            resolved = resolve_checkpoint_for_threshold(
                prev_root=prev_root,
                model_root=model_root,
                task_id=task_id,
                threshold=threshold,
                train_seed=args.train_seed,
                prev_model_name=args.prev_model_name,
                base_model_name=args.base_model_name,
                base_exp_name=args.base_exp_name,
            )
            source = str(resolved["source"])
            if source == "missing":
                skipped = {
                    "threshold": threshold,
                    "threshold_value": float(threshold),
                    "threshold_exp_name": exp_name,
                    "model_task_id": task_id,
                    "eval_task_id": args.eval_task_id if args.eval_task_id is not None else task_id,
                    "selected_source": source,
                    "used_reusable_modules": False,
                    "fallback_reason": str(resolved["fallback_reason"]),
                    "model_dir": "",
                    "threshold_model_dir": str(resolved["threshold_model_dir"]),
                    "base_model_dir": str(resolved["base_model_dir"]),
                }
                skipped_records.append(skipped)
                print(
                    f"*** task={task_id:02d} threshold={threshold}: "
                    "skip because neither threshold prev-units nor base HybridRARL checkpoint exists"
                )
                print(f"    threshold checkpoint: {resolved['threshold_model_dir']}")
                print(f"    base fallback:        {resolved['base_model_dir']}")
                continue

            model_dir = Path(resolved["model_dir"])
            if source == "base_hybridrarl_no_reuse":
                print(
                    f"*** task={task_id:02d} threshold={threshold}: "
                    f"no threshold prev-units checkpoint; fallback to {model_dir.name}"
                )
            else:
                print(
                    f"*** task={task_id:02d} threshold={threshold}: "
                    f"use prev-units checkpoint {model_dir.name}"
                )

            try:
                record = evaluate_one_task(
                    tasks_module=tasks_module,
                    models_module=models_module,
                    task_id=task_id,
                    eval_task_id=args.eval_task_id,
                    model_dir=model_dir,
                    model_root=model_root,
                    train_seed=args.train_seed,
                    test_seed=args.test_seed,
                    episodes=args.episodes,
                    max_steps=args.max_steps,
                    fallback_max_steps=args.fallback_max_steps,
                    deterministic=args.deterministic,
                    device=device,
                )
            except Exception as exc:
                if not is_model_load_failure(exc):
                    raise

                first_error = exception_chain_message(exc)
                base_dir = candidate_base_dir(
                    model_root=model_root,
                    task_id=task_id,
                    train_seed=args.train_seed,
                    base_model_name=args.base_model_name,
                    base_exp_name=args.base_exp_name,
                ).resolve()

                if source == "threshold_prev_units" and has_saved_actor(base_dir):
                    print(
                        f"*** task={task_id:02d} threshold={threshold}: "
                        "threshold checkpoint exists but cannot be loaded; try base HybridRARL fallback"
                    )
                    print(f"    failed threshold checkpoint: {model_dir}")
                    print(f"    load error: {first_error}")
                    print(f"    base fallback: {base_dir}")
                    try:
                        record = evaluate_one_task(
                            tasks_module=tasks_module,
                            models_module=models_module,
                            task_id=task_id,
                            eval_task_id=args.eval_task_id,
                            model_dir=base_dir,
                            model_root=model_root,
                            train_seed=args.train_seed,
                            test_seed=args.test_seed,
                            episodes=args.episodes,
                            max_steps=args.max_steps,
                            fallback_max_steps=args.fallback_max_steps,
                            deterministic=args.deterministic,
                            device=device,
                        )
                    except Exception as base_exc:
                        if not is_model_load_failure(base_exc):
                            raise
                        skipped = {
                            "threshold": threshold,
                            "threshold_value": float(threshold),
                            "threshold_exp_name": exp_name,
                            "model_task_id": task_id,
                            "eval_task_id": args.eval_task_id if args.eval_task_id is not None else task_id,
                            "selected_source": "load_failed",
                            "used_reusable_modules": False,
                            "fallback_reason": "threshold_checkpoint_load_failed_and_base_checkpoint_load_failed",
                            "model_dir": str(model_dir),
                            "threshold_model_dir": str(resolved["threshold_model_dir"]),
                            "base_model_dir": str(base_dir),
                            "load_error": first_error,
                            "base_load_error": exception_chain_message(base_exc),
                        }
                        skipped_records.append(skipped)
                        print(
                            f"*** task={task_id:02d} threshold={threshold}: "
                            "skip because threshold checkpoint and base fallback both failed to load"
                        )
                        print(f"    threshold error: {first_error}")
                        print(f"    base error:      {exception_chain_message(base_exc)}")
                        continue

                    source = "base_hybridrarl_no_reuse"
                    model_dir = base_dir
                    resolved["used_reusable_modules"] = False
                    resolved["fallback_reason"] = "threshold_checkpoint_load_failed"
                else:
                    skipped = {
                        "threshold": threshold,
                        "threshold_value": float(threshold),
                        "threshold_exp_name": exp_name,
                        "model_task_id": task_id,
                        "eval_task_id": args.eval_task_id if args.eval_task_id is not None else task_id,
                        "selected_source": "load_failed",
                        "used_reusable_modules": False,
                        "fallback_reason": "selected_checkpoint_load_failed",
                        "model_dir": str(model_dir),
                        "threshold_model_dir": str(resolved["threshold_model_dir"]),
                        "base_model_dir": str(base_dir),
                        "load_error": first_error,
                    }
                    skipped_records.append(skipped)
                    print(
                        f"*** task={task_id:02d} threshold={threshold}: "
                        "skip because selected checkpoint exists but cannot be loaded"
                    )
                    print(f"    selected checkpoint: {model_dir}")
                    print(f"    load error: {first_error}")
                    continue

            record.update(
                {
                    "threshold": threshold,
                    "threshold_value": float(threshold),
                    "threshold_exp_name": exp_name,
                    "selected_source": source,
                    "used_reusable_modules": bool(resolved["used_reusable_modules"]),
                    "fallback_reason": str(resolved["fallback_reason"]),
                    "threshold_model_dir": str(resolved["threshold_model_dir"]),
                }
            )
            threshold_records.append(record)
            all_records.append(record)

        summary = overall_success_summary(threshold_records)
        summary.update(
            {
                "threshold": threshold,
                "threshold_value": float(threshold),
                "threshold_exp_name": exp_name,
                "num_threshold_prev_units": int(
                    sum(1 for row in threshold_records if row["used_reusable_modules"])
                ),
                "num_base_fallback": int(
                    sum(1 for row in threshold_records if not row["used_reusable_modules"])
                ),
                "num_missing_skipped": int(
                    sum(
                        1
                        for row in skipped_records
                        if parse_threshold(str(row["threshold"])) == threshold
                    )
                ),
            }
        )
        summaries.append(summary)
        print(
            f"\nThreshold {threshold} summary: "
            f"success(tasks)={summary['mean_success_across_tasks']:.6f}, "
            f"return(tasks)={summary['mean_return_across_tasks']:.6f}, "
            f"prev_units={summary['num_threshold_prev_units']}, "
            f"fallback={summary['num_base_fallback']}, "
            f"skipped={summary['num_missing_skipped']}"
        )

    write_records(all_records, output_csv)
    write_summary(summaries, output_csv)

    print("\nOverall threshold comparison")
    for summary in sorted(summaries, key=lambda row: float(row["threshold_value"])):
        print(
            f"threshold={summary['threshold']}, "
            f"success={summary['mean_success_across_tasks']:.6f}, "
            f"return={summary['mean_return_across_tasks']:.6f}, "
            f"prev_units={summary['num_threshold_prev_units']}, "
            f"fallback={summary['num_base_fallback']}, "
            f"skipped={summary['num_missing_skipped']}"
        )

    print(f"\n*** Saved records CSV: {output_csv}")
    print(f"*** Saved records JSON: {output_csv.with_suffix('.json')}")
    print(f"*** Saved summary CSV: {output_csv.with_name(output_csv.stem + '__summary.csv')}")
    print(f"*** Saved summary JSON: {output_csv.with_name(output_csv.stem + '__summary.json')}")
    if skipped_records:
        skipped_path = output_csv.with_name(output_csv.stem + "__skipped.json")
        with skipped_path.open("w", encoding="utf-8") as f:
            json.dump(skipped_records, f, indent=2, sort_keys=True)
        print(f"*** Saved skipped-task JSON: {skipped_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
