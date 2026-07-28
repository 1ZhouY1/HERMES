#!/usr/bin/env python3
"""
Collect transitions on each current task using the previous task's saved policy.

For target task t:
    t == 0: use random actions
    t > 0: load source policy for task t-1 and interact with task t

The source policy directories are selected with --source-pattern. The pattern can be:
    1. a glob containing '*', for example:
       "task_*__rarl__SAC_hybridRAGRL__42"
    2. a format string containing "{task_id}", for example:
       "task_{task_id}__rarl__SAC_hybridRAGRL__42"

Each target task is saved separately.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from models import PairMemory, RewardReplacingPairMemory
from tasks import get_task, get_task_name


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class LoadedPolicyFromParts(nn.Module):
    def __init__(self, net_mean: nn.Module, net_logstd: Optional[nn.Module]):
        super().__init__()
        self.net_mean = net_mean
        self.net_logstd = net_logstd

    def forward(self, x: torch.Tensor):
        mean = self.net_mean(x)
        if isinstance(mean, (tuple, list)):
            mean = mean[0]
        if self.net_logstd is None:
            return mean, torch.zeros_like(mean)
        return mean, self.net_logstd(x)


def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_task_id(path: Path) -> int:
    match = re.search(r"task_(\d+)__", path.name)
    if match is None:
        raise ValueError(f"Cannot infer task id from model directory name: {path}")
    return int(match.group(1))


def default_agents_root() -> Path:
    return Path(__file__).resolve().parent / "agents"


def resolve_source_dirs(
    agents_root: Path,
    source_pattern: str,
    source_task_ids: Sequence[int],
) -> Dict[int, Path]:
    source_dirs: Dict[int, Path] = {}

    if "{task_id}" in source_pattern:
        for task_id in source_task_ids:
            source_dirs[int(task_id)] = agents_root / source_pattern.format(task_id=int(task_id))
        return source_dirs

    if "*" not in source_pattern:
        raise ValueError("--source-pattern must contain '*' or '{task_id}'.")

    for path in sorted(agents_root.glob(source_pattern)):
        if not path.is_dir():
            continue
        task_id = parse_task_id(path)
        if int(task_id) in set(int(t) for t in source_task_ids):
            source_dirs.setdefault(int(task_id), path)

    missing = [int(task_id) for task_id in source_task_ids if int(task_id) not in source_dirs]
    if missing:
        raise FileNotFoundError(
            f"Could not find source model dirs for task ids {missing} under {agents_root} "
            f"with pattern {source_pattern!r}."
        )

    return source_dirs


def load_pair_memory_file(path: Path):
    data = safe_torch_load(path, map_location="cpu")
    memory_cls = RewardReplacingPairMemory if "rewards" in data else PairMemory
    kwargs = {
        "capacity": data["capacity"],
        "obs_dim": data["obs_dim"],
        "action_dim": data["action_dim"],
        "device": "cpu",
        "dtype": data.get("dtype", torch.float32),
    }
    if memory_cls is RewardReplacingPairMemory:
        kwargs["replace_neighbors"] = data.get("replace_neighbors", 50)

    memory = memory_cls(**kwargs)
    memory.ptr = int(data["ptr"])
    memory.size = int(data["size"])
    memory.states.copy_(data["states"])
    memory.actions.copy_(data["actions"])
    if hasattr(memory, "rewards") and "rewards" in data:
        memory.rewards.copy_(data["rewards"])
        memory.num_replacements = int(data.get("num_replacements", 0))
        memory.num_skips = int(data.get("num_skips", 0))
    return memory


def attach_pair_memory_if_needed(model: nn.Module, model_dir: Path) -> None:
    if not hasattr(model, "retrieve_context"):
        return
    if getattr(model, "memory", None) is not None:
        return

    for filename in ["reward_replacing_pair_memory.pt", "pair_memory.pt"]:
        memory_path = model_dir / filename
        if memory_path.exists():
            model.memory = load_pair_memory_file(memory_path)
            return


def load_policy(model_dir: Path, device: torch.device) -> nn.Module:
    if not model_dir.exists():
        raise FileNotFoundError(f"Source model dir does not exist: {model_dir}")

    model_path = model_dir / "model.pt"
    if model_path.exists():
        payload = safe_torch_load(model_path, map_location=device)
        if isinstance(payload, nn.Module):
            model = payload
        elif isinstance(payload, dict) and "net_mean" in payload:
            model = LoadedPolicyFromParts(payload["net_mean"], payload.get("net_logstd"))
        elif isinstance(payload, dict) and "state_dict" in payload:
            raise TypeError(
                f"{model_path} is a state_dict checkpoint. Please load it into a module before using this script."
            )
        else:
            raise TypeError(f"Unsupported checkpoint in {model_path}: {type(payload)}")
        attach_pair_memory_if_needed(model, model_dir)
        return model.to(device).eval()

    net_mean_path = model_dir / "net_mean.pt"
    net_logstd_path = model_dir / "net_logstd.pt"
    if net_mean_path.exists():
        net_mean = safe_torch_load(net_mean_path, map_location=device)
        net_logstd = safe_torch_load(net_logstd_path, map_location=device) if net_logstd_path.exists() else None
        return LoadedPolicyFromParts(net_mean, net_logstd).to(device).eval()

    raise FileNotFoundError(f"No model.pt or net_mean.pt found in {model_dir}")


def reset_env(env, seed=None):
    out = env.reset(seed=seed) if seed is not None else env.reset()
    return out[0] if isinstance(out, tuple) else out


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, bool(terminated or truncated), info
    obs, reward, done, info = out
    return obs, reward, bool(done), info


def sac_log_std(raw_log_std: torch.Tensor) -> torch.Tensor:
    raw_log_std = torch.tanh(raw_log_std)
    return LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (raw_log_std + 1)


@torch.no_grad()
def policy_action(
    model: nn.Module,
    obs,
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
    action_mode: str,
    device: torch.device,
) -> np.ndarray:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    out = model(obs_t)

    if isinstance(out, (tuple, list)):
        mean = out[0]
        raw_log_std = out[1] if len(out) > 1 else torch.zeros_like(mean)
    else:
        mean = out
        raw_log_std = torch.zeros_like(mean)

    if action_mode == "deterministic":
        pre_tanh_action = mean.squeeze(0)
    elif action_mode == "stochastic":
        log_std = sac_log_std(raw_log_std)
        normal = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh_action = normal.sample().squeeze(0)
    else:
        raise ValueError(f"Unknown action mode: {action_mode}")

    action = torch.tanh(pre_tanh_action) * action_scale + action_bias
    return action.cpu().numpy().astype(np.float32)


def random_action(env) -> np.ndarray:
    return np.asarray(env.action_space.sample(), dtype=np.float32).reshape(-1)


def collect_for_target_task(
    target_task_id: int,
    source_model_dir: Optional[Path],
    samples_per_task: int,
    episodes: int,
    episode_len: int,
    seed: int,
    action_mode: str,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    env = get_task(target_task_id)
    model = None if source_model_dir is None else load_policy(source_model_dir, device=device)

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

    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []
    returns = []
    successes = []
    lengths = []

    for ep in range(episodes):
        if len(observations) >= samples_per_task:
            break

        obs = reset_env(env, seed=seed + target_task_id * 1000 + ep)
        ep_return = 0.0
        ep_success = 0.0
        ep_len = 0

        for step in range(episode_len):
            if len(observations) >= samples_per_task:
                break

            if model is None:
                action = random_action(env)
            else:
                action = policy_action(
                    model=model,
                    obs=obs,
                    action_scale=action_scale,
                    action_bias=action_bias,
                    action_mode=action_mode,
                    device=device,
                )

            next_obs, reward, done, info = step_env(env, action)

            observations.append(np.asarray(obs, dtype=np.float32).reshape(-1))
            actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
            rewards.append(float(reward))
            next_observations.append(np.asarray(next_obs, dtype=np.float32).reshape(-1))
            dones.append(float(done))

            ep_return += float(reward)
            ep_success = max(ep_success, float(info.get("success", 0.0)))
            ep_len = step + 1
            obs = next_obs
            if done:
                break

        returns.append(ep_return)
        successes.append(ep_success)
        lengths.append(ep_len)
        source_label = "random" if source_model_dir is None else source_model_dir.name
        print(
            f"target task {target_task_id:02d} {get_task_name(target_task_id)} "
            f"episode {ep + 1:03d}: source={source_label}, "
            f"samples={len(observations)}/{samples_per_task}, "
            f"return={ep_return:.2f}, success={ep_success:.0f}, length={ep_len}"
        )

    env.close()

    if not observations:
        raise RuntimeError(f"No samples collected for target task {target_task_id}.")

    data = {
        "observations": torch.as_tensor(np.stack(observations, axis=0), dtype=torch.float32),
        "actions": torch.as_tensor(np.stack(actions, axis=0), dtype=torch.float32),
        "rewards": torch.as_tensor(np.asarray(rewards, dtype=np.float32)).unsqueeze(-1),
        "next_observations": torch.as_tensor(np.stack(next_observations, axis=0), dtype=torch.float32),
        "dones": torch.as_tensor(np.asarray(dones, dtype=np.float32)).unsqueeze(-1),
    }
    meta = {
        "target_task_id": int(target_task_id),
        "target_task_name": get_task_name(target_task_id),
        "source_task_id": None if source_model_dir is None else int(target_task_id) - 1,
        "source_model_dir": None if source_model_dir is None else str(source_model_dir),
        "num_samples": int(data["observations"].size(0)),
        "episodes_used": int(len(returns)),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_success": float(np.mean(successes)) if successes else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
    }
    return data, meta


def save_task_dataset(output_dir: Path, target_task_id: int, data: Dict[str, torch.Tensor], meta: Dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"target_task_{target_task_id:02d}.pt"
    torch.save({"data": data, "meta": meta}, path)
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect current-task transitions using the previous task's saved policy."
    )
    parser.add_argument("--agents-root", type=Path, default=None)
    parser.add_argument(
        "--source-pattern",
        required=True,
        help=(
            "Source model directory glob or format string. Examples: "
            "'task_*__rarl__SAC_hybridRAGRL__42' or "
            "'task_{task_id}__rarl__SAC_hybridRAGRL__42'."
        ),
    )
    parser.add_argument("--tasks", type=int, nargs="*", default=list(range(10)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-task", type=int, default=5000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--episode-len", type=int, default=500)
    parser.add_argument("--action-mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("transfer_datasets/prev_policy_on_current_task"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    agents_root = args.agents_root.expanduser() if args.agents_root else default_agents_root()
    tasks = [int(task_id) for task_id in args.tasks]
    needed_source_task_ids = sorted({task_id - 1 for task_id in tasks if task_id > 0})
    source_dirs = resolve_source_dirs(
        agents_root=agents_root,
        source_pattern=args.source_pattern,
        source_task_ids=needed_source_task_ids,
    )

    output_dir = args.output_dir.expanduser()
    datasets_dir = output_dir / "tasks"
    manifest = {
        "description": "target task transitions collected by previous task policy; task 0 is random",
        "agents_root": str(agents_root),
        "source_pattern": args.source_pattern,
        "tasks": tasks,
        "samples_per_task": int(args.samples_per_task),
        "episodes": int(args.episodes),
        "episode_len": int(args.episode_len),
        "action_mode": args.action_mode,
        "seed": int(args.seed),
        "files": [],
    }

    print("*** Previous-policy current-task collection ***")
    print(f"agents_root: {agents_root}")
    print(f"source_pattern: {args.source_pattern}")
    print(f"tasks: {tasks}")
    print(f"device: {device}")
    print(f"output_dir: {output_dir}")

    for target_task_id in tasks:
        source_model_dir = None if target_task_id == 0 else source_dirs[target_task_id - 1]
        data, meta = collect_for_target_task(
            target_task_id=target_task_id,
            source_model_dir=source_model_dir,
            samples_per_task=args.samples_per_task,
            episodes=args.episodes,
            episode_len=args.episode_len,
            seed=args.seed,
            action_mode=args.action_mode,
            device=device,
        )
        path = save_task_dataset(datasets_dir, target_task_id, data, meta)
        manifest["files"].append({**meta, "path": str(path)})
        print(f"saved target task {target_task_id:02d}: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"\nsaved manifest: {manifest_path}")
    print(f"saved task datasets: {datasets_dir}")


if __name__ == "__main__":
    main()
