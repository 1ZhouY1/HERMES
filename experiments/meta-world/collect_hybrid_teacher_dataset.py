#!/usr/bin/env python3
"""
Stage 1 for policy distillation.

Use saved teacher policies to interact with their own Meta-World tasks and save
a fixed distillation dataset:

    observation -> teacher Gaussian policy parameters

Hybrid teachers still use LongTermMemory while acting, but memory samples are
not used as student training observations. Simple teachers do not use memory.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from models import LongTermMemory
from tasks import get_task, get_task_name


LOG_STD_MAX = 2
LOG_STD_MIN = -20
MEMORY_TEACHER_TYPES = {"hybrid"}


def default_teacher_glob(teacher_model_type: str) -> str:
    if teacher_model_type == "simple":
        return "task_*__simple__*"
    if teacher_model_type == "hybrid":
        return "task_*__hybrid__SAC_hybridRAGRL__*"
    raise ValueError(f"Unsupported teacher model type: {teacher_model_type}")


def default_output_dir(teacher_model_type: str) -> str:
    return f"distill_datasets/{teacher_model_type}_teachers__42"


def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class LoadedHybridTeacher(nn.Module):
    def __init__(self, net_mean: nn.Module, net_logstd: Optional[nn.Module]):
        super().__init__()
        self.net_mean = net_mean
        self.net_logstd = net_logstd

    def forward(self, x: torch.Tensor, memory_batch: Dict[str, torch.Tensor]):
        if self.net_logstd is None:
            mean = self.net_mean(x, memory_batch=memory_batch)
            if isinstance(mean, (tuple, list)):
                mean = mean[0]
            return mean, torch.zeros_like(mean)

        mean, info = self.net_mean(
            x,
            memory_batch=memory_batch,
            return_info=True,
        )
        log_std = self.net_logstd(info["encoded_state"])
        return mean, log_std


def parse_task_id(path: Path) -> int:
    match = re.search(r"task_(\d+)__", path.name)
    if match is None:
        raise ValueError(f"Cannot infer task id from teacher directory name: {path}")
    return int(match.group(1))


def discover_teacher_dirs(
    teacher_root: Path,
    teacher_glob: str,
    task_ids: Optional[Sequence[int]],
    require_memory: bool,
) -> List[Path]:
    if not teacher_root.exists():
        raise FileNotFoundError(f"Teacher root does not exist: {teacher_root}")

    wanted = None if task_ids is None else set(int(t) for t in task_ids)
    teacher_dirs = []
    for path in sorted(teacher_root.glob(teacher_glob)):
        if not path.is_dir():
            continue
        if not (path / "model.pt").exists():
            continue
        if require_memory and not (path / "long_term_memory.pt").exists():
            continue
        task_id = parse_task_id(path)
        if wanted is not None and task_id not in wanted:
            continue
        teacher_dirs.append(path)

    if not teacher_dirs:
        raise FileNotFoundError(
            f"No teacher directories matching {teacher_glob!r} found under {teacher_root}"
        )
    return teacher_dirs


def load_simple_teacher(model_dir: Path, device: torch.device) -> Tuple[nn.Module, Dict]:
    payload = safe_torch_load(model_dir / "model.pt", map_location=device)
    if not isinstance(payload, nn.Module):
        raise TypeError(
            f"Unsupported simple checkpoint format in {model_dir / 'model.pt'}: {type(payload)}"
        )
    return payload.to(device).eval(), {}


def load_hybrid_teacher(model_dir: Path, device: torch.device) -> Tuple[nn.Module, Dict]:
    payload = safe_torch_load(model_dir / "model.pt", map_location=device)

    if isinstance(payload, dict) and "net_mean" in payload:
        model = LoadedHybridTeacher(
            net_mean=payload["net_mean"],
            net_logstd=payload.get("net_logstd"),
        )
        return model.to(device).eval(), payload.get("meta", {})

    if isinstance(payload, nn.Module):
        return payload.to(device).eval(), {}

    raise TypeError(
        f"Unsupported model checkpoint format in {model_dir / 'model.pt'}: {type(payload)}"
    )


def load_teacher(
    teacher_model_type: str,
    model_dir: Path,
    device: torch.device,
) -> Tuple[nn.Module, Dict]:
    if teacher_model_type == "simple":
        return load_simple_teacher(model_dir, device=device)
    if teacher_model_type == "hybrid":
        return load_hybrid_teacher(model_dir, device=device)
    raise ValueError(f"Unsupported teacher model type: {teacher_model_type}")


def load_memory(model_dir: Path) -> LongTermMemory:
    data = safe_torch_load(model_dir / "long_term_memory.pt", map_location="cpu")
    memory = LongTermMemory(
        capacity=data["capacity"],
        traj_len=data["traj_len"],
        obs_dim=data["obs_dim"],
        action_dim=data["action_dim"],
        reward_dim=data["reward_dim"],
        device="cpu",
        dtype=data.get("dtype", torch.float32),
    )
    memory.ptr = data["ptr"]
    memory.size = data["size"]
    memory.states.copy_(data["states"])
    memory.actions.copy_(data["actions"])
    memory.rewards.copy_(data["rewards"])
    return memory


def sample_memory_batch(
    memory: LongTermMemory,
    n_traj: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    effective_n_traj = min(int(n_traj), len(memory))
    traj_batch = memory.sample_trajectories(
        n_traj=effective_n_traj,
        device=device,
        replace=False,
    )
    return {
        "obs": traj_batch.states,
        "states": traj_batch.states,
        "actions": traj_batch.actions,
        "rewards": traj_batch.rewards,
    }


def sac_log_std(raw_log_std: torch.Tensor) -> torch.Tensor:
    log_std = torch.tanh(raw_log_std)
    return LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)


def forward_policy(model: nn.Module, obs: torch.Tensor, **kwargs):
    mean, raw_log_std = model(obs, **kwargs)
    return mean, sac_log_std(raw_log_std)


def forward_teacher_policy(
    teacher_model_type: str,
    teacher: nn.Module,
    obs: torch.Tensor,
    memory: Optional[LongTermMemory],
    n_traj: Optional[int],
    device: torch.device,
):
    if teacher_model_type in MEMORY_TEACHER_TYPES:
        assert memory is not None
        assert n_traj is not None
        memory_batch = sample_memory_batch(memory, n_traj=n_traj, device=device)
        return forward_policy(
            teacher,
            obs,
            memory_batch=memory_batch,
        )
    return forward_policy(teacher, obs)


def reset_env(env, seed=None):
    out = env.reset(seed=seed) if seed is not None else env.reset()
    return out[0] if isinstance(out, tuple) else out


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, reward, done, info
    obs, reward, done, info = out
    return obs, reward, bool(done), info


def teacher_action(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "deterministic":
        pre_tanh = mean
    elif mode == "stochastic":
        normal = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh = normal.sample()
    else:
        raise ValueError(f"Unknown action mode: {mode}")
    return torch.tanh(pre_tanh) * action_scale + action_bias


@torch.no_grad()
def collect_one_teacher(
    teacher_model_type: str,
    model_dir: Path,
    task_id: int,
    episodes: int,
    max_steps: int,
    seed: int,
    n_traj_override: Optional[int],
    action_mode: str,
    device: torch.device,
) -> Dict:
    teacher, meta = load_teacher(
        teacher_model_type=teacher_model_type,
        model_dir=model_dir,
        device=device,
    )

    memory = None
    n_traj = None
    if teacher_model_type in MEMORY_TEACHER_TYPES:
        memory = load_memory(model_dir)
        if len(memory) == 0:
            raise ValueError(f"Teacher memory is empty: {model_dir}")

        n_traj = int(meta.get("n_traj", min(512, len(memory))))
        if n_traj_override is not None:
            n_traj = int(n_traj_override)
        n_traj = max(1, min(n_traj, len(memory)))

    env = get_task(task_id)
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
    teacher_means = []
    teacher_log_stds = []
    actions = []
    successes = []
    returns = []
    lengths = []

    for ep in range(episodes):
        obs = reset_env(env, seed=seed + task_id * 1000 + ep)
        ep_success = 0.0
        ep_return = 0.0
        ep_len = 0

        for step in range(1, max_steps + 1):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mean, log_std = forward_teacher_policy(
                teacher_model_type=teacher_model_type,
                teacher=teacher,
                obs=obs_t,
                memory=memory,
                n_traj=n_traj,
                device=device,
            )
            action = teacher_action(
                mean=mean.squeeze(0),
                log_std=log_std.squeeze(0),
                action_scale=action_scale,
                action_bias=action_bias,
                mode=action_mode,
            )

            observations.append(obs_t.squeeze(0).cpu())
            teacher_means.append(mean.squeeze(0).cpu())
            teacher_log_stds.append(log_std.squeeze(0).cpu())
            actions.append(action.cpu())

            obs, reward, done, info = step_env(env, action.cpu().numpy().astype(np.float32))
            ep_return += float(reward)
            ep_success = max(ep_success, float(info.get("success", 0.0)))
            ep_len = step
            if done:
                break

        successes.append(ep_success)
        returns.append(ep_return)
        lengths.append(ep_len)
        print(
            f"task {task_id:02d} episode {ep + 1:02d}: "
            f"success={ep_success:.0f}, return={ep_return:.2f}, length={ep_len}"
        )

    env.close()

    if not observations:
        raise RuntimeError(f"No samples collected for task {task_id}.")

    return {
        "task_id": int(task_id),
        "task_name": get_task_name(task_id),
        "teacher_model_type": teacher_model_type,
        "teacher_dir": str(model_dir),
        "teacher_memory_size": None if memory is None else len(memory),
        "n_traj": n_traj,
        "episodes": int(episodes),
        "max_steps": int(max_steps),
        "teacher_action_mode": action_mode,
        "observations": torch.stack(observations, dim=0),
        "teacher_means": torch.stack(teacher_means, dim=0),
        "teacher_log_stds": torch.stack(teacher_log_stds, dim=0),
        "teacher_actions": torch.stack(actions, dim=0),
        "successes": successes,
        "returns": returns,
        "lengths": lengths,
        "success_rate": float(np.mean(successes)),
        "average_return": float(np.mean(returns)),
        "average_length": float(np.mean(lengths)),
        "num_samples": len(observations),
    }


def collect(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    teacher_root = Path(args.teacher_root).expanduser()
    teacher_glob = args.teacher_glob or default_teacher_glob(args.teacher_model_type)
    output_dir = Path(args.output_dir or default_output_dir(args.teacher_model_type)).expanduser()
    require_memory = args.teacher_model_type in MEMORY_TEACHER_TYPES
    if args.teacher_dirs:
        teacher_dirs = [Path(p).expanduser() for p in args.teacher_dirs]
        if args.teacher_task_ids is None or len(args.teacher_task_ids) != len(teacher_dirs):
            raise ValueError("--teacher-task-ids must be provided with the same length as --teacher-dirs.")
        task_ids = [int(t) for t in args.teacher_task_ids]
    else:
        teacher_dirs = discover_teacher_dirs(
            teacher_root=teacher_root,
            teacher_glob=teacher_glob,
            task_ids=args.tasks,
            require_memory=require_memory,
        )
        task_ids = [parse_task_id(path) for path in teacher_dirs]

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format": "policy_distillation_dataset_v1",
        "teacher_model_type": args.teacher_model_type,
        "teacher_root": str(teacher_root),
        "teacher_glob": teacher_glob,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "teacher_action_mode": args.teacher_action_mode,
        "seed": args.seed,
        "tasks": [],
    }

    print("*** Collecting teacher distillation datasets ***")
    print(f"teacher_model_type: {args.teacher_model_type}")
    print(f"teacher_glob: {teacher_glob}")
    print(f"device: {device}")
    print(f"output_dir: {output_dir}")

    for model_dir, task_id in zip(teacher_dirs, task_ids):
        print(f"\n*** Collect task {task_id}: {model_dir} ***")
        data = collect_one_teacher(
            teacher_model_type=args.teacher_model_type,
            model_dir=model_dir,
            task_id=task_id,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            n_traj_override=args.n_traj,
            action_mode=args.teacher_action_mode,
            device=device,
        )
        dataset_path = output_dir / f"task_{task_id:02d}.pt"
        torch.save(data, dataset_path)
        manifest["tasks"].append(
            {
                "task_id": task_id,
                "task_name": data["task_name"],
                "teacher_model_type": data["teacher_model_type"],
                "path": str(dataset_path.name),
                "teacher_dir": str(model_dir),
                "num_samples": data["num_samples"],
                "success_rate": data["success_rate"],
                "average_return": data["average_return"],
                "average_length": data["average_length"],
                "teacher_memory_size": data["teacher_memory_size"],
                "n_traj": data["n_traj"],
            }
        )
        print(
            f"saved {dataset_path}: samples={data['num_samples']}, "
            f"success_rate={data['success_rate']:.4f}"
        )

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nsaved manifest: {manifest_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect rollout datasets from teacher policies for policy distillation."
    )
    parser.add_argument(
        "--teacher-model-type",
        choices=["simple", "hybrid"],
        default="hybrid",
        help="Teacher architecture to load. Add new branches here for future models.",
    )
    parser.add_argument("--teacher-root", default="agents")
    parser.add_argument(
        "--teacher-glob",
        default=None,
        help="Optional glob under --teacher-root. Defaults depend on --teacher-model-type.",
    )
    parser.add_argument("--teacher-dirs", nargs="*", default=None)
    parser.add_argument("--teacher-task-ids", type=int, nargs="*", default=None)
    parser.add_argument("--tasks", type=int, nargs="*", default=list(range(10)))
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output dataset directory. Defaults depend on --teacher-model-type.",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--n-traj", type=int, default=None)
    parser.add_argument(
        "--teacher-action-mode",
        choices=["deterministic", "stochastic"],
        default="deterministic",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    collect(parse_args())
