#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from tasks import get_task, get_task_name


def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def first_output(out):
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def parse_task_id(path: Path) -> int:
    match = re.search(r"task_(\d+)__", path.name)
    if match is None:
        raise ValueError(f"Cannot infer task id from teacher directory name: {path}")
    return int(match.group(1))


def default_teacher_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    local_agents = script_dir / "agents"
    if local_agents.exists():
        return local_agents
    return Path("experiments/meta-world/agents")


def discover_teacher_dirs(
    teacher_root: Path,
    teacher_glob: str,
    task_ids: Optional[Sequence[int]],
) -> List[Path]:
    if not teacher_root.exists():
        raise FileNotFoundError(f"Teacher root does not exist: {teacher_root}")

    wanted = None if task_ids is None else set(int(task_id) for task_id in task_ids)
    dirs = []
    for path in sorted(teacher_root.glob(teacher_glob)):
        if not path.is_dir():
            continue
        if not (path / "model.pt").exists():
            continue
        task_id = parse_task_id(path)
        if wanted is not None and task_id not in wanted:
            continue
        dirs.append(path)

    if not dirs:
        raise FileNotFoundError(
            f"No simple teacher directories matching {teacher_glob!r} found under {teacher_root}"
        )
    return dirs


def load_simple_teacher(model_dir: Path, device: torch.device) -> nn.Module:
    payload = safe_torch_load(model_dir / "model.pt", map_location=device)
    if not isinstance(payload, nn.Module):
        raise TypeError(
            f"Expected a saved SimpleAgent module in {model_dir / 'model.pt'}, got {type(payload)}"
        )
    return payload.to(device).eval()


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


@torch.no_grad()
def deterministic_action(model, obs, action_scale, action_bias, device: torch.device) -> np.ndarray:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    mean = first_output(model(obs_t))
    action = torch.tanh(mean.squeeze(0)) * action_scale + action_bias
    return action.cpu().numpy().astype(np.float32)


@torch.no_grad()
def stochastic_action(model, obs, action_scale, action_bias, device: torch.device) -> np.ndarray:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    mean, raw_log_std = model(obs_t)
    log_std = torch.tanh(raw_log_std)
    log_std = -20 + 0.5 * (2 - (-20)) * (log_std + 1)
    normal = torch.distributions.Normal(mean, log_std.exp())
    pre_tanh = normal.sample().squeeze(0)
    action = torch.tanh(pre_tanh) * action_scale + action_bias
    return action.cpu().numpy().astype(np.float32)


def collect_teacher(
    model_dir: Path,
    task_id: int,
    episodes: int,
    episode_len: int,
    seed: int,
    action_mode: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    env = get_task(task_id)
    model = load_simple_teacher(model_dir, device=device)

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

    obs_episodes = []
    action_episodes = []
    returns = []
    successes = []

    for ep in range(episodes):
        obs = reset_env(env, seed=seed + int(task_id) * 1000 + ep)
        ep_obs = [np.asarray(obs, dtype=np.float32).reshape(-1)]
        ep_actions = []
        ep_return = 0.0
        ep_success = 0.0

        for step in range(episode_len):
            if action_mode == "deterministic":
                action = deterministic_action(
                    model=model,
                    obs=obs,
                    action_scale=action_scale,
                    action_bias=action_bias,
                    device=device,
                )
            elif action_mode == "stochastic":
                action = stochastic_action(
                    model=model,
                    obs=obs,
                    action_scale=action_scale,
                    action_bias=action_bias,
                    device=device,
                )
            else:
                raise ValueError(f"Unknown action mode: {action_mode}")

            next_obs, reward, done, info = step_env(env, action)
            ep_actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
            ep_obs.append(np.asarray(next_obs, dtype=np.float32).reshape(-1))
            ep_return += float(reward)
            ep_success = max(ep_success, float(info.get("success", 0.0)))
            obs = next_obs

            if done and step + 1 < episode_len:
                raise RuntimeError(
                    f"Task {task_id} episode ended at step {step + 1}, "
                    f"but --episode-len={episode_len}. Use the environment horizon or lower --episode-len."
                )

        obs_episodes.append(np.stack(ep_obs, axis=0))
        action_episodes.append(np.stack(ep_actions, axis=0))
        returns.append(ep_return)
        successes.append(ep_success)
        print(
            f"collect task {task_id:02d} {get_task_name(task_id)} "
            f"episode {ep + 1:03d}: return={ep_return:.2f}, success={ep_success:.0f}"
        )

    env.close()

    return {
        "obs": torch.as_tensor(np.stack(obs_episodes, axis=0), dtype=torch.float32),
        "actions": torch.as_tensor(np.stack(action_episodes, axis=0), dtype=torch.float32),
        "returns": torch.as_tensor(returns, dtype=torch.float32),
        "successes": torch.as_tensor(successes, dtype=torch.float32),
        "task_ids": torch.full((episodes,), int(task_id), dtype=torch.long),
    }


def merge_collected(items: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = ["obs", "actions", "returns", "successes", "task_ids"]
    return {key: torch.cat([item[key] for item in items], dim=0) for key in keys}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect fixed-length MetaWorld trajectories from saved simple teachers for Summary pretraining."
    )
    parser.add_argument("--teacher-root", type=Path, default=None)
    parser.add_argument("--teacher-glob", default="task_*__simple__*")
    parser.add_argument("--teacher-dirs", type=Path, nargs="*", default=None)
    parser.add_argument("--tasks", type=int, nargs="*", default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-len", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("summary_datasets/simple_teachers_summary_dataset.pt"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.teacher_dirs:
        teacher_dirs = [path.expanduser() for path in args.teacher_dirs]
    else:
        teacher_root = args.teacher_root.expanduser() if args.teacher_root else default_teacher_root()
        teacher_dirs = discover_teacher_dirs(
            teacher_root=teacher_root,
            teacher_glob=args.teacher_glob,
            task_ids=args.tasks,
        )

    collected = []
    meta_teachers = []
    for model_dir in teacher_dirs:
        task_id = parse_task_id(model_dir)
        item = collect_teacher(
            model_dir=model_dir,
            task_id=task_id,
            episodes=args.episodes,
            episode_len=args.episode_len,
            seed=args.seed,
            action_mode=args.action_mode,
            device=device,
        )
        collected.append(item)
        meta_teachers.append({"task_id": task_id, "task_name": get_task_name(task_id), "path": str(model_dir)})

    dataset = merge_collected(collected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **dataset,
            "meta": {
                "source": "simple_teacher_rollouts",
                "teachers": meta_teachers,
                "episodes_per_teacher": args.episodes,
                "episode_len": args.episode_len,
                "action_mode": args.action_mode,
                "seed": args.seed,
                "obs_shape": tuple(dataset["obs"].shape),
                "actions_shape": tuple(dataset["actions"].shape),
            },
        },
        args.output,
    )
    with (args.output.with_suffix(".json")).open("w", encoding="utf-8") as f:
        json.dump(
            {
                "output": str(args.output),
                "num_episodes": int(dataset["obs"].size(0)),
                "obs_shape": tuple(dataset["obs"].shape),
                "actions_shape": tuple(dataset["actions"].shape),
                "teachers": meta_teachers,
            },
            f,
            indent=2,
        )

    print(f"\nsaved dataset: {args.output}")
    print(f"obs: {tuple(dataset['obs'].shape)}")
    print(f"actions: {tuple(dataset['actions'].shape)}")


if __name__ == "__main__":
    main()
