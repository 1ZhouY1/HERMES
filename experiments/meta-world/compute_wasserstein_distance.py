#!/usr/bin/env python3
import argparse
import csv
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
    by_task: Dict[int, Path] = {}
    for path in sorted(teacher_root.glob(teacher_glob)):
        if not path.is_dir():
            continue
        if not (path / "model.pt").exists():
            continue
        task_id = parse_task_id(path)
        if wanted is not None and task_id not in wanted:
            continue
        by_task.setdefault(task_id, path)

    if not by_task:
        raise FileNotFoundError(
            f"No simple teacher directories matching {teacher_glob!r} found under {teacher_root}"
        )
    return [by_task[task_id] for task_id in sorted(by_task)]


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
def deterministic_action(model: nn.Module, obs, action_scale, action_bias, device: torch.device) -> np.ndarray:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    mean = first_output(model(obs_t))
    action = torch.tanh(mean.squeeze(0)) * action_scale + action_bias
    return action.cpu().numpy().astype(np.float32)


@torch.no_grad()
def stochastic_action(model: nn.Module, obs, action_scale, action_bias, device: torch.device) -> np.ndarray:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    mean, raw_log_std = model(obs_t)
    log_std = torch.tanh(raw_log_std)
    log_std = -20 + 0.5 * (2 - (-20)) * (log_std + 1)
    normal = torch.distributions.Normal(mean, log_std.exp())
    pre_tanh = normal.sample().squeeze(0)
    action = torch.tanh(pre_tanh) * action_scale + action_bias
    return action.cpu().numpy().astype(np.float32)


def random_action(env) -> np.ndarray:
    return np.asarray(env.action_space.sample(), dtype=np.float32).reshape(-1)


def collect_states_for_task(
    model_dir: Path,
    task_id: int,
    states_per_task: int,
    episodes: int,
    episode_len: int,
    seed: int,
    action_mode: str,
    device: torch.device,
) -> Tuple[np.ndarray, Dict]:
    env = get_task(task_id)
    model = None if action_mode == "random" else load_simple_teacher(model_dir, device=device)

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

    states = []
    returns = []
    successes = []
    lengths = []

    for ep in range(episodes):
        if len(states) >= states_per_task:
            break

        obs = reset_env(env, seed=seed + int(task_id) * 1000 + ep)
        ep_return = 0.0
        ep_success = 0.0
        ep_len = 0

        for step in range(episode_len):
            if len(states) >= states_per_task:
                break

            states.append(np.asarray(obs, dtype=np.float32).reshape(-1))

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
            elif action_mode == "random":
                action = random_action(env)
            else:
                raise ValueError(f"Unknown action mode: {action_mode}")

            obs, reward, done, info = step_env(env, action)
            ep_return += float(reward)
            ep_success = max(ep_success, float(info.get("success", 0.0)))
            ep_len = step + 1

            if done:
                break

        returns.append(ep_return)
        successes.append(ep_success)
        lengths.append(ep_len)
        print(
            f"task {task_id:02d} {get_task_name(task_id)} episode {ep + 1:03d}: "
            f"states={len(states)}/{states_per_task}, return={ep_return:.2f}, "
            f"success={ep_success:.0f}, length={ep_len}"
        )

    env.close()

    if len(states) == 0:
        raise RuntimeError(f"No states collected for task {task_id}.")

    states_np = np.stack(states, axis=0).astype(np.float32)
    meta = {
        "task_id": int(task_id),
        "task_name": get_task_name(task_id),
        "model_dir": str(model_dir),
        "num_states": int(states_np.shape[0]),
        "episodes_used": int(len(returns)),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_success": float(np.mean(successes)) if successes else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
    }
    return states_np, meta


def normalize_task_states(task_states: Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    all_states = np.concatenate(list(task_states.values()), axis=0)
    mean = all_states.mean(axis=0, keepdims=True)
    std = all_states.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    normalized = {
        task_id: ((states - mean) / std).astype(np.float32)
        for task_id, states in task_states.items()
    }
    return normalized, mean.squeeze(0).astype(np.float32), std.squeeze(0).astype(np.float32)


def sliced_wasserstein_distance(
    x: np.ndarray,
    y: np.ndarray,
    num_projections: int,
    seed: int,
) -> float:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"x and y should be 2D arrays, got {x.shape}, {y.shape}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"state dim mismatch: {x.shape[1]} vs {y.shape[1]}")

    n = min(x.shape[0], y.shape[0])
    rng = np.random.default_rng(seed)
    if x.shape[0] != n:
        x = x[rng.choice(x.shape[0], size=n, replace=False)]
    if y.shape[0] != n:
        y = y[rng.choice(y.shape[0], size=n, replace=False)]

    directions = rng.normal(size=(num_projections, x.shape[1])).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1e-8)

    x_proj = np.sort(x @ directions.T, axis=0)
    y_proj = np.sort(y @ directions.T, axis=0)
    return float(np.sqrt(np.mean((x_proj - y_proj) ** 2)))


def pairwise_sliced_wasserstein(
    task_states: Dict[int, np.ndarray],
    num_projections: int,
    seed: int,
) -> Tuple[List[int], np.ndarray]:
    task_ids = sorted(task_states)
    distances = np.zeros((len(task_ids), len(task_ids)), dtype=np.float32)

    for i, task_i in enumerate(task_ids):
        for j in range(i + 1, len(task_ids)):
            task_j = task_ids[j]
            distance = sliced_wasserstein_distance(
                task_states[task_i],
                task_states[task_j],
                num_projections=num_projections,
                seed=seed + i * 1009 + j,
            )
            distances[i, j] = distance
            distances[j, i] = distance
            print(
                f"Wasserstein task {task_i:02d} vs {task_j:02d}: {distance:.6f}"
            )

    return task_ids, distances


def write_distance_csv(path: Path, task_ids: Sequence[int], distances: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{task_id}:{get_task_name(task_id)}" for task_id in task_ids]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task"] + labels)
        for label, row in zip(labels, distances):
            writer.writerow([label] + [f"{value:.8f}" for value in row.tolist()])


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sample states from saved simple MetaWorld teachers and compute "
            "pairwise sliced Wasserstein distances between task state distributions."
        )
    )
    parser.add_argument("--teacher-root", type=Path, default=None)
    parser.add_argument("--teacher-glob", default="task_*__simple__*")
    parser.add_argument("--teacher-dirs", type=Path, nargs="*", default=None)
    parser.add_argument("--tasks", type=int, nargs="*", default=None)
    parser.add_argument("--states-per-task", type=int, default=5000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-len", type=int, default=500)
    parser.add_argument("--action-mode", choices=["deterministic", "stochastic", "random"], default="deterministic")
    parser.add_argument("--num-projections", type=int, default=512)
    parser.add_argument("--normalize", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-csv", type=Path, default=Path("analysis/simple_state_wasserstein.csv"))
    parser.add_argument("--output-pt", type=Path, default=Path("analysis/simple_state_wasserstein.pt"))
    parser.add_argument(
        "--save-states",
        default=False,
        action="store_true",
        help="Also store sampled states in --output-pt. This can make the file large.",
    )
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

    task_states: Dict[int, np.ndarray] = {}
    task_meta = []
    for model_dir in teacher_dirs:
        task_id = parse_task_id(model_dir)
        states, meta = collect_states_for_task(
            model_dir=model_dir,
            task_id=task_id,
            states_per_task=args.states_per_task,
            episodes=args.episodes,
            episode_len=args.episode_len,
            seed=args.seed,
            action_mode=args.action_mode,
            device=device,
        )
        task_states[task_id] = states
        task_meta.append(meta)

    obs_mean = None
    obs_std = None
    distance_states = task_states
    if args.normalize:
        distance_states, obs_mean, obs_std = normalize_task_states(task_states)

    task_ids, distances = pairwise_sliced_wasserstein(
        task_states=distance_states,
        num_projections=args.num_projections,
        seed=args.seed,
    )

    write_distance_csv(args.output_csv, task_ids, distances)
    args.output_pt.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metric": "sliced_wasserstein_2",
        "task_ids": task_ids,
        "task_names": [get_task_name(task_id) for task_id in task_ids],
        "distances": torch.as_tensor(distances),
        "normalized": bool(args.normalize),
        "obs_mean": None if obs_mean is None else torch.as_tensor(obs_mean),
        "obs_std": None if obs_std is None else torch.as_tensor(obs_std),
        "meta": {
            "teacher_dirs": [str(path) for path in teacher_dirs],
            "task_meta": task_meta,
            "states_per_task": args.states_per_task,
            "episodes": args.episodes,
            "episode_len": args.episode_len,
            "action_mode": args.action_mode,
            "num_projections": args.num_projections,
            "seed": args.seed,
        },
    }
    if args.save_states:
        payload["states"] = {
            int(task_id): torch.as_tensor(states)
            for task_id, states in task_states.items()
        }
    torch.save(payload, args.output_pt)

    with args.output_pt.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "output_csv": str(args.output_csv),
                "output_pt": str(args.output_pt),
                "metric": "sliced_wasserstein_2",
                "normalized": bool(args.normalize),
                "task_ids": task_ids,
                "task_names": [get_task_name(task_id) for task_id in task_ids],
                "task_meta": task_meta,
            },
            f,
            indent=2,
        )

    print(f"\nsaved distance csv: {args.output_csv}")
    print(f"saved distance pt:  {args.output_pt}")
    print("distance matrix:")
    print(distances)


if __name__ == "__main__":
    main()
