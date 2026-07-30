#!/usr/bin/env python3
"""
Compute pairwise sliced Wasserstein distances from cached task-state .pt files.

Example:
    python compute_state_cache_wasserstein.py \
        --state-dir state_cache \
        --output-csv analysis/state_cache_wasserstein.csv

The script scans only existing files, so the cache can contain a partial set of
tasks, such as task_0..task_4 while later tasks are still training.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from tasks import get_task_name


def safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def task_name(task_id: int) -> str:
    try:
        return get_task_name(task_id)
    except Exception:
        return f"task_{task_id}"


def infer_task_id(path: Path, payload) -> int:
    if isinstance(payload, dict):
        meta = payload.get("meta", {})
        if meta.get("task_id") is not None:
            return int(meta["task_id"])
        if payload.get("task_id") is not None:
            return int(payload["task_id"])

    match = re.search(r"task_(\d+)", path.stem)
    if match is not None:
        return int(match.group(1))

    raise ValueError(f"Cannot infer task id from file name or payload: {path}")


def to_numpy_2d(value, path: Path, key: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)

    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{path}: expected {key} to be 2D, got shape {array.shape}")
    return array


def extract_states(payload, path: Path, state_key: str) -> np.ndarray:
    if isinstance(payload, torch.Tensor):
        return to_numpy_2d(payload, path, "tensor_payload")
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected dict or tensor payload, got {type(payload)}")

    data = payload.get("data", payload)
    if state_key in data:
        return to_numpy_2d(data[state_key], path, state_key)
    if "states" in data:
        return to_numpy_2d(data["states"], path, "states")
    if "observations" in data:
        return to_numpy_2d(data["observations"], path, "observations")

    keys = ", ".join(sorted(str(key) for key in data.keys()))
    raise KeyError(f"{path}: cannot find state key {state_key!r}; available keys: {keys}")


def discover_state_files(
    state_dir: Path,
    file_glob: str,
    tasks: Optional[Sequence[int]],
) -> List[Path]:
    if not state_dir.exists():
        raise FileNotFoundError(f"State directory does not exist: {state_dir}")

    files = sorted(path for path in state_dir.glob(file_glob) if path.is_file())
    if not files and file_glob != "*.pt":
        files = sorted(path for path in state_dir.glob("*.pt") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No .pt state files found under {state_dir}")

    if tasks is None:
        return files

    wanted = set(int(task_id) for task_id in tasks)
    selected = []
    found = set()
    for path in files:
        payload = safe_torch_load(path, map_location="cpu")
        task_id = infer_task_id(path, payload)
        if task_id in wanted:
            selected.append(path)
            found.add(task_id)

    missing = sorted(wanted - found)
    if missing:
        print(f"warning: requested task ids not found and skipped: {missing}")
    if not selected:
        raise FileNotFoundError(f"No requested task ids found under {state_dir}")
    return selected


def load_task_states(
    files: Sequence[Path],
    state_key: str,
    max_states_per_task: Optional[int],
    seed: int,
) -> Tuple[Dict[int, np.ndarray], List[Dict]]:
    rng = np.random.default_rng(seed)
    task_states: Dict[int, np.ndarray] = {}
    task_meta: List[Dict] = []

    for path in files:
        payload = safe_torch_load(path.expanduser(), map_location="cpu")
        task_id = infer_task_id(path, payload)
        states = extract_states(payload, path, state_key=state_key)

        if max_states_per_task is not None and states.shape[0] > max_states_per_task:
            indices = rng.choice(states.shape[0], size=max_states_per_task, replace=False)
            states = states[indices]

        if task_id in task_states:
            raise ValueError(f"Duplicate state file for task {task_id}: {path}")

        task_states[task_id] = states
        task_meta.append(
            {
                "task_id": int(task_id),
                "task_name": task_name(task_id),
                "file": str(path),
                "num_states": int(states.shape[0]),
                "state_dim": int(states.shape[1]),
            }
        )
        print(
            f"loaded task {task_id:02d} {task_name(task_id)}: "
            f"{states.shape[0]} states, dim={states.shape[1]}, file={path}"
        )

    dims = {states.shape[1] for states in task_states.values()}
    if len(dims) != 1:
        detail = {task_id: states.shape for task_id, states in task_states.items()}
        raise ValueError(f"All state files must have the same state dim, got: {detail}")

    return task_states, sorted(task_meta, key=lambda item: item["task_id"])


def normalize_task_states(task_states: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    all_states = np.concatenate(list(task_states.values()), axis=0)
    mean = all_states.mean(axis=0, keepdims=True)
    std = np.maximum(all_states.std(axis=0, keepdims=True), 1e-6)
    return {
        task_id: ((states - mean) / std).astype(np.float32)
        for task_id, states in task_states.items()
    }


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

    rng = np.random.default_rng(seed)
    n = min(x.shape[0], y.shape[0])
    if x.shape[0] != n:
        x = x[rng.choice(x.shape[0], size=n, replace=False)]
    if y.shape[0] != n:
        y = y[rng.choice(y.shape[0], size=n, replace=False)]

    directions = rng.normal(size=(int(num_projections), x.shape[1])).astype(np.float32)
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
            print(f"Wasserstein task {task_i:02d} vs task {task_j:02d}: {distance:.6f}")

    return task_ids, distances


def write_distance_csv(path: Path, task_ids: Sequence[int], distances: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"task_{task_id}:{task_name(task_id)}" for task_id in task_ids]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task"] + labels)
        for label, row in zip(labels, distances):
            writer.writerow([label] + [f"{value:.8f}" for value in row.tolist()])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute pairwise sliced Wasserstein distances from cached task-state files."
    )
    parser.add_argument("--state-dir", type=Path, default=Path("state_cache"))
    parser.add_argument("--file-glob", default="task_*__sampled_states.pt")
    parser.add_argument("--tasks", type=int, nargs="*", default=None)
    parser.add_argument("--state-key", default="states")
    parser.add_argument("--max-states-per-task", type=int, default=None)
    parser.add_argument("--num-projections", type=int, default=512)
    parser.add_argument("--normalize", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path, default=Path("analysis/state_cache_wasserstein.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    files = discover_state_files(
        state_dir=args.state_dir.expanduser(),
        file_glob=args.file_glob,
        tasks=args.tasks,
    )
    task_states, task_meta = load_task_states(
        files=files,
        state_key=args.state_key,
        max_states_per_task=args.max_states_per_task,
        seed=args.seed,
    )
    if len(task_states) < 2:
        raise RuntimeError("Need at least two task-state files to compute pairwise distances.")

    distance_states = normalize_task_states(task_states) if args.normalize else task_states
    task_ids, distances = pairwise_sliced_wasserstein(
        task_states=distance_states,
        num_projections=args.num_projections,
        seed=args.seed,
    )
    write_distance_csv(args.output_csv.expanduser(), task_ids, distances)

    print(f"\nloaded tasks: {[item['task_id'] for item in task_meta]}")
    print(f"saved csv: {args.output_csv}")
    print("distance matrix:")
    print(distances)


if __name__ == "__main__":
    main()
