#!/usr/bin/env python3
"""Compute pairwise sliced Wasserstein distances between task state sets.

Each task is represented by a state matrix of shape ``(num_states, state_dim)``.
For every random unit direction theta, the states are projected to one
dimension and the empirical 1D W2 distance is computed.  The reported metric
is the root mean square over all projections:

    SWD_2(P, Q) = sqrt(mean_theta(W2(P_theta, Q_theta)^2))

The script accepts the state files produced by train_simple_sac_collect_states.py
and other files containing a tensor under ``--state-key``.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from tasks import get_task_name


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def task_name(task_id: int) -> str:
    try:
        return get_task_name(task_id)
    except Exception:
        return f"task_{task_id}"


def infer_task_id(path: Path, payload) -> int:
    if isinstance(payload, dict):
        meta = payload.get("meta", {})
        if isinstance(meta, dict) and meta.get("task_id") is not None:
            return int(meta["task_id"])
        if payload.get("task_id") is not None:
            return int(payload["task_id"])

    match = re.search(r"task[_-](\d+)", path.stem)
    if match is None:
        raise ValueError(f"Cannot infer task id from {path}")
    return int(match.group(1))


def extract_states(payload, path: Path, state_key: str) -> np.ndarray:
    if isinstance(payload, torch.Tensor):
        value = payload
    elif isinstance(payload, dict):
        data = payload.get("data", payload)
        if state_key in data:
            value = data[state_key]
        elif "states" in data:
            value = data["states"]
        elif "observations" in data:
            value = data["observations"]
        else:
            raise KeyError(
                f"{path}: cannot find state key {state_key!r}; "
                f"available keys: {sorted(data.keys())}"
            )
    else:
        raise TypeError(f"{path}: expected a tensor or dict, got {type(payload)}")

    states = np.asarray(
        value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value,
        dtype=np.float64,
    )
    if states.ndim == 1:
        states = states.reshape(1, -1)
    if states.ndim != 2:
        raise ValueError(f"{path}: expected states with shape (N,D), got {states.shape}")
    if states.shape[0] == 0 or states.shape[1] == 0:
        raise ValueError(f"{path}: state matrix is empty")
    if not np.isfinite(states).all():
        raise ValueError(f"{path}: state matrix contains NaN or infinite values")
    return states


def discover_state_files(
    state_dir: Path,
    file_glob: str,
    task_ids: Optional[Sequence[int]],
) -> List[Path]:
    files = sorted(path for path in state_dir.glob(file_glob) if path.is_file())
    if not files:
        raise FileNotFoundError(
            f"No state files matching {file_glob!r} found under {state_dir}"
        )

    wanted = None if task_ids is None else {int(task_id) for task_id in task_ids}
    selected = []
    found = set()
    for path in files:
        payload = safe_torch_load(path)
        task_id = infer_task_id(path, payload)
        if wanted is None or task_id in wanted:
            selected.append(path)
            found.add(task_id)

    if wanted is not None:
        missing = sorted(wanted - found)
        if missing:
            print(f"warning: requested tasks not found: {missing}")
    if not selected:
        raise FileNotFoundError("No requested task state files were found")
    return selected


def load_task_states(
    files: Sequence[Path],
    state_key: str,
    max_states_per_task: Optional[int],
    seed: int,
) -> Tuple[Dict[int, np.ndarray], List[Dict]]:
    rng = np.random.default_rng(seed)
    task_states: Dict[int, np.ndarray] = {}
    metadata = []

    for path in files:
        payload = safe_torch_load(path)
        task_id = infer_task_id(path, payload)
        if task_id in task_states:
            raise ValueError(f"Duplicate state file for task {task_id}: {path}")

        states = extract_states(payload, path, state_key)
        if max_states_per_task is not None and states.shape[0] > max_states_per_task:
            indices = rng.choice(states.shape[0], max_states_per_task, replace=False)
            states = states[indices]

        task_states[task_id] = states
        metadata.append({
            "task_id": task_id,
            "task_name": task_name(task_id),
            "file": str(path),
            "num_states": int(states.shape[0]),
            "state_dim": int(states.shape[1]),
        })
        print(
            f"loaded task {task_id:02d} {task_name(task_id)}: "
            f"{states.shape[0]} states, dim={states.shape[1]}"
        )

    dims = {states.shape[1] for states in task_states.values()}
    if len(dims) != 1:
        raise ValueError(f"All tasks must have the same state dimension: {dims}")
    return task_states, sorted(metadata, key=lambda item: item["task_id"])


def normalize_states(task_states: Dict[int, np.ndarray]):
    all_states = np.concatenate(list(task_states.values()), axis=0)
    mean = all_states.mean(axis=0)
    std = np.maximum(all_states.std(axis=0), 1e-12)
    normalized = {
        task_id: (states - mean) / std
        for task_id, states in task_states.items()
    }
    return normalized, mean, std


def empirical_w2_squared(x: np.ndarray, y: np.ndarray) -> float:
    """Return squared 1D W2, using empirical quantile interpolation."""
    x = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(y, dtype=np.float64).reshape(-1))
    n = max(x.size, y.size)
    quantiles = (np.arange(n, dtype=np.float64) + 0.5) / n

    def quantile_values(values):
        positions = (np.arange(values.size, dtype=np.float64) + 0.5) / values.size
        return np.interp(quantiles, positions, values)

    diff = quantile_values(x) - quantile_values(y)
    return float(np.mean(diff * diff))


def sliced_wasserstein_2(
    x: np.ndarray,
    y: np.ndarray,
    directions: np.ndarray,
) -> float:
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(f"Expected x and y as (N,D) with equal D, got {x.shape}, {y.shape}")
    if directions.ndim != 2 or directions.shape[1] != x.shape[1]:
        raise ValueError(f"Direction shape {directions.shape} does not match state dim {x.shape[1]}")

    x_proj = x @ directions.T
    y_proj = y @ directions.T
    per_projection = [
        empirical_w2_squared(x_proj[:, i], y_proj[:, i])
        for i in range(directions.shape[0])
    ]
    return float(np.sqrt(np.mean(per_projection)))


def pairwise_swd(task_states: Dict[int, np.ndarray], num_projections: int, seed: int):
    task_ids = sorted(task_states)
    state_dim = next(iter(task_states.values())).shape[1]
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(num_projections, state_dim))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)

    distances = np.zeros((len(task_ids), len(task_ids)), dtype=np.float64)
    for i, task_i in enumerate(task_ids):
        for j in range(i + 1, len(task_ids)):
            distance = sliced_wasserstein_2(
                task_states[task_i], task_states[task_ids[j]], directions
            )
            distances[i, j] = distances[j, i] = distance
            print(f"SWD task {task_i:02d} vs {task_ids[j]:02d}: {distance:.8f}")
    return task_ids, distances, directions


def write_csv(path: Path, task_ids: Sequence[int], distances: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"task_{task_id}:{task_name(task_id)}" for task_id in task_ids]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task"] + labels)
        for label, row in zip(labels, distances):
            writer.writerow([label] + [f"{value:.10f}" for value in row])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=Path("simple_sac_20k_states"))
    parser.add_argument("--file-glob", default="task_*_simple_sac_20k_states.pt")
    parser.add_argument("--tasks", type=int, nargs="*", default=None)
    parser.add_argument("--state-key", default="states")
    parser.add_argument("--max-states-per-task", type=int, default=None)
    parser.add_argument("--num-projections", type=int, default=512)
    parser.add_argument("--normalize", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path, default=Path("analysis/state_swd.csv"))
    parser.add_argument("--output-pt", type=Path, default=Path("analysis/state_swd.pt"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_projections <= 0:
        raise ValueError("--num-projections must be positive")

    files = discover_state_files(args.state_dir.expanduser(), args.file_glob, args.tasks)
    task_states, metadata = load_task_states(
        files, args.state_key, args.max_states_per_task, args.seed
    )
    if len(task_states) < 2:
        raise RuntimeError("At least two tasks are required")

    distance_states = task_states
    mean = std = None
    if args.normalize:
        distance_states, mean, std = normalize_states(task_states)

    task_ids, distances, directions = pairwise_swd(
        distance_states, args.num_projections, args.seed
    )
    write_csv(args.output_csv.expanduser(), task_ids, distances)

    args.output_pt.expanduser().parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "metric": "sliced_wasserstein_2",
        "task_ids": task_ids,
        "task_names": [task_name(task_id) for task_id in task_ids],
        "distances": torch.as_tensor(distances, dtype=torch.float32),
        "directions": torch.as_tensor(directions, dtype=torch.float32),
        "normalized": bool(args.normalize),
        "normalization_mean": None if mean is None else torch.as_tensor(mean, dtype=torch.float32),
        "normalization_std": None if std is None else torch.as_tensor(std, dtype=torch.float32),
        "metadata": metadata,
    }, args.output_pt.expanduser())

    with args.output_pt.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump({
            "metric": "sliced_wasserstein_2",
            "task_ids": task_ids,
            "task_names": [task_name(task_id) for task_id in task_ids],
            "normalized": bool(args.normalize),
            "num_projections": args.num_projections,
            "state_key": args.state_key,
            "metadata": metadata,
        }, handle, indent=2)

    print(f"\nsaved CSV: {args.output_csv}")
    print(f"saved PT:  {args.output_pt}")
    print(distances)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
