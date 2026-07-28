#!/usr/bin/env python3
"""
Compute pairwise sliced Wasserstein distances from saved transfer datasets.

This script reads the .pt files produced by collect_prev_policy_on_current_task.py.
By default it looks for:

    transfer_datasets/prev_policy_on_current_task/tasks/target_task_*.pt

Each file should contain:

    {
        "data": {
            "observations": Tensor[N, obs_dim],
            "next_observations": Tensor[N, obs_dim],
            ...
        },
        "meta": {"target_task_id": int, ...}
    }

The output contains a CSV distance matrix, a .pt payload, and a JSON summary.
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
        for key in ["target_task_id", "task_id", "source_task_id"]:
            if key in meta and meta[key] is not None:
                return int(meta[key])
        for key in ["target_task_id", "task_id"]:
            if key in payload and payload[key] is not None:
                return int(payload[key])

    for pattern in [r"target_task_(\d+)", r"task_(\d+)"]:
        match = re.search(pattern, path.stem)
        if match is not None:
            return int(match.group(1))

    raise ValueError(f"Cannot infer task id from {path}")


def as_numpy_2d(value, key: str, path: Path) -> np.ndarray:
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


def extract_states(
    payload,
    path: Path,
    state_key: str,
    include_next_observations: bool,
) -> np.ndarray:
    if isinstance(payload, torch.Tensor):
        return as_numpy_2d(payload, "tensor_payload", path)

    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected dict or tensor payload, got {type(payload)}")

    data = payload.get("data", payload)
    if state_key in data:
        states = as_numpy_2d(data[state_key], state_key, path)
    elif "states" in data:
        states = as_numpy_2d(data["states"], "states", path)
    elif "observations" in data:
        states = as_numpy_2d(data["observations"], "observations", path)
    else:
        keys = ", ".join(sorted(str(key) for key in data.keys()))
        raise KeyError(f"{path}: cannot find state key {state_key!r}; available keys: {keys}")

    if include_next_observations:
        if "next_observations" not in data:
            raise KeyError(f"{path}: --include-next-observations requires next_observations")
        next_states = as_numpy_2d(data["next_observations"], "next_observations", path)
        if next_states.shape[1] != states.shape[1]:
            raise ValueError(
                f"{path}: observations dim {states.shape[1]} != next_observations dim {next_states.shape[1]}"
            )
        states = np.concatenate([states, next_states], axis=0)

    return states.astype(np.float32, copy=False)


def discover_dataset_files(dataset_dir: Path, tasks: Optional[Sequence[int]]) -> List[Path]:
    candidates_dir = dataset_dir / "tasks" if (dataset_dir / "tasks").exists() else dataset_dir
    if not candidates_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {candidates_dir}")

    files = sorted(candidates_dir.glob("target_task_*.pt"))
    if not files:
        files = sorted(candidates_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt dataset files found in {candidates_dir}")

    if tasks is None:
        return files

    wanted = set(int(task_id) for task_id in tasks)
    selected = []
    for path in files:
        payload = safe_torch_load(path, map_location="cpu")
        task_id = infer_task_id(path, payload)
        if task_id in wanted:
            selected.append(path)

    missing = sorted(wanted - {infer_task_id(path, safe_torch_load(path, map_location="cpu")) for path in selected})
    if missing:
        raise FileNotFoundError(f"Missing dataset files for task ids: {missing}")
    return selected


def load_task_states(
    files: Sequence[Path],
    state_key: str,
    include_next_observations: bool,
    max_states_per_task: Optional[int],
    seed: int,
) -> Tuple[Dict[int, np.ndarray], List[Dict]]:
    task_states: Dict[int, np.ndarray] = {}
    task_meta: List[Dict] = []
    rng = np.random.default_rng(seed)

    for path in files:
        payload = safe_torch_load(path.expanduser(), map_location="cpu")
        task_id = infer_task_id(path, payload)
        states = extract_states(
            payload=payload,
            path=path,
            state_key=state_key,
            include_next_observations=include_next_observations,
        )

        if max_states_per_task is not None and states.shape[0] > max_states_per_task:
            indices = rng.choice(states.shape[0], size=max_states_per_task, replace=False)
            states = states[indices]

        if task_id in task_states:
            raise ValueError(f"Duplicate task id {task_id} from {path}")
        task_states[task_id] = states

        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        task_meta.append(
            {
                "task_id": int(task_id),
                "task_name": task_name(task_id),
                "file": str(path),
                "num_states": int(states.shape[0]),
                "state_dim": int(states.shape[1]),
                "source_task_id": meta.get("source_task_id"),
                "source_model_dir": meta.get("source_model_dir"),
                "mean_return": meta.get("mean_return"),
                "mean_success": meta.get("mean_success"),
            }
        )
        print(
            f"loaded task {task_id:02d} {task_name(task_id)}: "
            f"{states.shape[0]} states, dim={states.shape[1]}, file={path}"
        )

    dims = {states.shape[1] for states in task_states.values()}
    if len(dims) != 1:
        detail = {task_id: states.shape for task_id, states in task_states.items()}
        raise ValueError(f"All tasks must have the same state dim, got: {detail}")

    return task_states, sorted(task_meta, key=lambda item: item["task_id"])


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

    rng = np.random.default_rng(seed)
    n = min(x.shape[0], y.shape[0])
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
            print(f"Wasserstein task {task_i:02d} vs {task_j:02d}: {distance:.6f}")

    return task_ids, distances


def write_distance_csv(path: Path, task_ids: Sequence[int], distances: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{task_id}:{task_name(task_id)}" for task_id in task_ids]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task"] + labels)
        for label, row in zip(labels, distances):
            writer.writerow([label] + [f"{value:.8f}" for value in row.tolist()])


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load saved per-task .pt datasets and compute pairwise sliced "
            "Wasserstein distances between their state distributions."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("transfer_datasets/prev_policy_on_current_task"),
        help="Directory containing task .pt files, or a parent directory with a tasks/ subdirectory.",
    )
    parser.add_argument(
        "--dataset-files",
        type=Path,
        nargs="*",
        default=None,
        help="Explicit .pt files. If given, --dataset-dir discovery is skipped.",
    )
    parser.add_argument("--tasks", type=int, nargs="*", default=None)
    parser.add_argument("--state-key", default="observations")
    parser.add_argument("--include-next-observations", action="store_true", default=False)
    parser.add_argument("--max-states-per-task", type=int, default=None)
    parser.add_argument("--num-projections", type=int, default=512)
    parser.add_argument("--normalize", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/prev_policy_state_wasserstein"))
    parser.add_argument("--output-name", default="wasserstein")
    parser.add_argument(
        "--save-states",
        default=False,
        action="store_true",
        help="Also store loaded states in the output .pt. This can make the file large.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset_files:
        files = [path.expanduser() for path in args.dataset_files]
    else:
        files = discover_dataset_files(
            dataset_dir=args.dataset_dir.expanduser(),
            tasks=args.tasks,
        )

    task_states, task_meta = load_task_states(
        files=files,
        state_key=args.state_key,
        include_next_observations=args.include_next_observations,
        max_states_per_task=args.max_states_per_task,
        seed=args.seed,
    )

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

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{args.output_name}.csv"
    output_pt = output_dir / f"{args.output_name}.pt"
    output_json = output_dir / f"{args.output_name}.json"

    write_distance_csv(output_csv, task_ids, distances)

    payload = {
        "metric": "sliced_wasserstein_2",
        "task_ids": task_ids,
        "task_names": [task_name(task_id) for task_id in task_ids],
        "distances": torch.as_tensor(distances),
        "normalized": bool(args.normalize),
        "obs_mean": None if obs_mean is None else torch.as_tensor(obs_mean),
        "obs_std": None if obs_std is None else torch.as_tensor(obs_std),
        "meta": {
            "dataset_files": [str(path) for path in files],
            "task_meta": task_meta,
            "state_key": args.state_key,
            "include_next_observations": bool(args.include_next_observations),
            "max_states_per_task": args.max_states_per_task,
            "num_projections": int(args.num_projections),
            "seed": int(args.seed),
        },
    }
    if args.save_states:
        payload["states"] = {
            int(task_id): torch.as_tensor(states)
            for task_id, states in task_states.items()
        }
    torch.save(payload, output_pt)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "output_csv": str(output_csv),
                "output_pt": str(output_pt),
                "metric": "sliced_wasserstein_2",
                "normalized": bool(args.normalize),
                "state_key": args.state_key,
                "include_next_observations": bool(args.include_next_observations),
                "task_ids": task_ids,
                "task_names": [task_name(task_id) for task_id in task_ids],
                "distances": distances.tolist(),
                "task_meta": task_meta,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    print(f"\nsaved distance csv: {output_csv}")
    print(f"saved distance pt:  {output_pt}")
    print(f"saved summary json: {output_json}")
    print("distance matrix:")
    print(distances)


if __name__ == "__main__":
    main()
