#!/usr/bin/env python3
"""Compute pairwise sliced Wasserstein distances for HalfCheetah task states.

This is the HalfCheetah counterpart of the Meta-World ``compute_state_swd.py``.
Each task is represented by a state matrix ``X_i`` with shape ``(N, D)``.  For
every random unit direction ``theta``, states are projected to 1-D and an
empirical 1-D W2 distance is computed.  The reported sliced Wasserstein metric
is:

    SWD_2(P_i, P_j) = sqrt(mean_theta(W2(P_i^theta, P_j^theta)^2))

The script is designed for files saved by
``experiments/halfcheetah/train_simple_sac_collect_states.py`` but also accepts
plain tensors or dict-like payloads containing ``states``, ``observations``, or
``visited_states``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from tasks import get_task_name, tasks


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
        if isinstance(meta, dict):
            for key in ("task_id", "target_task_id", "source_task_id"):
                if meta.get(key) is not None:
                    return int(meta[key])
        for key in ("task_id", "target_task_id", "source_task_id"):
            if payload.get(key) is not None:
                return int(payload[key])

    for pattern in (r"task[_-](\d+)", r"target_task[_-](\d+)"):
        match = re.search(pattern, path.stem)
        if match is not None:
            return int(match.group(1))
    raise ValueError(f"Cannot infer task id from file name or payload: {path}")


def to_numpy_2d(value, path: Path, key: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)

    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{path}: expected {key} with shape (N,D), got {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{path}: empty state matrix for key {key!r}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: state matrix contains NaN or infinite values")
    return array


def extract_states(payload, path: Path, state_key: str) -> np.ndarray:
    if isinstance(payload, torch.Tensor):
        return to_numpy_2d(payload, path, "tensor_payload")
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected tensor or dict payload, got {type(payload)}")

    data = payload.get("data", payload)
    for key in (state_key, "states", "observations", "visited_states"):
        if key in data:
            return to_numpy_2d(data[key], path, key)

    keys = ", ".join(sorted(str(key) for key in data.keys()))
    raise KeyError(
        f"{path}: cannot find state key {state_key!r}; available keys: {keys}"
    )


def discover_state_files(
    state_dir: Path,
    file_glob: str,
    task_ids: Optional[Sequence[int]],
) -> List[Path]:
    state_dir = state_dir.expanduser()
    if not state_dir.exists():
        raise FileNotFoundError(f"State directory does not exist: {state_dir}")

    files = sorted(path for path in state_dir.glob(file_glob) if path.is_file())
    if not files and file_glob != "*.pt":
        files = sorted(path for path in state_dir.glob("*.pt") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No state files found under {state_dir}")

    if task_ids is None:
        return files

    wanted = {int(task_id) for task_id in task_ids}
    selected = []
    found = set()
    for path in files:
        payload = safe_torch_load(path)
        task_id = infer_task_id(path, payload)
        if task_id in wanted:
            selected.append(path)
            found.add(task_id)

    missing = sorted(wanted - found)
    if missing:
        print(f"warning: requested task ids not found: {missing}")
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
    metadata: List[Dict] = []

    for path in files:
        payload = safe_torch_load(path)
        task_id = infer_task_id(path, payload)
        if task_id in task_states:
            raise ValueError(f"Duplicate state file for task {task_id}: {path}")

        states = extract_states(payload, path, state_key=state_key)
        original_num_states = int(states.shape[0])
        if max_states_per_task is not None and states.shape[0] > max_states_per_task:
            indices = rng.choice(states.shape[0], int(max_states_per_task), replace=False)
            states = states[indices]

        task_states[task_id] = states
        metadata.append(
            {
                "task_id": int(task_id),
                "task_name": task_name(task_id),
                "file": str(path),
                "num_states": int(states.shape[0]),
                "original_num_states": original_num_states,
                "state_dim": int(states.shape[1]),
            }
        )
        print(
            f"loaded task {task_id:02d} {task_name(task_id)}: "
            f"{states.shape[0]} states, dim={states.shape[1]}"
        )

    dims = {states.shape[1] for states in task_states.values()}
    if len(dims) != 1:
        detail = {task_id: states.shape for task_id, states in task_states.items()}
        raise ValueError(f"All tasks must have the same state dimension: {detail}")
    return task_states, sorted(metadata, key=lambda item: item["task_id"])


def normalize_states_global(
    task_states: Dict[int, np.ndarray],
    eps: float,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    all_states = np.concatenate(list(task_states.values()), axis=0)
    mean = all_states.mean(axis=0)
    std = np.maximum(all_states.std(axis=0), float(eps))
    normalized = {
        task_id: (states - mean) / std
        for task_id, states in task_states.items()
    }
    return normalized, mean, std


def empirical_w2_squared(x: np.ndarray, y: np.ndarray) -> float:
    """Squared empirical 1-D W2 using quantile interpolation."""
    x_sorted = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    y_sorted = np.sort(np.asarray(y, dtype=np.float64).reshape(-1))
    n = max(x_sorted.size, y_sorted.size)
    quantiles = (np.arange(n, dtype=np.float64) + 0.5) / n

    def interp_quantiles(values: np.ndarray) -> np.ndarray:
        positions = (np.arange(values.size, dtype=np.float64) + 0.5) / values.size
        return np.interp(quantiles, positions, values)

    diff = interp_quantiles(x_sorted) - interp_quantiles(y_sorted)
    return float(np.mean(diff * diff))


def sliced_wasserstein_2(
    x: np.ndarray,
    y: np.ndarray,
    directions: np.ndarray,
) -> float:
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(f"Expected x and y as (N,D) with equal D, got {x.shape}, {y.shape}")
    if directions.ndim != 2 or directions.shape[1] != x.shape[1]:
        raise ValueError(
            f"Direction shape {directions.shape} does not match state dim {x.shape[1]}"
        )

    x_proj = x @ directions.T
    y_proj = y @ directions.T
    per_projection = [
        empirical_w2_squared(x_proj[:, i], y_proj[:, i])
        for i in range(directions.shape[0])
    ]
    return float(np.sqrt(np.mean(per_projection)))


def pairwise_swd(
    task_states: Dict[int, np.ndarray],
    num_projections: int,
    seed: int,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    task_ids = sorted(task_states)
    state_dim = next(iter(task_states.values())).shape[1]
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(int(num_projections), state_dim)).astype(np.float64)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)

    distances = np.zeros((len(task_ids), len(task_ids)), dtype=np.float64)
    for row, task_i in enumerate(task_ids):
        for col in range(row + 1, len(task_ids)):
            task_j = task_ids[col]
            distance = sliced_wasserstein_2(
                task_states[task_i],
                task_states[task_j],
                directions,
            )
            distances[row, col] = distances[col, row] = distance
            print(f"SWD task {task_i:02d} vs task {task_j:02d}: {distance:.8f}")
    return task_ids, distances, directions


def distances_from_target(
    target_task: int,
    task_states: Dict[int, np.ndarray],
    num_projections: int,
    seed: int,
) -> Tuple[List[Dict], np.ndarray]:
    if target_task not in task_states:
        raise KeyError(f"target task {target_task} was not loaded")

    state_dim = next(iter(task_states.values())).shape[1]
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(int(num_projections), state_dim)).astype(np.float64)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)

    rows = []
    for task_id in sorted(task_states):
        distance = sliced_wasserstein_2(
            task_states[target_task],
            task_states[task_id],
            directions,
        )
        rows.append(
            {
                "target_task": int(target_task),
                "task_id": int(task_id),
                "task_name": task_name(task_id),
                "distance": float(distance),
            }
        )
        print(f"SWD target task {target_task:02d} vs task {task_id:02d}: {distance:.8f}")
    return sorted(rows, key=lambda item: item["distance"]), directions


def write_pairwise_csv(path: Path, task_ids: Sequence[int], distances: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"task_{task_id}:{task_name(task_id)}" for task_id in task_ids]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task"] + labels)
        for label, row in zip(labels, distances):
            writer.writerow([label] + [f"{value:.10f}" for value in row])


def write_target_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["target_task", "task_id", "task_name", "distance"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "target_task": int(row["target_task"]),
                    "task_id": int(row["task_id"]),
                    "task_name": row["task_name"],
                    "distance": f"{float(row['distance']):.10f}",
                }
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=Path("simple_sac_20k_states"))
    parser.add_argument("--file-glob", default="task_*_simple_sac_20k_states.pt")
    parser.add_argument("--tasks", type=int, nargs="*", default=list(range(len(tasks))))
    parser.add_argument(
        "--state-key",
        default="states",
        help="State key inside each .pt file: states, observations, or visited_states.",
    )
    parser.add_argument("--max-states-per-task", type=int, default=None)
    parser.add_argument("--num-projections", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--norm-eps",
        type=float,
        default=1e-12,
        help="Minimum global std when --normalize is enabled.",
    )
    parser.add_argument(
        "--target-task",
        type=int,
        default=None,
        help="If set, compute/rank distances from this task only.",
    )
    parser.add_argument("--output-csv", type=Path, default=Path("analysis/halfcheetah_state_swd.csv"))
    parser.add_argument("--output-pt", type=Path, default=Path("analysis/halfcheetah_state_swd.pt"))
    parser.add_argument("--output-json", type=Path, default=Path("analysis/halfcheetah_state_swd_summary.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.num_projections) <= 0:
        raise ValueError("--num-projections must be positive")

    files = discover_state_files(
        state_dir=args.state_dir,
        file_glob=args.file_glob,
        task_ids=args.tasks,
    )
    task_states, metadata = load_task_states(
        files=files,
        state_key=args.state_key,
        max_states_per_task=args.max_states_per_task,
        seed=int(args.seed),
    )
    if len(task_states) < 2:
        raise RuntimeError("Need at least two task-state files to compute distances.")

    distance_states = task_states
    mean = std = None
    if args.normalize:
        distance_states, mean, std = normalize_states_global(
            task_states,
            eps=float(args.norm_eps),
        )

    args.output_csv.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output_pt.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output_json.expanduser().parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metric": "sliced_wasserstein_2",
        "formula": "sqrt(mean_theta(W2(P_i^theta, P_j^theta)^2))",
        "state_key": args.state_key,
        "normalized": bool(args.normalize),
        "normalization": "global per-dimension z-score over all loaded task states" if args.normalize else "none",
        "normalization_mean": None if mean is None else torch.as_tensor(mean, dtype=torch.float32),
        "normalization_std": None if std is None else torch.as_tensor(std, dtype=torch.float32),
        "num_projections": int(args.num_projections),
        "seed": int(args.seed),
        "metadata": metadata,
    }

    if args.target_task is None:
        task_ids, distances, directions = pairwise_swd(
            task_states=distance_states,
            num_projections=int(args.num_projections),
            seed=int(args.seed),
        )
        write_pairwise_csv(args.output_csv.expanduser(), task_ids, distances)
        payload.update(
            {
                "mode": "pairwise",
                "task_ids": task_ids,
                "task_names": [task_name(task_id) for task_id in task_ids],
                "distances": torch.as_tensor(distances, dtype=torch.float32),
                "directions": torch.as_tensor(directions, dtype=torch.float32),
            }
        )
        json_summary = {
            "mode": "pairwise",
            "metric": payload["metric"],
            "task_ids": task_ids,
            "task_names": [task_name(task_id) for task_id in task_ids],
            "state_key": args.state_key,
            "normalized": bool(args.normalize),
            "num_projections": int(args.num_projections),
            "seed": int(args.seed),
            "output_csv": str(args.output_csv),
            "output_pt": str(args.output_pt),
            "metadata": metadata,
            "nearest_neighbors": {
                int(task_id): [
                    {
                        "task_id": int(task_ids[j]),
                        "task_name": task_name(task_ids[j]),
                        "distance": float(distances[i, j]),
                    }
                    for j in np.argsort(distances[i]).tolist()
                    if j != i
                ]
                for i, task_id in enumerate(task_ids)
            },
        }
    else:
        rows, directions = distances_from_target(
            target_task=int(args.target_task),
            task_states=distance_states,
            num_projections=int(args.num_projections),
            seed=int(args.seed),
        )
        write_target_csv(args.output_csv.expanduser(), rows)
        payload.update(
            {
                "mode": "target",
                "target_task": int(args.target_task),
                "distances": rows,
                "directions": torch.as_tensor(directions, dtype=torch.float32),
            }
        )
        json_summary = {
            "mode": "target",
            "metric": payload["metric"],
            "target_task": int(args.target_task),
            "target_task_name": task_name(int(args.target_task)),
            "ranked_distances": rows,
            "state_key": args.state_key,
            "normalized": bool(args.normalize),
            "num_projections": int(args.num_projections),
            "seed": int(args.seed),
            "output_csv": str(args.output_csv),
            "output_pt": str(args.output_pt),
            "metadata": metadata,
        }

    torch.save(payload, args.output_pt.expanduser())
    with args.output_json.expanduser().open("w", encoding="utf-8") as handle:
        json.dump(json_summary, handle, indent=2, sort_keys=True)

    print(f"\nsaved csv: {args.output_csv}")
    print(f"saved pt: {args.output_pt}")
    print(f"saved json: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
