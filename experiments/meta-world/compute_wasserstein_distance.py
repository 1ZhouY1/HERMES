#!/usr/bin/env python3
"""
Compute task distances with temporal-dimension normalized state distributions.

For each task T_i, the saved state matrix is:

    Q_i in R^{N x L}

where N is the number of training interaction steps and L is the state length.
This script first normalizes each state dimension along the temporal dimension:

    norm[q_i(l)] = (q_i[:, l] - mean(q_i[:, l])) / std(q_i[:, l])

Then it treats every state dimension l as one 1-D empirical distribution and
computes:

    S_i = (1 / L) * sum_l phi(norm[q_k(l)], norm[q_i(l)])

where phi is the 1-D empirical Wasserstein distance.
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
        for key in ["task_id", "target_task_id", "source_task_id"]:
            if meta.get(key) is not None:
                return int(meta[key])
            if payload.get(key) is not None:
                return int(payload[key])

    for pattern in [r"task_(\d+)", r"target_task_(\d+)"]:
        match = re.search(pattern, path.stem)
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
        raise ValueError(f"{path}: expected {key} to be 2D Q=NxL, got shape {array.shape}")
    return array


def extract_state_matrix(payload, path: Path, state_key: str) -> np.ndarray:
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
        raise FileNotFoundError(f"Missing state files for task ids: {missing}")
    return selected


def load_task_matrices(
    files: Sequence[Path],
    state_key: str,
    expected_states: Optional[int],
    allow_count_mismatch: bool,
) -> Tuple[Dict[int, np.ndarray], List[Dict]]:
    task_matrices: Dict[int, np.ndarray] = {}
    task_meta: List[Dict] = []

    for path in files:
        payload = safe_torch_load(path.expanduser(), map_location="cpu")
        task_id = infer_task_id(path, payload)
        matrix = extract_state_matrix(payload, path, state_key=state_key)

        if expected_states is not None and matrix.shape[0] != int(expected_states):
            message = (
                f"{path}: expected N={expected_states} states, got N={matrix.shape[0]}. "
                "Use --allow-count-mismatch to compare with the minimum shared N."
            )
            if allow_count_mismatch:
                print(f"warning: {message}")
            else:
                raise ValueError(message)

        if task_id in task_matrices:
            raise ValueError(f"Duplicate state file for task {task_id}: {path}")

        task_matrices[task_id] = matrix
        task_meta.append(
            {
                "task_id": int(task_id),
                "task_name": task_name(task_id),
                "file": str(path),
                "shape": list(matrix.shape),
                "num_states": int(matrix.shape[0]),
                "state_dim": int(matrix.shape[1]),
            }
        )
        print(
            f"loaded task {task_id:02d} {task_name(task_id)}: "
            f"Q={tuple(matrix.shape)}, file={path}"
        )

    dims = {matrix.shape[1] for matrix in task_matrices.values()}
    if len(dims) != 1:
        detail = {task_id: matrix.shape for task_id, matrix in task_matrices.items()}
        raise ValueError(f"All task matrices must have the same state length L, got: {detail}")

    return task_matrices, sorted(task_meta, key=lambda item: item["task_id"])


def normalize_along_temporal_dim(q: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if q.ndim != 2:
        raise ValueError(f"Q should be 2D, got {q.shape}")
    mean = q.mean(axis=0, keepdims=True)
    std = q.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    normalized = ((q - mean) / std).astype(np.float32)
    return normalized, mean.squeeze(0).astype(np.float32), std.squeeze(0).astype(np.float32)


def normalize_task_matrices(
    task_matrices: Dict[int, np.ndarray],
    eps: float,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[str, np.ndarray]]]:
    normalized = {}
    stats = {}
    for task_id, matrix in task_matrices.items():
        norm_matrix, mean, std = normalize_along_temporal_dim(matrix, eps=eps)
        normalized[task_id] = norm_matrix
        stats[task_id] = {"temporal_mean": mean, "temporal_std": std}
    return normalized, stats


def empirical_wasserstein_1d(
    x: np.ndarray,
    y: np.ndarray,
    p: int,
) -> float:
    if p not in [1, 2]:
        raise ValueError("--wasserstein-p currently supports only 1 or 2.")

    x_sorted = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    y_sorted = np.sort(np.asarray(y, dtype=np.float64).reshape(-1))

    n = min(x_sorted.size, y_sorted.size)
    if x_sorted.size != n:
        quantiles = (np.arange(n, dtype=np.float64) + 0.5) / n
        x_positions = (np.arange(x_sorted.size, dtype=np.float64) + 0.5) / x_sorted.size
        x_sorted = np.interp(quantiles, x_positions, x_sorted)
    if y_sorted.size != n:
        quantiles = (np.arange(n, dtype=np.float64) + 0.5) / n
        y_positions = (np.arange(y_sorted.size, dtype=np.float64) + 0.5) / y_sorted.size
        y_sorted = np.interp(quantiles, y_positions, y_sorted)

    diff = np.abs(x_sorted - y_sorted)
    if p == 1:
        return float(diff.mean())
    return float(np.sqrt(np.mean(diff ** 2)))


def temporal_dim_wasserstein(
    q_k: np.ndarray,
    q_i: np.ndarray,
    p: int,
) -> Tuple[float, np.ndarray]:
    if q_k.ndim != 2 or q_i.ndim != 2:
        raise ValueError(f"Q matrices should be 2D, got {q_k.shape}, {q_i.shape}")
    if q_k.shape[1] != q_i.shape[1]:
        raise ValueError(f"state length mismatch: {q_k.shape[1]} vs {q_i.shape[1]}")

    state_dim = q_k.shape[1]
    per_dim = np.zeros(state_dim, dtype=np.float32)
    for dim in range(state_dim):
        per_dim[dim] = empirical_wasserstein_1d(q_k[:, dim], q_i[:, dim], p=p)
    return float(per_dim.mean()), per_dim


def pairwise_temporal_dim_wasserstein(
    task_matrices: Dict[int, np.ndarray],
    p: int,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    task_ids = sorted(task_matrices)
    num_tasks = len(task_ids)
    state_dim = next(iter(task_matrices.values())).shape[1]
    distances = np.zeros((num_tasks, num_tasks), dtype=np.float32)
    per_dim_distances = np.zeros((num_tasks, num_tasks, state_dim), dtype=np.float32)

    for row, task_k in enumerate(task_ids):
        for col in range(row + 1, num_tasks):
            task_i = task_ids[col]
            distance, per_dim = temporal_dim_wasserstein(
                task_matrices[task_k],
                task_matrices[task_i],
                p=p,
            )
            distances[row, col] = distance
            distances[col, row] = distance
            per_dim_distances[row, col] = per_dim
            per_dim_distances[col, row] = per_dim
            print(
                f"S(task {task_k:02d}, task {task_i:02d}) "
                f"= {distance:.8f} over L={state_dim}"
            )

    return task_ids, distances, per_dim_distances


def distances_from_target(
    target_task: int,
    task_matrices: Dict[int, np.ndarray],
    p: int,
) -> Tuple[List[Dict], Dict[int, np.ndarray]]:
    if target_task not in task_matrices:
        raise KeyError(f"target task {target_task} was not loaded.")

    rows = []
    per_dim_by_task = {}
    for task_id in sorted(task_matrices):
        distance, per_dim = temporal_dim_wasserstein(
            task_matrices[target_task],
            task_matrices[task_id],
            p=p,
        )
        rows.append(
            {
                "target_task": int(target_task),
                "task_id": int(task_id),
                "task_name": task_name(task_id),
                "distance": float(distance),
            }
        )
        per_dim_by_task[task_id] = per_dim
        print(f"S_{task_id} for target task {target_task:02d}: {distance:.8f}")
    return rows, per_dim_by_task


def write_pairwise_csv(path: Path, task_ids: Sequence[int], distances: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"task_{task_id}:{task_name(task_id)}" for task_id in task_ids]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task"] + labels)
        for label, row in zip(labels, distances):
            writer.writerow([label] + [f"{value:.8f}" for value in row.tolist()])


def write_target_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["target_task", "task_id", "task_name", "distance"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "target_task": row["target_task"],
                    "task_id": row["task_id"],
                    "task_name": row["task_name"],
                    "distance": f"{row['distance']:.8f}",
                }
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the formula S_i = 1/L sum_l phi(norm[q_k(l)], norm[q_i(l)]) "
            "from saved NxL task-state matrices."
        )
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("simple_sac_20k_states"),
        help="Directory containing saved task state .pt files.",
    )
    parser.add_argument(
        "--file-glob",
        default="task_*_simple_sac_20k_states.pt",
        help="Glob used to discover task state files under --state-dir.",
    )
    parser.add_argument("--tasks", type=int, nargs="*", default=list(range(10)))
    parser.add_argument(
        "--state-key",
        default="states",
        help="State matrix key inside each .pt file. Defaults to 'states'.",
    )
    parser.add_argument(
        "--expected-states",
        type=int,
        default=20_000,
        help="Expected temporal length N. Set to 0 to disable the check.",
    )
    parser.add_argument(
        "--allow-count-mismatch",
        default=False,
        action="store_true",
        help="Allow different N across tasks by comparing empirical quantiles.",
    )
    parser.add_argument(
        "--wasserstein-p",
        type=int,
        choices=[1, 2],
        default=1,
        help="Use empirical W1 or W2 for the per-dimension phi.",
    )
    parser.add_argument(
        "--norm-eps",
        type=float,
        default=1e-6,
        help="Minimum temporal std used during per-task, per-dimension normalization.",
    )
    parser.add_argument(
        "--target-task",
        type=int,
        default=None,
        help="If set, write only S_i distances from this target task. Otherwise write pairwise matrix.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("analysis/temporal_dim_wasserstein.csv"),
    )
    parser.add_argument(
        "--output-pt",
        type=Path,
        default=Path("analysis/temporal_dim_wasserstein.pt"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("analysis/temporal_dim_wasserstein_summary.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected_states = None if int(args.expected_states) <= 0 else int(args.expected_states)
    files = discover_state_files(
        state_dir=args.state_dir.expanduser(),
        file_glob=args.file_glob,
        tasks=args.tasks,
    )
    task_matrices, task_meta = load_task_matrices(
        files=files,
        state_key=args.state_key,
        expected_states=expected_states,
        allow_count_mismatch=args.allow_count_mismatch,
    )
    if len(task_matrices) < 2:
        raise RuntimeError("Need at least two task-state files to compute distances.")

    normalized_matrices, norm_stats = normalize_task_matrices(
        task_matrices,
        eps=float(args.norm_eps),
    )

    args.output_csv.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output_pt.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output_json.expanduser().parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metric": f"temporal_dim_empirical_wasserstein_{args.wasserstein_p}",
        "formula": "S_i = (1 / L) * sum_l phi(norm[q_k(l)], norm[q_i(l)])",
        "state_key": args.state_key,
        "expected_states": expected_states,
        "task_meta": task_meta,
        "normalization": "per task and per state dimension over temporal axis N",
        "normalization_eps": float(args.norm_eps),
    }

    if args.target_task is None:
        task_ids, distances, per_dim_distances = pairwise_temporal_dim_wasserstein(
            normalized_matrices,
            p=int(args.wasserstein_p),
        )
        write_pairwise_csv(args.output_csv.expanduser(), task_ids, distances)
        payload.update(
            {
                "mode": "pairwise",
                "task_ids": task_ids,
                "distances": torch.as_tensor(distances, dtype=torch.float32),
                "per_dim_distances": torch.as_tensor(per_dim_distances, dtype=torch.float32),
                "temporal_mean": {
                    int(task_id): torch.as_tensor(stats["temporal_mean"], dtype=torch.float32)
                    for task_id, stats in norm_stats.items()
                },
                "temporal_std": {
                    int(task_id): torch.as_tensor(stats["temporal_std"], dtype=torch.float32)
                    for task_id, stats in norm_stats.items()
                },
            }
        )
        json_summary = {
            "mode": "pairwise",
            "metric": payload["metric"],
            "task_ids": task_ids,
            "task_names": [task_name(task_id) for task_id in task_ids],
            "output_csv": str(args.output_csv),
            "output_pt": str(args.output_pt),
            "state_key": args.state_key,
            "expected_states": expected_states,
            "task_meta": task_meta,
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
        rows, per_dim_by_task = distances_from_target(
            target_task=int(args.target_task),
            task_matrices=normalized_matrices,
            p=int(args.wasserstein_p),
        )
        rows_sorted = sorted(rows, key=lambda item: item["distance"])
        write_target_csv(args.output_csv.expanduser(), rows_sorted)
        payload.update(
            {
                "mode": "target",
                "target_task": int(args.target_task),
                "distances": rows_sorted,
                "per_dim_distances": {
                    int(task_id): torch.as_tensor(per_dim, dtype=torch.float32)
                    for task_id, per_dim in per_dim_by_task.items()
                },
                "temporal_mean": {
                    int(task_id): torch.as_tensor(stats["temporal_mean"], dtype=torch.float32)
                    for task_id, stats in norm_stats.items()
                },
                "temporal_std": {
                    int(task_id): torch.as_tensor(stats["temporal_std"], dtype=torch.float32)
                    for task_id, stats in norm_stats.items()
                },
            }
        )
        json_summary = {
            "mode": "target",
            "metric": payload["metric"],
            "target_task": int(args.target_task),
            "target_task_name": task_name(int(args.target_task)),
            "ranked_distances": rows_sorted,
            "output_csv": str(args.output_csv),
            "output_pt": str(args.output_pt),
            "state_key": args.state_key,
            "expected_states": expected_states,
            "task_meta": task_meta,
        }

    torch.save(payload, args.output_pt.expanduser())
    with args.output_json.expanduser().open("w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2, sort_keys=True)

    print(f"\nsaved csv: {args.output_csv}")
    print(f"saved pt: {args.output_pt}")
    print(f"saved json: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
