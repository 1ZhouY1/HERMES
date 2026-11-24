#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

import torch


def safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute coarse-retrieval coverage: C_epsilon = 1/T sum_t "
            "1[min_i d(s_t, s_i_start) < epsilon]."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--snapshot-glob", default="snapshot_*")
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--top-k", type=int, nargs="*", default=[1, 5, 10, 50, 100])
    parser.add_argument("--metric", choices=["l2", "l1", "cosine"], default="l2")
    parser.add_argument("--episode-file", default="episode.pt")
    parser.add_argument("--memory-file", default="trajectory_memory.pt")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-pt", type=Path, default=None)
    return parser.parse_args()


def snapshot_step(path: Path) -> int:
    match = re.search(r"snapshot_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def discover_snapshots(run_dir: Path, snapshot_glob: str) -> List[Path]:
    snapshots = sorted(
        [path for path in run_dir.glob(snapshot_glob) if path.is_dir()],
        key=snapshot_step,
    )
    if not snapshots:
        raise FileNotFoundError(f"No snapshot directories found in {run_dir}")
    return snapshots


def load_episode_states(path: Path) -> torch.Tensor:
    payload = safe_torch_load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        states = payload
    elif isinstance(payload, dict):
        if "states" in payload:
            states = payload["states"]
        elif "episode_states" in payload:
            states = payload["episode_states"]
        else:
            raise KeyError(f"{path}: cannot find states or episode_states.")
    else:
        raise TypeError(f"{path}: unsupported episode payload type {type(payload)}")

    states = torch.as_tensor(states, dtype=torch.float32)
    if states.dim() != 2:
        raise ValueError(f"{path}: expected episode states as T x L, got {tuple(states.shape)}")
    return states


def load_memory_start_states(path: Path) -> torch.Tensor:
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected trajectory memory dict, got {type(payload)}")
    if "states" not in payload:
        raise KeyError(f"{path}: cannot find memory states.")

    states = torch.as_tensor(payload["states"], dtype=torch.float32)
    size = int(payload.get("size", states.shape[0]))
    if states.dim() != 3:
        raise ValueError(f"{path}: expected memory states as M x H x L, got {tuple(states.shape)}")
    if size <= 0:
        raise ValueError(f"{path}: memory is empty.")
    return states[:size, 0, :]


def pairwise_distances(x: torch.Tensor, y: torch.Tensor, metric: str) -> torch.Tensor:
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"state dimension mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    if metric == "l2":
        return torch.cdist(x, y, p=2)
    if metric == "l1":
        return torch.cdist(x, y, p=1)

    x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
    y_norm = torch.nn.functional.normalize(y, p=2, dim=1)
    return 1.0 - x_norm @ y_norm.T


def compute_snapshot_coverage(
    snapshot_dir: Path,
    epsilon: float,
    top_ks: Sequence[int],
    metric: str,
    episode_file: str,
    memory_file: str,
) -> Dict:
    episode_states = load_episode_states(snapshot_dir / episode_file)
    memory_start_states = load_memory_start_states(snapshot_dir / memory_file)
    distances = pairwise_distances(episode_states, memory_start_states, metric=metric)
    min_distances, nearest_indices = distances.min(dim=1)
    covered = min_distances < float(epsilon)

    max_k = min(max(int(k) for k in top_ks), memory_start_states.shape[0])
    top_values, top_indices = torch.topk(distances, k=max_k, dim=1, largest=False)

    topk_summary = {}
    for raw_k in top_ks:
        k = min(int(raw_k), max_k)
        kth_distances = top_values[:, k - 1]
        any_covered = top_values[:, :k].lt(float(epsilon)).any(dim=1)
        all_covered = top_values[:, :k].lt(float(epsilon)).all(dim=1)
        topk_summary[int(raw_k)] = {
            "effective_k": int(k),
            "coverage_any": float(any_covered.float().mean().item()),
            "coverage_all": float(all_covered.float().mean().item()),
            "kth_coverage": float(kth_distances.lt(float(epsilon)).float().mean().item()),
            "mean_topk_distance": float(top_values[:, :k].mean().item()),
            "mean_kth_distance": float(kth_distances.mean().item()),
        }

    result = {
        "snapshot": snapshot_dir.name,
        "global_step": int(snapshot_step(snapshot_dir)),
        "epsilon": float(epsilon),
        "metric": metric,
        "episode_length": int(episode_states.shape[0]),
        "memory_size": int(memory_start_states.shape[0]),
        "coverage": float(covered.float().mean().item()),
        "covered_steps": int(covered.sum().item()),
        "mean_min_distance": float(min_distances.mean().item()),
        "median_min_distance": float(min_distances.median().item()),
        "max_min_distance": float(min_distances.max().item()),
        "topk": topk_summary,
        "nearest_indices": nearest_indices,
        "min_distances": min_distances,
        "top_values": top_values,
        "top_indices": top_indices,
    }
    return result


def write_csv(path: Path, rows: Sequence[Dict], top_ks: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "snapshot",
        "global_step",
        "epsilon",
        "metric",
        "episode_length",
        "memory_size",
        "coverage",
        "covered_steps",
        "mean_min_distance",
        "median_min_distance",
        "max_min_distance",
    ]
    for k in top_ks:
        fieldnames.extend(
            [
                f"top{k}_coverage_any",
                f"top{k}_coverage_all",
                f"top{k}_kth_coverage",
                f"top{k}_mean_topk_distance",
                f"top{k}_mean_kth_distance",
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in fieldnames if key in row}
            for k in top_ks:
                summary = row["topk"][int(k)]
                flat[f"top{k}_coverage_any"] = summary["coverage_any"]
                flat[f"top{k}_coverage_all"] = summary["coverage_all"]
                flat[f"top{k}_kth_coverage"] = summary["kth_coverage"]
                flat[f"top{k}_mean_topk_distance"] = summary["mean_topk_distance"]
                flat[f"top{k}_mean_kth_distance"] = summary["mean_kth_distance"]
            writer.writerow(flat)


def json_ready(row: Dict) -> Dict:
    return {
        key: value
        for key, value in row.items()
        if not isinstance(value, torch.Tensor)
    }


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser()
    top_ks = sorted({int(k) for k in args.top_k if int(k) > 0})
    if not top_ks:
        raise ValueError("--top-k must contain at least one positive integer.")

    rows = [
        compute_snapshot_coverage(
            snapshot_dir=snapshot_dir,
            epsilon=float(args.epsilon),
            top_ks=top_ks,
            metric=args.metric,
            episode_file=args.episode_file,
            memory_file=args.memory_file,
        )
        for snapshot_dir in discover_snapshots(run_dir, args.snapshot_glob)
    ]

    output_csv = args.output_csv or (run_dir / "memory_coarse_coverage.csv")
    output_json = args.output_json or (run_dir / "memory_coarse_coverage.json")
    output_pt = args.output_pt or (run_dir / "memory_coarse_coverage.pt")

    write_csv(output_csv.expanduser(), rows, top_ks=top_ks)
    output_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
    with output_json.expanduser().open("w", encoding="utf-8") as f:
        json.dump([json_ready(row) for row in rows], f, indent=2, sort_keys=True)
    torch.save(rows, output_pt.expanduser())

    for row in rows:
        print(
            f"{row['snapshot']}: C_epsilon={row['coverage']:.6f}, "
            f"T={row['episode_length']}, memory={row['memory_size']}, "
            f"mean_min_d={row['mean_min_distance']:.6f}"
        )
    print(f"saved csv: {output_csv}")
    print(f"saved json: {output_json}")
    print(f"saved pt: {output_pt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
