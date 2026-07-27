#!/usr/bin/env python3
"""
Visualize state distributions stored in LongTermMemory across Meta-World tasks.

The script samples states from each task memory, standardizes features, applies
PCA to 2D/3D, and writes:

    memory_state_pca.html
    memory_state_pca_stats.json

The HTML is self-contained and can be opened without Python.
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from models import LongTermMemory
from tasks import get_task_name


COLORS = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc949",
    "#af7aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ab",
]


def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_task_id(path: Path) -> int:
    match = re.search(r"task_(\d+)__", path.name)
    if match is None:
        raise ValueError(f"Cannot infer task id from memory directory name: {path}")
    return int(match.group(1))


def discover_memory_dirs(
    memory_root: Path,
    memory_glob: str,
    task_ids: Sequence[int],
) -> Dict[int, Path]:
    if not memory_root.exists():
        raise FileNotFoundError(f"Memory root does not exist: {memory_root}")

    wanted = set(int(task_id) for task_id in task_ids)
    dirs: Dict[int, Path] = {}
    for path in sorted(memory_root.glob(memory_glob)):
        if not path.is_dir():
            continue
        if not (path / "long_term_memory.pt").exists():
            continue
        task_id = parse_task_id(path)
        if task_id in wanted:
            dirs[task_id] = path

    missing = sorted(wanted - set(dirs))
    if missing:
        raise FileNotFoundError(
            f"Missing memory directories for tasks {missing} under {memory_root} with glob {memory_glob!r}"
        )
    return dirs


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


def sample_memory_states(memory: LongTermMemory, samples_per_task: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    states = memory.states[: len(memory)].reshape(-1, memory.obs_dim)
    total = int(states.shape[0])
    n = min(int(samples_per_task), total)
    indices = rng.choice(total, size=n, replace=False)
    return states[indices].cpu().numpy().astype(np.float32)


def standardize(x: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return (x - mean) / std, mean.squeeze(0), std.squeeze(0)


def pca(x: np.ndarray, dims: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_centered = x - x.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(x_centered, full_matrices=False)
    components = vt[:dims]
    projected = x_centered @ components.T
    explained_variance = (singular_values**2) / max(1, x.shape[0] - 1)
    explained_ratio = explained_variance / np.maximum(explained_variance.sum(), 1e-12)
    return projected, components, explained_ratio[:dims]


def compute_stats(points: np.ndarray, labels: np.ndarray, task_ids: Sequence[int]) -> Dict:
    centroids = {}
    within_radius = {}
    for task_id in task_ids:
        task_points = points[labels == task_id]
        centroid = task_points.mean(axis=0)
        centroids[int(task_id)] = centroid
        within_radius[int(task_id)] = float(np.linalg.norm(task_points - centroid, axis=1).mean())

    centroid_matrix = np.stack([centroids[int(task_id)] for task_id in task_ids], axis=0)
    distances = np.linalg.norm(
        centroid_matrix[:, None, :] - centroid_matrix[None, :, :],
        axis=-1,
    )

    nearest_correct = 0
    for point, label in zip(points, labels):
        d = np.linalg.norm(centroid_matrix - point[None, :], axis=1)
        pred = int(task_ids[int(np.argmin(d))])
        nearest_correct += int(pred == int(label))
    nearest_centroid_accuracy = nearest_correct / max(1, int(points.shape[0]))

    pairwise = []
    for i, task_i in enumerate(task_ids):
        for j, task_j in enumerate(task_ids):
            if j <= i:
                continue
            pairwise.append(
                {
                    "task_i": int(task_i),
                    "task_j": int(task_j),
                    "distance": float(distances[i, j]),
                }
            )
    pairwise.sort(key=lambda item: item["distance"])

    return {
        "nearest_centroid_accuracy": float(nearest_centroid_accuracy),
        "within_radius": {str(k): v for k, v in within_radius.items()},
        "centroid_distances_sorted": pairwise,
    }


def prepare_plot_points(
    projected: np.ndarray,
    labels: np.ndarray,
    task_names: Dict[int, str],
    max_plot_points_per_task: int,
    seed: int,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    plot_points = []
    for task_id in sorted(set(int(x) for x in labels.tolist())):
        idx = np.where(labels == task_id)[0]
        n = min(max_plot_points_per_task, len(idx))
        chosen = rng.choice(idx, size=n, replace=False)
        for row in chosen:
            coords = projected[row]
            plot_points.append(
                {
                    "task": int(task_id),
                    "name": task_names[int(task_id)],
                    "x": round(float(coords[0]), 5),
                    "y": round(float(coords[1]), 5),
                    "z": round(float(coords[2]), 5) if projected.shape[1] >= 3 else 0.0,
                }
            )
    return plot_points


def write_html(path: Path, plot_points: List[Dict], tasks: List[Dict], explained_ratio: Sequence[float]) -> None:
    data_json = json.dumps(plot_points, separators=(",", ":"))
    tasks_json = json.dumps(tasks, separators=(",", ":"))
    explained_json = json.dumps([round(float(x), 5) for x in explained_ratio], separators=(",", ":"))
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Memory State PCA</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #202124;
      background: #f7f7f8;
    }}
    .wrap {{
      max-width: 1180px;
      margin: 24px auto;
      padding: 0 18px;
    }}
    .bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      margin-bottom: 12px;
    }}
    button {{
      border: 1px solid #c9ccd1;
      border-radius: 6px;
      background: #fff;
      padding: 7px 11px;
      cursor: pointer;
    }}
    button.active {{
      background: #202124;
      color: #fff;
      border-color: #202124;
    }}
    .meta {{
      color: #5f6368;
      font-size: 13px;
    }}
    canvas {{
      width: 100%;
      height: 680px;
      display: block;
      background: #fff;
      border: 1px solid #dadce0;
      border-radius: 8px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      margin-top: 12px;
      font-size: 13px;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .dot {{
      width: 9px;
      height: 9px;
      border-radius: 50%;
      display: inline-block;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="bar">
      <button id="mode2d" class="active">2D PCA</button>
      <button id="mode3d">3D PCA</button>
      <span class="meta" id="meta"></span>
    </div>
    <canvas id="plot" width="1120" height="680"></canvas>
    <div class="legend" id="legend"></div>
  </div>
  <script>
    const points = {data_json};
    const tasks = {tasks_json};
    const explained = {explained_json};
    const colors = {json.dumps(COLORS, separators=(",", ":"))};
    const canvas = document.getElementById("plot");
    const ctx = canvas.getContext("2d");
    const meta = document.getElementById("meta");
    const legend = document.getElementById("legend");
    let mode = "2d";
    let yaw = 0.7;
    let pitch = 0.35;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    legend.innerHTML = tasks.map((t, i) =>
      `<span><i class="dot" style="background:${{colors[i % colors.length]}}"></i>task ${{String(t.task).padStart(2, "0")}} ${{t.name}}</span>`
    ).join("");

    function project(p) {{
      if (mode === "2d") return [p.x, p.y];
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = cy * p.x + sy * p.z;
      const z1 = -sy * p.x + cy * p.z;
      const y1 = cp * p.y - sp * z1;
      return [x1, y1];
    }}

    function draw() {{
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      const projected = points.map(project);
      const xs = projected.map(p => p[0]);
      const ys = projected.map(p => p[1]);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const pad = 42;
      const sx = (w - 2 * pad) / Math.max(1e-9, maxX - minX);
      const sy = (h - 2 * pad) / Math.max(1e-9, maxY - minY);
      const s = Math.min(sx, sy);
      ctx.globalAlpha = 0.72;
      for (let i = 0; i < points.length; i++) {{
        const p = points[i];
        const q = projected[i];
        const x = pad + (q[0] - minX) * s;
        const y = h - pad - (q[1] - minY) * s;
        ctx.fillStyle = colors[p.task % colors.length];
        ctx.beginPath();
        ctx.arc(x, y, 2.2, 0, Math.PI * 2);
        ctx.fill();
      }}
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#5f6368";
      ctx.font = "13px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      const ev = explained.map((v, i) => `PC${{i + 1}}=${{(v * 100).toFixed(1)}}%`).join(", ");
      meta.textContent = `${{points.length}} plotted states | ${{ev}}` + (mode === "3d" ? " | drag to rotate" : "");
    }}

    document.getElementById("mode2d").onclick = () => {{
      mode = "2d";
      document.getElementById("mode2d").classList.add("active");
      document.getElementById("mode3d").classList.remove("active");
      draw();
    }};
    document.getElementById("mode3d").onclick = () => {{
      mode = "3d";
      document.getElementById("mode3d").classList.add("active");
      document.getElementById("mode2d").classList.remove("active");
      draw();
    }};
    canvas.onpointerdown = (e) => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; }};
    canvas.onpointerup = () => {{ dragging = false; }};
    canvas.onpointerleave = () => {{ dragging = false; }};
    canvas.onpointermove = (e) => {{
      if (!dragging || mode !== "3d") return;
      yaw += (e.clientX - lastX) * 0.008;
      pitch += (e.clientY - lastY) * 0.008;
      pitch = Math.max(-1.3, Math.min(1.3, pitch));
      lastX = e.clientX;
      lastY = e.clientY;
      draw();
    }};
    draw();
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def analyze(args):
    rng_seed = int(args.seed)
    task_ids = [int(task_id) for task_id in args.tasks]
    memory_dirs = discover_memory_dirs(
        memory_root=Path(args.memory_root).expanduser(),
        memory_glob=args.memory_glob,
        task_ids=task_ids,
    )

    states_by_task = {}
    task_names = {}
    memory_sizes = {}
    for task_id in task_ids:
        memory = load_memory(memory_dirs[task_id])
        states = sample_memory_states(
            memory=memory,
            samples_per_task=args.samples_per_task,
            seed=rng_seed + task_id,
        )
        states_by_task[task_id] = states
        task_names[task_id] = get_task_name(task_id)
        memory_sizes[task_id] = len(memory)
        print(
            f"task {task_id:02d} {task_names[task_id]}: "
            f"memory={len(memory)} trajectories, sampled_states={states.shape[0]}"
        )

    all_states = np.concatenate([states_by_task[task_id] for task_id in task_ids], axis=0)
    labels = np.concatenate(
        [
            np.full(states_by_task[task_id].shape[0], task_id, dtype=np.int64)
            for task_id in task_ids
        ],
        axis=0,
    )
    standardized, feature_mean, feature_std = standardize(all_states)
    projected, components, explained_ratio = pca(standardized, dims=3)
    stats = compute_stats(projected, labels, task_ids)
    plot_points = prepare_plot_points(
        projected=projected,
        labels=labels,
        task_names=task_names,
        max_plot_points_per_task=args.plot_samples_per_task,
        seed=rng_seed,
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "memory_state_pca.html"
    stats_path = output_dir / "memory_state_pca_stats.json"
    npz_path = output_dir / "memory_state_pca_data.npz"

    tasks = [
        {
            "task": int(task_id),
            "name": task_names[task_id],
            "memory_trajectories": int(memory_sizes[task_id]),
            "sampled_states": int(states_by_task[task_id].shape[0]),
        }
        for task_id in task_ids
    ]

    write_html(
        path=html_path,
        plot_points=plot_points,
        tasks=tasks,
        explained_ratio=explained_ratio,
    )

    output_stats = {
        "memory_root": args.memory_root,
        "memory_glob": args.memory_glob,
        "samples_per_task": args.samples_per_task,
        "plot_samples_per_task": args.plot_samples_per_task,
        "tasks": tasks,
        "pca_explained_variance_ratio": [float(x) for x in explained_ratio],
        "nearest_centroid_accuracy": stats["nearest_centroid_accuracy"],
        "within_radius": stats["within_radius"],
        "closest_task_pairs": stats["centroid_distances_sorted"][:10],
        "farthest_task_pairs": list(reversed(stats["centroid_distances_sorted"][-10:])),
    }
    stats_path.write_text(json.dumps(output_stats, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(
        npz_path,
        projected=projected.astype(np.float32),
        labels=labels.astype(np.int64),
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        components=components.astype(np.float32),
        explained_ratio=np.asarray(explained_ratio, dtype=np.float32),
    )

    print(f"\nnearest_centroid_accuracy: {output_stats['nearest_centroid_accuracy']:.4f}")
    print("closest task pairs:")
    for item in output_stats["closest_task_pairs"][:5]:
        print(
            f"  task {item['task_i']:02d} - task {item['task_j']:02d}: "
            f"distance={item['distance']:.4f}"
        )
    print(f"\nsaved html: {html_path}")
    print(f"saved stats: {stats_path}")
    print(f"saved npz: {npz_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample LongTermMemory states and visualize task distributions with PCA."
    )
    parser.add_argument("--memory-root", default="agents")
    parser.add_argument("--memory-glob", default="task_*__hybrid__SAC_hybridRAGRL__42")
    parser.add_argument("--tasks", type=int, nargs="*", default=list(range(10)))
    parser.add_argument("--samples-per-task", type=int, default=5000)
    parser.add_argument("--plot-samples-per-task", type=int, default=1000)
    parser.add_argument("--output-dir", default="memory_visualizations/hybrid_states__42")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
