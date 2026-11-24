#!/usr/bin/env python3
"""Plot Meta-World ablation curves for HybridRARL variants.

Supported comparisons include:

1. Full ``hybridrarl``: retrieval plus selectively reused prev-units.
2. ``hybridrarl-no-retrieval``: selective module reuse only, with a simple
   fallback when no reusable module was selected.
3. ``hybridrarl-random-retrieval``: no prev-units, with random coarse
   trajectory retrieval instead of state-related top-k retrieval.
4. ``hybridrarl-random-state-retrieval``: no prev-units, randomly sample
   state positions from trajectory memory and feed them directly to attention,
   without the GRU trajectory summarizer.
5. ``hybridrarl-no-selective-reuse``: the original per-task HybridRARL actor,
   retaining retrieval but never loading prev-units.
6. ``hybridrarl-direct-reuse``: directly reuse all previous modules as
   prev-units, without similarity-based module selection.

The script reads only these TensorBoard scalars:

    charts/episodic_return
    charts/success

It creates publication-style figures with one subplot per task and all three
methods overlaid in each subplot. With one training seed the translucent band
is a locally aggregated variation envelope, not a confidence interval across
independent seeds. Raw per-episode scalars are never drawn.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# TensorBoard versions bundled with older Meta-World environments still refer
# to these NumPy 1.x aliases.  Keep the plotting utility runnable with NumPy
# 2.x without changing how scalar event files are read.
if not hasattr(np, "string_"):
    np.string_ = np.bytes_
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TASK_NAMES = [
    "hammer-v2",
    "push-wall-v2",
    "faucet-close-v2",
    "push-back-v2",
    "stick-pull-v2",
    "handle-press-side-v2",
    "push-v2",
    "shelf-place-v2",
    "window-close-v2",
    "peg-unplug-side-v2",
]

RETURN_TAG = "charts/episodic_return"
SUCCESS_TAG = "charts/success"

METHODS = {
    "hybridrarl": {
        "label": "Ours",
        "color": "#009E73",
        "linestyle": "-",
        "zorder": 4,
        "model_names": (
            # Full HybridRARL should use the selective prev-unit run whenever
            # reusable modules exist. The bare `hybridrarl` run is only the
            # no-reuse fallback and is searched below as a fallback source.
            "hybridrarl-prev-units",
            "hybridrarl-simple-prev-units",
        ),
        "exp_names": (
            "csv_threshold_transfer",
        ),
        "parent_names": (),
        # A full HybridRARL task without selected prev-units is saved as the
        # original hybridrarl model.
        "fallback_model_names": ("hybridrarl",),
        "fallback_exp_names": ("SAC_hybridRAGRL",),
        "fallback_parent_names": (),
    },
    "hybridrarl-traj10": {
        "label": "L=10",
        "color": "#D55E00",
        "linestyle": "-",
        "zorder": 4,
        "model_names": (
            "hybridrarl-prev-units",
            "hybridrarl-simple-prev-units",
        ),
        "exp_names": ("csv_threshold_transfer_traj10",),
        "parent_names": (),
        "fallback_model_names": ("hybridrarl",),
        "fallback_exp_names": ("SAC_hybridRAGRL_traj10",),
        "fallback_parent_names": (),
    },
    "hybridrarl-traj20": {
        "label": "L=20",
        "color": "#009E73",
        "linestyle": "-",
        "zorder": 4,
        "model_names": (
            "hybridrarl-prev-units",
            "hybridrarl-simple-prev-units",
        ),
        "exp_names": ("csv_threshold_transfer",),
        "parent_names": (),
        "fallback_model_names": ("hybridrarl",),
        "fallback_exp_names": ("SAC_hybridRAGRL",),
        "fallback_parent_names": (),
    },
    "hybridrarl-traj50": {
        "label": "L=50",
        "color": "#0072B2",
        "linestyle": "-",
        "zorder": 4,
        "model_names": (
            "hybridrarl-prev-units",
            "hybridrarl-simple-prev-units",
        ),
        "exp_names": ("csv_threshold_transfer_traj50",),
        "parent_names": (),
        "fallback_model_names": ("hybridrarl",),
        "fallback_exp_names": ("SAC_hybridRAGRL_traj50",),
        "fallback_parent_names": (),
    },
    "hybridrarl-no-retrieval": {
        "label": "Ours w/o HER",
        "color": "#D55E00",
        "linestyle": "-",
        "zorder": 3,
        "model_names": (
            "no-retrieval-prev-units",
            "no-retrieval-simple-prev-units",
        ),
        "exp_names": (
            "csv_simple_prev",
        ),
        "parent_names": (
            "no_retrieval_simple_prev_units",
            "no-retrieval-simple-prev-units",
        ),
        # No-retrieval runs without reusable modules are intentionally not
        # saved; those tasks are exactly the simple baseline.
        "fallback_model_names": ("simple",),
        "fallback_exp_names": (),
        "fallback_parent_names": (),
    },
    "hybridrarl-random-retrieval": {
        "label": "Ours w/ random retrieval",
        "color": "#E69F00",
        "linestyle": "-",
        "zorder": 3,
        # Random coarse retrieval ablation: no prev-units; only replace
        # state-related trajectory selection with random top-r choices.
        "model_names": ("hybridrarl-random-retrieval",),
        "exp_names": ("SAC_hybridRAGRL", "SAC"),
        "parent_names": (),
        "fallback_model_names": (),
        "fallback_exp_names": (),
        "fallback_parent_names": (),
    },
    "hybridrarl-random-state-retrieval": {
        "label": "Ours w/ random states",
        "color": "#56B4E9",
        "linestyle": "-",
        "zorder": 3,
        # Random state retrieval ablation: no prev-units, no trajectory
        # summarizer; randomly selected memory states are fed to attention.
        "model_names": ("hybridrarl-random-state-retrieval",),
        "exp_names": ("SAC_hybridRAGRL", "SAC"),
        "parent_names": (),
        "fallback_model_names": (),
        "fallback_exp_names": (),
        "fallback_parent_names": (),
    },
    "hybridrarl-direct-reuse": {
        "label": "Ours w/o select modules",
        "color": "#CC79A7",
        "linestyle": "-",
        "zorder": 3,
        # Direct reuse ablation: skip the similarity-based selection step and
        # load all previous task modules as prev-units.
        "model_names": ("hybridrarl-ablation-no-select",),
        "exp_names": ("all_prev_units",),
        "parent_names": (),
        "fallback_model_names": (),
        "fallback_exp_names": (),
        "fallback_parent_names": (),
    },
    "hybridrarl-no-selective-reuse": {
        "label": "Ours w/o SMR",
        "color": "#0072B2",
        "linestyle": "-",
        "zorder": 3,
        # This must be the base per-task actor; prev-unit checkpoints are
        # intentionally excluded from the search.
        "model_names": ("hybridrarl",),
        "exp_names": ("SAC_hybridRAGRL",),
        "parent_names": (),
        "fallback_model_names": (),
        "fallback_exp_names": (),
        "fallback_parent_names": (),
    },
}

METHOD_ALIASES = {
    "hybridrarl-random-retrieval": "hybridrarl-random-retrieval",
    "hybridrarl_random_retrieval": "hybridrarl-random-retrieval",
    "hybridrarl-random-retrieval-method": "hybridrarl-random-retrieval",
    "random-retrieval": "hybridrarl-random-retrieval",
    "random_retrieval": "hybridrarl-random-retrieval",
    "hybridrarl-random-state-retrieval": "hybridrarl-random-state-retrieval",
    "hybridrarl_random_state_retrieval": "hybridrarl-random-state-retrieval",
    "random-state-retrieval": "hybridrarl-random-state-retrieval",
    "random_state_retrieval": "hybridrarl-random-state-retrieval",
    "random-states": "hybridrarl-random-state-retrieval",
    "random_states": "hybridrarl-random-state-retrieval",
    "hybridrarl-no-retrieval": "hybridrarl-no-retrieval",
    "hybridrarl_no_retrieval": "hybridrarl-no-retrieval",
    "hybridrarl-no-selective-reuse": "hybridrarl-no-selective-reuse",
    "hybridrarl_no_selective_reuse": "hybridrarl-no-selective-reuse",
    "hybridrarl-direct-reuse": "hybridrarl-direct-reuse",
    "hybridrarl_direct_reuse": "hybridrarl-direct-reuse",
    "hybridrarl-traj10": "hybridrarl-traj10",
    "hybridrarl_traj10": "hybridrarl-traj10",
    "traj10": "hybridrarl-traj10",
    "l10": "hybridrarl-traj10",
    "hybridrarl-traj20": "hybridrarl-traj20",
    "hybridrarl_traj20": "hybridrarl-traj20",
    "traj20": "hybridrarl-traj20",
    "l20": "hybridrarl-traj20",
    "hybridrarl-traj50": "hybridrarl-traj50",
    "hybridrarl_traj50": "hybridrarl-traj50",
    "traj50": "hybridrarl-traj50",
    "l50": "hybridrarl-traj50",
}


def normalize_method_key(raw_method: str) -> str:
    method = str(raw_method).strip().lower()
    method = METHOD_ALIASES.get(method, method)
    if method not in METHODS:
        valid = ", ".join(sorted(METHODS))
        raise ValueError(f"Unknown method `{raw_method}`. Valid methods: {valid}.")
    return method


def normalize_method_list(raw_methods: Sequence[str]) -> List[str]:
    normalized = []
    for raw_method in raw_methods:
        method = normalize_method_key(raw_method)
        if method not in normalized:
            normalized.append(method)
    return normalized


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot episodic-return and success curves for HybridRARL ablations."
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--train-seed", "--seed", dest="train_seed", type=int, required=True)
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="*",
        default=list(range(10)),
        help="Meta-World task ids to plot. Default: 0 1 ... 9.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=list(METHODS),
        help=(
            "Methods to overlay. For example: hybridrarl "
            "hybridrarl-random-retrieval. Valid methods: "
            + ", ".join(sorted(METHODS))
        ),
    )
    parser.add_argument(
        "--hybridrarl-no-selective-reuse-parent",
        "--only-retrieval-parent",
        dest="hybridrarl_no_selective_reuse_parent",
        default=None,
        help="Optional parent directory name for retrieval-only HybridRARL runs.",
    )
    parser.add_argument(
        "--hybridrarl-no-retrieval-parent",
        "--only-reuse-parent",
        dest="hybridrarl_no_retrieval_parent",
        default=None,
        help="Optional parent directory name for no-retrieval runs.",
    )
    parser.add_argument(
        "--hybridrarl-random-retrieval-parent",
        dest="hybridrarl_random_retrieval_parent",
        default=None,
        help="Optional parent directory name for random-retrieval HybridRARL runs.",
    )
    parser.add_argument(
        "--hybridrarl-direct-reuse-parent",
        dest="hybridrarl_direct_reuse_parent",
        default=None,
        help="Optional parent directory name for direct-reuse HybridRARL runs.",
    )
    parser.add_argument(
        "--hybridrarl-parent",
        default=None,
        help="Optional parent directory name for the full HybridRARL runs.",
    )
    parser.add_argument(
        "--smoothing",
        "--smooth-factor",
        dest="smoothing",
        type=float,
        default=0.6,
        help=(
            "TensorBoard-style smoothing weight in [0, 1]. 0 shows the raw "
            "curve; larger values give more weight to the previous smoothed value."
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=None,
        help="Legacy moving-average window. Do not combine with --smoothing.",
    )
    parser.add_argument(
        "--aggregation-window",
        type=int,
        default=25,
        help=(
            "Number of consecutive TensorBoard scalar events aggregated into one "
            "plot point before smoothing. This removes per-episode success spikes; "
            "set to 1 to disable."
        ),
    )
    parser.add_argument(
        "--band-scale",
        type=float,
        default=1.0,
        help="Scale of the filled variability band around the smoothed curve.",
    )
    parser.add_argument(
        "--no-band",
        action="store_true",
        help="Do not draw the filled variability band.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Optional maximum x-axis training step.",
    )
    parser.add_argument(
        "--min-run-max-step",
        type=int,
        default=900_000,
        help=(
            "Minimum TensorBoard step a run must contain to be considered "
            "usable. Candidate runs below this threshold are skipped."
        ),
    )
    parser.add_argument(
        "--strict",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Fail when any selected method/task curve is missing or two methods "
            "resolve to the same TensorBoard run. Use --no-strict to plot the "
            "available subset."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/ablation_curves"),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("png", "pdf", "svg"),
        choices=("png", "pdf", "svg", "eps"),
        help=(
            "Figure formats to save. PDF/SVG are recommended vector formats "
            "when translucent bands are enabled."
        ),
    )
    parser.add_argument(
        "--file-prefix",
        default=None,
        help="Output filename prefix. Defaults to hybridrarl_ablations__train<seed>.",
    )
    parser.add_argument(
        "--legend-loc",
        default="lower right",
        help="Matplotlib legend location inside each task subplot.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure-level title. By default only each task title is shown.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster output resolution.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def parse_run_dir_name(name: str) -> Optional[Tuple[int, str, str, int]]:
    """Parse task_<id>__<model>__<experiment>__<seed>."""
    parts = name.split("__")
    if len(parts) < 4 or not parts[0].startswith("task_"):
        return None
    match = re.fullmatch(r"task_(\d+)", parts[0])
    if match is None:
        return None
    try:
        return int(match.group(1)), parts[1], parts[2], int(parts[-1])
    except ValueError:
        return None


def find_run_dirs(
    runs_dir: Path,
    task_id: int,
    train_seed: int,
    model_names: Sequence[str],
    exp_names: Sequence[str],
    parent_names: Sequence[str],
) -> List[Path]:
    model_priority = {name: index for index, name in enumerate(model_names)}
    exp_priority = {name: index for index, name in enumerate(exp_names)}
    parent_names = set(parent_names)
    matches = []
    if not runs_dir.exists():
        return matches
    for candidate in runs_dir.rglob("*"):
        if not candidate.is_dir():
            continue
        parsed = parse_run_dir_name(candidate.name)
        if parsed is None:
            continue
        parsed_task, parsed_model, parsed_exp, parsed_seed = parsed
        if parsed_task != int(task_id) or parsed_seed != int(train_seed):
            continue
        if parsed_model not in model_priority:
            continue
        if exp_priority and parsed_exp not in exp_priority:
            continue
        if parent_names and candidate.parent.name not in parent_names:
            continue
        matches.append(candidate)
    return sorted(
        matches,
        key=lambda path: (
            model_priority[parse_run_dir_name(path.name)[1]],
            exp_priority.get(parse_run_dir_name(path.name)[2], 0),
            -path.stat().st_mtime,
            str(path),
        ),
    )


def read_scalar_file(event_file: Path, tag: str) -> List[Tuple[int, float]]:
    accumulator = EventAccumulator(
        str(event_file),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    return [(int(event.step), float(event.value)) for event in accumulator.Scalars(tag)]


def read_curve(run_dir: Path, tag: str) -> List[Tuple[int, float]]:
    """Merge all event files in one run and keep the newest value per step."""
    event_files = sorted(
        [
            path
            for path in run_dir.iterdir()
            if path.is_file() and path.name.startswith("events.out.tfevents")
        ],
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    by_step: Dict[int, float] = {}
    for event_file in event_files:
        try:
            for step, value in read_scalar_file(event_file, tag):
                by_step[step] = value
        except Exception as exc:
            print(f"*** Warning: cannot read {event_file}: {type(exc).__name__}: {exc}")
    return sorted(by_step.items())


def max_curve_step(curves: Dict[str, List[Tuple[int, float]]]) -> int:
    max_steps = [points[-1][0] for points in curves.values() if points]
    return max(max_steps) if max_steps else -1


def total_curve_points(curves: Dict[str, List[Tuple[int, float]]]) -> int:
    return sum(len(points) for points in curves.values())


def choose_run(
    runs_dir: Path,
    task_id: int,
    train_seed: int,
    method_key: str,
    parent_override: Optional[str],
    min_run_max_step: int,
) -> Tuple[Optional[Path], Dict[str, List[Tuple[int, float]]]]:
    config = METHODS[method_key]
    parent_names = (parent_override,) if parent_override else config["parent_names"]

    def try_source(model_names, exp_names, source_parent_names):
        candidates = find_run_dirs(
            runs_dir,
            task_id,
            train_seed,
            model_names,
            exp_names,
            source_parent_names,
        )
        # Some runs are stored directly under runs/, while the no-retrieval
        # ablation is commonly grouped under a named parent directory.
        if not candidates and source_parent_names:
            candidates = find_run_dirs(
                runs_dir, task_id, train_seed, model_names, exp_names, ()
            )
        checked = []
        had_candidates = bool(candidates)
        best = None
        for candidate_index, run_dir in enumerate(candidates):
            curves = {
                RETURN_TAG: read_curve(run_dir, RETURN_TAG),
                SUCCESS_TAG: read_curve(run_dir, SUCCESS_TAG),
            }
            run_max_step = max_curve_step(curves)
            checked.append(f"{run_dir} (max_step={run_max_step})")
            if curves[RETURN_TAG] or curves[SUCCESS_TAG]:
                if run_max_step < min_run_max_step:
                    print(
                        f"*** Warning: skip incomplete run for {method_key}, "
                        f"task {task_id}: max_step={run_max_step} < "
                        f"min_run_max_step={min_run_max_step}; run={run_dir}"
                    )
                    continue
                # Prefer the run that actually covers the longest training
                # horizon. This avoids selecting a short interrupted run just
                # because it was modified more recently.
                score = (
                    run_max_step,
                    total_curve_points(curves),
                    -candidate_index,
                )
                if best is None or score > best[0]:
                    best = (score, run_dir, curves)
        if best is not None:
            _, run_dir, curves = best
            return run_dir, curves, checked, had_candidates
        return None, {RETURN_TAG: [], SUCCESS_TAG: []}, checked, had_candidates

    path, curves, checked, had_primary_candidates = try_source(
        config["model_names"],
        config["exp_names"],
        parent_names,
    )
    if path is not None:
        return path, curves
    if had_primary_candidates:
        print(
            f"*** Warning: no primary candidate met min_run_max_step="
            f"{min_run_max_step} for {method_key}, task {task_id}; "
            f"checked {checked}"
        )
        return None, {RETURN_TAG: [], SUCCESS_TAG: []}

    fallback_model_names = config["fallback_model_names"]
    if fallback_model_names:
        fallback_path, fallback_curves, fallback_checked, _ = try_source(
            fallback_model_names,
            config["fallback_exp_names"],
            config["fallback_parent_names"],
        )
        if fallback_path is not None:
            return fallback_path, fallback_curves
        checked.extend(fallback_checked)
    if checked:
        print(
            f"*** Warning: no usable TensorBoard scalars for {method_key}, "
            f"task {task_id}; checked {checked}"
        )
    return None, {RETURN_TAG: [], SUCCESS_TAG: []}


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < 2:
        return values
    window = min(int(window), int(values.size))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    # Keep the same length as the event sequence and avoid shifting the x axis.
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def tensorboard_smoothing(values: np.ndarray, smoothing: float) -> np.ndarray:
    """Match TensorBoard's scalar smoothing convention.

    TensorBoard's slider value is the weight of the previous smoothed value,
    so a larger value means a smoother curve:

        y_s[t] = smoothing*y_s[t-1] + (1-smoothing)*y[t]
    """
    if values.size == 0:
        return values
    smoothed = np.empty_like(values, dtype=np.float64)
    smoothed[0] = values[0]
    for index in range(1, values.size):
        smoothed[index] = (
            smoothing * smoothed[index - 1]
            + (1.0 - smoothing) * values[index]
        )
    return smoothed


def aggregate_events(
    steps: np.ndarray,
    values: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate dense event scalars into smooth plot points and local spread."""
    if steps.size == 0:
        return steps, values, values
    window = max(1, min(int(window), int(steps.size)))
    grouped_steps = []
    grouped_means = []
    grouped_variances = []
    for start in range(0, int(steps.size), window):
        stop = min(start + window, int(steps.size))
        group_steps = steps[start:stop]
        group_values = values[start:stop]
        grouped_steps.append(float(np.mean(group_steps)))
        grouped_means.append(float(np.mean(group_values)))
        grouped_variances.append(float(np.var(group_values)))
    return (
        np.asarray(grouped_steps, dtype=np.float64),
        np.asarray(grouped_means, dtype=np.float64),
        np.asarray(grouped_variances, dtype=np.float64),
    )


def prepare_curve(
    points: Sequence[Tuple[int, float]],
    smoothing: Optional[float],
    smooth_window: Optional[int],
    aggregation_window: int,
    band_scale: float,
    max_step: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not points:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty, empty, empty
    by_step: Dict[int, float] = {}
    for step, value in points:
        if max_step is None or step <= max_step:
            by_step[int(step)] = float(value)
    if not by_step:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty, empty, empty
    raw_steps = np.asarray(sorted(by_step), dtype=np.float64)
    raw_values = np.asarray(
        [by_step[int(step)] for step in raw_steps], dtype=np.float64
    )
    steps, values, local_variance = aggregate_events(
        raw_steps, raw_values, aggregation_window
    )
    smoothed = (
        tensorboard_smoothing(values, smoothing)
        if smoothing is not None
        else moving_average(values, int(smooth_window))
    )
    smoothed_variance = (
        tensorboard_smoothing(local_variance, smoothing)
        if smoothing is not None
        else moving_average(local_variance, int(smooth_window))
    )
    local_std = np.sqrt(np.maximum(smoothed_variance, 0.0))
    lower = smoothed - float(band_scale) * local_std
    upper = smoothed + float(band_scale) * local_std
    return steps, values, smoothed, lower, upper


def style_axes(ax, metric: str, task_id: int, task_name: str) -> None:
    ax.set_title(f"Task {task_id}: {task_name}", fontsize=10, pad=7)
    ax.set_xlabel("Timestep", fontsize=9)
    ax.set_ylabel("Reward" if metric == "return" else "Success rate", fontsize=9)
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    if metric == "success":
        ax.set_ylim(-0.02, 1.02)


def plot_metric(
    plt,
    metric: str,
    task_ids: Sequence[int],
    curves: Dict[str, Dict[int, Dict[str, List[Tuple[int, float]]]]],
    smoothing: Optional[float],
    smooth_window: Optional[int],
    aggregation_window: int,
    band_scale: float,
    show_band: bool,
    max_step: Optional[int],
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
    legend_loc: str,
    title: Optional[str],
) -> None:
    import math

    if not formats:
        raise ValueError("At least one output format is required.")
    n_tasks = len(task_ids)
    ncols = 2 if n_tasks > 1 else 1
    nrows = int(math.ceil(n_tasks / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.8 * ncols, 3.9 * nrows),
        squeeze=False,
        sharex=False,
    )
    axes_flat = list(axes.ravel())
    for axis_index, task_id in enumerate(task_ids):
        ax = axes_flat[axis_index]
        for method_key in curves:
            points = curves[method_key][task_id][RETURN_TAG if metric == "return" else SUCCESS_TAG]
            steps, values, smoothed, lower, upper = prepare_curve(
                points,
                smoothing,
                smooth_window,
                aggregation_window,
                band_scale,
                max_step,
            )
            if steps.size == 0:
                continue
            config = METHODS[method_key]
            if show_band and steps.size > 1:
                if metric == "success":
                    lower = np.clip(lower, 0.0, 1.0)
                    upper = np.clip(upper, 0.0, 1.0)
                ax.fill_between(
                    steps,
                    lower,
                    upper,
                    color=config["color"],
                    alpha=0.20,
                    linewidth=0,
                    zorder=1,
                )
        line_method_keys = sorted(
            curves,
            key=lambda method_key: METHODS[method_key].get("zorder", 3),
        )
        for method_key in line_method_keys:
            points = curves[method_key][task_id][RETURN_TAG if metric == "return" else SUCCESS_TAG]
            steps, values, smoothed, lower, upper = prepare_curve(
                points,
                smoothing,
                smooth_window,
                aggregation_window,
                band_scale,
                max_step,
            )
            if steps.size == 0:
                continue
            config = METHODS[method_key]
            ax.plot(
                steps,
                smoothed,
                color=config["color"],
                linewidth=2.0,
                alpha=0.88,
                linestyle=config.get("linestyle", "-"),
                zorder=config.get("zorder", 3),
                label=config["label"],
            )
        style_axes(ax, metric, task_id, TASK_NAMES[task_id] if task_id < len(TASK_NAMES) else f"task-{task_id}")
        legend_handles = []
        legend_labels = []
        for method_key in curves:
            points = curves[method_key][task_id][RETURN_TAG if metric == "return" else SUCCESS_TAG]
            if not points:
                continue
            config = METHODS[method_key]
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=config["color"],
                    linewidth=2.0,
                    linestyle=config.get("linestyle", "-"),
                )
            )
            legend_labels.append(config["label"])
        if legend_handles:
            ax.legend(
                legend_handles,
                legend_labels,
                loc=legend_loc,
                frameon=True,
                framealpha=0.85,
                facecolor="white",
                edgecolor="#CCCCCC",
                fontsize=10,
            )
    for ax in axes_flat[n_tasks:]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for fmt in formats:
        output_path = output_stem.with_suffix(f".{fmt}")
        save_kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(output_path, **save_kwargs)
        saved_paths.append(output_path)
    plt.close(fig)
    for saved_path in saved_paths:
        print(f"*** Saved: {saved_path}")


def write_manifest(
    output_dir: Path,
    train_seed: int,
    task_ids: Sequence[int],
    curves: Dict[str, Dict[int, Dict[str, List[Tuple[int, float]]]]],
    selected_paths: Dict[str, Dict[int, Optional[Path]]],
) -> None:
    manifest_path = output_dir / f"ablation_curve_sources__train{train_seed}.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "task_id",
                "task_name",
                "run_dir",
                "return_points",
                "success_points",
                "max_step",
            ]
        )
        for method_key in curves:
            for task_id in task_ids:
                writer.writerow(
                    [
                        method_key,
                        task_id,
                        TASK_NAMES[task_id] if task_id < len(TASK_NAMES) else f"task-{task_id}",
                        "" if selected_paths[method_key][task_id] is None else str(selected_paths[method_key][task_id]),
                        len(curves[method_key][task_id][RETURN_TAG]),
                        len(curves[method_key][task_id][SUCCESS_TAG]),
                        max_curve_step(curves[method_key][task_id]),
                    ]
                )
    print(f"*** Saved: {manifest_path}")


def main() -> int:
    args = parse_args()
    args.methods = normalize_method_list(args.methods)
    if args.smoothing is not None and not 0.0 <= args.smoothing <= 1.0:
        raise ValueError("--smoothing must be in the interval [0, 1].")
    if args.smooth_window is not None and args.smooth_window <= 0:
        raise ValueError("--smooth-window must be positive.")
    if args.smoothing is not None and args.smooth_window is not None:
        raise ValueError("Use either --smoothing or --smooth-window, not both.")
    if args.aggregation_window <= 0:
        raise ValueError("--aggregation-window must be positive.")
    if args.band_scale < 0.0:
        raise ValueError("--band-scale must be non-negative.")
    if args.min_run_max_step < 0:
        raise ValueError("--min-run-max-step must be non-negative.")
    invalid_tasks = [task_id for task_id in args.tasks if task_id < 0 or task_id >= len(TASK_NAMES)]
    if invalid_tasks:
        raise ValueError(f"Invalid task ids {invalid_tasks}; valid ids are 0-{len(TASK_NAMES) - 1}.")

    runs_dir = resolve_path(args.runs_dir)
    output_dir = resolve_path(args.output_dir)
    parent_overrides = {
        "hybridrarl": args.hybridrarl_parent,
        "hybridrarl-no-retrieval": args.hybridrarl_no_retrieval_parent,
        "hybridrarl-random-retrieval": args.hybridrarl_random_retrieval_parent,
        "hybridrarl-direct-reuse": args.hybridrarl_direct_reuse_parent,
        "hybridrarl-no-selective-reuse": args.hybridrarl_no_selective_reuse_parent,
    }
    curves: Dict[str, Dict[int, Dict[str, List[Tuple[int, float]]]]] = {}
    selected_paths: Dict[str, Dict[int, Optional[Path]]] = {}
    missing = []

    print("*** Benchmark: meta-world")
    print("*** Curves: charts/episodic_return, charts/success")
    print(f"*** Runs dir: {runs_dir}")
    print(f"*** Train seed: {args.train_seed}")
    print(f"*** Tasks: {args.tasks}")
    if args.smoothing is not None:
        print(f"*** Smoothing: TensorBoard-style smoothing={args.smoothing}")
    else:
        print(f"*** Smoothing: moving average window={args.smooth_window} event points")
    print(f"*** Event aggregation window: {args.aggregation_window}")
    print(f"*** Filled variability band: {'disabled' if args.no_band else f'scale={args.band_scale}'}")
    print(f"*** Minimum usable run max_step: {args.min_run_max_step}")
    for method_key in args.methods:
        curves[method_key] = {}
        selected_paths[method_key] = {}
        for task_id in args.tasks:
            path, method_curves = choose_run(
                runs_dir,
                task_id,
                args.train_seed,
                method_key,
                parent_overrides.get(method_key),
                args.min_run_max_step,
            )
            curves[method_key][task_id] = method_curves
            selected_paths[method_key][task_id] = path
            if path is None:
                missing.append((method_key, task_id))
            else:
                method_max_step = max_curve_step(method_curves)
                parsed_source = parse_run_dir_name(path.name)
                source_model = parsed_source[1] if parsed_source else "unknown"
                source_exp = parsed_source[2] if parsed_source else "unknown"
                print(
                    f"*** {method_key:16s} task={task_id:02d} "
                    f"model={source_model} "
                    f"exp={source_exp} "
                    f"return_points={len(method_curves[RETURN_TAG]):4d} "
                    f"success_points={len(method_curves[SUCCESS_TAG]):4d} "
                    f"max_step={method_max_step:7d} "
                    f"run={path}"
                )

    if missing and args.strict:
        formatted = ", ".join(f"{method}/task_{task_id}" for method, task_id in missing)
        raise FileNotFoundError(f"Missing selected curves: {formatted}")
    if missing:
        print(
            "*** Warning: missing curves were skipped: "
            + ", ".join(f"{method}/task_{task_id}" for method, task_id in missing)
        )

    duplicate_sources = []
    for task_id in args.tasks:
        source_to_methods: Dict[Path, List[str]] = {}
        for method_key in args.methods:
            source = selected_paths[method_key][task_id]
            if source is not None:
                source_to_methods.setdefault(source, []).append(method_key)
        duplicate_sources.extend(
            (task_id, source, method_keys)
            for source, method_keys in source_to_methods.items()
            if len(method_keys) > 1
        )
    if duplicate_sources:
        formatted = "; ".join(
            f"task {task_id}: {source} -> {', '.join(method_keys)}"
            for task_id, source, method_keys in duplicate_sources
        )
        message = (
            "Different ablations resolved to the same TensorBoard run; refusing "
            "to overlay identical curves. " + formatted
        )
        if args.strict:
            raise RuntimeError(message)
        print(f"*** Warning: {message}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("This script requires matplotlib.") from exc

    prefix = args.file_prefix or f"hybridrarl_ablations__train{args.train_seed}"
    plot_metric(
        plt,
        "return",
        args.tasks,
        curves,
        args.smoothing,
        args.smooth_window,
        args.aggregation_window,
        args.band_scale,
        not args.no_band,
        args.max_step,
        output_dir / f"{prefix}__reward",
        args.formats,
        args.dpi,
        args.legend_loc,
        args.title,
    )
    plot_metric(
        plt,
        "success",
        args.tasks,
        curves,
        args.smoothing,
        args.smooth_window,
        args.aggregation_window,
        args.band_scale,
        not args.no_band,
        args.max_step,
        output_dir / f"{prefix}__success",
        args.formats,
        args.dpi,
        args.legend_loc,
        args.title,
    )
    write_manifest(output_dir, args.train_seed, args.tasks, curves, selected_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
