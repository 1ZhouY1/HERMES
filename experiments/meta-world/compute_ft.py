#!/usr/bin/env python3
"""Compute Meta-World forward transfer from TensorBoard training curves.

For task i, using performance p_i(t) in [0, 1]:

    AUC_i   = 1 / Delta * integral_0^Delta p_i(t) dt
    AUC_i_b = 1 / Delta * integral_0^Delta p_i^b(t) dt
    FTr_i   = (AUC_i - AUC_i_b) / (1 - AUC_i_b)

The method and the simple baseline are matched by task id and training seed.
The test seed is recorded in the output metadata; TensorBoard run names in
this project contain the training seed rather than the test seed.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Older TensorBoard versions still refer to NumPy 1.x aliases. Keep this
# script usable in environments with NumPy 2.x.
if not hasattr(np, "string_"):
    np.string_ = np.bytes_
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tasks import get_task_name, tasks


# `charts/success` is the per-training-step curve. Do not silently fall back
# to `charts/test_success`, because it may only contain a sparse endpoint
# evaluation value and is not the training curve used for FT.
DEFAULT_METRIC_TAGS = ("charts/success",)
TASK_ONE_START_MODELS = {
    "finetune",
    "componet",
    "component",
}
FINETUNE_VARIANTS_WITHOUT_FTR = {
    "finetune-1",
    "finetune-n",
}
MODEL_SEARCH_PRIORITIES = {
    "hybridrarl": (
        "hybridrarl-prev-units",
        "hybridrarl-simple-prev-units",
        "hybridrarl",
        "hybrid",
    ),
    "hybridrarl-random-retrieval": (
        "hybridrarl-random-retrieval",
    ),
}

MODEL_EXP_NAME_PRIORITIES = {
    "hybridrarl-prev-units": (
        "csv_threshold_transfer",
        "SAC_hybridrarl_prev_units",
    ),
    "hybridrarl-simple-prev-units": (
        "csv_threshold_transfer",
        "csv_simple_prev",
    ),
    "hybridrarl": (
        "SAC_hybridRAGRL",
        "SAC",
    ),
    "hybrid": (
        "SAC_hybridRAGRL",
        "SAC",
    ),
    "hybridrarl-random-retrieval": (
        "SAC_hybridRAGRL",
        "SAC",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Forward Transfer from Meta-World TensorBoard curves."
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help=(
            "Method model name in run directories, for example hybridrarl, "
            "hybridrarl-random-retrieval, or packnet."
        ),
    )
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--test-seed", type=int, required=True)
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Task ids. By default uses 0-9, except finetune/componet models "
            "which default to 1-9."
        ),
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--baseline-name", default="simple")
    parser.add_argument(
        "--metric-tag",
        default="charts/success",
        help="Training-curve scalar tag (default: charts/success).",
    )
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument(
        "--no-clip-performance",
        action="store_true",
        help="Do not clip TensorBoard values to the [0, 1] range.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def parse_run_dir_name(dirname: str) -> Optional[Tuple[int, str, str, int]]:
    parts = Path(dirname).name.split("__")
    if len(parts) < 4 or not parts[0].startswith("task_"):
        return None
    try:
        task_id = int(parts[0][len("task_"):])
        train_seed = int(parts[-1])
    except ValueError:
        return None
    return task_id, parts[1], parts[2], train_seed


def find_run_dirs(
    runs_dir: Path,
    task_id: int,
    model_name: str,
    train_seed: int,
    exp_names: Sequence[str] = (),
) -> List[Path]:
    exp_priority = {name: index for index, name in enumerate(exp_names)}
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
        if parsed_task != int(task_id) or parsed_model != str(model_name) or parsed_seed != int(train_seed):
            continue
        if exp_priority and parsed_exp not in exp_priority:
            continue
        matches.append(candidate)
    return sorted(
        matches,
        key=lambda path: (
            exp_priority.get(parse_run_dir_name(path.name)[2], 0),
            -path.stat().st_mtime,
            str(path),
        ),
    )


def event_files(run_dir: Path) -> List[Path]:
    return sorted(
        [path for path in run_dir.iterdir() if path.is_file() and path.name.startswith("events.out.tfevents")],
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )


def scalar_points(event_file: Path, tags: Sequence[str]):
    accumulator = EventAccumulator(
        str(event_file),
        # TensorBoard stores scalar summaries under the literal "scalars"
        # key; EventAccumulator.SCALARS is not available in all versions.
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    for tag in tags:
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        events = accumulator.Scalars(tag)
        if events:
            points = sorted((int(event.step), float(event.value)) for event in events)
            return tag, points
    return None, []


def choose_curve_file(
    run_dirs: Sequence[Path],
    requested_tag: str,
    total_timesteps: int,
) -> Dict:
    tags = [requested_tag] + [tag for tag in DEFAULT_METRIC_TAGS if tag != requested_tag]
    checked = []
    for run_dir in run_dirs:
        for event_file in event_files(run_dir):
            try:
                selected_tag, points = scalar_points(event_file, tags)
            except Exception as exc:
                checked.append(f"{event_file}: {type(exc).__name__}: {exc}")
                continue
            if not points:
                checked.append(f"{event_file}: no requested scalar tag")
                continue
            max_step = max(step for step, _ in points)
            checked.append(f"{event_file}: max_step={max_step}, tag={selected_tag}")
            # Most training loops use range(total_timesteps), so the final
            # logged step is total_timesteps - 1. Accept both 0-based and
            # 1-based logging conventions.
            required_last_step = max(0, int(total_timesteps) - 1)
            if max_step >= required_last_step:
                return {
                    "run_dir": run_dir,
                    "event_file": event_file,
                    "tag": selected_tag,
                    "points": points,
                    "max_step": max_step,
                    "checked": checked,
                }

    details = "\n".join(f"    {item}" for item in checked)
    raise RuntimeError(
        f"No complete TensorBoard curve found in run dirs {list(run_dirs)}. "
        f"Required max step: at least {max(0, int(total_timesteps) - 1)} "
        f"(0-based logging) or {total_timesteps} (1-based logging).\n{details}"
    )


def clean_points(points: Sequence[Tuple[int, float]], clip_performance: bool):
    # Keep the last scalar value when a writer emits duplicate steps.
    by_step = {}
    for step, value in points:
        by_step[int(step)] = float(value)
    steps = np.asarray(sorted(by_step), dtype=np.float64)
    values = np.asarray([by_step[int(step)] for step in steps], dtype=np.float64)
    if clip_performance:
        values = np.clip(values, 0.0, 1.0)
    return steps, values


def normalized_auc(
    points: Sequence[Tuple[int, float]],
    total_timesteps: int,
    clip_performance: bool,
) -> float:
    """Integrate one task's curve over its local training interval [0, Delta].

    The task index in the paper only describes the corresponding interval on
    a global timeline. Each TensorBoard run resets its local step counter, so
    no additional (i - 1) * Delta shift is needed here.
    """
    steps, values = clean_points(points, clip_performance=clip_performance)
    horizon = float(total_timesteps)
    valid = (steps >= 0.0) & (steps <= horizon)
    steps = steps[valid]
    values = values[valid]
    if steps.size == 0:
        raise ValueError("The selected curve has no points inside the training horizon.")

    # np.interp supplies the nearest observed value before the first point and
    # after the last point, making the integral well-defined at both boundaries.
    integration_steps = np.unique(np.concatenate(([0.0], steps, [horizon])))
    integration_values = np.interp(integration_steps, steps, values)
    return float(np.trapz(integration_values, integration_steps) / horizon)


def ftr(method_auc: float, baseline_auc: float) -> float:
    denominator = 1.0 - float(baseline_auc)
    if abs(denominator) < 1e-12:
        return float("nan")
    return float((float(method_auc) - float(baseline_auc)) / denominator)


def normalized_model_name(model_name: str) -> str:
    return str(model_name).strip().lower().replace("_", "-")


def model_name_candidates(model_name: str) -> Tuple[str, ...]:
    """Return model-directory names in the requested search priority."""
    normalized_name = normalized_model_name(model_name)
    return MODEL_SEARCH_PRIORITIES.get(normalized_name, (str(model_name),))


def exp_name_candidates(model_name: str) -> Tuple[str, ...]:
    return MODEL_EXP_NAME_PRIORITIES.get(str(model_name), ())


def find_preferred_curve(
    runs_dir: Path,
    task_id: int,
    model_name: str,
    train_seed: int,
    requested_tag: str,
    total_timesteps: int,
) -> Tuple[str, Dict]:
    """Choose the first complete curve following the model priority list."""
    candidates = model_name_candidates(model_name)
    errors = []
    for candidate_name in candidates:
        exp_names = exp_name_candidates(candidate_name)
        candidate_dirs = find_run_dirs(
            runs_dir,
            task_id,
            candidate_name,
            train_seed,
            exp_names=exp_names,
        )
        if not candidate_dirs:
            exp_text = "" if not exp_names else f" with exp in {list(exp_names)}"
            errors.append(f"{candidate_name}: no run directory{exp_text}")
            continue
        try:
            curve = choose_curve_file(
                candidate_dirs, requested_tag, total_timesteps
            )
        except RuntimeError as exc:
            errors.append(f"{candidate_name}: {exc}")
            continue
        return candidate_name, curve

    details = "\n".join(f"    {item}" for item in errors)
    raise FileNotFoundError(
        f"No complete curve found for task {task_id}, requested model "
        f"{model_name}, train seed {train_seed}. Search priority: "
        f"{list(candidates)}\n{details}"
    )


def main() -> int:
    args = parse_args()
    normalized_name = normalized_model_name(args.model_name)
    if normalized_name in FINETUNE_VARIANTS_WITHOUT_FTR:
        raise ValueError(
            "Forward Transfer is only computed for the `finetune` model; "
            "finetune-1 and finetune-n are excluded."
        )

    starts_from_task_one = normalized_name in {
        normalized_model_name(name) for name in TASK_ONE_START_MODELS
    }
    if args.tasks is None:
        start_task = 1 if starts_from_task_one else 0
        task_ids = list(range(start_task, min(10, len(tasks))))
    else:
        task_ids = list(args.tasks)

    invalid = [task_id for task_id in task_ids if task_id < 0 or task_id >= len(tasks)]
    if invalid:
        raise ValueError(f"Invalid task ids {invalid}; valid ids are 0-{len(tasks) - 1}.")
    runs_dir = resolve_path(args.runs_dir)
    clip_performance = not args.no_clip_performance
    rows = []

    print("*** Benchmark: meta-world")
    print(f"*** Method: {args.model_name}")
    print(f"*** Baseline: {args.baseline_name}")
    print(f"*** Train seed: {args.train_seed}")
    print(f"*** Test seed: {args.test_seed}")
    print(f"*** Tasks: {task_ids}")
    if args.tasks is None and task_ids and task_ids[0] == 1:
        print("*** Task 0 skipped: this model family starts training from task 1")
    print(f"*** Runs dir: {runs_dir}")
    print(f"*** Metric tag: {args.metric_tag}")
    method_candidates = model_name_candidates(args.model_name)
    print(f"*** Method search priority: {list(method_candidates)}")
    for candidate_name in method_candidates:
        exp_names = exp_name_candidates(candidate_name)
        if exp_names:
            print(f"*** Exp-name priority for {candidate_name}: {list(exp_names)}")

    for task_id in task_ids:
        baseline_dirs = find_run_dirs(runs_dir, task_id, args.baseline_name, args.train_seed)
        if not baseline_dirs:
            raise FileNotFoundError(
                f"No baseline run directory found for task {task_id}, model {args.baseline_name}, "
                f"train seed {args.train_seed} under {runs_dir}."
            )

        selected_method_name, method_curve = find_preferred_curve(
            runs_dir,
            task_id,
            args.model_name,
            args.train_seed,
            args.metric_tag,
            args.total_timesteps,
        )
        baseline_curve = choose_curve_file(baseline_dirs, args.metric_tag, args.total_timesteps)
        method_auc = normalized_auc(
            method_curve["points"], args.total_timesteps, clip_performance
        )
        baseline_auc = normalized_auc(
            baseline_curve["points"], args.total_timesteps, clip_performance
        )
        task_ftr = ftr(method_auc, baseline_auc)
        row = {
            "task_id": int(task_id),
            "task_name": get_task_name(task_id),
            "model_name": args.model_name,
            "selected_model_name": selected_method_name,
            "baseline_name": args.baseline_name,
            "train_seed": int(args.train_seed),
            "test_seed": int(args.test_seed),
            "metric_tag": method_curve["tag"],
            "total_timesteps": int(args.total_timesteps),
            "method_run_dir": str(method_curve["run_dir"]),
            "method_event_file": str(method_curve["event_file"]),
            "method_max_step": int(method_curve["max_step"]),
            "baseline_run_dir": str(baseline_curve["run_dir"]),
            "baseline_event_file": str(baseline_curve["event_file"]),
            "baseline_max_step": int(baseline_curve["max_step"]),
            "auc_method": method_auc,
            "auc_baseline": baseline_auc,
            "forward_transfer": task_ftr,
        }
        rows.append(row)
        print(
            f"task={task_id} ({get_task_name(task_id)}), "
            f"model={selected_method_name}, AUC={method_auc:.6f}, "
            f"AUC_b={baseline_auc:.6f}, FTr={task_ftr:.6f}"
        )
        print(f"  method curve: {method_curve['event_file']}")
        print(f"  baseline curve: {baseline_curve['event_file']}")

    ftr_values = np.asarray(
        [row["forward_transfer"] for row in rows if np.isfinite(row["forward_transfer"])],
        dtype=np.float64,
    )
    # Finetune/CompoNet reuse the simple model on task 0.  There is no
    # separate task-0 curve, but its FT contribution is exactly zero and must
    # still be included in the average over all benchmark tasks.
    total_task_count = len(tasks)
    implicit_baseline_task_count = 1 if starts_from_task_one and 0 not in task_ids else 0
    summary_ftr_values = (
        np.concatenate((ftr_values, np.zeros(implicit_baseline_task_count, dtype=np.float64)))
        if implicit_baseline_task_count
        else ftr_values
    )
    average_denominator = (
        total_task_count if starts_from_task_one else summary_ftr_values.size
    )
    ftr_mean = (
        float(np.sum(summary_ftr_values)) / float(average_denominator)
        if average_denominator
        else float("nan")
    )
    summary = {
        "benchmark": "meta-world",
        "model_name": args.model_name,
        "baseline_name": args.baseline_name,
        "train_seed": int(args.train_seed),
        "test_seed": int(args.test_seed),
        "tasks": [int(task_id) for task_id in task_ids],
        "total_timesteps": int(args.total_timesteps),
        "metric_tag": args.metric_tag,
        "mean_forward_transfer": ftr_mean,
        "var_forward_transfer": (
            float(np.var(summary_ftr_values)) if summary_ftr_values.size else float("nan")
        ),
        "num_valid_forward_transfer": int(ftr_values.size),
        "num_total_tasks_for_average": int(total_task_count),
        "num_implicit_baseline_tasks": int(implicit_baseline_task_count),
    }
    print("\nOverall Forward Transfer")
    print(
        f"mean={summary['mean_forward_transfer']:.6f}, "
        f"variance={summary['var_forward_transfer']:.6f}"
    )

    output = args.output
    if output is None:
        output = Path.cwd() / "analysis" / (
            f"forward_transfer__{args.model_name}__train{args.train_seed}__test{args.test_seed}.csv"
        )
    output = resolve_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_output = output.with_suffix(".json")
    json_output.write_text(json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8")
    summary_output = output.with_name(output.stem + "__summary.json")
    summary_output.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(f"*** Saved CSV: {output}")
    print(f"*** Saved JSON: {json_output}")
    print(f"*** Saved summary JSON: {summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
