#!/usr/bin/env python3
"""
Evaluate Meta-World HybridRARL direct-reuse ablation on task0-task9.

Protocol:
  * task 0 uses the original first-stage HybridRARL checkpoint, because it has
    no previous tasks to reuse.
  * task 1..9 use checkpoints saved as
      agents/hybridrarl_prev_units/
        task_i__hybridrarl-ablation-no-select__all_prev_units__seed

This script intentionally resolves those paths explicitly so the direct-reuse
ablation is not confused with:
  * full HybridRARL selected-reuse checkpoints
  * base HybridRARL without prev-units
  * no-retrieval / no-selective-reuse ablations

Typical usage from experiments/meta-world:

  python evaluate_hybridrarl_ablation_no_select.py \
      --train-seed 42 \
      --test-seed 0 \
      --episodes 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

import torch


def import_eval_lib():
    experiments_dir = Path(__file__).resolve().parents[1]
    if str(experiments_dir) not in sys.path:
        sys.path.insert(0, str(experiments_dir))
    import test_rl_model as eval_lib

    return eval_lib


def resolve_path(path: Path, exp_dir: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    return (exp_dir / path).resolve()


def model_dir_name(task_id: int, model_type: str, exp_name: str, seed: int) -> str:
    return f"task_{int(task_id)}__{model_type}__{exp_name}__{int(seed)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate hybridrarl-ablation-no-select on Meta-World task0-task9."
    )
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--test-seed", type=int, required=True)
    parser.add_argument(
        "--episodes",
        "--episode",
        dest="episodes",
        type=int,
        required=True,
        help="Number of evaluation episodes per task.",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="*",
        default=list(range(10)),
        help="Evaluation task ids. Default: 0 1 ... 9.",
    )
    parser.add_argument(
        "--base-root",
        type=Path,
        default=Path("agents"),
        help="Root containing original first-stage HybridRARL checkpoints.",
    )
    parser.add_argument(
        "--base-model-type",
        default="hybridrarl",
        help="Model-type component for task0's original HybridRARL checkpoint.",
    )
    parser.add_argument(
        "--base-exp-name",
        default="SAC_hybridRAGRL",
        help="Experiment-name component for task0's original HybridRARL checkpoint.",
    )
    parser.add_argument(
        "--transfer-root",
        type=Path,
        default=Path("agents/hybridrarl_prev_units"),
        help="Root containing hybridrarl-ablation-no-select checkpoints.",
    )
    parser.add_argument(
        "--transfer-model-type",
        default="hybridrarl-ablation-no-select",
        help="Model-type component for task1-9 direct-reuse checkpoints.",
    )
    parser.add_argument(
        "--transfer-exp-name",
        default="all_prev_units",
        help="Experiment-name component for task1-9 direct-reuse checkpoints.",
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--fallback-max-steps", type=int, default=500)
    parser.add_argument("--deterministic", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--recursive-search", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--cuda", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--skip-missing",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Skip tasks whose explicitly requested checkpoint is missing or "
            "unusable. Enabled by default; pass --no-skip-missing to fail fast."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path. Defaults to analysis/eval_results/...",
    )
    return parser.parse_args()


def make_eval_args(
    *,
    user_args: argparse.Namespace,
    model_name: str,
    model_root: Path,
    model_dir: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        benchmark="meta-world",
        model_name=model_name,
        train_seed=int(user_args.train_seed),
        test_seed=int(user_args.test_seed),
        episodes=int(user_args.episodes),
        tasks=None,
        eval_task_id=None,
        source_task_id=None,
        model_root=model_root,
        model_dir=str(model_dir),
        scenario="",
        componet_prev_tasks=None,
        exp_name=None,
        exp_name_candidates=None,
        max_steps=int(user_args.max_steps),
        fallback_max_steps=int(user_args.fallback_max_steps),
        deterministic=bool(user_args.deterministic),
        prefer_prev_units=False,
        recursive_search=bool(user_args.recursive_search),
        cuda=bool(user_args.cuda),
        output=None,
    )


def expected_model_dir(args: argparse.Namespace, exp_dir: Path, task_id: int) -> Path:
    if int(task_id) == 0:
        root = resolve_path(args.base_root, exp_dir)
        return root / model_dir_name(
            task_id=0,
            model_type=args.base_model_type,
            exp_name=args.base_exp_name,
            seed=args.train_seed,
        )

    root = resolve_path(args.transfer_root, exp_dir)
    return root / model_dir_name(
        task_id=task_id,
        model_type=args.transfer_model_type,
        exp_name=args.transfer_exp_name,
        seed=args.train_seed,
    )


def main() -> int:
    args = parse_args()
    eval_lib = import_eval_lib()
    exp_dir = Path(__file__).resolve().parent
    tasks_module, models_module = eval_lib.import_experiment_modules(exp_dir)
    task_names = getattr(tasks_module, "tasks", [])

    invalid_tasks = [task_id for task_id in args.tasks if task_id < 0 or task_id >= len(task_names)]
    if invalid_tasks:
        raise ValueError(f"Invalid task ids: {invalid_tasks}; valid range is 0..{len(task_names) - 1}.")

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    eval_lib.set_seed(args.test_seed)

    if args.output is None:
        output = (
            exp_dir
            / "analysis"
            / "eval_results"
            / (
                f"meta-world__hybridrarl-ablation-no-select"
                f"__train{args.train_seed}__test{args.test_seed}.csv"
            )
        )
    else:
        output = resolve_path(args.output, exp_dir)

    print("*** Benchmark: meta-world")
    print("*** Method: hybridrarl-ablation-no-select / direct all-prev-units reuse")
    print(f"*** task0 source: {args.base_model_type}__{args.base_exp_name}")
    print(f"*** task1-9 source: {args.transfer_model_type}__{args.transfer_exp_name}")
    print(f"*** Train seed: {args.train_seed}")
    print(f"*** Test seed: {args.test_seed}")
    print(f"*** Episodes: {args.episodes}")
    print(f"*** Device: {device}")

    records: List[dict] = []
    skipped: List[dict] = []
    for task_id in [int(task_id) for task_id in args.tasks]:
        model_dir = expected_model_dir(args, exp_dir, task_id)
        model_name = args.base_model_type if task_id == 0 else args.transfer_model_type
        model_root = resolve_path(args.base_root if task_id == 0 else args.transfer_root, exp_dir)

        if not eval_lib.has_saved_actor(model_dir):
            message = f"task {task_id}: missing checkpoint at {model_dir}"
            if args.skip_missing:
                print(f"*** Skip: {message}")
                skipped.append({"task_id": task_id, "model_dir": str(model_dir), "reason": "missing"})
                continue
            raise FileNotFoundError(message)

        eval_args = make_eval_args(
            user_args=args,
            model_name=model_name,
            model_root=model_root,
            model_dir=model_dir,
        )
        try:
            record = eval_lib.evaluate_one_task(
                eval_args,
                exp_dir,
                tasks_module,
                models_module,
                task_id,
                device,
            )
        except (FileNotFoundError, RuntimeError, EOFError, OSError) as exc:
            message = (
                f"task {task_id}: unusable checkpoint at {model_dir}; "
                f"{type(exc).__name__}: {exc}"
            )
            if args.skip_missing:
                print(f"*** Skip: {message}")
                skipped.append(
                    {
                        "task_id": task_id,
                        "model_dir": str(model_dir),
                        "reason": "unusable",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue
            raise
        record["requested_method"] = "hybridrarl-ablation-no-select"
        record["task0_fallback_to_base_hybridrarl"] = bool(task_id == 0)
        records.append(record)

    if not records:
        raise RuntimeError("No tasks were evaluated. Check model paths or remove --skip-missing.")

    print("\nSummary")
    for record in records:
        print(
            f"task={record['model_task_id']} ({record['task_name']}), "
            f"ckpt_model={record['checkpoint_model_name']}, "
            f"return={record['mean_return']:.4f}±{record['std_return']:.4f}, "
            f"success={record['mean_success']:.4f}±{record['std_success']:.4f}, "
            f"model={record['model_dir']}"
        )

    overall_summary = eval_lib.overall_success_summary(records)
    if skipped:
        overall_summary["skipped"] = skipped
    print(
        "\nOverall Performance"
        f"\nmean_return_across_tasks={overall_summary['mean_return_across_tasks']:.6f}, "
        f"var_return_across_tasks={overall_summary['var_return_across_tasks']:.6f}"
        f"\nmean_success_across_tasks={overall_summary['mean_success_across_tasks']:.6f}, "
        f"var_success_across_tasks={overall_summary['var_success_across_tasks']:.6f}"
        f"\nmean_return_across_episodes={overall_summary['mean_return_across_episodes']:.6f}, "
        f"mean_success_across_episodes={overall_summary['mean_success_across_episodes']:.6f}"
    )

    output = Path(output)
    eval_lib.write_results(records, output)
    summary_path = eval_lib.write_overall_summary(overall_summary, output)
    print(f"\n*** Saved CSV: {output}")
    print(f"*** Saved JSON: {output.with_suffix('.json')}")
    print(f"*** Saved summary JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
