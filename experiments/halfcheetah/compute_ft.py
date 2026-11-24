#!/usr/bin/env python3
"""Compute test-performance Forward Transfer for HalfCheetah scenarios.

For a scenario with T unique tasks, this script evaluates the model at each
training stage and implements

    FT = 1 / T * sum_i (P_{T,i} - b_i),

where P_i is the performance of the checkpoint corresponding to task i,
evaluated on task i.  For CompoNet, that checkpoint contains all previous
units available at stage i.  b_i is the fixed SAC-N reference reward for task
i from the supplied reference table.

HalfCheetah has no binary success metric, so performance is the mean
cumulative episode reward.  The reference rewards b_i are the fixed SAC-N
values supplied in the HalfCheetah reference table and are scenario specific.
This implementation intentionally does not use TensorBoard training curves or
AUC.

The scenario definitions contain repeated passes.  Only the first occurrence
of each task is evaluated, so T is 4 for forgetting, transfer, and
compositionality, and T is 2 for robustness.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from tasks import SCENARIOS, get_task_name
from test_halfcheetah_models import (
    evaluate_task,
    model_plan,
    resolve_model_dir,
    scenario_previous_paths,
    set_test_seed,
    unique_scenario_task_ids,
)


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SIMPLE_EXP_NAMES = ["SAC", "run_sac", "SAC_hybridRAGRL", "sac_20k"]

# Fixed b_i values from the supplied HalfCheetah reference-reward table.
# They are scenario-specific because a task name can have a different
# reference reward in different scenarios.
REFERENCE_REWARDS = {
    "forgetting": {
        "hugefoot": 2209.0,
        "moon": 2982.0,
        "carry_stuff": 6309.0,
        "rainfall": 1001.0,
    },
    "transfer": {
        "carry_stuff_hugegravity": 7233.0,
        "moon": 3599.0,
        # The table calls this task defective_sensors; tasks.py calls it
        # defective_module.
        "defective_module": 5909.0,
        "hugefoot_rainfall": 2942.0,
    },
    "robustness": {
        "normal": 4932.0,
        "inverted_actions": 5833.0,
    },
    "compositionality": {
        "tinyfoot": 6311.0,
        "moon": 3932.0,
        "carry_stuff_hugegravity": 6319.0,
        "tinyfoot_moon": 1355.0,
    },
}


def normalize_model_name(name: str) -> str:
    aliases = {
        "finetune_1": "finetune-1",
        "finetune1": "finetune-1",
        "finetune_n": "finetune-n",
        "finetunen": "finetune-n",
        "component": "componet",
    }
    normalized = str(name).strip().lower().replace("_", "-")
    return aliases.get(normalized, normalized)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute HalfCheetah Forward Transfer from final test performance."
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help=(
            "Continual model whose final checkpoint is evaluated: finetune, "
            "componet, hybridrarl, hybrid, packnet, or prognet."
        ),
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=sorted(SCENARIOS),
    )
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--test-seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--model-root", type=Path, default=Path("agents"))
    parser.add_argument(
        "--exp-name-candidates",
        nargs="*",
        default=[],
        help="Additional continual-model experiment names to search.",
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the actor mean rather than sampling during evaluation.",
    )
    parser.add_argument(
        "--cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path; JSON and summary JSON are written beside it.",
    )
    return parser.parse_args()


def resolve_model_root(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def continual_exp_names(model_name: str, extra: Sequence[str]) -> List[str]:
    if model_name == "hybridrarl":
        defaults = [
            "SAC_hybridrarl_prev_units",
            "SAC_hybridRAGRL",
            "SAC",
            "run_sac",
        ]
    else:
        defaults = ["SAC", "run_sac"]
    return list(dict.fromkeys(defaults + list(extra)))


def evaluate_one_checkpoint(
    args,
    task_id: int,
    source_task_id: int,
    model_dir: Path,
    previous_paths: Sequence[Path],
    device: torch.device,
) -> Dict:
    """Evaluate one checkpoint on one HalfCheetah task."""
    # evaluate_task uses the actor mean in the same way as the standalone
    # evaluator, so all P_T_i and b_i values share one protocol.
    return evaluate_task(
        args,
        task_id=int(task_id),
        source_task_id=int(source_task_id),
        model_dir=model_dir,
        previous_paths=previous_paths,
        device=device,
    )


def write_outputs(rows: Sequence[Dict], summary: Dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scenario",
        "scenario_position",
        "task_id",
        "task_name",
        "train_seed",
        "test_seed",
        "episodes",
        "stage_model_name",
        "checkpoint_task_id",
        "stage_model_dir",
        "mean_return_P_i",
        "reference_reward_b_i",
        "reference_reward_source",
        "forward_transfer_difference",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    json_output = output.with_suffix(".json")
    json_output.write_text(
        json.dumps(list(rows), indent=2, sort_keys=True), encoding="utf-8"
    )
    summary_output = output.with_name(output.stem + "__summary.json")
    summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"*** Saved CSV: {output}")
    print(f"*** Saved JSON: {json_output}")
    print(f"*** Saved summary JSON: {summary_output}")


def main() -> int:
    args = parse_args()
    args.model_name = normalize_model_name(args.model_name)
    supported = {"finetune", "componet", "hybridrarl", "hybrid", "packnet", "prognet"}
    if args.model_name not in supported:
        raise ValueError(
            f"Unsupported model `{args.model_name}`. Choose from {sorted(supported)}."
        )
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")

    task_ids = unique_scenario_task_ids(args.scenario)
    model_root = resolve_model_root(args.model_root)
    args.model_root = model_root
    # This is only used to reconstruct CompoNet's first simple prev unit;
    # b_i itself comes from REFERENCE_REWARDS below.
    args.simple_exp_names = list(DEFAULT_SIMPLE_EXP_NAMES)
    continual_names = continual_exp_names(
        args.model_name, args.exp_name_candidates
    )
    model_names, _ = model_plan(args.model_name)
    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )
    set_test_seed(args.test_seed)

    print("*** Benchmark: halfcheetah")
    print(f"*** Method: {args.model_name}")
    print(f"*** Scenario: {args.scenario}")
    print(f"*** Train seed: {args.train_seed}")
    print(f"*** Test seed: {args.test_seed}")
    print(f"*** Episodes per task: {args.episodes}")
    print(f"*** Device: {device}")
    print(f"*** Unique tasks: {[get_task_name(task_id) for task_id in task_ids]}")
    print("*** Metric: mean cumulative episode return")
    print("*** b_i source: supplied SAC-N reference-reward table")
    print("*** FT: mean_i(P_i - b_i), using the checkpoint at each task stage")

    rows = []
    for position, task_id in enumerate(task_ids):
        task_name = get_task_name(task_id)
        try:
            reference_reward = float(REFERENCE_REWARDS[args.scenario][task_name])
        except KeyError as exc:
            raise KeyError(
                f"No reference reward b_i is defined for scenario={args.scenario!r}, "
                f"task={task_name!r}."
            ) from exc

        # Each P_i must be produced by the checkpoint at the corresponding
        # training stage. CompoNet and Finetune use the already-trained simple
        # model for the first task and add current-method checkpoints later.
        stage_model_names = model_names
        stage_exp_names = continual_names
        stage_scenario = args.scenario
        previous_paths = []
        if args.model_name in {"componet", "finetune"} and position == 0:
            stage_model_names = ["simple"]
            stage_exp_names = args.simple_exp_names
            stage_scenario = ""
        elif args.model_name == "componet":
            previous_paths = scenario_previous_paths(
                args,
                task_ids,
                position,
                model_root,
                continual_names,
            )
        elif args.model_name == "prognet":
            previous_paths = [
                resolve_model_dir(
                    model_root,
                    args.scenario,
                    previous_task_id,
                    ["prognet"],
                    continual_names,
                    args.train_seed,
                )
                for previous_task_id in task_ids[:position]
            ]

        stage_model_dir = resolve_model_dir(
            model_root,
            stage_scenario,
            task_id,
            stage_model_names,
            stage_exp_names,
            args.train_seed,
        )

        print(
            f"\n*** Task {position + 1}/{len(task_ids)}: "
            f"{task_name} (task {task_id}, stage checkpoint task {task_id})"
        )
        final_record = evaluate_one_checkpoint(
            args,
            task_id,
            task_id,
            stage_model_dir,
            previous_paths,
            device,
        )
        p_i = float(final_record["mean_return"])
        difference = p_i - reference_reward
        row = {
            "scenario": args.scenario,
            "scenario_position": position + 1,
            "task_id": int(task_id),
            "task_name": get_task_name(task_id),
            "train_seed": int(args.train_seed),
            "test_seed": int(args.test_seed),
            "episodes": int(args.episodes),
            "stage_model_name": stage_model_dir.name.split("__")[1],
            "checkpoint_task_id": int(task_id),
            "stage_model_dir": str(stage_model_dir),
            "mean_return_P_i": p_i,
            "reference_reward_b_i": reference_reward,
            "reference_reward_source": "supplied SAC-N reference table",
            "forward_transfer_difference": difference,
        }
        rows.append(row)
        print(
            f"*** P_i={p_i:.6f}, b_i={reference_reward:.6f}, "
            f"P_i-b_i={difference:.6f}"
        )

    differences = np.asarray(
        [row["forward_transfer_difference"] for row in rows], dtype=np.float64
    )
    summary = {
        "benchmark": "halfcheetah",
        "metric": "mean_cumulative_episode_return",
        "definition": "FT = (1/T) * sum_i (P_i - b_i)",
        "reference_reward_source": "supplied SAC-N reference table",
        "model_name": args.model_name,
        "scenario": args.scenario,
        "train_seed": int(args.train_seed),
        "test_seed": int(args.test_seed),
        "episodes_per_task": int(args.episodes),
        "tasks": [int(task_id) for task_id in task_ids],
        "task_names": [get_task_name(task_id) for task_id in task_ids],
        "num_tasks_T": len(task_ids),
        "checkpoint_selection": "task-corresponding checkpoint at each training stage",
        "forward_transfer": float(np.mean(differences)),
        "variance_of_task_differences": float(np.var(differences)),
        "sum_of_task_differences": float(np.sum(differences)),
    }
    print("\nOverall Forward Transfer")
    print(
        f"FT={summary['forward_transfer']:.6f}, "
        f"variance={summary['variance_of_task_differences']:.6f}, "
        f"T={summary['num_tasks_T']}"
    )

    output = args.output
    if output is None:
        output = THIS_DIR / "analysis" / (
            f"forward_transfer__{args.model_name}__{args.scenario}__"
            f"train{args.train_seed}__test{args.test_seed}.csv"
        )
    output = resolve_model_root(output)
    write_outputs(rows, summary, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
