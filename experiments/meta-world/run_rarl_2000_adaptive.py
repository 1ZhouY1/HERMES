import argparse
import ast
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def load_tasks() -> List[str]:
    """
    Read task names without importing tasks.py, so dry-run works even before
    the Meta-World runtime is activated.
    """
    with open("tasks.py", "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename="tasks.py")

    single_tasks = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "single_tasks":
                    single_tasks = ast.literal_eval(node.value)

    if single_tasks is None:
        raise RuntimeError("Could not find single_tasks in tasks.py.")
    return single_tasks + single_tasks


def load_single_tasks() -> List[str]:
    return load_tasks()[:10]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Launch the sequential rarl-2000 adaptive-transfer experiment. "
            "Task 0 can use plain rarl-2000; later tasks use previous units, "
            "state Wasserstein matching, and optional policy-prefix reuse."
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="*",
        default=list(range(1, 10)),
        help="Task ids to launch. Default: 1..9, assuming task 0 rarl-2000 is already trained.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="agents",
        help="Directory passed to SAC_hybridRAGRL.py for saving agents.",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="SAC_hybridRAGRL",
        help="Experiment name used in saved directory names.",
    )
    parser.add_argument(
        "--task0-model-type",
        type=str,
        default="rarl-2000",
        help="Model type used for task 0 previous unit lookup.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="rarl-2000-adaptive",
        help="Model type used for tasks > 0.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save and overwrite the latest checkpoint every N SAC training steps. 0 disables periodic saving.",
    )
    parser.add_argument(
        "--save-memory",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Forward memory saving to SAC_hybridRAGRL.py. Enabled by default.",
    )
    parser.add_argument(
        "--adaptive-state-dir",
        type=str,
        default=None,
        help=(
            "Shared task-state cache directory or filename template. "
            "Forwarded to SAC_hybridRAGRL.py as --adaptive-state-dir."
        ),
    )
    parser.add_argument(
        "--allow-missing-prev",
        default=False,
        action="store_true",
        help="Only print a warning when a previous unit directory is missing.",
    )
    parser.add_argument(
        "--no-run",
        default=False,
        action="store_true",
        help="Only print commands without launching them.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help=(
            "Optional arguments forwarded to SAC_hybridRAGRL.py. "
            "Put them after '--', for example: -- --total-timesteps 100000"
        ),
    )
    return parser.parse_args()


def validate_task_ids(task_ids: List[int]) -> None:
    tasks = load_tasks()
    max_task_id = len(tasks) - 1
    bad_ids = [task_id for task_id in task_ids if task_id < 0 or task_id > max_task_id]
    if bad_ids:
        raise ValueError(f"Invalid task ids {bad_ids}. Valid range is 0..{max_task_id}.")


def model_dir_name(task_id: int, model_type: str, exp_name: str, seed: int) -> str:
    return f"task_{task_id}__{model_type}__{exp_name}__{seed}"


def previous_unit_dir(save_dir: Path, task_id: int, args) -> Path:
    if task_id == 0:
        primary = save_dir / model_dir_name(task_id, args.task0_model_type, args.exp_name, args.seed)
        fallback = save_dir / model_dir_name(task_id, args.algorithm, args.exp_name, args.seed)
        if primary.exists() or not fallback.exists():
            return primary
        return fallback
    return save_dir / model_dir_name(task_id, args.algorithm, args.exp_name, args.seed)


def resolve_prev_units(target_task_id: int, args) -> List[Path]:
    save_dir = Path(args.save_dir)
    prev_units = []
    missing = []
    for prev_task_id in range(target_task_id):
        prev_dir = previous_unit_dir(save_dir, prev_task_id, args)
        if args.no_run:
            prev_units.append(prev_dir)
            continue
        if (prev_dir / "model.pt").exists():
            prev_units.append(prev_dir)
        else:
            missing.append(prev_dir)

    if missing and not args.allow_missing_prev:
        formatted = "\n".join(f"    {path}" for path in missing)
        raise FileNotFoundError(
            f"Missing previous units for target task {target_task_id}:\n{formatted}"
        )
    if missing:
        for path in missing:
            print(f"*** Warning: missing previous unit: {path}")
    return prev_units


def build_command(args, task_id: int) -> List[str]:
    if task_id == 0:
        model_type = args.task0_model_type
        prev_units = []
    else:
        model_type = args.algorithm
        prev_units = resolve_prev_units(task_id, args)

    cmd = [
        sys.executable,
        "SAC_hybridRAGRL.py",
        f"--model-type={model_type}",
        f"--task-id={task_id}",
        f"--seed={args.seed}",
        f"--save-dir={args.save_dir}",
    ]
    if args.save_every > 0:
        cmd.append(f"--save-every={args.save_every}")
    if args.save_memory:
        cmd.append("--save-memory")
    if args.adaptive_state_dir is not None:
        cmd.append(f"--adaptive-state-dir={args.adaptive_state_dir}")
    if prev_units:
        cmd.append("--prev-units")
        cmd.extend(str(prev_unit) for prev_unit in prev_units)

    extra_args = args.extra_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    cmd.extend(extra_args)
    return cmd


def format_command(cmd: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def launch_all(args, task_ids: List[int]) -> int:
    tasks = load_tasks()
    for task_id in task_ids:
        cmd = build_command(args, task_id)
        print(f"[task {task_id}] {format_command(cmd)}")
        if args.no_run:
            continue

        print(f"*** Launching task {task_id}: {tasks[task_id]}")
        start = time.time()
        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"*** Task {task_id} failed with code {ret}.")
            return ret
        print(f"*** Task {task_id} finished in {(time.time() - start) / 60.0:.1f} minutes.")
    return 0


if __name__ == "__main__":
    args = parse_args()
    task_ids = args.tasks if args.tasks is not None else list(range(1, len(load_single_tasks())))
    validate_task_ids(task_ids)
    raise SystemExit(launch_all(args, task_ids))
