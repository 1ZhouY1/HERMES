import argparse
import ast
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List


HYBRID_ALGORITHMS = [
    "hybridrarl",
    "hybridrarl-wo-gate",
    "hybrid_add_norm",
    "hybrid-encoder",
    "hybrid-context",
    "hybrid-raw-encoder",
    "hybrid-raw-context",
    "hybrid-raw",
    "hybrid",
    "gate",
]


def load_tasks() -> List[str]:
    """
    Read the task list without importing tasks.py, so this launcher can print
    dry-run commands even before the Meta-World environment is activated.
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
            "Launch independent Hybrid RAGRL Meta-World tasks in parallel. "
            "Each task owns its own LongTermMemory."
        )
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=HYBRID_ALGORITHMS,
        default="hybridrarl",
        help="Hybrid model type passed to SAC_hybridRAGRL.py.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="*",
        default=None,
        help="Task ids to launch. Default: 0..9, the ten unique Meta-World tasks.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=3,
        help="Maximum number of simultaneously running tasks. Default: 3.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="agents",
        help="Directory passed to SAC_hybridRAGRL.py for saving agents.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save and overwrite the latest checkpoint every N environment steps. 0 disables periodic saving.",
    )
    parser.add_argument(
        "--save-memory",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Forward memory saving to SAC_hybridRAGRL.py. Enabled by default for hybrid experiments.",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="SAC_hybridRAGRL",
        help="Experiment name used by SAC_hybridRAGRL.py when constructing saved model directories.",
    )
    parser.add_argument(
        "--hybridrarl-state-dir",
        type=str,
        default="hybridrarl_state_cache",
        help=(
            "Shared state-cache directory for hybridrarl Wasserstein matching. "
            "Existing task files are reused; missing task files are sampled by SAC_hybridRAGRL.py."
        ),
    )
    parser.add_argument(
        "--allow-missing-prev",
        default=False,
        action="store_true",
        help="For hybridrarl, warn instead of failing when an expected previous model directory is missing.",
    )
    parser.add_argument(
        "--independent-hybridrarl",
        default=False,
        action="store_true",
        help=(
            "Run hybridrarl tasks independently and in parallel without --prev-units. "
            "Use this for the first stage where all task-specific models are trained before manual reuse selection."
        ),
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
    return save_dir / model_dir_name(
        task_id=task_id,
        model_type=args.algorithm,
        exp_name=args.exp_name,
        seed=args.seed,
    )


def resolve_hybridrarl_prev_units(target_task_id: int, args) -> List[Path]:
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
            f"Missing previous hybridrarl units for target task {target_task_id}:\n{formatted}\n"
            "Run earlier tasks first, include them in --tasks, or pass --allow-missing-prev."
        )

    for path in missing:
        print(f"*** Warning: missing previous hybridrarl unit: {path}")
    return prev_units


def normalize_extra_args(extra_args: List[str]) -> List[str]:
    if extra_args and extra_args[0] == "--":
        return extra_args[1:]
    return extra_args


def build_command(args, task_id: int) -> List[str]:
    cmd = [
        sys.executable,
        "SAC_hybridRAGRL.py",
        f"--model-type={args.algorithm}",
        f"--task-id={task_id}",
        f"--seed={args.seed}",
        f"--save-dir={args.save_dir}",
        f"--exp-name={args.exp_name}",
    ]
    if args.save_every > 0:
        cmd.append(f"--save-every={args.save_every}")
    if args.save_memory:
        cmd.append("--save-memory")

    if args.algorithm == "hybridrarl":
        cmd.append(f"--hybridrarl-state-dir={args.hybridrarl_state_dir}")
        if not args.independent_hybridrarl:
            prev_units = resolve_hybridrarl_prev_units(task_id, args)
            if prev_units:
                cmd.append("--prev-units")
                cmd.extend(str(prev_unit) for prev_unit in prev_units)

    cmd.extend(normalize_extra_args(args.extra_args))
    return cmd


def format_command(cmd: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def launch_all(args, task_ids: List[int]) -> int:
    if args.algorithm == "hybridrarl" and not args.independent_hybridrarl:
        return launch_hybridrarl_sequential(args, task_ids)

    tasks = load_tasks()
    commands = [(task_id, build_command(args, task_id)) for task_id in task_ids]

    for task_id, cmd in commands:
        print(f"[task {task_id}] {format_command(cmd)}")

    if args.no_run:
        return 0

    max_parallel = len(commands) if args.max_parallel <= 0 else args.max_parallel
    running = []
    pending = list(commands)
    failed = []

    while pending or running:
        while pending and len(running) < max_parallel:
            task_id, cmd = pending.pop(0)
            print(f"*** Launching task {task_id}: {tasks[task_id]}")
            proc = subprocess.Popen(cmd)
            running.append((task_id, proc))

        still_running = []
        for task_id, proc in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((task_id, proc))
            elif ret != 0:
                failed.append((task_id, ret))
                print(f"*** Task {task_id} failed with code {ret}.")
            else:
                print(f"*** Task {task_id} finished.")

        running = still_running
        if pending or running:
            time.sleep(2)

    if failed:
        print("*** Failed tasks:")
        for task_id, ret in failed:
            print(f"    task {task_id}: return code {ret}")
        return 1
    return 0


def launch_hybridrarl_sequential(args, task_ids: List[int]) -> int:
    tasks = load_tasks()
    for task_id in task_ids:
        cmd = build_command(args, task_id)
        print(f"[task {task_id}] {format_command(cmd)}")
        if args.no_run:
            continue

        print(f"*** Launching hybridrarl task {task_id}: {tasks[task_id]}")
        start = time.time()
        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"*** Task {task_id} failed with code {ret}.")
            return ret
        print(f"*** Task {task_id} finished in {(time.time() - start) / 60.0:.1f} minutes.")
    return 0


if __name__ == "__main__":
    args = parse_args()
    task_ids = args.tasks if args.tasks is not None else list(range(len(load_single_tasks())))
    validate_task_ids(task_ids)
    raise SystemExit(launch_all(args, task_ids))
