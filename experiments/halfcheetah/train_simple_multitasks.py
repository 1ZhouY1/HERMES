#!/usr/bin/env python3
"""Launch independent run_sac.py processes for multiple Simple SAC tasks.

This launcher intentionally delegates the actual SAC implementation to
run_sac.py, so its TensorBoard logging, terminal logging, evaluation, and
checkpoint saving remain exactly the same as a normal run_sac.py experiment.

Example:
    python train_simple_multitasks.py \
        --tasks 0 1 2 \
        --seed 42 \
        --total-timesteps 1000000 \
        --save-dir agents
"""

import argparse
import subprocess
import sys
import threading
from pathlib import Path

from tasks import get_task_name, tasks


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one independent run_sac.py Simple SAC process per selected "
            "HalfCheetah task."
        )
    )
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="+",
        required=True,
        help="Task ids to train simultaneously, for example: 0 1 2",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--save-dir", type=Path, default=Path("agents"))
    parser.add_argument("--exp-name", type=str, default="run_sac")
    parser.add_argument("--run-sac", type=Path, default=Path("run_sac.py"))
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--random-actions-end", type=int, default=10_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-lr", type=float, default=1e-3)
    parser.add_argument("--q-lr", type=float, default=1e-3)
    parser.add_argument("--policy-frequency", type=int, default=2)
    parser.add_argument("--target-network-frequency", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--autotune", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--cuda", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--torch-deterministic",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    return parser.parse_args()


def command_for_task(args, task_id: int):
    run_sac = args.run_sac.expanduser()
    if not run_sac.is_absolute():
        run_sac = (Path.cwd() / run_sac).resolve()

    command = [
        sys.executable,
        str(run_sac),
        "--model-type",
        "simple",
        "--task-id",
        str(task_id),
        "--seed",
        str(args.seed),
        "--total-timesteps",
        str(args.total_timesteps),
        "--save-dir",
        str(args.save_dir),
        "--exp-name",
        str(args.exp_name),
        "--buffer-size",
        str(args.buffer_size),
        "--batch-size",
        str(args.batch_size),
        "--learning-starts",
        str(args.learning_starts),
        "--random-actions-end",
        str(args.random_actions_end),
        "--gamma",
        str(args.gamma),
        "--tau",
        str(args.tau),
        "--policy-lr",
        str(args.policy_lr),
        "--q-lr",
        str(args.q_lr),
        "--policy-frequency",
        str(args.policy_frequency),
        "--target-network-frequency",
        str(args.target_network_frequency),
        "--alpha",
        str(args.alpha),
    ]
    if args.autotune:
        command.append("--autotune")
    else:
        command.append("--no-autotune")
    if args.cuda:
        command.append("--cuda")
    else:
        command.append("--no-cuda")
    if args.torch_deterministic:
        command.append("--torch-deterministic")
    else:
        command.append("--no-torch-deterministic")
    return command


def stream_output(task_id: int, process: subprocess.Popen) -> None:
    prefix = f"[task {task_id} {get_task_name(task_id)}]"
    assert process.stdout is not None
    for line in process.stdout:
        print(f"{prefix} {line}", end="", flush=True)


def main() -> int:
    args = parse_args()
    invalid = [task_id for task_id in args.tasks if task_id < 0 or task_id >= len(tasks)]
    if invalid:
        raise ValueError(f"Invalid task ids {invalid}; valid ids are 0-{len(tasks) - 1}.")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    run_sac = args.run_sac.expanduser()
    if not run_sac.is_absolute():
        run_sac = (Path.cwd() / run_sac).resolve()
    if not run_sac.exists():
        raise FileNotFoundError(f"Cannot find run_sac.py: {run_sac}")

    print("*** Benchmark: HalfCheetah")
    print(f"*** Tasks: {[(task_id, get_task_name(task_id)) for task_id in args.tasks]}")
    print(f"*** Seed: {args.seed}")
    print(f"*** Timesteps per task: {args.total_timesteps}")
    print(f"*** Save dir: {args.save_dir.resolve()}")
    print("*** Extra state sampling: disabled")
    print("*** TensorBoard/logging: delegated to run_sac.py (runs/<run_name>)")

    processes = []
    output_threads = []
    for task_id in args.tasks:
        command = command_for_task(args, task_id)
        print(f"*** Launch task {task_id}: {' '.join(command)}")
        process = subprocess.Popen(
            command,
            cwd=str(run_sac.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes.append((task_id, process))
        thread = threading.Thread(
            target=stream_output,
            args=(task_id, process),
            daemon=True,
        )
        thread.start()
        output_threads.append(thread)

    failed = []
    for task_id, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failed.append((task_id, return_code))
    for thread in output_threads:
        thread.join()

    if failed:
        print(f"*** Failed tasks: {failed}")
        return 1
    print("*** All selected Simple SAC tasks finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
