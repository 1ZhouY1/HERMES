#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

from tasks import tasks


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train SimpleAgent SAC on HalfCheetah tasks for 20k steps, save "
            "state traces, then compute temporal-dimension Wasserstein distances."
        )
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tasks", type=int, nargs="*", default=list(range(len(tasks))))
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--save-dir", type=Path, default=Path("agents/simple_sac_20k"))
    parser.add_argument("--state-dir", type=Path, default=Path("simple_sac_20k_states"))
    parser.add_argument("--output-csv", type=Path, default=Path("analysis/halfcheetah_temporal_dim_wasserstein.csv"))
    parser.add_argument("--output-pt", type=Path, default=Path("analysis/halfcheetah_temporal_dim_wasserstein.pt"))
    parser.add_argument("--output-json", type=Path, default=Path("analysis/halfcheetah_temporal_dim_wasserstein_summary.json"))
    parser.add_argument("--skip-train", default=False, action="store_true")
    parser.add_argument("--skip-compute", default=False, action="store_true")
    return parser.parse_args()


def run(cmd):
    print("\n" + " ".join(str(part) for part in cmd))
    result = subprocess.run([str(part) for part in cmd])
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> int:
    args = parse_args()

    if not args.skip_train:
        run(
            [
                sys.executable,
                "train_simple_sac_collect_states.py",
                "--seed",
                args.seed,
                "--total-timesteps",
                args.total_timesteps,
                "--save-dir",
                args.save_dir,
                "--state-dir",
                args.state_dir,
                "--tasks",
                *args.tasks,
            ]
        )

    if not args.skip_compute:
        run(
            [
                sys.executable,
                "compute_temporal_dim_wasserstein.py",
                "--state-dir",
                args.state_dir,
                "--tasks",
                *args.tasks,
                "--expected-states",
                args.total_timesteps,
                "--output-csv",
                args.output_csv,
                "--output-pt",
                args.output_pt,
                "--output-json",
                args.output_json,
            ]
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
