#!/usr/bin/env python3
"""
Evaluate the final Meta-World CompoNet checkpoint on task0-task9.

This script is intentionally separate from experiments/test_rl_model.py because
the target protocol is different: load one source model (normally task 9, the
last learned CompoNet model) and evaluate that same actor on multiple
Meta-World tasks.

Typical usage from experiments/meta-world:

  python eval_final_componet_all_tasks.py \
      --train-seed 42 \
      --test-seed 0 \
      --episodes 50 \
      --final-task-id 9 \
      --tasks 0 1 2 3 4 5 6 7 8 9

The loader reconstructs CompoNet's previous units by default:
  task0: simple checkpoint
  task1..final_task_id-1: CompoNet checkpoints

If a checkpoint naming convention differs from the defaults, use
--final-model-dir and/or --prev-dirs to provide exact paths.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class TakeFirst:
    def __call__(self, x):
        return x[0] if isinstance(x, (tuple, list)) else x


class LoadedCompoNetAgent(nn.Module):
    """Wrap separately saved CompoNet mean/log-std modules as a SAC actor."""

    def __init__(self, net_mean: nn.Module, net_logstd: nn.Module):
        super().__init__()
        self.net_mean = net_mean
        self.net_logstd = net_logstd

    def forward(self, x, **unused_kwargs):
        mean_out = self.net_mean(x)
        mean = mean_out[0] if isinstance(mean_out, (tuple, list)) else mean_out
        logstd = self.net_logstd(x)
        return mean, logstd


def mlp_net(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(int(input_dim), 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, int(output_dim)),
    )


def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def ordered_unique(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def resolve_path(path: Path, exp_dir: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    return (exp_dir / path).resolve()


def model_dir_name(task_id: int, model_name: str, exp_name: str, train_seed: int) -> str:
    return f"task_{int(task_id)}__{model_name}__{exp_name}__{int(train_seed)}"


def has_simple_actor(model_dir: Path) -> bool:
    return (model_dir / "model.pt").exists()


def has_componet_actor(model_dir: Path) -> bool:
    return (model_dir / "net_mean.pt").exists() and (model_dir / "net_logstd.pt").exists()


def has_expected_actor(model_dir: Path, model_name: str) -> bool:
    if model_name == "simple":
        return has_simple_actor(model_dir)
    if model_name == "componet":
        return has_componet_actor(model_dir)
    raise ValueError(f"Unsupported model_name for this script: {model_name}")


def candidate_dirs(
    model_root: Path,
    task_id: int,
    model_name: str,
    exp_names: Sequence[str],
    train_seed: int,
) -> List[Path]:
    return [
        model_root / model_dir_name(task_id, model_name, exp_name, train_seed)
        for exp_name in exp_names
    ]


def resolve_model_dir(
    model_root: Path,
    task_id: int,
    model_name: str,
    exp_names: Sequence[str],
    train_seed: int,
    recursive_search: bool,
) -> Path:
    candidates = candidate_dirs(model_root, task_id, model_name, exp_names, train_seed)
    for candidate in candidates:
        if has_expected_actor(candidate, model_name):
            return candidate.resolve()

    if recursive_search and model_root.exists():
        target_names = {candidate.name for candidate in candidates}
        patterns = ["model.pt"] if model_name == "simple" else ["net_mean.pt", "net_logstd.pt"]
        for pattern in patterns:
            for model_file in model_root.rglob(pattern):
                parent = model_file.parent
                if parent.name in target_names and has_expected_actor(parent, model_name):
                    return parent.resolve()

    tried = "\n".join(str(candidate) for candidate in candidates)
    expected = "model.pt" if model_name == "simple" else "net_mean.pt + net_logstd.pt"
    raise FileNotFoundError(
        f"Cannot find {model_name} checkpoint for task {task_id} "
        f"(expected {expected}). Tried:\n{tried}"
    )


def import_metaworld_tasks(exp_dir: Path):
    if str(exp_dir) not in sys.path:
        sys.path.insert(0, str(exp_dir))
    return importlib.import_module("tasks")


def import_componet(exp_dir: Path):
    if str(exp_dir) not in sys.path:
        sys.path.insert(0, str(exp_dir))
    try:
        from componet import CompoNet, FirstModuleWrapper

        return CompoNet, FirstModuleWrapper
    except ImportError:
        # In this repository the shared CompoNet implementation may live under
        # experiments/halfcheetah/componet, while Meta-World checkpoints still
        # serialize references to module name "componet".
        halfcheetah_dir = exp_dir.parent / "halfcheetah"
        if str(halfcheetah_dir) not in sys.path:
            sys.path.insert(0, str(halfcheetah_dir))
        from componet import CompoNet, FirstModuleWrapper

        return CompoNet, FirstModuleWrapper


def resolve_prev_dirs(args, exp_dir: Path) -> List[Path]:
    if args.prev_dirs:
        prev_dirs = [resolve_path(path, exp_dir) for path in args.prev_dirs]
        if len(prev_dirs) != int(args.final_task_id):
            raise ValueError(
                f"--prev-dirs should contain exactly {args.final_task_id} paths "
                f"for final task {args.final_task_id}, got {len(prev_dirs)}."
            )
        return prev_dirs

    model_root = resolve_path(args.model_root, exp_dir)
    simple_exp_names = ordered_unique(args.simple_exp_names)
    componet_exp_names = ordered_unique(args.componet_exp_names)

    prev_dirs = []
    for prev_task_id in range(int(args.final_task_id)):
        if prev_task_id == 0:
            prev_dirs.append(
                resolve_model_dir(
                    model_root=model_root,
                    task_id=prev_task_id,
                    model_name="simple",
                    exp_names=simple_exp_names,
                    train_seed=args.train_seed,
                    recursive_search=args.recursive_search,
                )
            )
        else:
            prev_dirs.append(
                resolve_model_dir(
                    model_root=model_root,
                    task_id=prev_task_id,
                    model_name="componet",
                    exp_names=componet_exp_names,
                    train_seed=args.train_seed,
                    recursive_search=args.recursive_search,
                )
            )
    return prev_dirs


def load_final_componet(
    args,
    exp_dir: Path,
    model_dir: Path,
    prev_dirs: Sequence[Path],
    obs_dim: int,
    act_dim: int,
    device,
) -> nn.Module:
    CompoNet, FirstModuleWrapper = import_componet(exp_dir)

    if not prev_dirs:
        raise ValueError("CompoNet evaluation needs at least the task0 simple checkpoint as a previous unit.")
    if not has_simple_actor(prev_dirs[0]):
        raise FileNotFoundError(f"First previous unit must be a simple checkpoint with model.pt: {prev_dirs[0]}")
    for prev_dir in prev_dirs[1:]:
        if not has_componet_actor(prev_dir):
            raise FileNotFoundError(f"Previous CompoNet unit is missing net_mean.pt/net_logstd.pt: {prev_dir}")

    first_model = safe_torch_load(prev_dirs[0] / "model.pt", map_location=device)
    prev_units = [
        FirstModuleWrapper(
            model=first_model,
            ret_probs=False,
            transform_output=TakeFirst(),
        )
    ]
    for prev_dir in prev_dirs[1:]:
        prev_units.append(safe_torch_load(prev_dir / "net_mean.pt", map_location=device))

    net_mean = CompoNet(
        previous_units=prev_units,
        input_dim=int(obs_dim),
        hidden_dim=int(args.hidden_dim),
        out_dim=int(act_dim),
        internal_policy=mlp_net(int(obs_dim) + int(args.hidden_dim), int(act_dim)),
        ret_probs=False,
        device=device,
    )
    saved_net_mean = safe_torch_load(model_dir / "net_mean.pt", map_location=device)
    load_result = net_mean.load_state_dict(saved_net_mean.state_dict(), strict=False)
    if load_result.missing_keys:
        print(f"*** Missing keys while loading CompoNet mean: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"*** Unexpected keys while loading CompoNet mean: {load_result.unexpected_keys}")

    net_logstd = safe_torch_load(model_dir / "net_logstd.pt", map_location=device)
    model = LoadedCompoNetAgent(net_mean=net_mean, net_logstd=net_logstd).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def reset_env(env, seed: Optional[int]):
    try:
        out = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        if seed is not None and hasattr(env, "seed"):
            env.seed(seed)
        out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated or truncated), info
    obs, reward, done, info = out
    return obs, float(reward), bool(done), info


def infer_episode_limit(env, fallback: int) -> int:
    spec = getattr(env, "spec", None)
    max_steps = getattr(spec, "max_episode_steps", None) if spec is not None else None
    if max_steps is not None:
        return int(max_steps)
    max_path_length = getattr(env, "max_path_length", None)
    if max_path_length is not None:
        return int(max_path_length)
    return int(fallback)


def first_two_outputs(out) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(out, (tuple, list)):
        mean = out[0]
        logstd = out[1] if len(out) > 1 and torch.is_tensor(out[1]) else None
        return mean, logstd
    return out, None


@torch.no_grad()
def select_action(
    model: nn.Module,
    obs: np.ndarray,
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
    device,
    deterministic: bool,
) -> np.ndarray:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).reshape(1, -1)
    mean, raw_logstd = first_two_outputs(model(obs_t))

    if deterministic or raw_logstd is None:
        squashed = torch.tanh(mean)
    else:
        logstd = torch.tanh(raw_logstd)
        logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (logstd + 1)
        normal = torch.distributions.Normal(mean, logstd.exp())
        squashed = torch.tanh(normal.sample())

    action = squashed * action_scale + action_bias
    return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


def close_env(env) -> None:
    close_fn = getattr(env, "close", None)
    if callable(close_fn):
        close_fn()


def env_dims(tasks_module, task_id: int) -> Tuple[int, int]:
    env = tasks_module.get_task(task_id)
    try:
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        return obs_dim, act_dim
    finally:
        close_env(env)


def evaluate_on_task(
    tasks_module,
    model: nn.Module,
    task_id: int,
    task_name: str,
    source_model_task_id: int,
    episodes: int,
    test_seed: int,
    fallback_max_steps: int,
    max_steps_override: int,
    deterministic: bool,
    device,
) -> Dict:
    env = tasks_module.get_task(task_id)
    try:
        action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device)
        action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=device)
        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        max_steps = int(max_steps_override) if int(max_steps_override) > 0 else infer_episode_limit(env, fallback_max_steps)

        returns: List[float] = []
        successes: List[float] = []
        lengths: List[int] = []
        for episode in range(int(episodes)):
            obs = reset_env(env, int(test_seed) + episode)
            episode_return = 0.0
            episode_success = 0.0
            length = 0
            for _ in range(max_steps):
                action = select_action(
                    model=model,
                    obs=np.asarray(obs, dtype=np.float32),
                    action_scale=action_scale,
                    action_bias=action_bias,
                    device=device,
                    deterministic=deterministic,
                )
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, done, info = step_env(env, action)
                episode_return += reward
                length += 1
                if isinstance(info, dict) and "success" in info:
                    episode_success = max(episode_success, float(info["success"]))
                if done:
                    break
            returns.append(float(episode_return))
            successes.append(float(episode_success))
            lengths.append(int(length))

        return {
            "source_model_task_id": int(source_model_task_id),
            "eval_task_id": int(task_id),
            "task_name": task_name,
            "episodes": int(episodes),
            "mean_return": float(np.mean(returns)) if returns else float("nan"),
            "std_return": float(np.std(returns)) if returns else float("nan"),
            "mean_success": float(np.mean(successes)) if successes else float("nan"),
            "std_success": float(np.std(successes)) if successes else float("nan"),
            "mean_length": float(np.mean(lengths)) if lengths else float("nan"),
            "returns": returns,
            "successes": successes,
            "lengths": lengths,
        }
    finally:
        close_env(env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the final Meta-World CompoNet checkpoint on task0-task9."
    )
    parser.add_argument("--model-root", type=Path, default=Path("agents"), help="Directory containing saved models.")
    parser.add_argument("--final-task-id", type=int, default=9, help="Which CompoNet task checkpoint to load.")
    parser.add_argument("--tasks", type=int, nargs="*", default=None, help="Evaluation task ids. Default: 0..9.")
    parser.add_argument("--train-seed", type=int, required=True, help="Seed suffix of the saved training checkpoints.")
    parser.add_argument("--test-seed", type=int, default=0, help="Base seed for evaluation episodes.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes per task.")
    parser.add_argument(
        "--componet-exp-names",
        nargs="*",
        default=["SAC", "run_sac"],
        help="Experiment-name candidates for CompoNet directories.",
    )
    parser.add_argument(
        "--simple-exp-names",
        nargs="*",
        default=["SAC", "run_sac", "SAC_hybridRAGRL"],
        help="Experiment-name candidates for the task0 simple directory.",
    )
    parser.add_argument(
        "--final-model-dir",
        type=Path,
        default=None,
        help="Exact final CompoNet model directory. Overrides --model-root lookup for the final task.",
    )
    parser.add_argument(
        "--prev-dirs",
        type=Path,
        nargs="*",
        default=None,
        help="Exact previous-unit directories, ordered from task0 to task(final_task_id-1).",
    )
    parser.add_argument(
        "--recursive-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search recursively under --model-root if exact candidate directories are not present.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256, help="CompoNet hidden dimension used during training.")
    parser.add_argument(
        "--fallback-max-steps",
        type=int,
        default=500,
        help="Episode length if the env does not expose max_episode_steps/max_path_length.",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="Override episode length when > 0.")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use tanh(mean) actions. Disable to sample from SAC Gaussian policy.",
    )
    parser.add_argument(
        "--cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA when available.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path. JSON sidecar files are written next to it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp_dir = Path(__file__).resolve().parent
    tasks_module = import_metaworld_tasks(exp_dir)
    task_names = getattr(tasks_module, "tasks")

    task_ids = args.tasks if args.tasks is not None else list(range(min(10, len(task_names))))
    invalid_tasks = [task_id for task_id in task_ids if task_id < 0 or task_id >= len(task_names)]
    if invalid_tasks:
        raise ValueError(f"Invalid task ids: {invalid_tasks}; available range is 0..{len(task_names) - 1}.")

    set_seed(args.test_seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    model_root = resolve_path(args.model_root, exp_dir)

    if args.final_model_dir is not None:
        final_model_dir = resolve_path(args.final_model_dir, exp_dir)
        if not has_componet_actor(final_model_dir):
            raise FileNotFoundError(f"Final model dir is missing net_mean.pt/net_logstd.pt: {final_model_dir}")
    else:
        final_model_dir = resolve_model_dir(
            model_root=model_root,
            task_id=args.final_task_id,
            model_name="componet",
            exp_names=ordered_unique(args.componet_exp_names),
            train_seed=args.train_seed,
            recursive_search=args.recursive_search,
        )

    prev_dirs = resolve_prev_dirs(args, exp_dir)
    obs_dim, act_dim = env_dims(tasks_module, task_ids[0])
    model = load_final_componet(
        args=args,
        exp_dir=exp_dir,
        model_dir=final_model_dir,
        prev_dirs=prev_dirs,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
    )

    output = args.output
    if output is None:
        output = (
            exp_dir
            / "analysis"
            / "componet_final_eval"
            / f"task_{args.final_task_id}__componet_on_task0_9__train{args.train_seed}__test{args.test_seed}.csv"
        )
    output = resolve_path(output, exp_dir)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"*** Final CompoNet model: {final_model_dir}")
    print("*** Previous units:")
    for index, prev_dir in enumerate(prev_dirs):
        print(f"    {index}: {prev_dir}")
    print(f"*** Device: {device}")
    print(f"*** Evaluation tasks: {task_ids}")
    print(f"*** Episodes per task: {args.episodes}")

    records: List[Dict] = []
    for task_id in task_ids:
        task_name = tasks_module.get_task_name(task_id) if hasattr(tasks_module, "get_task_name") else task_names[task_id]
        record = evaluate_on_task(
            tasks_module=tasks_module,
            model=model,
            task_id=task_id,
            task_name=task_name,
            source_model_task_id=args.final_task_id,
            episodes=args.episodes,
            test_seed=args.test_seed,
            fallback_max_steps=args.fallback_max_steps,
            max_steps_override=args.max_steps,
            deterministic=args.deterministic,
            device=device,
        )
        records.append(record)
        print(
            f"task={task_id:02d} {task_name:24s} "
            f"return={record['mean_return']:.3f}±{record['std_return']:.3f} "
            f"success={record['mean_success']:.3f}±{record['std_success']:.3f}"
        )

    csv_fields = [
        "source_model_task_id",
        "eval_task_id",
        "task_name",
        "episodes",
        "mean_return",
        "std_return",
        "mean_success",
        "std_success",
        "mean_length",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in csv_fields})

    details_path = output.with_suffix(".details.json")
    summary_path = output.with_suffix(".summary.json")
    details = {
        "final_model_dir": str(final_model_dir),
        "prev_dirs": [str(path) for path in prev_dirs],
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "records": records,
    }
    details_path.write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "source_model_task_id": int(args.final_task_id),
        "num_eval_tasks": len(records),
        "average_success": float(np.mean([record["mean_success"] for record in records])) if records else float("nan"),
        "average_return": float(np.mean([record["mean_return"] for record in records])) if records else float("nan"),
        "csv": str(output),
        "details_json": str(details_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"*** Saved CSV: {output}")
    print(f"*** Saved details JSON: {details_path}")
    print(f"*** Saved summary JSON: {summary_path}")
    print(
        f"*** Average over {len(records)} tasks: "
        f"return={summary['average_return']:.3f}, success={summary['average_success']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
