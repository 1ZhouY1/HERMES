#!/usr/bin/env python3
"""Standalone HalfCheetah evaluator.

This file intentionally does not import the Meta-World evaluator. HalfCheetah
performance is measured with cumulative episode reward, not success rate.

The repeated entries in each scenario are removed before evaluation:
simple evaluates all 11 tasks; scenario-based methods evaluate the unique
tasks in the selected scenario in their original order.
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from componet import FirstModuleWrapper
from models import (
    HybridRARLAgent,
    HybridRARLPrevUnitsAgent,
    ReservoirTrajectoryMemory,
)
from models.hybridrarl_impl import build_policy_trunk
from tasks import SCENARIOS, get_task, get_task_name, tasks


THIS_DIR = Path(__file__).resolve().parent
MODEL_ALIASES = {
    "finetune_1": "finetune-1",
    "finetune1": "finetune-1",
    "finetune_n": "finetune-n",
    "finetunen": "finetune-n",
    "sac_n": "sac-n",
    "sacn": "sac-n",
}


def normalize_model_name(name: str) -> str:
    name = str(name).strip().lower()
    return MODEL_ALIASES.get(name, name)


def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_test_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def unique_scenario_task_ids(scenario: str) -> List[int]:
    seen = set()
    result = []
    for task_name in SCENARIOS[scenario]:
        if task_name in seen:
            continue
        seen.add(task_name)
        result.append(tasks.index(task_name))
    return result


def evaluation_task_ids(model_name: str, scenario: str) -> List[int]:
    if model_name == "simple":
        return list(range(len(tasks)))
    if not scenario:
        raise ValueError(f"--scenario is required for `{model_name}`.")
    return unique_scenario_task_ids(scenario)


def model_plan(model_name: str) -> Tuple[List[str], List[str]]:
    if model_name in {"simple", "sac-n", "baseline"}:
        return ["simple"], ["SAC", "run_sac", "SAC_hybridRAGRL", "sac_20k"]
    if model_name in {"finetune", "finetune-1", "finetune-n"}:
        return ["finetune"], ["SAC", "run_sac"]
    if model_name == "componet":
        return ["componet"], ["SAC", "run_sac"]
    if model_name == "packnet":
        return ["packnet"], ["SAC", "run_sac"]
    if model_name == "prognet":
        return ["prognet"], ["SAC", "run_sac"]
    if model_name == "hybridrarl":
        return [
            "hybridrarl-prev-units",
            "hybridrarl-simple-prev-units",
            "hybridrarl",
            "hybrid",
        ], [
            "SAC_hybridrarl_prev_units",
            "SAC_hybridRAGRL",
            "SAC",
            "run_sac",
        ]
    if model_name == "hybrid":
        return ["hybrid"], ["SAC_hybridRAGRL", "SAC", "run_sac"]
    raise ValueError(f"Unsupported model: {model_name}")


def model_directory_name(task_id: int, model_name: str, exp_name: str, seed: int) -> str:
    return f"task_{int(task_id)}__{model_name}__{exp_name}__{int(seed)}"


def has_checkpoint(path: Path, model_name: str) -> bool:
    if model_name in {"simple", "finetune"}:
        return (path / "model.pt").exists()
    if model_name == "componet":
        return (path / "net_mean.pt").exists() and (path / "net_logstd.pt").exists()
    if model_name == "packnet":
        return (path / "packnet.pt").exists()
    if model_name == "prognet":
        return all((path / name).exists() for name in ("prognet.pt", "fc_mean.pt", "fc_logstd.pt"))
    if model_name in {"hybrid", "hybridrarl", "hybridrarl-prev-units", "hybridrarl-simple-prev-units"}:
        return any(
            (path / name).exists()
            for name in ("model.pt", "hybridrarl_modules.pt", "hybridrarl_prev_units_modules.pt")
        )
    return (path / "model.pt").exists()


def search_roots(model_root: Path) -> List[Path]:
    result = []
    seen = set()
    for root in [model_root, model_root / "hybridrarl_prev_units", model_root / "hybridrarl", model_root / "hybrid"]:
        root = root.resolve()
        if root.exists() and root not in seen:
            result.append(root)
            seen.add(root)
    return result


def resolve_model_dir(
    model_root: Path,
    scenario: str,
    task_id: int,
    model_names: Sequence[str],
    exp_names: Sequence[str],
    seed: int,
) -> Path:
    roots = search_roots(model_root)
    candidates = []
    for model_name in model_names:
        for exp_name in exp_names:
            dirname = model_directory_name(task_id, model_name, exp_name, seed)
            for root in roots:
                if scenario:
                    candidates.append(root / scenario / dirname)
                candidates.append(root / dirname)

    for candidate in candidates:
        candidate_model_name = candidate.name.split("__")[1]
        if has_checkpoint(candidate, candidate_model_name):
            return candidate.resolve()

    for model_name in model_names:
        for exp_name in exp_names:
            dirname = model_directory_name(task_id, model_name, exp_name, seed)
            for root in roots:
                if not root.exists():
                    continue
                for candidate in root.rglob(dirname):
                    if candidate.is_dir() and has_checkpoint(candidate, model_name):
                        if not scenario or candidate.parent.name == scenario:
                            return candidate.resolve()

    tried = "\n".join(f"    {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"No checkpoint found for task {task_id} ({get_task_name(task_id)}), "
        f"models={list(model_names)}, seed={seed}. Tried:\n{tried}"
    )


def first_output(value):
    return value[0] if isinstance(value, (tuple, list)) else value


class CompoNetEvalAgent(nn.Module):
    def __init__(self, net_mean: nn.Module, net_logstd: nn.Module):
        super().__init__()
        self.net_mean = net_mean
        self.net_logstd = net_logstd

    def forward(self, x, **unused_kwargs):
        return self.net_mean(x)[0], self.net_logstd(x)


class FeatureEvalAgent(nn.Module):
    def __init__(self, feature: nn.Module, fc_mean: nn.Module, fc_logstd: nn.Module):
        super().__init__()
        self.feature = feature
        self.fc_mean = fc_mean
        self.fc_logstd = fc_logstd

    def forward(self, x, **unused_kwargs):
        hidden = self.feature(x)
        return self.fc_mean(hidden), self.fc_logstd(hidden)


def has_any_actor_file(path: Path) -> bool:
    return any(
        (path / name).exists()
        for name in (
            "model.pt", "net_mean.pt", "packnet.pt", "prognet.pt",
            "hybridrarl_modules.pt", "hybridrarl_prev_units_modules.pt",
        )
    )


def resolve_stored_path(path: Path, roots: Sequence[Path]) -> Path:
    path = Path(path).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates += [root / path for root in roots]
    candidates += [root / path.name for root in roots]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists() and has_any_actor_file(candidate):
            return candidate
    for root in roots:
        if not root.exists():
            continue
        for candidate in root.rglob(path.name):
            if candidate.is_dir() and has_any_actor_file(candidate):
                return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve saved previous unit: {path}")


def load_prognet_column(path: Path, previous_paths: Sequence[Path], device, stack=None) -> nn.Module:
    stack = set() if stack is None else stack
    path = path.resolve()
    if path in stack:
        raise RuntimeError(f"Recursive ProgNet checkpoint cycle: {path}")
    stack.add(path)
    feature = safe_torch_load(path / "prognet.pt", device)
    if not isinstance(feature, nn.Module):
        raise TypeError(f"Expected ProgNet module in {path / 'prognet.pt'}")
    feature.previous_models = [
        load_prognet_column(previous_path, previous_paths[:index], device, stack)
        for index, previous_path in enumerate(previous_paths)
    ]
    stack.remove(path)
    return feature


def load_hybrid_modules(path: Path, device, roots: Sequence[Path], stack=None) -> nn.Module:
    module_path = path / "hybridrarl_prev_units_modules.pt"
    is_prev_units = module_path.exists()
    if not is_prev_units:
        module_path = path / "hybridrarl_modules.pt"
    checkpoint = safe_torch_load(module_path, device)
    if not isinstance(checkpoint, dict) or "meta" not in checkpoint:
        raise TypeError(f"Invalid HybridRARL checkpoint: {module_path}")
    meta = checkpoint["meta"]
    memory = None
    if (path / "hybridrarl_memory.pt").exists():
        memory = ReservoirTrajectoryMemory.load(path, map_location=device)
    kwargs = {
        key: meta[key]
        for key in (
            "obs_dim", "act_dim", "memory_capacity", "traj_len", "top_r",
            "traj_hidden_dim", "att_dim", "value_dim", "policy_hidden_dim",
            "policy_layers", "coarse_metric",
        )
        if key in meta
    }
    kwargs["memory"] = memory
    if is_prev_units:
        previous_paths = [
            resolve_stored_path(Path(saved_path), roots)
            for saved_path in meta.get("prev_unit_paths", [])
        ]
        kwargs["prev_units"] = [
            load_actor(previous_path, device, roots, stack=stack)
            for previous_path in previous_paths
        ]
        kwargs["prev_unit_paths"] = previous_paths
        if "prev_att_dim" in meta:
            kwargs["prev_att_dim"] = meta["prev_att_dim"]
        model = HybridRARLPrevUnitsAgent(**kwargs).to(device)
        state_names = ("trajectory_rnn", "attention_head", "prev_unit_attention", "policy", "fc_mean", "fc_logstd")
    else:
        model = HybridRARLAgent(**kwargs).to(device)
        state_names = ("trajectory_rnn", "attention_head", "policy", "fc_mean", "fc_logstd")
    for name in state_names:
        if name in checkpoint:
            getattr(model, name).load_state_dict(checkpoint[name])
    return model.eval()


def load_componet(path: Path, previous_paths: Sequence[Path], device) -> nn.Module:
    if not previous_paths:
        raise ValueError(f"CompoNet checkpoint {path} requires previous units.")
    first_model = safe_torch_load(previous_paths[0] / "model.pt", device)
    first_unit = FirstModuleWrapper(
        model=first_model,
        ret_probs=False,
        transform_output=lambda value: first_output(value),
    )
    # CompoNet expects every previous unit, including the first simple unit,
    # to return (phi, state) in its internal ``is_prev`` mode.
    first_unit.is_prev = True
    previous_units = [first_unit]
    for previous_path in previous_paths[1:]:
        previous_unit = safe_torch_load(previous_path / "net_mean.pt", device)
        previous_unit.is_prev = True
        previous_units.append(previous_unit)
    net_mean = safe_torch_load(path / "net_mean.pt", device)
    net_mean.previous_units = nn.Sequential(*previous_units)
    net_mean.is_prev = False
    net_logstd = safe_torch_load(path / "net_logstd.pt", device)
    return CompoNetEvalAgent(net_mean, net_logstd).to(device).eval()


def load_actor(path: Path, device, roots: Sequence[Path], previous_paths=None, stack=None) -> nn.Module:
    path = path.resolve()
    directory_model_name = path.name.split("__")[1] if "__" in path.name else ""
    if (path / "hybridrarl_prev_units_modules.pt").exists() or (path / "hybridrarl_modules.pt").exists():
        model = load_hybrid_modules(path, device, roots, stack=stack)
    elif directory_model_name == "componet" or ((path / "net_mean.pt").exists() and (path / "net_logstd.pt").exists()):
        model = load_componet(path, previous_paths or [], device)
    elif (path / "packnet.pt").exists():
        model = safe_torch_load(path / "packnet.pt", device)
    elif (path / "prognet.pt").exists():
        feature = load_prognet_column(path, previous_paths or [], device)
        model = FeatureEvalAgent(
            feature,
            safe_torch_load(path / "fc_mean.pt", device),
            safe_torch_load(path / "fc_logstd.pt", device),
        )
    elif (path / "model.pt").exists():
        model = safe_torch_load(path / "model.pt", device)
    else:
        raise FileNotFoundError(f"No supported actor checkpoint in {path}")
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module in {path}, got {type(model)}")
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def reset_env(env, seed: int):
    output = env.reset(seed=int(seed))
    return output[0] if isinstance(output, tuple) else output


def step_env(env, action):
    output = env.step(action)
    if len(output) == 5:
        obs, reward, terminated, truncated, info = output
        return obs, float(reward), bool(terminated or truncated), info
    obs, reward, done, info = output
    return obs, float(reward), bool(done), info


def episode_limit(env, fallback: int) -> int:
    spec = getattr(env, "spec", None)
    value = getattr(spec, "max_episode_steps", None) if spec is not None else None
    return int(value) if value is not None else int(fallback)


def scenario_previous_paths(args, task_ids: Sequence[int], task_index: int, model_root: Path, exp_names: Sequence[str]) -> List[Path]:
    paths = []
    for index, previous_task_id in enumerate(task_ids[:task_index]):
        previous_model_name = "simple" if index == 0 else "componet"
        previous_scenario = "" if previous_model_name == "simple" else args.scenario
        previous_exp_names = args.simple_exp_names if previous_model_name == "simple" else exp_names
        paths.append(
            resolve_model_dir(
                model_root,
                previous_scenario,
                previous_task_id,
                [previous_model_name],
                previous_exp_names,
                args.train_seed,
            )
        )
    return paths


def evaluate_task(args, task_id: int, source_task_id: int, model_dir: Path, previous_paths: Sequence[Path], device) -> Dict:
    env = get_task(task_id)
    model = load_actor(model_dir, device, search_roots(args.model_root), previous_paths=previous_paths)
    scale = torch.as_tensor(
        (env.action_space.high - env.action_space.low) / 2.0,
        dtype=torch.float32,
        device=device,
    )
    bias = torch.as_tensor(
        (env.action_space.high + env.action_space.low) / 2.0,
        dtype=torch.float32,
        device=device,
    )
    max_steps = args.max_steps if args.max_steps > 0 else episode_limit(env, 1000)
    returns = []
    lengths = []
    print(f"\n*** Eval task {task_id}: {get_task_name(task_id)} (checkpoint task {source_task_id})")
    print(f"*** Model: {model_dir}")
    for episode_index in range(args.episodes):
        episode_seed = args.test_seed + episode_index
        obs = reset_env(env, episode_seed)
        total_reward = 0.0
        length = 0
        for length in range(1, max_steps + 1):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).reshape(1, -1)
            mean = first_output(model(obs_tensor))
            action = torch.tanh(mean) * scale + bias
            action = action.squeeze(0).cpu().numpy().astype(np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, _ = step_env(env, action)
            total_reward += reward
            if done:
                break
        returns.append(total_reward)
        lengths.append(length)
        print(f"episode={episode_index + 1:03d}, seed={episode_seed}, return={total_reward:.4f}, length={length}")
    env.close()
    result = {
        "task_id": int(task_id),
        "task_name": get_task_name(task_id),
        "source_task_id": int(source_task_id),
        "source_task_name": get_task_name(source_task_id),
        "model_dir": str(model_dir),
        "episodes": int(args.episodes),
        "train_seed": int(args.train_seed),
        "test_seed": int(args.test_seed),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "returns": [float(value) for value in returns],
        "lengths": [int(value) for value in lengths],
    }
    print(f"*** Task summary: return={result['mean_return']:.4f} +/- {result['std_return']:.4f}")
    return result


def summarize(records: Sequence[Dict], args, task_ids: Sequence[int]) -> Dict:
    task_means = np.asarray([record["mean_return"] for record in records], dtype=np.float64)
    episode_values = np.asarray([value for record in records for value in record["returns"]], dtype=np.float64)
    return {
        "benchmark": "halfcheetah",
        "model_name": args.model_name,
        "scenario": args.scenario or "all_tasks",
        "train_seed": int(args.train_seed),
        "test_seed": int(args.test_seed),
        "episodes_per_task": int(args.episodes),
        "task_ids": [int(task_id) for task_id in task_ids],
        "task_names": [get_task_name(task_id) for task_id in task_ids],
        "num_tasks": int(task_means.size),
        "mean_return_across_tasks": float(np.mean(task_means)) if task_means.size else 0.0,
        "var_return_across_tasks": float(np.var(task_means)) if task_means.size else 0.0,
        "mean_return_across_episodes": float(np.mean(episode_values)) if episode_values.size else 0.0,
        "var_return_across_episodes": float(np.var(episode_values)) if episode_values.size else 0.0,
    }


def write_outputs(records: Sequence[Dict], summary: Dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_id", "task_name", "source_task_id", "source_task_name", "model_dir",
        "episodes", "train_seed", "test_seed", "mean_return", "std_return", "mean_length",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})
    output.with_suffix(".json").write_text(json.dumps(list(records), indent=2, sort_keys=True), encoding="utf-8")
    output.with_name(output.stem + "__summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n*** Saved CSV: {output}")
    print(f"*** Saved JSON: {output.with_suffix('.json')}")
    print(f"*** Saved summary JSON: {output.with_name(output.stem + '__summary.json')}")


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone HalfCheetah evaluator using cumulative reward.")
    parser.add_argument(
        "--model-name",
        required=True,
        choices=["simple", "sac-n", "baseline", "finetune-1", "finetune-n", "componet", "hybridrarl", "hybrid", "packnet", "prognet"],
    )
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="")
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--test-seed", type=int, required=True)
    parser.add_argument("--episodes", "--episode", dest="episodes", type=int, required=True)
    parser.add_argument("--model-root", type=Path, default=Path("agents"))
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--exp-name-candidates", nargs="*", default=[])
    parser.add_argument("--simple-exp-names", nargs="*", default=["SAC", "run_sac", "SAC_hybridRAGRL", "sac_20k"])
    parser.add_argument("--source-task-id", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--deterministic", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--cuda", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.model_name = normalize_model_name(args.model_name)
    if args.model_name != "simple" and not args.scenario:
        raise ValueError(f"--scenario is required for `{args.model_name}`.")
    task_ids = evaluation_task_ids(args.model_name, args.scenario)
    scenario_task_ids = unique_scenario_task_ids(args.scenario) if args.scenario else []
    source_task_id = (
        int(args.source_task_id)
        if args.source_task_id is not None
        else (scenario_task_ids[-1] if args.model_name in {"baseline", "finetune-1"} else None)
    )
    if source_task_id is not None and not 0 <= source_task_id < len(tasks):
        raise ValueError(f"Invalid source task id {source_task_id}.")

    model_names, default_exp_names = model_plan(args.model_name)
    exp_names = [args.exp_name] if args.exp_name else list(default_exp_names)
    exp_names += list(args.exp_name_candidates)
    exp_names = list(dict.fromkeys(exp_names))
    model_root = args.model_root.expanduser()
    if not model_root.is_absolute():
        model_root = (Path.cwd() / model_root).resolve()
    args.model_root = model_root
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    set_test_seed(args.test_seed)

    print("*** Benchmark: halfcheetah")
    print(f"*** Model name: {args.model_name}")
    print(f"*** Scenario: {args.scenario or 'all_tasks'}")
    print(f"*** Train seed: {args.train_seed}")
    print(f"*** Test seed: {args.test_seed}")
    print(f"*** Episodes: {args.episodes}")
    print(f"*** Device: {device}")
    print(f"*** Evaluation tasks: {[get_task_name(task_id) for task_id in task_ids]}")
    if args.model_name == "hybridrarl":
        print("*** HybridRARL priority: prev-units -> hybridrarl -> hybrid")
    if source_task_id is not None:
        print(f"*** Shared source task: {source_task_id} ({get_task_name(source_task_id)})")

    records = []
    for task_index, task_id in enumerate(task_ids):
        current_source_task_id = source_task_id if source_task_id is not None else int(task_id)
        current_model_names = model_names
        current_scenario = args.scenario if args.model_name not in {"simple", "sac-n", "baseline"} else ""
        current_exp_names = args.simple_exp_names if args.model_name in {"simple", "sac-n", "baseline"} else exp_names
        previous_paths = []

        if args.model_name == "componet" and task_index == 0:
            current_model_names = ["simple"]
            current_scenario = ""
            current_exp_names = args.simple_exp_names
        elif args.model_name == "componet":
            previous_paths = scenario_previous_paths(
                args, task_ids, task_index, model_root, exp_names
            )
        elif args.model_name == "prognet":
            previous_paths = [
                resolve_model_dir(
                    model_root,
                    args.scenario,
                    previous_task_id,
                    ["prognet"],
                    exp_names,
                    args.train_seed,
                )
                for previous_task_id in task_ids[:task_index]
            ]

        model_dir = resolve_model_dir(
            model_root,
            current_scenario,
            current_source_task_id,
            current_model_names,
            current_exp_names,
            args.train_seed,
        )
        records.append(
            evaluate_task(
                args,
                int(task_id),
                int(current_source_task_id),
                model_dir,
                previous_paths,
                device,
            )
        )

    summary = summarize(records, args, task_ids)
    print("\nOverall Performance")
    print(
        f"mean_return_across_tasks={summary['mean_return_across_tasks']:.6f}, "
        f"variance={summary['var_return_across_tasks']:.6f}"
    )
    print(
        f"mean_return_across_episodes={summary['mean_return_across_episodes']:.6f}, "
        f"variance={summary['var_return_across_episodes']:.6f}"
    )
    output = args.output.expanduser() if args.output else (
        THIS_DIR / "analysis" / "eval_results" /
        f"halfcheetah__{args.model_name}__{args.scenario or 'all_tasks'}__"
        f"train{args.train_seed}__test{args.test_seed}.csv"
    )
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    write_outputs(records, summary, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
