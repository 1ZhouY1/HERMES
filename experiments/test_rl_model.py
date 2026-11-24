#!/usr/bin/env python3
"""
Evaluate saved Meta-World or HalfCheetah SAC actors.

Default behavior:
  task_i is evaluated on environment task i.

Examples:
  python experiments/test_rl_model.py --benchmark meta-world --model-name hybridrarl --train-seed 42 --test-seed 0 --episodes 10
  python experiments/test_rl_model.py --benchmark meta-world --model-name hybridrarl-no-retrieval --train-seed 42 --test-seed 0 --episodes 50
  python experiments/test_rl_model.py --benchmark meta-world --model-name hybridrarl-no-selective-reuse --train-seed 42 --test-seed 0 --episodes 50
  python experiments/test_rl_model.py --benchmark halfcheetah --scenario forgetting --model-root experiments/halfcheetah/agents/hybridrarl_prev_units --model-name hybridrarl --train-seed 42 --test-seed 0 --episodes 10
  python experiments/test_rl_model.py --benchmark meta-world --model-name finetune-1 --train-seed 42 --test-seed 0 --episodes 10
  python experiments/test_rl_model.py --benchmark meta-world --model-name finetune-n --train-seed 42 --test-seed 0 --episodes 10
"""

import argparse
import csv
import importlib
import importlib.util
import json
import pickle
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


def mlp_net(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(int(input_dim), 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, int(output_dim)),
    )


class LoadedCompoNetAgent(nn.Module):
    def __init__(self, net_mean: nn.Module, net_logstd: nn.Module):
        super().__init__()
        self.net_mean = net_mean
        self.net_logstd = net_logstd

    def forward(self, x, **unused_kwargs):
        mean = self.net_mean(x)[0]
        logstd = self.net_logstd(x)
        return mean, logstd


class LoadedFeatureActor(nn.Module):
    """Actor wrapper for feature network + SAC mean/log-std heads."""

    def __init__(self, feature: nn.Module, fc_mean: nn.Module, fc_logstd: nn.Module):
        super().__init__()
        self.feature = feature
        self.fc_mean = fc_mean
        self.fc_logstd = fc_logstd

    def forward(self, x, **unused_kwargs):
        hidden = self.feature(x)
        return self.fc_mean(hidden), self.fc_logstd(hidden)


class LoadedHybridDictAgent(nn.Module):
    def __init__(self, net_mean: nn.Module, net_logstd: Optional[nn.Module], memory, n_traj: int):
        super().__init__()
        self.net_mean = net_mean
        self.net_logstd = net_logstd
        self.memory = memory
        self.n_traj = int(n_traj)

    def forward(self, x, memory_batch=None, return_info: bool = False, **unused_kwargs):
        if memory_batch is None:
            assert self.memory is not None, "Hybrid model requires long_term_memory.pt or memory_batch."
            memory_batch = self.memory.sample_trajectories(n_traj=self.n_traj, device=x.device)

        if return_info:
            mean, info = self.net_mean(x, memory_batch=memory_batch, return_info=True)
            encoded_state = info["encoded_state"]
        else:
            mean = self.net_mean(x, memory_batch=memory_batch, return_info=False)
            encoded_state = self.net_mean.encode_state(x)

        if self.net_logstd is None:
            logstd = torch.zeros_like(mean)
        else:
            logstd = self.net_logstd(encoded_state)
        if return_info:
            return mean, logstd, info
        return mean, logstd


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


def experiment_dir(benchmark: str) -> Path:
    root = Path(__file__).resolve().parent
    if benchmark == "meta-world":
        return root / "meta-world"
    if benchmark == "halfcheetah":
        return root / "halfcheetah"
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def import_experiment_modules(exp_dir: Path):
    sys.path.insert(0, str(exp_dir))
    tasks_module = importlib.import_module("tasks")
    models_module = importlib.import_module("models")
    return tasks_module, models_module


def default_task_ids(benchmark: str, task_names: Sequence[str]) -> List[int]:
    if benchmark == "meta-world":
        return list(range(min(10, len(task_names))))
    return list(range(len(task_names)))


def model_dir_name(task_id: int, model_name: str, exp_name: str, train_seed: int) -> str:
    return f"task_{task_id}__{model_name}__{exp_name}__{train_seed}"


def normalized_model_name(model_name: str) -> str:
    name = str(model_name).strip().lower()
    aliases = {
        "finetune_1": "finetune-1",
        "finetune1": "finetune-1",
        "finetune_n": "finetune-n",
        "finetunen": "finetune-n",
        "hybridrarl_no_retrieval": "hybridrarl-no-retrieval",
        "hybridrarl_no_selective_reuse": "hybridrarl-no-selective-reuse",
        "hybridrarl_random_retrieval": "hybridrarl-random-retrieval",
    }
    return aliases.get(name, name)


def checkpoint_model_name(model_name: str) -> str:
    name = normalized_model_name(model_name)
    if name in {"finetune-1", "finetune-n"}:
        return "finetune"
    if name == "baseline":
        return "simple"
    return name


def checkpoint_model_names_for_task(args, load_task_id: int) -> List[str]:
    name = normalized_model_name(args.model_name)
    if name in {"finetune-n", "componet"} and int(load_task_id) == 0:
        return ["simple", "finetune"]
    return candidate_model_names(args)


def model_name_from_dir(dirname: str) -> str:
    parts = Path(dirname).name.split("__")
    if len(parts) >= 4:
        return parts[1]
    return ""


def checkpoint_task_id(args, eval_task_id: int) -> int:
    if normalized_model_name(args.model_name) == "finetune-1":
        if args.source_task_id is None:
            raise ValueError("finetune-1 requires --source-task-id to be resolved before evaluation.")
        return int(args.source_task_id)
    if normalized_model_name(args.model_name) == "baseline":
        if args.source_task_id is None:
            raise ValueError("baseline requires the final simple checkpoint task id to be resolved before evaluation.")
        return int(args.source_task_id)
    return int(eval_task_id)


def default_exp_names(benchmark: str, model_name: str) -> List[str]:
    model_name = checkpoint_model_name(model_name)
    # Some simple actors were trained with SAC_hybridRAGRL.py and therefore
    # keep that experiment name even though the saved actor is a simple model.
    common_simple = ["SAC", "run_sac", "SAC_hybridRAGRL"]
    meta_defaults = {
        "simple": common_simple,
        "finetune": ["run_sac", "SAC"],
        "componet": ["run_sac", "SAC"],
        "packnet": ["run_sac", "SAC"],
        "prognet": ["run_sac", "SAC"],
        "hybrid": ["SAC_hybridRAGRL", "SAC"],
        "hybridrarl": ["SAC_hybridRAGRL", "SAC_hybridrarl_prev_units", "SAC"],
        "hybridrarl-prev-units": ["csv_threshold_transfer", "SAC_hybridrarl_prev_units"],
        "hybridrarl-simple-prev-units": ["csv_threshold_transfer", "csv_simple_prev"],
        "hybridrarl-random-retrieval": ["SAC_hybridRAGRL", "SAC"],
        "no-retrieval": ["csv_threshold_transfer", "SAC"],
        "no-retrieval-prev-units": ["csv_threshold_transfer", "csv_simple_prev"],
        "no-retrieval-simple-prev-units": ["csv_threshold_transfer", "csv_simple_prev"],
    }
    halfcheetah_defaults = {
        "simple": common_simple,
        "finetune": ["SAC", "run_sac"],
        "componet": ["SAC", "run_sac"],
        "packnet": ["run_sac", "SAC"],
        "prognet": ["run_sac", "SAC"],
        "hybrid": ["SAC_hybridRAGRL", "SAC_hybridrarl_prev_units", "SAC"],
        "hybridrarl": ["SAC_hybridrarl_prev_units", "SAC_hybridRAGRL", "SAC"],
        "hybridrarl-prev-units": ["SAC_hybridrarl_prev_units"],
        "hybridrarl-random-retrieval": ["SAC_hybridRAGRL", "SAC"],
        "no-retrieval": ["SAC_hybridrarl_prev_units", "SAC"],
        "no-retrieval-prev-units": ["SAC_hybridrarl_prev_units"],
    }
    table = meta_defaults if benchmark == "meta-world" else halfcheetah_defaults
    return table.get(model_name, ["SAC", "run_sac", "csv_threshold_transfer", "SAC_hybridRAGRL", "SAC_hybridrarl_prev_units"])


def candidate_model_names(args) -> List[str]:
    model_name = checkpoint_model_name(args.model_name)
    if model_name == "finetune":
        return ["finetune"]
    if model_name == "hybridrarl-no-selective-reuse":
        # This ablation must not load a HybridRARLPrevUnitsAgent.  It evaluates
        # only the original per-task HybridRARL actor, with its own retrieval
        # memory but no cross-task module reuse.
        return ["hybridrarl"]
    if not args.prefer_prev_units:
        return ["hybridrarl", "hybrid"] if model_name == "hybridrarl" else [model_name]
    if model_name == "hybridrarl":
        return [
            "hybridrarl-prev-units",
            "hybridrarl-simple-prev-units",
            "hybridrarl",
            "hybrid",
        ]
    if model_name == "no-retrieval":
        return [
            "no-retrieval-prev-units",
            "no-retrieval-simple-prev-units",
            "no-retrieval",
        ]
    if model_name == "hybridrarl-no-retrieval":
        # This ablation has no retrieval memory.  A saved prev-unit actor is
        # used only when selective reuse found at least one source task;
        # otherwise the method is exactly the task's simple SAC actor.
        # Do not include the bare no-retrieval actor here: it is not the
        # specified simple fallback for a task without reusable modules.
        return [
            "no-retrieval-prev-units",
            "no-retrieval-simple-prev-units",
            "simple",
        ]
    return [model_name]


def ordered_unique(values: Iterable[str]) -> List[str]:
    out = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def resolve_root(path: Path, exp_dir: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    return (exp_dir / path).resolve()


def format_model_dir_arg(raw: str, task_id: int, model_name: str, train_seed: int) -> Path:
    model_name_for_dir = checkpoint_model_name(model_name)
    return Path(
        raw.format(
            task_id=task_id,
            task=task_id,
            model_name=model_name_for_dir,
            requested_model_name=model_name,
            train_seed=train_seed,
            seed=train_seed,
        )
    ).expanduser()


def candidate_model_dirs(args, exp_dir: Path, task_id: int) -> List[Path]:
    load_task_id = checkpoint_task_id(args, task_id)
    if args.model_dir:
        return [format_model_dir_arg(args.model_dir, load_task_id, args.model_name, args.train_seed)]

    model_root = resolve_root(args.model_root, exp_dir)

    candidates = []
    for model_name in checkpoint_model_names_for_task(args, load_task_id):
        exp_names = [args.exp_name] if args.exp_name else default_exp_names(args.benchmark, model_name)
        exp_names = ordered_unique(exp_names + list(args.exp_name_candidates or []))
        for exp_name in exp_names:
            dirname = model_dir_name(load_task_id, model_name, exp_name, args.train_seed)
            if args.scenario:
                candidates.append(model_root / args.scenario / dirname)
            candidates.append(model_root / dirname)
    return candidates


def candidate_named_model_dirs(args, exp_dir: Path, task_id: int, model_name: str) -> List[Path]:
    model_root = resolve_root(args.model_root, exp_dir)
    exp_names = [args.exp_name] if args.exp_name else default_exp_names(args.benchmark, model_name)
    exp_names = ordered_unique(exp_names + list(args.exp_name_candidates or []))
    candidates = []
    for exp_name in exp_names:
        dirname = model_dir_name(task_id, model_name, exp_name, args.train_seed)
        if args.scenario:
            candidates.append(model_root / args.scenario / dirname)
        candidates.append(model_root / dirname)
    return candidates


def has_saved_actor(model_dir: Path) -> bool:
    return (
        (model_dir / "model.pt").exists()
        or ((model_dir / "net_mean.pt").exists() and (model_dir / "net_logstd.pt").exists())
        or (model_dir / "packnet.pt").exists()
        or (
            (model_dir / "prognet.pt").exists()
            and (model_dir / "fc_mean.pt").exists()
            and (model_dir / "fc_logstd.pt").exists()
        )
        or (model_dir / "hybridrarl_modules.pt").exists()
        or (model_dir / "hybridrarl_prev_units_modules.pt").exists()
    )


def resolve_model_dir(args, exp_dir: Path, task_id: int) -> Path:
    candidates = candidate_model_dirs(args, exp_dir, task_id)
    for candidate in candidates:
        if has_saved_actor(candidate):
            return candidate.resolve()

    if args.recursive_search and not args.model_dir:
        model_root = resolve_root(args.model_root, exp_dir)
        target_names = {candidate.name for candidate in candidates}
        for pattern in [
            "model.pt",
            "net_mean.pt",
            "packnet.pt",
            "prognet.pt",
            "hybridrarl_modules.pt",
            "hybridrarl_prev_units_modules.pt",
        ]:
            for model_path in model_root.rglob(pattern):
                if model_path.parent.name in target_names and has_saved_actor(model_path.parent):
                    return model_path.parent.resolve()

    tried = "\n".join(
        f"    {candidate / 'model.pt'} or {candidate / 'net_mean.pt'}"
        for candidate in candidates
    )
    raise FileNotFoundError(f"No saved actor found for task {task_id}. Tried:\n{tried}")


def resolve_named_model_dir(args, exp_dir: Path, task_id: int, model_name: str) -> Path:
    candidates = candidate_named_model_dirs(args, exp_dir, task_id, model_name)
    for candidate in candidates:
        if has_saved_actor(candidate):
            return candidate.resolve()
    if args.recursive_search:
        model_root = resolve_root(args.model_root, exp_dir)
        target_names = {candidate.name for candidate in candidates}
        for pattern in [
            "model.pt",
            "net_mean.pt",
            "packnet.pt",
            "prognet.pt",
            "hybridrarl_modules.pt",
            "hybridrarl_prev_units_modules.pt",
        ]:
            for model_path in model_root.rglob(pattern):
                if model_path.parent.name in target_names and has_saved_actor(model_path.parent):
                    return model_path.parent.resolve()
    tried = "\n".join(
        f"    {candidate / 'model.pt'} or {candidate / 'net_mean.pt'}"
        for candidate in candidates
    )
    raise FileNotFoundError(f"No {model_name} checkpoint found for task {task_id}. Tried:\n{tried}")


def scenario_task_ids(args, tasks_module) -> Optional[List[int]]:
    if not args.scenario or not hasattr(tasks_module, "SCENARIOS"):
        return None
    scenarios = getattr(tasks_module, "SCENARIOS")
    if args.scenario not in scenarios:
        raise KeyError(f"Unknown scenario `{args.scenario}`. Available: {sorted(scenarios)}")
    scenario = scenarios[args.scenario]
    task_names = getattr(tasks_module, "tasks")
    return [int(task_names.index(name)) for name in scenario]


def componet_previous_task_ids(args, tasks_module, load_task_id: int) -> List[int]:
    if args.componet_prev_tasks is not None:
        return [int(task_id) for task_id in args.componet_prev_tasks]

    sequence_ids = scenario_task_ids(args, tasks_module)
    if sequence_ids is None:
        return list(range(int(load_task_id)))

    matching_indices = [
        index for index, task_id in enumerate(sequence_ids)
        if int(task_id) == int(load_task_id)
    ]
    if not matching_indices:
        return list(range(int(load_task_id)))
    return [int(task_id) for task_id in sequence_ids[: matching_indices[-1]]]


def resolve_componet_prev_dirs(args, exp_dir: Path, tasks_module, load_task_id: int) -> List[Path]:
    prev_task_ids = componet_previous_task_ids(args, tasks_module, load_task_id)
    prev_dirs = []
    for index, prev_task_id in enumerate(prev_task_ids):
        prev_model_name = "simple" if index == 0 else "componet"
        prev_dirs.append(resolve_named_model_dir(args, exp_dir, prev_task_id, prev_model_name))
    return prev_dirs


def relocate_saved_model_dir(saved_path: Path, search_roots: Sequence[Path]) -> Path:
    saved_path = Path(saved_path).expanduser()
    if has_saved_actor(saved_path):
        return saved_path.resolve()

    for root in search_roots:
        for candidate in [root / saved_path.name, root.parent / saved_path.name]:
            if has_saved_actor(candidate):
                return candidate.resolve()
        for pattern in [
            "model.pt",
            "net_mean.pt",
            "packnet.pt",
            "prognet.pt",
            "hybridrarl_modules.pt",
            "hybridrarl_prev_units_modules.pt",
        ]:
            for model_path in root.rglob(pattern):
                if model_path.parent.name == saved_path.name and has_saved_actor(model_path.parent):
                    return model_path.parent.resolve()

    return saved_path.resolve()


def attach_hybridrarl_memory(model: nn.Module, model_dir: Path, models_module, map_location) -> None:
    if not hasattr(model, "memory"):
        return
    memory_path = model_dir / "hybridrarl_memory.pt"
    if not memory_path.exists():
        return
    memory_cls = getattr(models_module, "ReservoirTrajectoryMemory", None)
    if memory_cls is None:
        return
    model.memory = memory_cls.load(model_dir, map_location=map_location)


def ensure_packnet_prognet_modules() -> None:
    """Make legacy ``models.packnet/prognet`` pickle classes importable.

    Meta-World's current models package does not export these classes, while
    the saved files retain their original ``models.*`` pickle module names.
    """
    package = sys.modules.get("models")
    if package is None:
        return
    halfcheetah_dir = Path(__file__).resolve().parent / "halfcheetah"
    for module_name in ("packnet", "prognet"):
        qualified_name = f"models.{module_name}"
        if qualified_name in sys.modules:
            continue
        module_path = halfcheetah_dir / "models" / f"{module_name}.py"
        if not module_path.exists():
            continue
        spec = importlib.util.spec_from_file_location(qualified_name, module_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        setattr(package, module_name, module)
        spec.loader.exec_module(module)


def load_packnet_agent(model_dir: Path, map_location) -> nn.Module:
    ensure_packnet_prognet_modules()
    model = safe_torch_load(model_dir / "packnet.pt", map_location=map_location)
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module in {model_dir / 'packnet.pt'}, got {type(model)}")
    return model


def load_prognet_core(
    model_dir: Path,
    previous_dirs: Sequence[Path],
    map_location,
    loading_stack: Optional[set] = None,
) -> nn.Module:
    """Load a saved ProgNet column and restore its previous columns."""
    ensure_packnet_prognet_modules()
    loading_stack = set() if loading_stack is None else loading_stack
    model_dir = Path(model_dir).resolve()
    if model_dir in loading_stack:
        raise RuntimeError(f"Recursive ProgNet checkpoint cycle detected at {model_dir}")
    loading_stack.add(model_dir)

    feature = safe_torch_load(model_dir / "prognet.pt", map_location=map_location)
    if not isinstance(feature, nn.Module):
        raise TypeError(f"Expected nn.Module in {model_dir / 'prognet.pt'}, got {type(feature)}")

    # ProgNet's save() removes previous_models to avoid serializing the whole
    # history. Reattach the earlier feature columns in task order.
    restored_previous = []
    for index, previous_dir in enumerate(previous_dirs):
        previous_dir = Path(previous_dir).resolve()
        restored_previous.append(
            load_prognet_core(
                previous_dir,
                previous_dirs=previous_dirs[:index],
                map_location=map_location,
                loading_stack=loading_stack,
            )
        )
    if not hasattr(feature, "previous_models"):
        feature.previous_models = restored_previous
    else:
        feature.previous_models = restored_previous

    loading_stack.remove(model_dir)
    return feature


def load_prognet_agent(
    args,
    exp_dir: Path,
    model_dir: Path,
    load_task_id: int,
    map_location,
) -> nn.Module:
    previous_task_ids = getattr(args, "_prognet_previous_task_ids", None)
    if previous_task_ids is None:
        previous_task_ids = range(int(load_task_id))
    previous_dirs = []
    for previous_task_id in previous_task_ids:
        previous_dirs.append(
            resolve_named_model_dir(args, exp_dir, previous_task_id, "prognet")
        )

    feature = load_prognet_core(
        model_dir=model_dir,
        previous_dirs=previous_dirs,
        map_location=map_location,
    )
    fc_mean = safe_torch_load(model_dir / "fc_mean.pt", map_location=map_location)
    fc_logstd = safe_torch_load(model_dir / "fc_logstd.pt", map_location=map_location)
    model = LoadedFeatureActor(feature, fc_mean, fc_logstd)
    return model


def load_long_term_memory(model_dir: Path, models_module, map_location):
    memory_path = model_dir / "long_term_memory.pt"
    if not memory_path.exists():
        return None
    memory_cls = getattr(models_module, "LongTermMemory", None)
    if memory_cls is None:
        return None
    try:
        return memory_cls.load(model_dir, map_location=map_location)
    except Exception as exc:
        raise RuntimeError(
            f"Corrupted or incomplete memory checkpoint: {memory_path}. "
            "The file should be regenerated or copied again."
        ) from exc


def load_hybridrarl_modules(
    model_dir: Path,
    models_module,
    map_location,
    search_roots: Sequence[Path],
    loading_stack: set,
) -> nn.Module:
    """Rebuild a HybridRARL actor when only its module checkpoint was saved."""
    prev_modules_path = model_dir / "hybridrarl_prev_units_modules.pt"
    base_modules_path = model_dir / "hybridrarl_modules.pt"
    modules_path = prev_modules_path if prev_modules_path.exists() else base_modules_path
    if not modules_path.exists():
        raise FileNotFoundError(
            "No HybridRARL actor checkpoint found in "
            f"{model_dir}. Expected model.pt, hybridrarl_modules.pt, or "
            "hybridrarl_prev_units_modules.pt."
        )

    checkpoint = safe_torch_load(modules_path, map_location=map_location)
    if not isinstance(checkpoint, dict) or "meta" not in checkpoint:
        raise TypeError(f"Invalid HybridRARL module checkpoint: {modules_path}")

    meta = checkpoint["meta"]
    is_prev_units = modules_path.name == "hybridrarl_prev_units_modules.pt"
    actor_cls_name = "HybridRARLPrevUnitsAgent" if is_prev_units else "HybridRARLAgent"
    actor_cls = getattr(models_module, actor_cls_name, None)
    memory_cls = getattr(models_module, "ReservoirTrajectoryMemory", None)
    if actor_cls is None or memory_cls is None:
        raise ImportError(
            f"The imported models module does not expose {actor_cls_name} "
            "and ReservoirTrajectoryMemory."
        )

    memory = None
    if (model_dir / "hybridrarl_memory.pt").exists():
        memory = memory_cls.load(model_dir, map_location=map_location)

    common_kwargs = {
        key: meta[key]
        for key in (
            "obs_dim",
            "act_dim",
            "memory_capacity",
            "traj_len",
            "top_r",
            "traj_hidden_dim",
            "att_dim",
            "value_dim",
            "policy_hidden_dim",
            "policy_layers",
            "coarse_metric",
            "random_coarse_retrieval",
        )
        if key in meta
    }
    # Backward compatibility: older random-retrieval checkpoints may have
    # been saved before `random_coarse_retrieval` was added to the module
    # metadata.  The directory name is still unambiguous, so preserve the
    # intended ablation behavior during evaluation.
    if (
        "random_coarse_retrieval" not in common_kwargs
        and normalized_model_name(model_name_from_dir(model_dir.name))
        == "hybridrarl-random-retrieval"
    ):
        common_kwargs["random_coarse_retrieval"] = True
    common_kwargs["memory"] = memory

    prev_units = []
    prev_paths = []
    if is_prev_units:
        prev_paths = [
            relocate_saved_model_dir(Path(path), search_roots)
            for path in meta.get("prev_unit_paths", [])
        ]
        prev_units = [
            load_policy_unit(
                path,
                models_module=models_module,
                map_location=map_location,
                search_roots=search_roots,
                loading_stack=loading_stack,
            )
            for path in prev_paths
        ]
        common_kwargs["prev_units"] = prev_units
        common_kwargs["prev_unit_paths"] = prev_paths
        if "prev_att_dim" in meta:
            common_kwargs["prev_att_dim"] = meta["prev_att_dim"]

    model = actor_cls(**common_kwargs)
    for name in (
        "trajectory_rnn",
        "attention_head",
        "prev_unit_attention",
        "policy",
        "fc_mean",
        "fc_logstd",
    ):
        state_dict = checkpoint.get(name)
        if state_dict is not None and hasattr(model, name):
            getattr(model, name).load_state_dict(state_dict)

    if is_prev_units and hasattr(model, "prev_unit_task_ids"):
        model.prev_unit_task_ids = [int(value) for value in meta.get("prev_unit_task_ids", [])]
    return model


def load_policy_unit(
    model_dir: Path,
    models_module,
    map_location,
    search_roots: Sequence[Path],
    loading_stack: Optional[set] = None,
) -> nn.Module:
    loading_stack = set() if loading_stack is None else loading_stack
    model_dir = relocate_saved_model_dir(model_dir, search_roots)
    if model_dir in loading_stack:
        raise RuntimeError(f"Recursive prev-unit cycle detected at {model_dir}")

    loading_stack.add(model_dir)
    model_path = model_dir / "model.pt"
    if model_path.exists():
        model = safe_torch_load(model_path, map_location=map_location)
    elif (
        (model_dir / "hybridrarl_modules.pt").exists()
        or (model_dir / "hybridrarl_prev_units_modules.pt").exists()
    ):
        model = load_hybridrarl_modules(
            model_dir=model_dir,
            models_module=models_module,
            map_location=map_location,
            search_roots=search_roots,
            loading_stack=loading_stack,
        )
    else:
        available = sorted(path.name for path in model_dir.iterdir()) if model_dir.exists() else []
        raise FileNotFoundError(
            f"No actor checkpoint found in {model_dir}. "
            f"Available files: {available}"
        )

    if isinstance(model, dict) and "net_mean" in model:
        memory = load_long_term_memory(model_dir, models_module=models_module, map_location=map_location)
        n_traj = int(model.get("meta", {}).get("n_traj", 64))
        model = LoadedHybridDictAgent(
            net_mean=model["net_mean"],
            net_logstd=model.get("net_logstd", None),
            memory=memory,
            n_traj=n_traj,
        )
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module in {model_path}, got {type(model)}")

    attach_hybridrarl_memory(model, model_dir, models_module=models_module, map_location=map_location)

    if hasattr(model, "set_prev_units") and hasattr(model, "prev_unit_paths"):
        nested_paths = [
            relocate_saved_model_dir(Path(path), search_roots)
            for path in getattr(model, "prev_unit_paths", [])
        ]
        nested_units = [
            load_policy_unit(
                path,
                models_module=models_module,
                map_location=map_location,
                search_roots=search_roots,
                loading_stack=loading_stack,
            )
            for path in nested_paths
        ]
        model.prev_unit_paths = [str(path) for path in nested_paths]
        model.set_prev_units(nested_units)

    model = model.to(map_location).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    loading_stack.remove(model_dir)
    return model


def load_componet_agent(
    args,
    exp_dir: Path,
    tasks_module,
    model_dir: Path,
    load_task_id: int,
    obs_dim: int,
    act_dim: int,
    map_location,
) -> nn.Module:
    if not (model_dir / "net_mean.pt").exists() or not (model_dir / "net_logstd.pt").exists():
        return load_policy_unit(
            model_dir,
            models_module=importlib.import_module("models"),
            map_location=map_location,
            search_roots=ordered_unique_paths([model_dir.parent, resolve_root(args.model_root, exp_dir)]),
        )

    try:
        from componet import CompoNet, FirstModuleWrapper
    except ImportError:
        halfcheetah_dir = Path(__file__).resolve().parent / "halfcheetah"
        if str(halfcheetah_dir) not in sys.path:
            sys.path.insert(0, str(halfcheetah_dir))
        try:
            from componet import CompoNet, FirstModuleWrapper
        except ImportError as exc:
            raise ImportError(
                "Cannot import `componet`. CompoNet evaluation needs the "
                "CompoNet package used during training."
            ) from exc

    prev_dirs = resolve_componet_prev_dirs(args, exp_dir, tasks_module, load_task_id)
    if len(prev_dirs) == 0:
        raise ValueError(
            f"CompoNet task {load_task_id} has no previous units. "
            "Use the task 0 simple checkpoint, or pass --componet-prev-tasks."
        )

    first_model = safe_torch_load(prev_dirs[0] / "model.pt", map_location=map_location)
    prev_units = [
        FirstModuleWrapper(
            model=first_model,
            ret_probs=False,
            transform_output=TakeFirst(),
        )
    ]
    for prev_dir in prev_dirs[1:]:
        prev_units.append(safe_torch_load(prev_dir / "net_mean.pt", map_location=map_location))

    net_mean = CompoNet(
        previous_units=prev_units,
        input_dim=int(obs_dim),
        hidden_dim=256,
        out_dim=int(act_dim),
        internal_policy=mlp_net(int(obs_dim) + 256, int(act_dim)),
        ret_probs=False,
        device=map_location,
    )
    saved_net_mean = safe_torch_load(model_dir / "net_mean.pt", map_location=map_location)
    load_result = net_mean.load_state_dict(saved_net_mean.state_dict(), strict=False)
    if load_result.unexpected_keys:
        print(f"*** CompoNet unexpected keys while loading {model_dir}: {load_result.unexpected_keys}")

    net_logstd = safe_torch_load(model_dir / "net_logstd.pt", map_location=map_location)
    model = LoadedCompoNetAgent(net_mean=net_mean, net_logstd=net_logstd).to(map_location).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    print("*** CompoNet prev units:")
    for index, prev_dir in enumerate(prev_dirs):
        print(f"    {index}: {prev_dir}")
    return model


def load_evaluation_model(
    args,
    exp_dir: Path,
    tasks_module,
    models_module,
    model_dir: Path,
    load_task_id: int,
    obs_dim: int,
    act_dim: int,
    map_location,
    search_roots: Sequence[Path],
) -> nn.Module:
    actual_model_name = model_name_from_dir(model_dir.name) or checkpoint_model_name(args.model_name)
    if actual_model_name == "packnet":
        return load_packnet_agent(model_dir, map_location=map_location).to(map_location).eval()
    if actual_model_name == "prognet":
        return load_prognet_agent(
            args=args,
            exp_dir=exp_dir,
            model_dir=model_dir,
            load_task_id=load_task_id,
            map_location=map_location,
        ).to(map_location).eval()
    if actual_model_name == "componet":
        return load_componet_agent(
            args=args,
            exp_dir=exp_dir,
            tasks_module=tasks_module,
            model_dir=model_dir,
            load_task_id=load_task_id,
            obs_dim=obs_dim,
            act_dim=act_dim,
            map_location=map_location,
        )
    return load_policy_unit(
        model_dir,
        models_module=models_module,
        map_location=map_location,
        search_roots=search_roots,
    )


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


def model_kwargs(model: nn.Module, device) -> Dict[str, Dict[str, torch.Tensor]]:
    memory = getattr(model, "memory", None)
    if memory is not None and hasattr(memory, "__len__") and len(memory) > 0 and hasattr(memory, "current"):
        return {"memory_batch": memory.current(device=device)}
    return {}


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
    mean, raw_logstd = first_two_outputs(model(obs_t, **model_kwargs(model, device)))

    if deterministic or raw_logstd is None:
        squashed = torch.tanh(mean)
    else:
        logstd = torch.tanh(raw_logstd)
        logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (logstd + 1)
        normal = torch.distributions.Normal(mean, logstd.exp())
        squashed = torch.tanh(normal.sample())

    action = squashed * action_scale + action_bias
    return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


def evaluate_one_task(args, exp_dir: Path, tasks_module, models_module, task_id: int, device) -> Dict:
    load_task_id = checkpoint_task_id(args, task_id)
    env_task_id = args.eval_task_id if args.eval_task_id is not None else task_id
    env = tasks_module.get_task(env_task_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    model_dir = resolve_model_dir(args, exp_dir, task_id)
    load_model_name = model_name_from_dir(model_dir.name) or checkpoint_model_name(args.model_name)
    model_candidates = [model_dir]
    if not args.model_dir and normalized_model_name(args.model_name) == "hybridrarl":
        # A stale or truncated HybridRARL checkpoint must not prevent the
        # requested hybrid fallback from being considered.
        for candidate in candidate_model_dirs(args, exp_dir, task_id):
            if has_saved_actor(candidate):
                candidate = candidate.resolve()
                if candidate not in model_candidates:
                    model_candidates.append(candidate)

    model = None
    last_load_error = None
    for candidate in model_candidates:
        search_roots = ordered_unique_paths(
            [
                candidate.parent,
                resolve_root(args.model_root, exp_dir),
                exp_dir / "agents",
                exp_dir / "agents" / "hybridrarl_prev_units",
            ]
        )
        try:
            model = load_evaluation_model(
                args=args,
                exp_dir=exp_dir,
                tasks_module=tasks_module,
                models_module=models_module,
                model_dir=candidate,
                load_task_id=load_task_id,
                obs_dim=obs_dim,
                act_dim=act_dim,
                map_location=device,
                search_roots=search_roots,
            )
            model_dir = candidate
            load_model_name = model_name_from_dir(model_dir.name) or checkpoint_model_name(args.model_name)
            break
        except (FileNotFoundError, RuntimeError, EOFError, OSError, pickle.UnpicklingError) as exc:
            last_load_error = exc
            if normalized_model_name(args.model_name) != "hybridrarl" or args.model_dir:
                raise
            print(f"*** Skip unusable checkpoint: {candidate}")
            print(f"    {type(exc).__name__}: {exc}")

    if model is None:
        raise RuntimeError(
            f"No usable HybridRARL/hybrid checkpoint found for task {task_id}."
        ) from last_load_error
    action_scale = torch.as_tensor(
        (env.action_space.high - env.action_space.low) / 2.0,
        dtype=torch.float32,
        device=device,
    )
    action_bias = torch.as_tensor(
        (env.action_space.high + env.action_space.low) / 2.0,
        dtype=torch.float32,
        device=device,
    )
    max_steps = int(args.max_steps) if int(args.max_steps) > 0 else infer_episode_limit(env, args.fallback_max_steps)

    returns = []
    successes = []
    lengths = []
    task_name = tasks_module.get_task_name(env_task_id)

    print(f"\n*** Eval task {task_id} model on env task {env_task_id}: {task_name}")
    if load_task_id != task_id or load_model_name != args.model_name:
        print(f"*** Checkpoint source: task {load_task_id}, model `{load_model_name}`")
    print(f"*** Model: {model_dir}")

    for ep in range(int(args.episodes)):
        episode_seed = int(args.test_seed) + ep
        obs = reset_env(env, seed=episode_seed)
        ep_return = 0.0
        ep_success = 0.0
        step = 0

        for step in range(1, max_steps + 1):
            action = select_action(
                model=model,
                obs=np.asarray(obs, dtype=np.float32),
                action_scale=action_scale,
                action_bias=action_bias,
                device=device,
                deterministic=bool(args.deterministic),
            )
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, info = step_env(env, action)
            ep_return += float(reward)
            ep_success = max(ep_success, float(info.get("success", 0.0)))
            if done:
                break

        returns.append(ep_return)
        successes.append(ep_success)
        lengths.append(step)
        print(
            f"episode={ep + 1:03d}, seed={episode_seed}, "
            f"return={ep_return:.4f}, success={ep_success:.4f}, length={step}"
        )

    env.close()
    return {
        "benchmark": args.benchmark,
        "model_name": args.model_name,
        "checkpoint_model_name": load_model_name,
        "model_task_id": int(task_id),
        "checkpoint_task_id": int(load_task_id),
        "eval_task_id": int(env_task_id),
        "task_name": task_name,
        "train_seed": int(args.train_seed),
        "test_seed": int(args.test_seed),
        "episodes": int(args.episodes),
        "deterministic": bool(args.deterministic),
        "model_dir": str(model_dir),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_success": float(np.mean(successes)),
        "std_success": float(np.std(successes)),
        "mean_length": float(np.mean(lengths)),
        "returns": [float(x) for x in returns],
        "successes": [float(x) for x in successes],
        "lengths": [int(x) for x in lengths],
    }


def ordered_unique_paths(paths: Iterable[Path]) -> List[Path]:
    out = []
    seen = set()
    for path in paths:
        path = Path(path).expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            out.append(path)
    return out


def write_results(records: List[Dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark",
        "model_name",
        "checkpoint_model_name",
        "model_task_id",
        "checkpoint_task_id",
        "eval_task_id",
        "task_name",
        "train_seed",
        "test_seed",
        "episodes",
        "deterministic",
        "model_dir",
        "mean_return",
        "std_return",
        "mean_success",
        "std_success",
        "mean_length",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fieldnames})

    output_json = output_csv.with_suffix(".json")
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, sort_keys=True)


def overall_success_summary(records: List[Dict]) -> Dict[str, float]:
    task_mean_successes = np.asarray(
        [float(record["mean_success"]) for record in records],
        dtype=np.float64,
    )
    task_mean_returns = np.asarray(
        [float(record["mean_return"]) for record in records],
        dtype=np.float64,
    )
    episode_successes = np.asarray(
        [
            float(success)
            for record in records
            for success in record.get("successes", [])
        ],
        dtype=np.float64,
    )
    episode_returns = np.asarray(
        [
            float(ret)
            for record in records
            for ret in record.get("returns", [])
        ],
        dtype=np.float64,
    )
    if task_mean_successes.size == 0:
        return {
            "num_tasks": 0,
            "num_episodes": 0,
            "mean_return_across_tasks": 0.0,
            "var_return_across_tasks": 0.0,
            "mean_return_across_episodes": 0.0,
            "var_return_across_episodes": 0.0,
            "mean_success_across_tasks": 0.0,
            "var_success_across_tasks": 0.0,
            "mean_success_across_episodes": 0.0,
            "var_success_across_episodes": 0.0,
        }
    return {
        "num_tasks": int(task_mean_successes.size),
        "num_episodes": int(episode_successes.size),
        "mean_return_across_tasks": float(np.mean(task_mean_returns)),
        "var_return_across_tasks": float(np.var(task_mean_returns)),
        "mean_return_across_episodes": float(np.mean(episode_returns)) if episode_returns.size > 0 else 0.0,
        "var_return_across_episodes": float(np.var(episode_returns)) if episode_returns.size > 0 else 0.0,
        "mean_success_across_tasks": float(np.mean(task_mean_successes)),
        "var_success_across_tasks": float(np.var(task_mean_successes)),
        "mean_success_across_episodes": float(np.mean(episode_successes)) if episode_successes.size > 0 else 0.0,
        "var_success_across_episodes": float(np.var(episode_successes)) if episode_successes.size > 0 else 0.0,
    }


def write_overall_summary(summary: Dict[str, float], output_csv: Path) -> Path:
    summary_path = output_csv.with_name(output_csv.stem + "__summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved Meta-World or HalfCheetah models.")
    parser.add_argument("--benchmark", choices=["meta-world", "halfcheetah"], required=True)
    parser.add_argument(
        "--model-name",
        required=True,
        help=(
            "Model name in saved dir, e.g. simple, hybridrarl, or baseline. "
            "Use hybridrarl-no-retrieval to evaluate selective module reuse "
            "and automatically fall back to simple when a task has no saved prev-unit actor; "
            "use hybridrarl-no-selective-reuse to evaluate only the original HybridRARL actor; "
            "use hybridrarl-random-retrieval to evaluate random coarse retrieval."
        ),
    )
    parser.add_argument("--train-seed", type=int, required=True, help="Random seed used when training the model.")
    parser.add_argument("--test-seed", type=int, required=True, help="Environment reset seed used for testing.")
    parser.add_argument(
        "--episodes",
        "--episode",
        dest="episodes",
        type=int,
        required=True,
        help="Number of test episodes per task. `--episode` is retained as a compatibility alias.",
    )
    parser.add_argument("--tasks", type=int, nargs="*", default=None, help="Model task ids to evaluate. Defaults to all first-stage tasks.")
    parser.add_argument("--eval-task-id", type=int, default=None, help="Override test env task id. Default: env task id equals model task id.")
    parser.add_argument(
        "--source-task-id",
        type=int,
        default=None,
        help=(
            "Checkpoint task id used by finetune-1 or baseline. For baseline, "
            "this is the task whose simple actor is tested on every task. "
            "Default: final task in the benchmark evaluation set."
        ),
    )
    parser.add_argument("--model-root", type=Path, default=Path("agents"), help="Root containing saved model directories.")
    parser.add_argument("--model-dir", default=None, help="Direct model dir, optionally with {task_id}, {model_name}, {train_seed}.")
    parser.add_argument("--scenario", default="", help="HalfCheetah scenario subdir, e.g. forgetting or transfer.")
    parser.add_argument(
        "--componet-prev-tasks",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Manually specify previous task ids for CompoNet reconstruction. "
            "The first id is loaded as simple, later ids as componet."
        ),
    )
    parser.add_argument("--exp-name", default=None, help="Experiment name in task_i__model__exp__seed. If omitted, common names are tried.")
    parser.add_argument("--exp-name-candidates", nargs="*", default=None, help="Extra experiment names to try after defaults.")
    parser.add_argument("--max-steps", type=int, default=0, help="Episode cap. 0 uses env limit, then fallback.")
    parser.add_argument("--fallback-max-steps", type=int, default=500)
    parser.add_argument("--deterministic", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--prefer-prev-units",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "When --model-name is hybridrarl/no-retrieval, first try saved prev-units variants "
            "and fall back to the base model if no reusable-module checkpoint exists. "
            "hybridrarl-no-retrieval always uses its simple fallback instead of a bare "
            "no-retrieval actor."
        ),
    )
    parser.add_argument("--recursive-search", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--cuda", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp_dir = experiment_dir(args.benchmark)
    tasks_module, models_module = import_experiment_modules(exp_dir)
    task_names = getattr(tasks_module, "tasks", [])
    task_ids = args.tasks if args.tasks is not None else default_task_ids(args.benchmark, task_names)
    if normalized_model_name(args.model_name) == "finetune-1" and args.source_task_id is None:
        if len(task_ids) == 0:
            raise ValueError("finetune-1 cannot infer --source-task-id from an empty task list.")
        args.source_task_id = max(int(task_id) for task_id in task_ids)
    if normalized_model_name(args.model_name) == "baseline" and args.source_task_id is None:
        if len(task_ids) == 0:
            raise ValueError("baseline cannot infer the final task id from an empty evaluation task list.")
        # Meta-World keeps a duplicated task list in tasks.py, but the
        # evaluator's default task set is task 0-9. Use that set's final id.
        args.source_task_id = max(int(task_id) for task_id in default_task_ids(args.benchmark, task_names))
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    set_seed(args.test_seed)
    output = args.output
    if output is None:
        output = (
            exp_dir
            / "analysis"
            / "eval_results"
            / f"{args.benchmark}__{args.model_name}__train{args.train_seed}__test{args.test_seed}.csv"
        )
    elif not output.is_absolute():
        output = (Path.cwd() / output).resolve()

    print(f"*** Benchmark: {args.benchmark}")
    print(f"*** Model name: {args.model_name}")
    print(f"*** Model name search priority: {candidate_model_names(args)}")
    if normalized_model_name(args.model_name) == "hybridrarl-no-retrieval":
        print(
            "*** HybridRARL no-retrieval: evaluating a selective prev-unit actor when "
            "available; otherwise the corresponding simple actor."
        )
    if normalized_model_name(args.model_name) == "hybridrarl-no-selective-reuse":
        print(
            "*** HybridRARL no-selective-reuse: evaluating only the base HybridRARL "
            "checkpoint; prev-units are disabled."
        )
    if normalized_model_name(args.model_name) == "hybridrarl-random-retrieval":
        print(
            "*** HybridRARL random-retrieval: evaluating the per-task HybridRARL "
            "checkpoint with random coarse retrieval and no prev-units."
        )
    if normalized_model_name(args.model_name) == "baseline":
        print(
            f"*** Baseline: using simple checkpoint from final task "
            f"{args.source_task_id} on every evaluation task"
        )
    if normalized_model_name(args.model_name) == "finetune-1":
        print(f"*** Finetune-1 checkpoint task id: {args.source_task_id}")
    if normalized_model_name(args.model_name) == "finetune-n":
        print("*** Finetune-N: task 0 may use the initial simple checkpoint; later tasks use finetune checkpoints.")
    print(f"*** Train seed: {args.train_seed}")
    print(f"*** Test seed: {args.test_seed}")
    print(f"*** Episodes: {args.episodes}")
    print(f"*** Device: {device}")

    records = []
    for task_id in task_ids:
        records.append(evaluate_one_task(args, exp_dir, tasks_module, models_module, int(task_id), device))

    print("\nSummary")
    for record in records:
        print(
            f"task={record['model_task_id']} ({record['task_name']}), "
            f"ckpt_task={record['checkpoint_task_id']}, "
            f"ckpt_model={record['checkpoint_model_name']}, "
            f"return={record['mean_return']:.4f}±{record['std_return']:.4f}, "
            f"success={record['mean_success']:.4f}±{record['std_success']:.4f}, "
            f"length={record['mean_length']:.2f}"
        )

    overall_summary = overall_success_summary(records)
    print(
        "\nOverall Performance"
        f"\nmean_return_across_tasks={overall_summary['mean_return_across_tasks']:.6f}, "
        f"var_return_across_tasks={overall_summary['var_return_across_tasks']:.6f}"
        f"\nmean_return_across_episodes={overall_summary['mean_return_across_episodes']:.6f}, "
        f"var_return_across_episodes={overall_summary['var_return_across_episodes']:.6f}"
        f"\nmean_success_across_tasks={overall_summary['mean_success_across_tasks']:.6f}, "
        f"var_success_across_tasks={overall_summary['var_success_across_tasks']:.6f}"
        f"\nmean_success_across_episodes={overall_summary['mean_success_across_episodes']:.6f}, "
        f"var_success_across_episodes={overall_summary['var_success_across_episodes']:.6f}"
    )

    output = Path(output)
    write_results(records, output)
    summary_path = write_overall_summary(overall_summary, output)
    print(f"\n*** Saved CSV: {output}")
    print(f"*** Saved JSON: {output.with_suffix('.json')}")
    print(f"*** Saved summary JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
