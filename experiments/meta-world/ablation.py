#!/usr/bin/env python3
"""Train HybridRARL previous-unit transfer models.

The default mode selects previous units from a lower-triangular distance CSV:
every previous task T_i with distance(T_k, T_i) < threshold becomes a prev
unit.  ``--reuse-all-prev-units`` is the corresponding ablation: for T_k it
uses every T_0 .. T_(k-1) as a prev unit, without loading a CSV or computing a
similarity score.

If a previous task already has a transfer model from this script, that model is
used as a frozen prev unit. Otherwise the already-trained first-stage base
HybridRARL model is loaded as a frozen prev unit; base HybridRARL models are
never trained by this script.
"""

import argparse
import csv
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from models import HybridRARLAgent, HybridRARLPrevUnitsAgent, NoRetrievalAgent, NoRetrievalPrevUnitsAgent, ReservoirTrajectoryMemory, shared
from tasks import get_task, get_task_name


LOG_STD_MAX = 2
LOG_STD_MIN = -20


def make_env(task_id: int):
    def thunk():
        env = get_task(task_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.fc = shared(int(obs_dim) + int(act_dim))
        self.fc_out = nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=1)
        x = self.fc(x)
        return self.fc_out(x)


class Actor(nn.Module):
    def __init__(self, envs: gym.vector.SyncVectorEnv, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (envs.single_action_space.high - envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (envs.single_action_space.high + envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, obs: torch.Tensor, **kwargs):
        mean, log_std = self.model(obs, **kwargs)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, obs: torch.Tensor, **kwargs):
        mean, log_std = self(obs, **kwargs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed: int, torch_deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def parse_task_label(label: str) -> Dict:
    match = re.search(r"(?:task|t)[_\s-]*(\d+)", label.strip(), flags=re.IGNORECASE)
    task_id = int(match.group(1)) if match is not None else None
    task_name = label.split(":", 1)[1].strip() if ":" in label else label.strip()
    return {"id": task_id, "name": task_name, "label": label.strip()}


def read_distance_csv(path: Path) -> Tuple[List[Dict], List[List[float]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    if len(rows) < 2:
        raise ValueError(f"{path}: expected a square distance matrix CSV.")

    tasks = [parse_task_label(label) for label in rows[0][1:]]
    matrix: List[List[float]] = []
    for row_index, row in enumerate(rows[1:], start=2):
        if len(row) != len(tasks) + 1:
            raise ValueError(f"{path}: row {row_index} has wrong length.")
        matrix.append([float(value) for value in row[1:]])

    if len(matrix) != len(tasks):
        raise ValueError(f"{path}: matrix should be square.")
    return tasks, matrix


def related_previous_tasks(
    task_id: int,
    tasks: Sequence[Dict],
    matrix: Sequence[Sequence[float]],
    threshold: float,
) -> List[Dict]:
    row_index = None
    for index, task in enumerate(tasks):
        if task["id"] == task_id:
            row_index = index
            break
    if row_index is None:
        raise KeyError(f"task {task_id} is not present in the CSV.")

    related = []
    for col_index in range(row_index):
        distance = float(matrix[row_index][col_index])
        if distance < float(threshold):
            related.append(
                {
                    "task_id": int(tasks[col_index]["id"]),
                    "task_name": tasks[col_index]["name"],
                    "distance": distance,
                    "csv_row_index": row_index,
                    "csv_col_index": col_index,
                }
            )
    return related


def all_previous_tasks(task_id: int) -> List[Dict]:
    """Return every earlier Meta-World task as a prev unit for the ablation."""
    return [
        {
            "task_id": prev_task_id,
            "task_name": get_task_name(prev_task_id),
            "distance": None,
            "selection_mode": "all-previous",
        }
        for prev_task_id in range(task_id)
    ]


def format_distance(distance: Optional[float], precision: int = 8) -> str:
    """Format CSV distances while keeping all-prev-units metadata explicit."""
    if distance is None:
        return "n/a"
    return f"{float(distance):.{precision}f}"


def model_dir_name(task_id: int, model_type: str, exp_name: str, seed: int) -> str:
    return f"task_{task_id}__{model_type}__{exp_name}__{seed}"


def base_model_dir(args, task_id: int) -> Path:
    return Path(args.base_dir) / model_dir_name(
        task_id=task_id,
        model_type=args.base_model_type,
        exp_name=args.base_exp_name,
        seed=args.seed,
    )


def simple_exp_names(args) -> List[str]:
    exp_names = []
    for name in [args.simple_exp_name, "SAC", "run_sac"]:
        if name and name not in exp_names:
            exp_names.append(name)
    if args.simple_exp_name_candidates:
        for name in args.simple_exp_name_candidates:
            if name and name not in exp_names:
                exp_names.append(name)
    return exp_names


def simple_model_dir_candidates(args, task_id: int) -> List[Path]:
    simple_dir = Path(args.simple_dir) if args.simple_dir is not None else Path(args.base_dir)

    return [
        simple_dir
        / model_dir_name(
            task_id=task_id,
            model_type=args.simple_model_type,
            exp_name=exp_name,
            seed=args.seed,
        )
        for exp_name in simple_exp_names(args)
    ]


def transfer_model_dir(args, task_id: int) -> Path:
    return Path(args.transfer_save_dir) / model_dir_name(
        task_id=task_id,
        model_type=args.transfer_model_type,
        exp_name=args.transfer_exp_name,
        seed=args.seed,
    )


def resolve_prev_unit_dir(args, task_id: int) -> Tuple[Path, str]:
    if getattr(args, "prev_unit_source", "hybridrarl") == "simple":
        for simple_dir in simple_model_dir_candidates(args, task_id):
            if (simple_dir / "model.pt").exists():
                return simple_dir, "simple"
        tried = "\n".join(
            f"    {simple_dir / 'model.pt'}"
            for simple_dir in simple_model_dir_candidates(args, task_id)
        )
        raise FileNotFoundError(
            f"No simple prev unit model found for task {task_id}. Tried:\n{tried}"
        )

    if getattr(args, "ablation_no_retrieval", False):
        for model_type in ["no-retrieval-prev-units", "no-retrieval"]:
            ablation_dir = Path(args.transfer_save_dir) / model_dir_name(
                task_id=task_id,
                model_type=model_type,
                exp_name=args.transfer_exp_name,
                seed=args.seed,
            )
            if (ablation_dir / "model.pt").exists():
                return ablation_dir, model_type

    transfer_dir = transfer_model_dir(args, task_id)
    if (transfer_dir / "model.pt").exists():
        return transfer_dir, "transfer"
    base_dir = base_model_dir(args, task_id)
    if (base_dir / "model.pt").exists():
        return base_dir, "base"
    raise FileNotFoundError(
        f"No prev unit model found for task {task_id}. Tried:\n"
        f"    {transfer_dir / 'model.pt'}\n"
        f"    {base_dir / 'model.pt'}"
    )


def attach_memory_if_needed(model: nn.Module, model_dir: Path, map_location) -> None:
    if not hasattr(model, "memory"):
        return
    if getattr(model, "memory", None) is not None and len(getattr(model, "memory")) > 0:
        return
    memory_path = model_dir / "hybridrarl_memory.pt"
    if memory_path.exists():
        model.memory = ReservoirTrajectoryMemory.load(model_dir, map_location=map_location)


def relocate_saved_model_dir(model_dir: Path, args) -> Path:
    model_dir = Path(model_dir).expanduser()
    if (model_dir / "model.pt").exists():
        return model_dir.resolve()

    for root in (Path(args.transfer_save_dir), Path(args.base_dir)):
        relocated = root / model_dir.name
        if (relocated / "model.pt").exists():
            print(f"*** Relocated prev unit `{model_dir}` -> `{relocated}`")
            return relocated.resolve()

    return model_dir.resolve()


def load_policy_unit(
    model_dir: Path,
    map_location,
    args,
    loading_stack: Optional[set] = None,
) -> nn.Module:
    loading_stack = set() if loading_stack is None else loading_stack
    model_dir = relocate_saved_model_dir(model_dir, args)
    if model_dir in loading_stack:
        raise RuntimeError(f"Recursive prev-unit cycle detected at {model_dir}")

    model_path = model_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"No model.pt found in prev unit dir: {model_dir}")

    loading_stack.add(model_dir)
    model = safe_torch_load(model_path, map_location=map_location)
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module in {model_path}, got {type(model)}")

    attach_memory_if_needed(model, model_dir, map_location=map_location)

    if isinstance(model, (HybridRARLPrevUnitsAgent, NoRetrievalPrevUnitsAgent)):
        nested_paths = [
            relocate_saved_model_dir(Path(path), args)
            for path in getattr(model, "prev_unit_paths", [])
        ]
        nested_units = [
            load_policy_unit(
                path,
                map_location=map_location,
                args=args,
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


def sample_memory_batch(memory: ReservoirTrajectoryMemory, n_traj: int, device) -> Dict[str, torch.Tensor]:
    assert len(memory) > 0, "HybridRARL memory is empty."
    batch = memory.sample_trajectories(
        n_traj=min(int(n_traj), len(memory)),
        device=device,
        replace=False,
    )
    return batch.as_dict()


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def serializable_args(args) -> Dict:
    result = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def save_checkpoint(
    dirname: Path,
    actor: Actor,
    qf1: SoftQNetwork,
    qf2: SoftQNetwork,
    qf1_target: SoftQNetwork,
    qf2_target: SoftQNetwork,
    actor_optimizer,
    q_optimizer,
    args,
    task_id: int,
    global_step: int,
    alpha: float,
    log_alpha=None,
    alpha_optimizer=None,
    prev_unit_meta: Optional[List[Dict]] = None,
) -> None:
    dirname.mkdir(parents=True, exist_ok=True)
    actor.model.save(str(dirname), save_memory=True)
    checkpoint = {
        "type": "hybridrarl_prev_units_sac_checkpoint",
        "task_id": int(task_id),
        "task_name": get_task_name(task_id),
        "global_step": int(global_step),
        "args": serializable_args(args),
        "actor_state_dict": actor.state_dict(),
        "actor_model_state_dict": actor.model.state_dict(),
        "qf1_state_dict": qf1.state_dict(),
        "qf2_state_dict": qf2.state_dict(),
        "qf1_target_state_dict": qf1_target.state_dict(),
        "qf2_target_state_dict": qf2_target.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "q_optimizer_state_dict": q_optimizer.state_dict(),
        "alpha": float(alpha),
        "prev_units": prev_unit_meta or [],
    }
    if log_alpha is not None:
        checkpoint["log_alpha"] = log_alpha.detach().cpu()
    if alpha_optimizer is not None:
        checkpoint["alpha_optimizer_state_dict"] = alpha_optimizer.state_dict()
    torch.save(checkpoint, dirname / "sac_checkpoint.pt")
    with (dirname / "prev_units.json").open("w", encoding="utf-8") as f:
        json.dump(prev_unit_meta or [], f, indent=2, sort_keys=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HybridRARL prev-unit transfer models."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=(
            "Square task-distance CSV used for threshold selection. Required unless "
            "--reuse-all-prev-units is set."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument(
        "--reuse-all-prev-units",
        action="store_true",
        help=(
            "Ablation: use every earlier task T_0..T_(k-1) as frozen prev-units "
            "for task T_k. This bypasses CSV loading and all similarity calculations."
        ),
    )
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="*",
        default=None,
        help="Current tasks to train. Defaults to task 1-9; task 0 is excluded.",
    )
    parser.add_argument("--base-dir", type=Path, default=Path("agents"))
    parser.add_argument("--transfer-save-dir", type=Path, default=Path("agents/hybridrarl_prev_units"))
    parser.add_argument("--base-model-type", default="hybridrarl")
    parser.add_argument("--base-exp-name", default="SAC_hybridRAGRL")
    parser.add_argument(
        "--prev-unit-source",
        choices=["hybridrarl", "simple"],
        default="hybridrarl",
        help=(
            "hybridrarl reuses first-stage HybridRARL/transfer modules; "
            "simple reuses already-trained simple SAC modules selected by the CSV."
        ),
    )
    parser.add_argument(
        "--simple-dir",
        type=Path,
        default=None,
        help="Directory containing task_*__simple__... models. Defaults to --base-dir.",
    )
    parser.add_argument("--simple-model-type", default="simple")
    parser.add_argument("--simple-exp-name", default="SAC")
    parser.add_argument(
        "--simple-exp-name-candidates",
        nargs="*",
        default=None,
        help=(
            "Extra simple experiment names to try when resolving prev units. "
            "The script already tries --simple-exp-name, SAC, and run_sac. "
            "Do not pass 20k state-sampling teachers here unless you explicitly want to reuse them."
        ),
    )
    parser.add_argument(
        "--transfer-model-type",
        default=None,
        help=(
            "Model-type component of output directories. Defaults to "
            "hybridrarl-prev-units for CSV selection and "
            "hybridrarl-ablation-no-select for --reuse-all-prev-units."
        ),
    )
    parser.add_argument(
        "--transfer-exp-name",
        default=None,
        help=(
            "Experiment-name component of output directories. Defaults to "
            "csv_threshold_transfer for CSV selection and all_prev_units for the "
            "all-previous-units ablation."
        ),
    )
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--save-every", type=int, default=10_000)
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
    parser.add_argument("--torch-deterministic", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--memory-capacity", type=int, default=100)
    parser.add_argument("--traj-len", type=int, default=20)
    parser.add_argument("--top-r", type=int, default=5)
    parser.add_argument("--traj-hidden-dim", type=int, default=128)
    parser.add_argument("--att-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--prev-att-dim", type=int, default=128)
    parser.add_argument("--policy-hidden-dim", type=int, default=256)
    parser.add_argument("--policy-layers", type=int, default=4)
    parser.add_argument("--coarse-metric", choices=["cosine", "l2"], default="cosine")
    parser.add_argument(
        "--ablation-no-retrieval",
        default=False,
        action="store_true",
        help="Remove trajectory-memory retrieval and use only MLP plus optional selective modular reuse.",
    )
    parser.add_argument(
        "--no-run",
        default=False,
        action="store_true",
        help="Print selected prev units without launching SAC training.",
    )
    return parser.parse_args()


def build_model(
    args,
    obs_dim: int,
    act_dim: int,
    memory: Optional[ReservoirTrajectoryMemory],
    prev_units: List[nn.Module],
    prev_unit_paths: List[Path],
    prev_task_ids: List[int],
    device,
) -> nn.Module:
    if args.ablation_no_retrieval:
        if prev_units:
            model = NoRetrievalPrevUnitsAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                prev_units=prev_units,
                prev_unit_paths=[str(path) for path in prev_unit_paths],
                prev_att_dim=args.prev_att_dim,
                policy_hidden_dim=args.policy_hidden_dim,
                policy_layers=args.policy_layers,
            ).to(device)
            model.prev_unit_task_ids = [int(task_id) for task_id in prev_task_ids]
            return model
        return NoRetrievalAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            policy_hidden_dim=args.policy_hidden_dim,
            policy_layers=args.policy_layers,
        ).to(device)

    model = HybridRARLPrevUnitsAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        memory=memory,
        prev_units=prev_units,
        prev_unit_paths=[str(path) for path in prev_unit_paths],
        memory_capacity=args.memory_capacity,
        traj_len=args.traj_len,
        top_r=args.top_r,
        traj_hidden_dim=args.traj_hidden_dim,
        att_dim=args.att_dim,
        value_dim=args.value_dim,
        prev_att_dim=args.prev_att_dim,
        policy_hidden_dim=args.policy_hidden_dim,
        policy_layers=args.policy_layers,
        coarse_metric=args.coarse_metric,
    ).to(device)
    model.prev_unit_task_ids = [int(task_id) for task_id in prev_task_ids]
    return model


def output_model_type(args, has_prev_units: bool) -> str:
    if args.ablation_no_retrieval:
        if args.prev_unit_source == "simple" and has_prev_units:
            return "no-retrieval-simple-prev-units"
        return "no-retrieval-prev-units" if has_prev_units else "no-retrieval"
    if args.reuse_all_prev_units:
        return args.transfer_model_type
    if args.prev_unit_source == "simple" and has_prev_units:
        return "hybridrarl-simple-prev-units"
    return args.transfer_model_type


def train_task(
    args,
    task_id: int,
    related: List[Dict],
    device,
) -> None:
    task_seed = int(args.seed) + int(task_id) * 10_000
    set_seed(task_seed, args.torch_deterministic)

    prev_unit_paths = []
    prev_units = []
    prev_unit_meta = []
    for item in related:
        prev_dir, source_type = resolve_prev_unit_dir(args, item["task_id"])
        prev_unit_paths.append(prev_dir.resolve())
        prev_units.append(load_policy_unit(prev_dir, map_location=device, args=args))
        prev_unit_meta.append(
            {
                "task_id": int(item["task_id"]),
                "task_name": item["task_name"],
                "distance": (
                    None if item.get("distance") is None else float(item["distance"])
                ),
                "selection_mode": item.get("selection_mode", "csv-threshold"),
                "source_type": source_type,
                "model_dir": str(prev_dir.resolve()),
            }
        )

    if not prev_units and not args.ablation_no_retrieval:
        print(f"*** task {task_id}: no prev units; skipped.")
        return

    model_type = output_model_type(args, has_prev_units=bool(prev_units))

    run_name = model_dir_name(
        task_id=task_id,
        model_type=model_type,
        exp_name=args.transfer_exp_name,
        seed=args.seed,
    )
    print(f"\n*** Run name: {run_name} ***\n")
    print(f"*** Loading model `{model_type}` ***")
    if prev_unit_meta:
        print(f"*** Prev units for task {task_id} {get_task_name(task_id)} ***")
        for item in prev_unit_meta:
            selection = item["selection_mode"]
            distance = item["distance"]
            distance_text = format_distance(distance, precision=6)
            print(
                f"    task {item['task_id']} {item['task_name']} "
                f"selection={selection} distance={distance_text} "
                f"source={item['source_type']}"
            )
    else:
        print(f"*** No previous units available for task {task_id} {get_task_name(task_id)} ***")

    if args.no_run:
        return

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    writer.add_text(
        "prev_units",
        "|task|name|selection|distance|source|\n|-|-|-|-|-|\n%s"
        % (
            "\n".join(
                [
                    f"|{item['task_id']}|{item['task_name']}|"
                    f"{item['selection_mode']}|"
                    f"{format_distance(item['distance'])}|"
                    f"{item['source_type']}|"
                    for item in prev_unit_meta
                ]
            )
        ),
    )

    envs = gym.vector.SyncVectorEnv([make_env(task_id)])
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    envs.single_observation_space.dtype = np.float32

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    memory = None
    if not args.ablation_no_retrieval:
        memory = ReservoirTrajectoryMemory(
            capacity=args.memory_capacity,
            traj_len=args.traj_len,
            obs_dim=obs_dim,
            action_dim=act_dim,
            reward_dim=1,
            device="cpu",
        )
    model = build_model(
        args=args,
        obs_dim=obs_dim,
        act_dim=act_dim,
        memory=memory,
        prev_units=prev_units,
        prev_unit_paths=prev_unit_paths,
        prev_task_ids=[item["task_id"] for item in related],
        device=device,
    )
    actor = Actor(envs, model).to(device)
    qf1 = SoftQNetwork(obs_dim, act_dim).to(device)
    qf2 = SoftQNetwork(obs_dim, act_dim).to(device)
    qf1_target = SoftQNetwork(obs_dim, act_dim).to(device)
    qf2_target = SoftQNetwork(obs_dim, act_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)

    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        target_entropy = None
        log_alpha = None
        alpha = float(args.alpha)
        alpha_optimizer = None

    output_dir = Path(args.transfer_save_dir) / model_dir_name(
        task_id=task_id,
        model_type=model_type,
        exp_name=args.transfer_exp_name,
        seed=args.seed,
    )
    cache_states: List[np.ndarray] = []
    cache_actions: List[np.ndarray] = []
    cache_rewards: List[np.ndarray] = []
    obs, _ = envs.reset(seed=task_seed)
    start_time = time.time()
    actor_loss = torch.zeros((), device=device)
    alpha_loss = torch.zeros((), device=device)

    for global_step in range(int(args.total_timesteps)):
        if global_step < int(args.random_actions_end):
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                if args.ablation_no_retrieval:
                    actions, _, _ = actor.get_action(obs_tensor)
                else:
                    assert memory is not None and len(memory) > 0, (
                        "HybridRARLPrevUnits starts acting after random_actions_end, but memory is empty."
                    )
                    memory_batch = sample_memory_batch(memory, n_traj=args.memory_capacity, device=device)
                    actions, _, _ = actor.get_action(obs_tensor, memory_batch=memory_batch)
                actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if memory is not None:
            state_to_cache = np.asarray(obs[0], dtype=np.float32).reshape(-1)
            action_to_cache = np.asarray(actions[0], dtype=np.float32).reshape(-1)
            reward_to_cache = np.asarray([rewards[0]], dtype=np.float32).reshape(-1)
            cache_states.append(state_to_cache)
            cache_actions.append(action_to_cache)
            cache_rewards.append(reward_to_cache)
            if len(cache_states) == int(args.traj_len):
                memory.add_trajectory(
                    states=torch.as_tensor(np.stack(cache_states, axis=0), dtype=torch.float32),
                    actions=torch.as_tensor(np.stack(cache_actions, axis=0), dtype=torch.float32),
                    rewards=torch.as_tensor(np.stack(cache_rewards, axis=0), dtype=torch.float32),
                )
                cache_states.clear()
                cache_actions.clear()
                cache_rewards.clear()

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is None:
                    continue
                print(
                    f"global_step={global_step}, "
                    f"episodic_return={info['episode']['r']}, "
                    f"success={info.get('success', 0.0)}"
                )
                writer.add_scalar("charts/episodic_return", float(info["episode"]["r"]), global_step)
                writer.add_scalar("charts/episodic_length", float(info["episode"]["l"]), global_step)
                writer.add_scalar("charts/success", float(info.get("success", 0.0)), global_step)
                break

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and "final_observation" in infos:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        if global_step > int(args.learning_starts):
            data = rb.sample(int(args.batch_size))
            if args.ablation_no_retrieval:
                rag_kwargs = {}
            else:
                assert memory is not None
                rag_kwargs = {
                    "memory_batch": sample_memory_batch(
                        memory,
                        n_traj=args.memory_capacity,
                        device=device,
                    )
                }

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data.next_observations,
                    **rag_kwargs,
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - alpha * next_state_log_pi
                )
                next_q_value = data.rewards.flatten() + (
                    1.0 - data.dones.flatten()
                ) * float(args.gamma) * min_qf_next_target.view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % int(args.policy_frequency) == 0:
                for _ in range(int(args.policy_frequency)):
                    pi, log_pi, _ = actor.get_action(data.observations, **rag_kwargs)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=10.0)
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations, **rag_kwargs)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                        alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        alpha_optimizer.step()
                        alpha = log_alpha.exp().item()

            if global_step % int(args.target_network_frequency) == 0:
                soft_update(qf1, qf1_target, float(args.tau))
                soft_update(qf2, qf2_target, float(args.tau))

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if memory is not None:
                    writer.add_scalar("charts/hybridrarl_memory_size", len(memory), global_step)

                with torch.no_grad():
                    _, _, info = actor.model(data.observations, return_info=True, **rag_kwargs)
                    prev_weights = info.get("prev_attention_weights")
                    if prev_weights is not None and prev_weights.numel() > 0:
                        mean_weights = prev_weights.detach().mean(dim=0).cpu()
                        for unit_index, weight in enumerate(mean_weights.tolist()):
                            meta = prev_unit_meta[unit_index]
                            tag_name = re.sub(
                                r"[^A-Za-z0-9_]+",
                                "_",
                                f"task_{meta['task_id']}_{meta['task_name']}",
                            ).strip("_")
                            writer.add_scalar(f"att_{tag_name}", float(weight), global_step)

                    retrieval_weights = info.get("attention_weights")
                    if not args.ablation_no_retrieval and retrieval_weights is not None and retrieval_weights.numel() > 0:
                        mean_retrieval = retrieval_weights.detach().mean(dim=0).cpu()
                        for rank_index, weight in enumerate(mean_retrieval.tolist()):
                            writer.add_scalar(
                                f"att_memory_rank_{rank_index}",
                                float(weight),
                                global_step,
                            )

        current_step = global_step + 1
        if args.save_every > 0 and current_step % int(args.save_every) == 0:
            print(f"Saving checkpoint in `{output_dir}` at step {current_step}")
            save_checkpoint(
                dirname=output_dir,
                actor=actor,
                qf1=qf1,
                qf2=qf2,
                qf1_target=qf1_target,
                qf2_target=qf2_target,
                actor_optimizer=actor_optimizer,
                q_optimizer=q_optimizer,
                args=args,
                task_id=task_id,
                global_step=global_step,
                alpha=alpha,
                log_alpha=log_alpha,
                alpha_optimizer=alpha_optimizer,
                prev_unit_meta=prev_unit_meta,
            )

    save_checkpoint(
        dirname=output_dir,
        actor=actor,
        qf1=qf1,
        qf2=qf2,
        qf1_target=qf1_target,
        qf2_target=qf2_target,
        actor_optimizer=actor_optimizer,
        q_optimizer=q_optimizer,
        args=args,
        task_id=task_id,
        global_step=int(args.total_timesteps) - 1,
        alpha=alpha,
        log_alpha=log_alpha,
        alpha_optimizer=alpha_optimizer,
        prev_unit_meta=prev_unit_meta,
    )
    envs.close()
    writer.close()
    print(f"Saving trained agent in `{output_dir}` with name `{run_name}`")


def main() -> int:
    args = parse_args()
    if args.transfer_model_type is None:
        args.transfer_model_type = (
            "hybridrarl-ablation-no-select"
            if args.reuse_all_prev_units
            else "hybridrarl-prev-units"
        )
    if args.transfer_exp_name is None:
        args.transfer_exp_name = (
            "all_prev_units" if args.reuse_all_prev_units else "csv_threshold_transfer"
        )

    tasks = None
    matrix = None
    if args.reuse_all_prev_units:
        default_task_ids = list(range(1, 10))
    else:
        if args.csv is None:
            raise ValueError("--csv is required unless --reuse-all-prev-units is set.")
        tasks, matrix = read_distance_csv(args.csv.expanduser())
        available_task_ids = {
            int(task["id"]) for task in tasks if task["id"] is not None
        }
        default_task_ids = [
            task_id for task_id in range(1, 10) if task_id in available_task_ids
        ]

    task_ids = args.tasks if args.tasks is not None else default_task_ids
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    print(f"*** Seed: {args.seed}")
    print(
        "*** Prev-unit selection: all previous tasks (no similarity computation)"
        if args.reuse_all_prev_units
        else f"*** CSV: {args.csv}\n*** Threshold: {args.threshold}"
    )
    print(f"*** Prev unit source: {args.prev_unit_source}")
    if args.prev_unit_source == "simple":
        print(f"*** Simple model root: {args.simple_dir or args.base_dir}")
        print(f"*** Simple exp name priority: {simple_exp_names(args)}")
    print(f"*** Device: {device}")

    for task_id in sorted(int(task_id) for task_id in task_ids):
        if task_id not in range(1, 10):
            print(f"*** task {task_id}: skipped; this script only trains current tasks 1-9.")
            continue
        if args.reuse_all_prev_units:
            related = all_previous_tasks(task_id)
        else:
            related = related_previous_tasks(
                task_id=task_id,
                tasks=tasks,
                matrix=matrix,
                threshold=float(args.threshold),
            )
        if not related:
            print(f"*** task {task_id} {get_task_name(task_id)}: no previous units; skipped.")
            continue
        train_task(args=args, task_id=task_id, related=related, device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
