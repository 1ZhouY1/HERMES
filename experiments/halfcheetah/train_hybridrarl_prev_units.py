#!/usr/bin/env python3
import argparse
import json
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

from models import (
    HybridRARLAgent,
    HybridRARLPrevUnitsAgent,
    NoRetrievalAgent,
    NoRetrievalPrevUnitsAgent,
    ReservoirTrajectoryMemory,
    SimpleAgent,
    shared,
)
from tasks import SCENARIOS, get_task, get_task_name, tasks


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


def safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed: int, torch_deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def unique_scenario_sequence(scenario: str) -> List[str]:
    seen = set()
    sequence = []
    for task_name in SCENARIOS[scenario]:
        if task_name in seen:
            continue
        seen.add(task_name)
        sequence.append(task_name)
    return sequence


def task_id_from_name(task_name: str) -> int:
    if task_name not in tasks:
        raise KeyError(f"Unknown HalfCheetah task name: {task_name}")
    return tasks.index(task_name)


def model_dir_name(task_id: int, model_type: str, exp_name: str, seed: int) -> str:
    return f"task_{task_id}__{model_type}__{exp_name}__{seed}"


def simple_model_dir(args, task_id: int) -> Path:
    primary = Path(args.simple_dir) / model_dir_name(
        task_id=task_id,
        model_type="simple",
        exp_name=args.simple_exp_name,
        seed=args.seed,
    )
    if (primary / "model.pt").exists():
        return primary

    for exp_name in ["sac_20k", "SAC"]:
        fallback = Path(args.simple_dir) / model_dir_name(
            task_id=task_id,
            model_type="simple",
            exp_name=exp_name,
            seed=args.seed,
        )
        if (fallback / "model.pt").exists():
            return fallback
    return primary


def output_model_dir(args, task_id: int, model_type: str) -> Path:
    return Path(args.save_dir) / args.scenario / model_dir_name(
        task_id=task_id,
        model_type=model_type,
        exp_name=args.exp_name,
        seed=args.seed,
    )


def find_existing_state_file(args, task_id: int) -> Optional[Path]:
    candidates = [
        Path(args.sampled_state_dir) / f"task_{task_id:02d}_simple_policy_20k_states__{args.seed}.pt",
        Path(args.sampled_state_dir) / f"task_{task_id:02d}_simple_policy_20k_states.pt",
        Path(args.simple_state_dir) / f"task_{task_id:02d}_simple_sac_20k_states__{args.seed}.pt",
        Path(args.simple_state_dir) / f"task_{task_id:02d}_simple_sac_20k_states.pt",
        simple_model_dir(args, task_id) / f"task_{task_id:02d}_simple_sac_20k_states__{args.seed}.pt",
        simple_model_dir(args, task_id) / f"task_{task_id:02d}_simple_sac_20k_states.pt",
        simple_model_dir(args, task_id) / "simple_sac_20k_states.pt",
        simple_model_dir(args, task_id) / "states.pt",
    ]
    summary_path = simple_model_dir(args, task_id) / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        if summary.get("state_path"):
            candidates.append(Path(summary["state_path"]))

    for path in candidates:
        if path.exists():
            return path
    return None


def to_numpy_2d(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"expected NxL state matrix, got shape {array.shape}")
    return array


def extract_state_matrix(payload) -> np.ndarray:
    if isinstance(payload, torch.Tensor):
        return to_numpy_2d(payload)
    if not isinstance(payload, dict):
        raise TypeError(f"expected tensor or dict state payload, got {type(payload)}")
    data = payload.get("data", payload)
    for key in ["states", "observations", "visited_states"]:
        if key in data:
            return to_numpy_2d(data[key])
    raise KeyError(f"cannot find states/observations in payload keys: {sorted(data.keys())}")


def load_simple_agent(model_dir: Path, device) -> SimpleAgent:
    model_path = model_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Cannot find simple SAC model: {model_path}")
    model = safe_torch_load(model_path, map_location=device)
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module in {model_path}, got {type(model)}")
    return model.to(device).eval()


def sample_states_with_simple_policy(args, task_id: int, device) -> Tuple[np.ndarray, Path]:
    model_dir = simple_model_dir(args, task_id)
    model = load_simple_agent(model_dir, device=device)
    envs = gym.vector.SyncVectorEnv([make_env(task_id)])
    actor = Actor(envs, model).to(device).eval()

    observations = []
    obs, _ = envs.reset(seed=int(args.seed) + int(task_id) * 10_000 + 123)
    for _ in range(int(args.state_steps)):
        observations.append(np.asarray(obs[0], dtype=np.float32).reshape(-1))
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            actions, _, _ = actor.get_action(obs_tensor)
            actions = actions.detach().cpu().numpy()
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        obs = next_obs

    envs.close()
    states = np.stack(observations, axis=0).astype(np.float32)
    path = Path(args.sampled_state_dir) / f"task_{task_id:02d}_simple_policy_20k_states__{args.seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "states": torch.as_tensor(states, dtype=torch.float32),
            "observations": torch.as_tensor(states, dtype=torch.float32),
            "meta": {
                "task_id": int(task_id),
                "task_name": get_task_name(task_id),
                "seed": int(args.seed),
                "source_model_dir": str(model_dir),
                "num_observations": int(states.shape[0]),
                "state_dim": int(states.shape[1]),
            },
        },
        path,
    )
    print(f"*** Sampled simple-policy states for task {task_id} {get_task_name(task_id)}: {path}")
    return states, path


def ensure_simple_policy_states(args, task_id: int, device) -> Tuple[np.ndarray, Path]:
    existing = find_existing_state_file(args, task_id) if args.reuse_states else None
    if existing is not None:
        payload = safe_torch_load(existing, map_location="cpu")
        states = extract_state_matrix(payload).astype(np.float32)
        if states.shape[0] >= int(args.state_steps):
            states = states[: int(args.state_steps)]
        elif not args.allow_state_count_mismatch:
            raise ValueError(
                f"{existing}: expected at least {args.state_steps} states, got {states.shape[0]}"
            )
        print(f"*** Reusing states for task {task_id} {get_task_name(task_id)}: {existing}")
        return states, existing
    return sample_states_with_simple_policy(args, task_id, device=device)


def normalize_temporal(q: np.ndarray, eps: float) -> np.ndarray:
    mean = q.mean(axis=0, keepdims=True)
    std = np.maximum(q.std(axis=0, keepdims=True), float(eps))
    return ((q - mean) / std).astype(np.float32)


def empirical_wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    x_sorted = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    y_sorted = np.sort(np.asarray(y, dtype=np.float64).reshape(-1))
    n = min(x_sorted.size, y_sorted.size)
    if x_sorted.size != n:
        q = (np.arange(n, dtype=np.float64) + 0.5) / n
        pos = (np.arange(x_sorted.size, dtype=np.float64) + 0.5) / x_sorted.size
        x_sorted = np.interp(q, pos, x_sorted)
    if y_sorted.size != n:
        q = (np.arange(n, dtype=np.float64) + 0.5) / n
        pos = (np.arange(y_sorted.size, dtype=np.float64) + 0.5) / y_sorted.size
        y_sorted = np.interp(q, pos, y_sorted)
    return float(np.abs(x_sorted - y_sorted).mean())


def temporal_dim_wasserstein(q_k: np.ndarray, q_i: np.ndarray, eps: float) -> float:
    if q_k.shape[1] != q_i.shape[1]:
        raise ValueError(f"state dim mismatch: {q_k.shape} vs {q_i.shape}")
    q_k = normalize_temporal(q_k, eps=eps)
    q_i = normalize_temporal(q_i, eps=eps)
    per_dim = [
        empirical_wasserstein_1d(q_k[:, dim], q_i[:, dim])
        for dim in range(q_k.shape[1])
    ]
    return float(np.mean(per_dim))


def attach_memory_if_needed(model: nn.Module, model_dir: Path, map_location) -> None:
    if not hasattr(model, "memory"):
        return
    if getattr(model, "memory", None) is not None and len(getattr(model, "memory")) > 0:
        return
    memory_path = model_dir / "hybridrarl_memory.pt"
    if memory_path.exists():
        model.memory = ReservoirTrajectoryMemory.load(model_dir, map_location=map_location)


def relocate_model_dir(model_dir: Path, args) -> Path:
    model_dir = Path(model_dir).expanduser()
    if (model_dir / "model.pt").exists():
        return model_dir.resolve()
    for root in [Path(args.save_dir) / args.scenario, Path(args.save_dir)]:
        relocated = root / model_dir.name
        if (relocated / "model.pt").exists():
            print(f"*** Relocated prev unit `{model_dir}` -> `{relocated}`")
            return relocated.resolve()
    return model_dir.resolve()


def load_policy_unit(model_dir: Path, map_location, args, loading_stack: Optional[set] = None) -> nn.Module:
    loading_stack = set() if loading_stack is None else loading_stack
    model_dir = relocate_model_dir(model_dir, args)
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
            relocate_model_dir(Path(path), args)
            for path in getattr(model, "prev_unit_paths", [])
        ]
        nested_units = [
            load_policy_unit(path, map_location=map_location, args=args, loading_stack=loading_stack)
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
    model_type: str,
    log_alpha=None,
    alpha_optimizer=None,
    prev_unit_meta: Optional[List[Dict]] = None,
) -> None:
    dirname.mkdir(parents=True, exist_ok=True)
    actor.model.save(str(dirname), save_memory=True)
    checkpoint = {
        "type": "halfcheetah_hybridrarl_sac_checkpoint",
        "scenario": args.scenario,
        "model_type": model_type,
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
    parser = argparse.ArgumentParser(description="Train HalfCheetah HybridRARL prev-units by scenario.")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=0.11)
    parser.add_argument("--simple-dir", type=Path, default=Path("agents/simple_sac_20k"))
    parser.add_argument("--simple-exp-name", default="sac_20k")
    parser.add_argument("--simple-state-dir", type=Path, default=Path("simple_sac_20k_states"))
    parser.add_argument("--sampled-state-dir", type=Path, default=Path("simple_policy_20k_states"))
    parser.add_argument("--reuse-states", default=False, action="store_true")
    parser.add_argument("--state-steps", type=int, default=20_000)
    parser.add_argument("--allow-state-count-mismatch", default=False, action="store_true")
    parser.add_argument("--save-dir", type=Path, default=Path("agents/hybridrarl_prev_units"))
    parser.add_argument("--exp-name", default="SAC_hybridrarl_prev_units")
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
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument(
        "--ablation-no-retrieval",
        default=False,
        action="store_true",
        help="Remove trajectory-memory retrieval and use only MLP plus optional selective modular reuse.",
    )
    parser.add_argument("--no-run", default=False, action="store_true")
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
) -> Tuple[nn.Module, str]:
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
            return model, "no-retrieval-prev-units"
        return (
            NoRetrievalAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                policy_hidden_dim=args.policy_hidden_dim,
                policy_layers=args.policy_layers,
            ).to(device),
            "no-retrieval",
        )

    kwargs = dict(
        obs_dim=obs_dim,
        act_dim=act_dim,
        memory=memory,
        memory_capacity=args.memory_capacity,
        traj_len=args.traj_len,
        top_r=args.top_r,
        traj_hidden_dim=args.traj_hidden_dim,
        att_dim=args.att_dim,
        value_dim=args.value_dim,
        policy_hidden_dim=args.policy_hidden_dim,
        policy_layers=args.policy_layers,
        coarse_metric=args.coarse_metric,
    )
    if prev_units:
        model = HybridRARLPrevUnitsAgent(
            prev_units=prev_units,
            prev_unit_paths=[str(path) for path in prev_unit_paths],
            prev_att_dim=args.prev_att_dim,
            **kwargs,
        ).to(device)
        model.prev_unit_task_ids = [int(task_id) for task_id in prev_task_ids]
        return model, "hybridrarl-prev-units"
    return HybridRARLAgent(**kwargs).to(device), "hybridrarl"


def train_task(
    args,
    task_id: int,
    prev_items: List[Dict],
    trained_model_dirs: Dict[int, Path],
    device,
) -> Path:
    task_seed = int(args.seed) + int(task_id) * 10_000
    set_seed(task_seed, args.torch_deterministic)

    if args.ablation_no_retrieval:
        dry_model_type = "no-retrieval-prev-units" if prev_items else "no-retrieval"
    else:
        dry_model_type = "hybridrarl-prev-units" if prev_items else "hybridrarl"
    dry_output_dir = output_model_dir(args, task_id, model_type=dry_model_type)
    if args.no_run:
        print(f"\n*** Run name: {args.scenario}/{dry_output_dir.name} ***\n")
        print(f"*** Loading model `{dry_model_type}`")
        if prev_items:
            print(f"*** Prev units for task {task_id} {get_task_name(task_id)} ***")
            for item in prev_items:
                print(
                    f"    task {item['task_id']} {item['task_name']} "
                    f"distance={item['distance']:.6f}"
                )
        else:
            print(f"*** No prev unit below threshold for task {task_id} {get_task_name(task_id)}")
        return dry_output_dir

    prev_units = []
    prev_unit_paths = []
    prev_unit_meta = []
    for item in prev_items:
        prev_dir = trained_model_dirs[int(item["task_id"])]
        prev_unit_paths.append(prev_dir.resolve())
        prev_units.append(load_policy_unit(prev_dir, map_location=device, args=args))
        prev_unit_meta.append({**item, "model_dir": str(prev_dir.resolve())})

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
    model, model_type = build_model(
        args=args,
        obs_dim=obs_dim,
        act_dim=act_dim,
        memory=memory,
        prev_units=prev_units,
        prev_unit_paths=prev_unit_paths,
        prev_task_ids=[int(item["task_id"]) for item in prev_items],
        device=device,
    )
    run_name = f"{args.scenario}/{model_dir_name(task_id, model_type, args.exp_name, args.seed)}"
    output_dir = output_model_dir(args, task_id, model_type=model_type)

    print(f"\n*** Run name: {run_name} ***\n")
    print(f"*** Loading model `{model_type}`")
    if prev_unit_meta:
        print(f"*** Prev units for task {task_id} {get_task_name(task_id)} ***")
        for item in prev_unit_meta:
            print(
                f"    task {item['task_id']} {item['task_name']} "
                f"distance={item['distance']:.6f}"
            )
    else:
        print(f"*** No prev unit below threshold for task {task_id} {get_task_name(task_id)}")

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

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    writer.add_text(
        "prev_units",
        "|task|name|distance|\n|-|-|-|\n%s"
        % (
            "\n".join(
                [
                    f"|{item['task_id']}|{item['task_name']}|{item['distance']:.8f}|"
                    for item in prev_unit_meta
                ]
            )
        ),
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
                        "HybridRARL starts acting after random_actions_end, but memory is empty."
                    )
                    memory_batch = sample_memory_batch(memory, n_traj=args.memory_capacity, device=device)
                    actions, _, _ = actor.get_action(obs_tensor, memory_batch=memory_batch)
                actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if memory is not None:
            cache_states.append(np.asarray(obs[0], dtype=np.float32).reshape(-1))
            cache_actions.append(np.asarray(actions[0], dtype=np.float32).reshape(-1))
            cache_rewards.append(np.asarray([rewards[0]], dtype=np.float32).reshape(-1))
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
                    "memory_batch": sample_memory_batch(memory, n_traj=args.memory_capacity, device=device)
                }

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data.next_observations,
                    **rag_kwargs,
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
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
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
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
                            writer.add_scalar(f"att_memory_rank_{rank_index}", float(weight), global_step)

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
                model_type=model_type,
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
        model_type=model_type,
        log_alpha=log_alpha,
        alpha_optimizer=alpha_optimizer,
        prev_unit_meta=prev_unit_meta,
    )
    envs.close()
    writer.close()
    print(f"Saving trained agent in `{output_dir}` with name `{run_name}`")
    return output_dir


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    sequence_names = unique_scenario_sequence(args.scenario)
    sequence_task_ids = [task_id_from_name(name) for name in sequence_names]

    print(f"*** Scenario: {args.scenario}")
    print(f"*** Sequence: {[(task_id, get_task_name(task_id)) for task_id in sequence_task_ids]}")
    print(f"*** Seed: {args.seed}")
    print(f"*** Threshold: {args.threshold}")
    print(f"*** Device: {device}")

    state_cache: Dict[int, np.ndarray] = {}
    state_paths: Dict[int, str] = {}
    trained_model_dirs: Dict[int, Path] = {}
    similarity_rows = []

    for index, task_id in enumerate(sequence_task_ids):
        current_states, current_path = ensure_simple_policy_states(args, task_id, device=device)
        state_cache[task_id] = current_states
        state_paths[task_id] = str(current_path)

        prev_items = []
        for prev_task_id in sequence_task_ids[:index]:
            if prev_task_id not in state_cache:
                prev_states, prev_path = ensure_simple_policy_states(args, prev_task_id, device=device)
                state_cache[prev_task_id] = prev_states
                state_paths[prev_task_id] = str(prev_path)
            distance = temporal_dim_wasserstein(
                current_states,
                state_cache[prev_task_id],
                eps=float(args.norm_eps),
            )
            row = {
                "scenario": args.scenario,
                "task_id": int(task_id),
                "task_name": get_task_name(task_id),
                "prev_task_id": int(prev_task_id),
                "prev_task_name": get_task_name(prev_task_id),
                "distance": float(distance),
                "threshold": float(args.threshold),
                "selected": bool(distance < float(args.threshold)),
            }
            similarity_rows.append(row)
            print(
                f"S(task {task_id} {get_task_name(task_id)}, "
                f"prev task {prev_task_id} {get_task_name(prev_task_id)}) "
                f"= {distance:.8f}"
            )
            if distance < float(args.threshold):
                prev_items.append(
                    {
                        "task_id": int(prev_task_id),
                        "task_name": get_task_name(prev_task_id),
                        "distance": float(distance),
                    }
                )

        output_dir = train_task(
            args=args,
            task_id=task_id,
            prev_items=prev_items,
            trained_model_dirs=trained_model_dirs,
            device=device,
        )
        trained_model_dirs[task_id] = output_dir

    analysis_dir = Path(args.save_dir) / args.scenario
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_path = analysis_dir / "hybridrarl_similarity.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "scenario": args.scenario,
                "seed": int(args.seed),
                "threshold": float(args.threshold),
                "sequence": [
                    {"task_id": int(task_id), "task_name": get_task_name(task_id)}
                    for task_id in sequence_task_ids
                ],
                "state_paths": state_paths,
                "similarities": similarity_rows,
                "trained_model_dirs": {
                    int(task_id): str(path)
                    for task_id, path in trained_model_dirs.items()
                },
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"*** Saved similarity summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
