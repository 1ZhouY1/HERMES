#!/usr/bin/env python3
"""
Train HybridRARL previous-unit transfer models selected from a distance CSV.

For each current task T_k, only lower-triangular entries are considered. Every
previous task T_i with distance(T_k, T_i) < threshold becomes a prev unit.
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

from models import HybridRARLAgent, HybridRARLPrevUnitsAgent, ReservoirTrajectoryMemory, shared
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


def model_dir_name(task_id: int, model_type: str, exp_name: str, seed: int) -> str:
    return f"task_{task_id}__{model_type}__{exp_name}__{seed}"


def base_model_dir(args, task_id: int) -> Path:
    return Path(args.base_dir) / model_dir_name(
        task_id=task_id,
        model_type=args.base_model_type,
        exp_name=args.base_exp_name,
        seed=args.seed,
    )


def transfer_model_dir(args, task_id: int) -> Path:
    return Path(args.transfer_save_dir) / model_dir_name(
        task_id=task_id,
        model_type=args.transfer_model_type,
        exp_name=args.transfer_exp_name,
        seed=args.seed,
    )


def resolve_prev_unit_dir(args, task_id: int) -> Tuple[Path, str]:
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


def load_policy_unit(model_dir: Path, map_location, loading_stack: Optional[set] = None) -> nn.Module:
    loading_stack = set() if loading_stack is None else loading_stack
    model_dir = model_dir.expanduser().resolve()
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

    if isinstance(model, HybridRARLPrevUnitsAgent):
        nested_units = [
            load_policy_unit(Path(path), map_location=map_location, loading_stack=loading_stack)
            for path in getattr(model, "prev_unit_paths", [])
        ]
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
        description="Train CSV-selected HybridRARL prev-unit transfer models."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.2)
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
    parser.add_argument("--transfer-model-type", default="hybridrarl-prev-units")
    parser.add_argument("--transfer-exp-name", default="csv_threshold_transfer")
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
    memory: ReservoirTrajectoryMemory,
    prev_units: List[nn.Module],
    prev_unit_paths: List[Path],
    prev_task_ids: List[int],
    device,
) -> HybridRARLPrevUnitsAgent:
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
        prev_units.append(load_policy_unit(prev_dir, map_location=device))
        prev_unit_meta.append(
            {
                "task_id": int(item["task_id"]),
                "task_name": item["task_name"],
                "distance": float(item["distance"]),
                "source_type": source_type,
                "model_dir": str(prev_dir.resolve()),
            }
        )

    if not prev_units:
        print(f"*** task {task_id}: no prev units; skipped.")
        return

    print(f"\n*** Train task {task_id} {get_task_name(task_id)} with prev units:")
    for item in prev_unit_meta:
        print(
            f"    task {item['task_id']} {item['task_name']} "
            f"distance={item['distance']:.6f} source={item['source_type']}"
        )

    if args.no_run:
        return

    envs = gym.vector.SyncVectorEnv([make_env(task_id)])
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    envs.single_observation_space.dtype = np.float32

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
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

    output_dir = transfer_model_dir(args, task_id)
    cache_states: List[np.ndarray] = []
    cache_actions: List[np.ndarray] = []
    cache_rewards: List[np.ndarray] = []
    obs, _ = envs.reset(seed=task_seed)
    start_time = time.time()

    for global_step in range(int(args.total_timesteps)):
        if global_step < int(args.random_actions_end):
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            assert len(memory) > 0, (
                "HybridRARLPrevUnits starts acting after random_actions_end, but memory is empty."
            )
            memory_batch = sample_memory_batch(memory, n_traj=args.memory_capacity, device=device)
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                actions, _, _ = actor.get_action(obs_tensor, memory_batch=memory_batch)
                actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

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
                    f"task={task_id}, step={global_step}, "
                    f"episodic_return={float(info['episode']['r']):.2f}, "
                    f"success={float(info.get('success', 0.0)):.1f}"
                )
                break

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and "final_observation" in infos:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        if global_step > int(args.learning_starts):
            data = rb.sample(int(args.batch_size))
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

        current_step = global_step + 1
        if current_step % 1_000 == 0:
            sps = int(current_step / max(time.time() - start_time, 1e-6))
            print(
                f"task={task_id}, step={current_step}, memory={len(memory)}, "
                f"SPS={sps}, alpha={alpha:.4f}"
            )

        if args.save_every > 0 and current_step % int(args.save_every) == 0:
            print(f"*** Saving task {task_id} checkpoint at step {current_step}: {output_dir}")
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
    print(f"*** Finished task {task_id}; saved transfer model to {output_dir}")


def main() -> int:
    args = parse_args()
    tasks, matrix = read_distance_csv(args.csv.expanduser())
    available_task_ids = {
        int(task["id"]) for task in tasks if task["id"] is not None
    }
    default_task_ids = [task_id for task_id in range(1, 10) if task_id in available_task_ids]
    task_ids = args.tasks if args.tasks is not None else default_task_ids
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    print(f"csv: {args.csv}")
    print(f"seed: {args.seed}")
    print(f"threshold: {args.threshold}")
    print(f"device: {device}")

    for task_id in sorted(int(task_id) for task_id in task_ids):
        if task_id not in range(1, 10):
            print(f"*** task {task_id}: skipped; this script only trains current tasks 1-9.")
            continue
        related = related_previous_tasks(
            task_id=task_id,
            tasks=tasks,
            matrix=matrix,
            threshold=float(args.threshold),
        )
        if not related:
            print(f"*** task {task_id} {get_task_name(task_id)}: no distance below threshold; skipped.")
            continue
        train_task(args=args, task_id=task_id, related=related, device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
