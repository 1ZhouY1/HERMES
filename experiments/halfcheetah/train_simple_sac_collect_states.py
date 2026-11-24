import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer

from models import SimpleAgent, shared
from tasks import get_task, get_task_name, tasks


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

    def forward(self, obs: torch.Tensor):
        mean, log_std = self.model(obs)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, obs: torch.Tensor):
        mean, log_std = self(obs)
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train SimpleAgent with SAC on HalfCheetah tasks and save all "
            "interacted states for each task."
        )
    )
    parser.add_argument("--tasks", type=int, nargs="*", default=list(range(len(tasks))))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--save-dir", type=str, default="agents/simple_sac_20k")
    parser.add_argument("--state-dir", type=str, default="simple_sac_20k_states")
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--random-actions-end", type=int, default=1_000)
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
    return parser.parse_args()


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def set_seed(seed: int, torch_deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def to_numpy_stack(items: List[np.ndarray]) -> np.ndarray:
    return np.stack(items, axis=0).astype(np.float32)


def save_states(
    path: Path,
    task_id: int,
    seed: int,
    total_timesteps: int,
    obs_dim: int,
    act_dim: int,
    collected: Dict[str, List[np.ndarray]],
    episode_returns: List[float],
    episode_successes: List[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    observations = to_numpy_stack(collected["observations"])
    raw_next_observations = to_numpy_stack(collected["raw_next_observations"])
    next_observations = to_numpy_stack(collected["next_observations"])
    actions = to_numpy_stack(collected["actions"])
    rewards = np.asarray(collected["rewards"], dtype=np.float32).reshape(-1, 1)
    terminations = np.asarray(collected["terminations"], dtype=np.bool_)
    truncations = np.asarray(collected["truncations"], dtype=np.bool_)
    visited_states = np.concatenate(
        [observations[:1], raw_next_observations],
        axis=0,
    ).astype(np.float32)

    payload = {
        "observations": torch.as_tensor(observations, dtype=torch.float32),
        "states": torch.as_tensor(observations, dtype=torch.float32),
        "visited_states": torch.as_tensor(visited_states, dtype=torch.float32),
        "next_observations": torch.as_tensor(next_observations, dtype=torch.float32),
        "raw_next_observations": torch.as_tensor(raw_next_observations, dtype=torch.float32),
        "actions": torch.as_tensor(actions, dtype=torch.float32),
        "rewards": torch.as_tensor(rewards, dtype=torch.float32),
        "terminations": torch.as_tensor(terminations),
        "truncations": torch.as_tensor(truncations),
        "meta": {
            "task_id": int(task_id),
            "task_name": get_task_name(task_id),
            "seed": int(seed),
            "total_timesteps": int(total_timesteps),
            "num_observations": int(observations.shape[0]),
            "num_visited_states": int(visited_states.shape[0]),
            "obs_dim": int(obs_dim),
            "act_dim": int(act_dim),
            "mean_episodic_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "mean_success": float(np.mean(episode_successes)) if episode_successes else 0.0,
        },
    }
    torch.save(payload, path)


def save_checkpoint(
    task_dir: Path,
    actor: Actor,
    qf1: SoftQNetwork,
    qf2: SoftQNetwork,
    qf1_target: SoftQNetwork,
    qf2_target: SoftQNetwork,
    actor_optimizer: optim.Optimizer,
    q_optimizer: optim.Optimizer,
    args,
    task_id: int,
    alpha: float,
    log_alpha: torch.Tensor = None,
    alpha_optimizer: optim.Optimizer = None,
) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    actor.model.save(str(task_dir))
    checkpoint = {
        "type": "simple_sac_collect_states_checkpoint",
        "task_id": int(task_id),
        "task_name": get_task_name(task_id),
        "args": vars(args),
        "actor_state_dict": actor.state_dict(),
        "actor_model_state_dict": actor.model.state_dict(),
        "qf1_state_dict": qf1.state_dict(),
        "qf2_state_dict": qf2.state_dict(),
        "qf1_target_state_dict": qf1_target.state_dict(),
        "qf2_target_state_dict": qf2_target.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "q_optimizer_state_dict": q_optimizer.state_dict(),
        "alpha": float(alpha),
    }
    if log_alpha is not None:
        checkpoint["log_alpha"] = log_alpha.detach().cpu()
    if alpha_optimizer is not None:
        checkpoint["alpha_optimizer_state_dict"] = alpha_optimizer.state_dict()
    torch.save(checkpoint, task_dir / "sac_checkpoint.pt")


def train_one_task(args, task_id: int, device: torch.device) -> None:
    task_seed = int(args.seed) + int(task_id) * 10_000
    set_seed(task_seed, args.torch_deterministic)

    envs = gym.vector.SyncVectorEnv([make_env(task_id)])
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    envs.single_observation_space.dtype = np.float32

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    model = SimpleAgent(obs_dim=obs_dim, act_dim=act_dim).to(device)
    actor = Actor(envs, model).to(device)
    qf1 = SoftQNetwork(obs_dim, act_dim).to(device)
    qf2 = SoftQNetwork(obs_dim, act_dim).to(device)
    qf1_target = SoftQNetwork(obs_dim, act_dim).to(device)
    qf2_target = SoftQNetwork(obs_dim, act_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    rb = ReplayBuffer(
        buffer_size=int(args.buffer_size),
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        device=device,
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

    collected = {
        "observations": [],
        "raw_next_observations": [],
        "next_observations": [],
        "actions": [],
        "rewards": [],
        "terminations": [],
        "truncations": [],
    }
    episode_returns = []
    episode_successes = []
    ep_return = 0.0
    ep_success = 0.0

    obs, _ = envs.reset(seed=task_seed)
    start_time = time.time()
    print(
        f"\n*** Train simple SAC task {task_id}: {get_task_name(task_id)}, "
        f"steps={args.total_timesteps}, device={device} ***"
    )

    for global_step in range(int(args.total_timesteps)):
        if global_step < int(args.random_actions_end):
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                actions, _, _ = actor.get_action(obs_tensor)
                actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        raw_next_obs = next_obs.copy()
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and "final_observation" in infos:
                real_next_obs[idx] = infos["final_observation"][idx]

        collected["observations"].append(np.asarray(obs[0], dtype=np.float32).reshape(-1))
        collected["raw_next_observations"].append(np.asarray(raw_next_obs[0], dtype=np.float32).reshape(-1))
        collected["next_observations"].append(np.asarray(real_next_obs[0], dtype=np.float32).reshape(-1))
        collected["actions"].append(np.asarray(actions[0], dtype=np.float32).reshape(-1))
        collected["rewards"].append(float(rewards[0]))
        collected["terminations"].append(bool(terminations[0]))
        collected["truncations"].append(bool(truncations[0]))

        ep_return += float(rewards[0])
        if isinstance(infos, dict) and "success" in infos:
            ep_success = max(ep_success, float(np.asarray(infos["success"]).reshape(-1)[0]))

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is None:
                    continue
                episode_returns.append(ep_return)
                episode_successes.append(float(info.get("success", ep_success)))
                print(
                    f"task={task_id}, step={global_step}, "
                    f"episodic_return={ep_return:.2f}, success={episode_successes[-1]:.1f}"
                )
                ep_return = 0.0
                ep_success = 0.0
                break

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        if global_step > int(args.learning_starts):
            data = rb.sample(int(args.batch_size))
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
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
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                        alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        alpha_optimizer.step()
                        alpha = log_alpha.exp().item()

            if global_step % int(args.target_network_frequency) == 0:
                soft_update(qf1, qf1_target, float(args.tau))
                soft_update(qf2, qf2_target, float(args.tau))

        if (global_step + 1) % 1_000 == 0:
            sps = int((global_step + 1) / max(time.time() - start_time, 1e-6))
            print(f"task={task_id}, step={global_step + 1}, SPS={sps}, alpha={alpha:.4f}")

    if ep_return != 0.0 or not episode_returns:
        episode_returns.append(ep_return)
        episode_successes.append(ep_success)

    task_dir = Path(args.save_dir) / f"task_{task_id}__simple__sac_20k__{args.seed}"
    state_path = Path(args.state_dir) / f"task_{task_id:02d}_simple_sac_20k_states.pt"
    save_checkpoint(
        task_dir=task_dir,
        actor=actor,
        qf1=qf1,
        qf2=qf2,
        qf1_target=qf1_target,
        qf2_target=qf2_target,
        actor_optimizer=actor_optimizer,
        q_optimizer=q_optimizer,
        args=args,
        task_id=task_id,
        alpha=alpha,
        log_alpha=log_alpha,
        alpha_optimizer=alpha_optimizer,
    )
    save_states(
        path=state_path,
        task_id=task_id,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        obs_dim=obs_dim,
        act_dim=act_dim,
        collected=collected,
        episode_returns=episode_returns,
        episode_successes=episode_successes,
    )

    summary_path = task_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task_id": int(task_id),
                "task_name": get_task_name(task_id),
                "model_dir": str(task_dir),
                "state_path": str(state_path),
                "total_timesteps": int(args.total_timesteps),
                "num_states": len(collected["observations"]),
                "mean_episodic_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
                "mean_success": float(np.mean(episode_successes)) if episode_successes else 0.0,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    envs.close()
    print(f"*** Saved task {task_id} model to `{task_dir}`")
    print(f"*** Saved task {task_id} states to `{state_path}`")


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.state_dir).mkdir(parents=True, exist_ok=True)

    for task_id in args.tasks:
        if task_id < 0 or task_id >= len(tasks):
            raise ValueError(f"This script is intended for task ids 0-{len(tasks) - 1}.")
        train_one_task(args, task_id, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
