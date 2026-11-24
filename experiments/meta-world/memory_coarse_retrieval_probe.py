#!/usr/bin/env python3
import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from models import ReservoirTrajectoryMemory, SimpleAgent, shared
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

    @torch.no_grad()
    def get_deterministic_action(self, obs: torch.Tensor):
        mean, _ = self(obs)
        return torch.tanh(mean) * self.action_scale + self.action_bias


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a SimpleAgent SAC policy while maintaining a reservoir "
            "trajectory memory, then save memory and one frozen-policy episode "
            "at selected training steps."
        )
    )
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--snapshot-steps", type=int, nargs="*", default=[100_000, 500_000, 900_000])
    parser.add_argument("--save-dir", type=Path, default=Path("analysis/memory_coarse_retrieval_probe"))
    parser.add_argument("--exp-name", default="simple_memory_probe")
    parser.add_argument("--memory-capacity", type=int, default=2_000)
    parser.add_argument("--traj-len", type=int, default=20)
    parser.add_argument("--eval-deterministic", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval-max-steps", type=int, default=0)
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
    return parser.parse_args()


def set_seed(seed: int, torch_deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def run_name(args) -> str:
    return f"task_{args.task_id}__{args.exp_name}__{args.seed}"


def save_sac_checkpoint(
    path: Path,
    actor: Actor,
    qf1: SoftQNetwork,
    qf2: SoftQNetwork,
    qf1_target: SoftQNetwork,
    qf2_target: SoftQNetwork,
    actor_optimizer,
    q_optimizer,
    args,
    global_step: int,
    alpha: float,
    log_alpha=None,
    alpha_optimizer=None,
) -> None:
    checkpoint = {
        "type": "simple_memory_coarse_retrieval_probe",
        "task_id": int(args.task_id),
        "task_name": get_task_name(args.task_id),
        "global_step": int(global_step),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
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
    torch.save(checkpoint, path)


@torch.no_grad()
def collect_frozen_episode(
    actor: Actor,
    task_id: int,
    seed: int,
    device,
    deterministic: bool,
    max_steps: int,
) -> Dict:
    was_training = actor.training
    actor.eval()
    env = get_task(task_id)
    states = []
    actions = []
    rewards = []
    next_states = []
    terminations = []
    truncations = []

    obs, _ = env.reset(seed=seed)
    ep_return = 0.0
    step = 0
    while True:
        states.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        if deterministic:
            action = actor.get_deterministic_action(obs_tensor)
        else:
            action, _, _ = actor.get_action(obs_tensor)
        action_np = action.squeeze(0).detach().cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action_np)
        actions.append(np.asarray(action_np, dtype=np.float32).reshape(-1))
        rewards.append(float(reward))
        next_states.append(np.asarray(next_obs, dtype=np.float32).reshape(-1))
        terminations.append(bool(terminated))
        truncations.append(bool(truncated))
        ep_return += float(reward)
        step += 1

        if terminated or truncated or (max_steps > 0 and step >= max_steps):
            break
        obs = next_obs

    env.close()
    if was_training:
        actor.train()

    return {
        "states": torch.as_tensor(np.stack(states, axis=0), dtype=torch.float32),
        "actions": torch.as_tensor(np.stack(actions, axis=0), dtype=torch.float32),
        "rewards": torch.as_tensor(np.asarray(rewards, dtype=np.float32).reshape(-1, 1)),
        "next_states": torch.as_tensor(np.stack(next_states, axis=0), dtype=torch.float32),
        "terminations": torch.as_tensor(np.asarray(terminations, dtype=np.bool_)),
        "truncations": torch.as_tensor(np.asarray(truncations, dtype=np.bool_)),
        "episode_return": float(ep_return),
        "episode_length": int(len(states)),
        "deterministic": bool(deterministic),
    }


def save_snapshot(
    snapshot_dir: Path,
    actor: Actor,
    qf1: SoftQNetwork,
    qf2: SoftQNetwork,
    qf1_target: SoftQNetwork,
    qf2_target: SoftQNetwork,
    actor_optimizer,
    q_optimizer,
    memory: ReservoirTrajectoryMemory,
    args,
    global_step: int,
    alpha: float,
    device,
    log_alpha=None,
    alpha_optimizer=None,
) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    eval_seed = int(args.seed) + int(args.task_id) * 10_000 + int(global_step)
    episode = collect_frozen_episode(
        actor=actor,
        task_id=args.task_id,
        seed=eval_seed,
        device=device,
        deterministic=bool(args.eval_deterministic),
        max_steps=int(args.eval_max_steps),
    )
    episode_payload = {
        **episode,
        "meta": {
            "task_id": int(args.task_id),
            "task_name": get_task_name(args.task_id),
            "seed": int(args.seed),
            "eval_seed": int(eval_seed),
            "global_step": int(global_step),
        },
    }
    torch.save(episode_payload, snapshot_dir / "episode.pt")
    memory.save(str(snapshot_dir), filename="trajectory_memory.pt")
    actor.model.save(str(snapshot_dir))
    save_sac_checkpoint(
        path=snapshot_dir / "sac_checkpoint.pt",
        actor=actor,
        qf1=qf1,
        qf2=qf2,
        qf1_target=qf1_target,
        qf2_target=qf2_target,
        actor_optimizer=actor_optimizer,
        q_optimizer=q_optimizer,
        args=args,
        global_step=global_step,
        alpha=alpha,
        log_alpha=log_alpha,
        alpha_optimizer=alpha_optimizer,
    )
    with (snapshot_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task_id": int(args.task_id),
                "task_name": get_task_name(args.task_id),
                "global_step": int(global_step),
                "episode_length": int(episode["episode_length"]),
                "episode_return": float(episode["episode_return"]),
                "memory_size": int(len(memory)),
                "memory_capacity": int(memory.capacity),
                "memory_num_seen": int(memory.num_seen),
                "memory_num_replacements": int(memory.num_replacements),
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"Saved snapshot at step {global_step}: {snapshot_dir}")


def validate_snapshot_steps(snapshot_steps: Sequence[int], total_timesteps: int) -> List[int]:
    steps = sorted({int(step) for step in snapshot_steps})
    if not steps:
        raise ValueError("--snapshot-steps cannot be empty.")
    bad = [step for step in steps if step <= 0 or step > int(total_timesteps)]
    if bad:
        raise ValueError(f"snapshot steps must be in 1..{total_timesteps}, got {bad}")
    return steps


def main() -> int:
    args = parse_args()
    snapshot_steps = validate_snapshot_steps(args.snapshot_steps, args.total_timesteps)
    set_seed(args.seed, args.torch_deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    name = run_name(args)
    output_dir = Path(args.save_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n*** Run name: {name} ***\n")
    print(f"*** Device: {device}")
    print(f"*** Snapshot steps: {snapshot_steps}")

    writer = SummaryWriter(f"runs/{name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    envs = gym.vector.SyncVectorEnv([make_env(args.task_id)])
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
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    memory = ReservoirTrajectoryMemory(
        capacity=args.memory_capacity,
        traj_len=args.traj_len,
        obs_dim=obs_dim,
        action_dim=act_dim,
        reward_dim=1,
        device="cpu",
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

    cache_states: List[np.ndarray] = []
    cache_actions: List[np.ndarray] = []
    cache_rewards: List[np.ndarray] = []
    obs, _ = envs.reset(seed=args.seed)
    start_time = time.time()
    actor_loss = torch.zeros((), device=device)
    alpha_loss = torch.zeros((), device=device)
    saved_steps = set()

    for global_step in range(int(args.total_timesteps)):
        if global_step < int(args.random_actions_end):
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                actions, _, _ = actor.get_action(obs_tensor)
                actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and "final_observation" in infos:
                real_next_obs[idx] = infos["final_observation"][idx]

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

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        if global_step > int(args.learning_starts):
            data = rb.sample(int(args.batch_size))
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
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
                writer.add_scalar("charts/SPS", int(global_step / max(time.time() - start_time, 1e-6)), global_step)
                writer.add_scalar("charts/trajectory_memory_size", len(memory), global_step)
                print("SPS:", int(global_step / max(time.time() - start_time, 1e-6)))

        current_step = global_step + 1
        if current_step in snapshot_steps and current_step not in saved_steps:
            save_snapshot(
                snapshot_dir=output_dir / f"snapshot_{current_step}",
                actor=actor,
                qf1=qf1,
                qf2=qf2,
                qf1_target=qf1_target,
                qf2_target=qf2_target,
                actor_optimizer=actor_optimizer,
                q_optimizer=q_optimizer,
                memory=memory,
                args=args,
                global_step=current_step,
                alpha=alpha,
                device=device,
                log_alpha=log_alpha,
                alpha_optimizer=alpha_optimizer,
            )
            saved_steps.add(current_step)

    envs.close()
    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
