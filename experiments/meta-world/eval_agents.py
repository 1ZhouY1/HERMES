#!/usr/bin/env python3
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from models import LongTermMemory
from tasks import get_task


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class LoadedHybridAgent(nn.Module):
    def __init__(self, net_mean, net_logstd=None):
        super().__init__()
        self.net_mean = net_mean
        self.net_logstd = net_logstd

    def forward(self, x, memory_batch):
        return self.net_mean(x, memory_batch=memory_batch)


def first_output(out):
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def load_hybrid_agent(model_dir, device):
    model_path = os.path.join(model_dir, "model.pt")
    if os.path.exists(model_path):
        payload = safe_torch_load(model_path, map_location=device)
        if isinstance(payload, dict) and "net_mean" in payload:
            model = LoadedHybridAgent(
                net_mean=payload["net_mean"],
                net_logstd=payload.get("net_logstd", None),
            )
            return model.to(device).eval(), payload.get("meta", {})
        return payload.to(device).eval(), {}

    net_mean_path = os.path.join(model_dir, "net_mean.pt")
    net_logstd_path = os.path.join(model_dir, "net_logstd.pt")
    assert os.path.exists(net_mean_path), f"Cannot find {model_path} or {net_mean_path}"
    net_mean = safe_torch_load(net_mean_path, map_location=device)
    net_logstd = safe_torch_load(net_logstd_path, map_location=device) if os.path.exists(net_logstd_path) else None
    return LoadedHybridAgent(net_mean=net_mean, net_logstd=net_logstd).to(device).eval(), {}


def load_memory(model_dir):
    memory_path = os.path.join(model_dir, "long_term_memory.pt")
    assert os.path.exists(memory_path), f"Cannot find {memory_path}"
    try:
        return LongTermMemory.load(model_dir, map_location="cpu")
    except TypeError:
        data = safe_torch_load(memory_path, map_location="cpu")
        memory = LongTermMemory(
            capacity=data["capacity"],
            traj_len=data["traj_len"],
            obs_dim=data["obs_dim"],
            action_dim=data["action_dim"],
            reward_dim=data["reward_dim"],
            device="cpu",
            dtype=data.get("dtype", torch.float32),
        )
        memory.ptr = data["ptr"]
        memory.size = data["size"]
        memory.states.copy_(data["states"])
        memory.actions.copy_(data["actions"])
        memory.rewards.copy_(data["rewards"])
        return memory


def sample_memory_batch(memory, n_traj, device):
    effective_n_traj = min(int(n_traj), len(memory))
    traj_batch = memory.sample_trajectories(
        n_traj=effective_n_traj,
        device=device,
        replace=False,
    )
    return {
        "obs": traj_batch.states,
        "states": traj_batch.states,
        "actions": traj_batch.actions,
        "rewards": traj_batch.rewards,
    }


@torch.no_grad()
def deterministic_action(model, obs, memory, n_traj, action_scale, action_bias, device):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    memory_batch = sample_memory_batch(memory, n_traj=n_traj, device=device)
    mean = first_output(model(obs_t, memory_batch=memory_batch))
    action = torch.tanh(mean) * action_scale + action_bias
    return action.squeeze(0).cpu().numpy().astype(np.float32)


def reset_env(env, seed=None):
    out = env.reset(seed=seed) if seed is not None else env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, reward, done, info
    obs, reward, done, info = out
    return obs, reward, bool(done), info


def evaluate(args):
    device = torch.device(args.device)
    env = get_task(args.task_id)

    model, meta = load_hybrid_agent(args.model_dir, device=device)
    memory = load_memory(args.model_dir)
    n_traj = args.n_traj if args.n_traj is not None else int(meta.get("n_traj", 512))

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

    successes = []
    returns = []
    lengths = []

    print(f"model_dir: {args.model_dir}")
    print(f"task_id: {args.task_id}")
    print(f"episodes: {args.episodes}")
    print(f"memory_size: {len(memory)} trajectories")
    print(f"n_traj: {n_traj}")

    for ep in range(args.episodes):
        reset_seed = None if args.seed is None else args.seed + ep
        obs = reset_env(env, seed=reset_seed)
        ep_ret = 0.0
        ep_success = 0.0

        for step in range(1, args.max_steps + 1):
            action = deterministic_action(
                model=model,
                obs=obs,
                memory=memory,
                n_traj=n_traj,
                action_scale=action_scale,
                action_bias=action_bias,
                device=device,
            )
            obs, reward, done, info = step_env(env, action)
            ep_ret += float(reward)
            ep_success = max(ep_success, float(info.get("success", 0.0)))

            if done:
                break

        successes.append(ep_success)
        returns.append(ep_ret)
        lengths.append(step)
        print(
            f"episode {ep + 1:02d}: success={ep_success:.0f}, "
            f"return={ep_ret:.2f}, length={step}"
        )

    print("\nSummary")
    print(f"average_success: {np.mean(successes):.4f}")
    print(f"average_return:  {np.mean(returns):.4f}")
    print(f"average_length:  {np.mean(lengths):.2f}")

    env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="agents/task_0__hybrid__SAC_hybridRAGRL__42",
    )
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--n-traj", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
