#!/usr/bin/env python3
"""
record_componet_task2.py

Use a trained CompoNet actor to interact with Meta-World task 2 and save an mp4 video.

Run this file from your experiments/meta-world directory, where tasks.py and models/ exist.

Example:
python record_componet_task2.py \
  --model-dir agents/task_2__componet__SAC__42 \
  --prev-paths agents/task_0__simple__SAC__42 agents/task_1__componet__SAC__42 \
  --task-id 2 \
  --episodes 3 \
  --max-steps 500 \
  --output videos/task2_componet.mp4 \
  --device cuda
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn


LOG_STD_MAX = 2
LOG_STD_MIN = -20


def safe_torch_load(path: str, map_location):
    """torch.load wrapper compatible with older and newer PyTorch."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def import_componet_agent():
    """Try the same import style as your training code."""
    try:
        from models import CompoNetAgent
        return CompoNetAgent
    except Exception:
        from models.compo import CompoNetAgent
        return CompoNetAgent


class LoadedCompoNetAgent(nn.Module):
    """
    Direct wrapper for saved net_mean.pt / net_logstd.pt.

    This works only if loaded net_mean still has previous_units.
    If previous_units were deleted before saving, use load_componet_agent(...)
    with --prev-paths instead.
    """

    def __init__(self, net_mean: nn.Module, net_logstd: nn.Module):
        super().__init__()
        self.net_mean = net_mean
        self.net_logstd = net_logstd

    def forward(self, x):
        out = self.net_mean(x)
        if isinstance(out, (list, tuple)):
            mean = out[0]
        else:
            mean = out
        logstd = self.net_logstd(x)
        return mean, logstd


def load_componet_agent(
    model_dir: str,
    obs_dim: int,
    act_dim: int,
    prev_paths: Optional[List[str]],
    device: torch.device,
) -> nn.Module:
    """
    Load a trained CompoNet actor.

    Preferred route:
        Rebuild CompoNetAgent(prev_paths), then load saved net_mean/net_logstd weights.

    Fallback route:
        Directly load net_mean.pt and net_logstd.pt into LoadedCompoNetAgent.
    """
    model_dir = str(model_dir)
    net_mean_path = os.path.join(model_dir, "net_mean.pt")
    net_logstd_path = os.path.join(model_dir, "net_logstd.pt")

    assert os.path.exists(net_mean_path), f"Cannot find {net_mean_path}"
    assert os.path.exists(net_logstd_path), f"Cannot find {net_logstd_path}"

    map_location = device

    if prev_paths is not None and len(prev_paths) > 0:
        CompoNetAgent = import_componet_agent()

        model = CompoNetAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            prev_paths=prev_paths,
            map_location=map_location,
        )

        saved_net_mean = safe_torch_load(net_mean_path, map_location=map_location)
        saved_net_logstd = safe_torch_load(net_logstd_path, map_location=map_location)

        model.net_logstd = saved_net_logstd

        current = model.net_mean.state_dict()
        saved = saved_net_mean.state_dict()

        matched = {}
        skipped = []
        for k, v in saved.items():
            if k in current and current[k].shape == v.shape:
                matched[k] = v
            else:
                skipped.append(k)

        current.update(matched)
        model.net_mean.load_state_dict(current)

        print(f"[load] loaded {len(matched)} net_mean tensors from {net_mean_path}")
        if skipped:
            print(f"[load] skipped {len(skipped)} tensors due to missing key/shape mismatch")

        return model.to(device).eval()

    net_mean = safe_torch_load(net_mean_path, map_location=map_location)
    net_logstd = safe_torch_load(net_logstd_path, map_location=map_location)
    model = LoadedCompoNetAgent(net_mean=net_mean, net_logstd=net_logstd)
    return model.to(device).eval()


class DeterministicSACActor(nn.Module):
    """
    Minimal SAC actor wrapper for deterministic evaluation/video recording.

    It uses:
        action = tanh(mean) * action_scale + action_bias
    """

    def __init__(self, env, model: nn.Module, device: torch.device):
        super().__init__()
        self.model = model
        self.device = device

        high = env.action_space.high
        low = env.action_space.low

        self.action_scale = torch.tensor(
            (high - low) / 2.0,
            dtype=torch.float32,
            device=device,
        )
        self.action_bias = torch.tensor(
            (high + low) / 2.0,
            dtype=torch.float32,
            device=device,
        )

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean, logstd = self.model(obs_t)

        _ = torch.tanh(logstd)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        action = action.squeeze(0).detach().cpu().numpy()
        return action.astype(np.float32)


def render_frame(env, width: int, height: int, camera_name: str):
    """
    Meta-World/MuJoCo rendering is version-dependent, so try several render signatures.
    """
    render_trials = [
        lambda: env.render(offscreen=True, camera_name=camera_name, resolution=(width, height)),
        lambda: env.render(mode="rgb_array", camera_name=camera_name, width=width, height=height),
        lambda: env.render(mode="rgb_array", camera_name=camera_name),
        lambda: env.render(),
    ]

    last_error = None
    frame = None

    for fn in render_trials:
        try:
            frame = fn()
            if frame is not None:
                break
        except Exception as e:
            last_error = e
            frame = None

    if frame is None:
        raise RuntimeError(
            "Failed to render RGB frame. Try another camera_name, or make sure "
            "MuJoCo offscreen rendering is configured. Last error: "
            f"{last_error}"
        )

    frame = np.asarray(frame)

    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]

    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing net_mean.pt and net_logstd.pt.")
    parser.add_argument("--prev-paths", type=str, nargs="*", default=None, help="Previous module dirs. For task2, usually task0 and task1 dirs.")
    parser.add_argument("--task-id", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output", type=str, default="videos/task2_componet.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-name", type=str, default="corner2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from tasks import get_task, get_task_name

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = get_task(args.task_id)
    print(f"[env] task_id={args.task_id}, task_name={get_task_name(args.task_id)}")
    print(f"[env] obs_shape={env.observation_space.shape}, action_shape={env.action_space.shape}")

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    model = load_componet_agent(
        model_dir=args.model_dir,
        obs_dim=obs_dim,
        act_dim=act_dim,
        prev_paths=args.prev_paths,
        device=device,
    )
    policy = DeterministicSACActor(env=env, model=model, device=device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    episode_returns = []
    episode_successes = []

    for ep in range(args.episodes):
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        ep_return = 0.0
        ep_success = 0.0

        frames.append(render_frame(env, args.width, args.height, args.camera_name))

        for step in range(args.max_steps):
            action = policy.act(obs)
            step_out = env.step(action)

            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, info = step_out
            else:
                next_obs, reward, done, info = step_out
                terminated, truncated = done, False

            obs = next_obs
            ep_return += float(reward)
            ep_success = max(ep_success, float(info.get("success", 0.0)))

            frames.append(render_frame(env, args.width, args.height, args.camera_name))

            if terminated or truncated:
                break

        episode_returns.append(ep_return)
        episode_successes.append(ep_success)
        print(
            f"[episode {ep}] return={ep_return:.3f}, "
            f"success={ep_success}, steps={step + 1}"
        )

    imageio.mimsave(output_path, frames, fps=args.fps, macro_block_size=1)

    print(f"[saved] {output_path}")
    print(f"[summary] avg_return={np.mean(episode_returns):.3f}, avg_success={np.mean(episode_successes):.3f}")


if __name__ == "__main__":
    main()
