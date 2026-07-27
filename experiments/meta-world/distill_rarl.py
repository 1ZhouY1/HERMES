#!/usr/bin/env python3
"""
Distill a saved RARL actor into a memory-free SimpleAgent.

Teacher:
    RARLAgent loaded from <teacher-dir>/model.pt. The teacher uses its saved
    PairMemory / RewardReplacingPairMemory while collecting distillation labels.

Student:
    SimpleAgent with the same architecture as experiments/meta-world/models/simple.py.
    It only receives the current observation at inference time.

Critics are intentionally not loaded, trained, or saved.
"""

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import PairMemory, RewardReplacingPairMemory, SimpleAgent


LOG_STD_MAX = 2
LOG_STD_MIN = -20


def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_model_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_dir():
        path = path / "model.pt"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find teacher model: {path}")
    return path


def parse_task_id(path: Path) -> Optional[int]:
    match = re.search(r"task_(\d+)__", path.name)
    if match is None and path.name == "model.pt":
        match = re.search(r"task_(\d+)__", path.parent.name)
    return None if match is None else int(match.group(1))


def sac_log_std(raw_log_std: torch.Tensor) -> torch.Tensor:
    log_std = torch.tanh(raw_log_std)
    return LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)


def forward_policy(model: nn.Module, obs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    mean, raw_log_std = model(obs, **kwargs)
    return mean, sac_log_std(raw_log_std)


def gaussian_kl(
    teacher_mean: torch.Tensor,
    teacher_log_std: torch.Tensor,
    student_mean: torch.Tensor,
    student_log_std: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    log_temp = math.log(temperature)
    teacher_log_std = teacher_log_std + log_temp
    student_log_std = student_log_std + log_temp

    teacher_var = torch.exp(2.0 * teacher_log_std)
    student_var = torch.exp(2.0 * student_log_std).clamp_min(1e-12)
    kl_per_dim = (
        student_log_std
        - teacher_log_std
        + (teacher_var + (teacher_mean - student_mean).pow(2)) / (2.0 * student_var)
        - 0.5
    )
    return kl_per_dim.sum(dim=-1).mean()


def policy_kl(
    teacher_mean: torch.Tensor,
    teacher_log_std: torch.Tensor,
    student_mean: torch.Tensor,
    student_log_std: torch.Tensor,
    temperature: float,
    direction: str,
) -> torch.Tensor:
    if direction == "forward":
        return gaussian_kl(
            teacher_mean=teacher_mean,
            teacher_log_std=teacher_log_std,
            student_mean=student_mean,
            student_log_std=student_log_std,
            temperature=temperature,
        )
    if direction == "reverse":
        return gaussian_kl(
            teacher_mean=student_mean,
            teacher_log_std=student_log_std,
            student_mean=teacher_mean,
            student_log_std=teacher_log_std,
            temperature=temperature,
        )
    raise ValueError(f"Unknown KL direction: {direction}")


def teacher_action_from_policy(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "deterministic":
        pre_tanh_action = mean
    elif mode == "stochastic":
        normal = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh_action = normal.sample()
    else:
        raise ValueError(f"Unknown teacher action mode: {mode}")
    return torch.tanh(pre_tanh_action) * action_scale + action_bias


def load_pair_memory_file(path: Path):
    data = safe_torch_load(path, map_location="cpu")
    memory_cls = RewardReplacingPairMemory if "rewards" in data else PairMemory

    kwargs = {
        "capacity": data["capacity"],
        "obs_dim": data["obs_dim"],
        "action_dim": data["action_dim"],
        "device": "cpu",
        "dtype": data.get("dtype", torch.float32),
    }
    if memory_cls is RewardReplacingPairMemory:
        kwargs["replace_neighbors"] = data.get("replace_neighbors", 50)

    memory = memory_cls(**kwargs)
    memory.ptr = int(data["ptr"])
    memory.size = int(data["size"])
    memory.states.copy_(data["states"])
    memory.actions.copy_(data["actions"])
    if hasattr(memory, "rewards") and "rewards" in data:
        memory.rewards.copy_(data["rewards"])
        memory.num_replacements = int(data.get("num_replacements", 0))
        memory.num_skips = int(data.get("num_skips", 0))
    return memory


def attach_memory_from_sidecar(model: nn.Module, model_path: Path) -> None:
    if getattr(model, "memory", None) is not None:
        return

    model_dir = model_path.parent
    for filename in ["reward_replacing_pair_memory.pt", "pair_memory.pt"]:
        memory_path = model_dir / filename
        if memory_path.exists():
            model.memory = load_pair_memory_file(memory_path)
            return

    raise ValueError(
        f"Teacher model has no attached memory and no pair memory file was found in {model_dir}."
    )


def load_rarl_teacher(model_path: Path, device: torch.device) -> nn.Module:
    model = safe_torch_load(model_path, map_location=device)
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected model.pt to contain an nn.Module, got {type(model)}")
    if not hasattr(model, "retrieve_context"):
        raise TypeError(
            f"Expected a RARL-like model with retrieve_context(), got {type(model).__name__}"
        )

    attach_memory_from_sidecar(model, model_path)
    memory = getattr(model, "memory", None)
    if memory is None or len(memory) == 0:
        raise ValueError("RARL teacher memory is missing or empty.")

    return model.to(device).eval()


def reset_env(env, seed=None):
    out = env.reset(seed=seed) if seed is not None else env.reset()
    return out[0] if isinstance(out, tuple) else out


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, bool(terminated or truncated), info
    obs, reward, done, info = out
    return obs, reward, bool(done), info


@torch.no_grad()
def collect_teacher_dataset(
    teacher: nn.Module,
    task_id: int,
    episodes: int,
    max_steps: int,
    seed: int,
    teacher_action_mode: str,
    device: torch.device,
) -> Dict:
    from tasks import get_task

    env = get_task(task_id)
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

    observations = []
    teacher_means = []
    teacher_log_stds = []
    returns = []
    successes = []
    lengths = []
    memory_batch = teacher.memory.current(device=device)

    for ep in range(episodes):
        obs = reset_env(env, seed=seed + ep)
        ep_return = 0.0
        ep_success = 0.0
        ep_len = 0

        for step in range(1, max_steps + 1):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mean, log_std = forward_policy(
                teacher,
                obs_t,
                memory_batch=memory_batch,
            )
            action = teacher_action_from_policy(
                mean=mean.squeeze(0),
                log_std=log_std.squeeze(0),
                action_scale=action_scale,
                action_bias=action_bias,
                mode=teacher_action_mode,
            )

            observations.append(obs_t.squeeze(0).cpu())
            teacher_means.append(mean.squeeze(0).cpu())
            teacher_log_stds.append(log_std.squeeze(0).cpu())

            obs, reward, done, info = step_env(env, action.cpu().numpy().astype(np.float32))
            ep_return += float(reward)
            ep_success = max(ep_success, float(info.get("success", 0.0)))
            ep_len = step
            if done:
                break

        returns.append(ep_return)
        successes.append(ep_success)
        lengths.append(ep_len)
        print(
            f"collect episode {ep + 1:03d}/{episodes}: "
            f"success={ep_success:.0f}, return={ep_return:.2f}, length={ep_len}"
        )

    env.close()
    if not observations:
        raise RuntimeError("No distillation samples were collected.")

    return {
        "observations": torch.stack(observations, dim=0),
        "teacher_means": torch.stack(teacher_means, dim=0),
        "teacher_log_stds": torch.stack(teacher_log_stds, dim=0),
        "returns": returns,
        "successes": successes,
        "lengths": lengths,
    }


def sample_batch(dataset: Dict, batch_size: int, device: torch.device):
    n = int(dataset["observations"].size(0))
    idx = torch.randint(0, n, (batch_size,))
    return (
        dataset["observations"][idx].to(device=device, dtype=torch.float32),
        dataset["teacher_means"][idx].to(device=device, dtype=torch.float32),
        dataset["teacher_log_stds"][idx].to(device=device, dtype=torch.float32),
    )


def evaluate_student(
    student: SimpleAgent,
    task_id: int,
    episodes: int,
    max_steps: int,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    from tasks import get_task

    env = get_task(task_id)
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

    student.eval()
    returns = []
    successes = []
    with torch.no_grad():
        for ep in range(episodes):
            obs = reset_env(env, seed=seed + 10000 + ep)
            ep_return = 0.0
            ep_success = 0.0
            for _ in range(max_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                mean, _ = student(obs_t)
                action = torch.tanh(mean.squeeze(0)) * action_scale + action_bias
                obs, reward, done, info = step_env(env, action.cpu().numpy().astype(np.float32))
                ep_return += float(reward)
                ep_success = max(ep_success, float(info.get("success", 0.0)))
                if done:
                    break
            returns.append(ep_return)
            successes.append(ep_success)

    env.close()
    student.train()
    return {
        "success": float(np.mean(successes)),
        "return": float(np.mean(returns)),
    }


def build_meta(args, teacher_model_path: Path, teacher: nn.Module, dataset: Dict, step: int, best_loss: float):
    memory = getattr(teacher, "memory", None)
    return {
        "checkpoint_type": "rarl_distilled_simple",
        "student": "SimpleAgent",
        "teacher_model_path": str(teacher_model_path),
        "teacher_model_type": type(teacher).__name__,
        "task_id": int(args.task_id),
        "obs_dim": int(teacher.obs_dim),
        "act_dim": int(teacher.act_dim),
        "step": int(step),
        "best_loss": float(best_loss),
        "teacher_memory": {
            "type": type(memory).__name__ if memory is not None else None,
            "capacity": getattr(memory, "capacity", None),
            "size": len(memory) if memory is not None and hasattr(memory, "__len__") else None,
        },
        "distillation_data": {
            "source": "rarl_teacher_rollout",
            "samples": int(dataset["observations"].size(0)),
            "collect_episodes": int(args.collect_episodes),
            "collect_max_steps": int(args.collect_max_steps),
            "teacher_action_mode": args.teacher_action_mode,
            "success": float(np.mean(dataset["successes"])),
            "return": float(np.mean(dataset["returns"])),
            "length": float(np.mean(dataset["lengths"])),
        },
        "loss": {
            "type": "gaussian_policy_kl",
            "direction": args.kl_direction,
            "temperature": float(args.temperature),
            "mean_mse_coef": float(args.mean_mse_coef),
        },
        "args": vars(args),
    }


def save_student(student: SimpleAgent, save_dir: Path, meta: Dict) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    student.save(str(save_dir))
    with open(save_dir / "distill_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(args.device)
    teacher_model_path = resolve_model_path(Path(args.teacher))
    task_id = args.task_id
    if task_id is None:
        task_id = parse_task_id(teacher_model_path)
    if task_id is None:
        raise ValueError("Could not infer task id. Pass --task-id explicitly.")
    args.task_id = int(task_id)

    teacher = load_rarl_teacher(teacher_model_path, device=device)
    student = SimpleAgent(obs_dim=teacher.obs_dim, act_dim=teacher.act_dim).to(device)
    optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("*** RARL -> SimpleAgent distillation ***")
    print(f"teacher: {teacher_model_path}")
    print(f"task_id: {args.task_id}")
    print(f"device: {device}")
    print(f"teacher_memory: {type(teacher.memory).__name__}(size={len(teacher.memory)}, capacity={teacher.memory.capacity})")
    print(f"student: SimpleAgent(obs_dim={student.obs_dim}, act_dim={student.act_dim})")
    print("critic: not distilled")

    collect_episodes = 1 if args.dry_run else args.collect_episodes
    collect_max_steps = min(args.collect_max_steps, 5) if args.dry_run else args.collect_max_steps
    dataset = collect_teacher_dataset(
        teacher=teacher,
        task_id=args.task_id,
        episodes=collect_episodes,
        max_steps=collect_max_steps,
        seed=args.seed,
        teacher_action_mode=args.teacher_action_mode,
        device=device,
    )

    print(
        f"dataset: samples={int(dataset['observations'].size(0))}, "
        f"success={float(np.mean(dataset['successes'])):.4f}, "
        f"return={float(np.mean(dataset['returns'])):.2f}"
    )

    obs, teacher_mean, teacher_log_std = sample_batch(
        dataset,
        batch_size=min(args.batch_size, int(dataset["observations"].size(0))),
        device=device,
    )
    student_mean, student_log_std = forward_policy(student, obs)
    print(
        "shape check: "
        f"obs={tuple(obs.shape)}, teacher_mean={tuple(teacher_mean.shape)}, "
        f"student_mean={tuple(student_mean.shape)}"
    )

    if args.dry_run:
        print("dry run passed; no student was trained or saved.")
        return

    running_loss = 0.0
    running_kl = 0.0
    running_mse = 0.0
    best_loss = float("inf")
    student.train()

    for step in range(1, args.steps + 1):
        obs, teacher_mean, teacher_log_std = sample_batch(
            dataset,
            batch_size=args.batch_size,
            device=device,
        )
        student_mean, student_log_std = forward_policy(student, obs)
        kl_loss = policy_kl(
            teacher_mean=teacher_mean,
            teacher_log_std=teacher_log_std,
            student_mean=student_mean,
            student_log_std=student_log_std,
            temperature=args.temperature,
            direction=args.kl_direction,
        )
        mean_mse = (student_mean - teacher_mean).pow(2).mean()
        loss = kl_loss + args.mean_mse_coef * mean_mse

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
        optimizer.step()

        running_loss += float(loss.item())
        running_kl += float(kl_loss.item())
        running_mse += float(mean_mse.item())
        best_loss = min(best_loss, float(loss.item()))

        if step == 1 or step % args.log_every == 0:
            denom = 1 if step == 1 else args.log_every
            print(
                f"step {step:06d}/{args.steps}: "
                f"loss={running_loss / denom:.6f}, "
                f"kl={running_kl / denom:.6f}, "
                f"mean_mse={running_mse / denom:.6f}"
            )
            running_loss = 0.0
            running_kl = 0.0
            running_mse = 0.0

        if args.save_every > 0 and step % args.save_every == 0:
            checkpoint_dir = Path(args.save_dir).expanduser() / f"checkpoint_step_{step}"
            save_student(
                student,
                checkpoint_dir,
                build_meta(args, teacher_model_path, teacher, dataset, step, best_loss),
            )
            print(f"saved checkpoint: {checkpoint_dir / 'model.pt'}")

    save_dir = Path(args.save_dir).expanduser()
    meta = build_meta(args, teacher_model_path, teacher, dataset, args.steps, best_loss)
    save_student(student, save_dir, meta)
    print(f"saved distilled student: {save_dir / 'model.pt'}")
    print(f"saved metadata: {save_dir / 'distill_meta.json'}")

    if args.eval_episodes > 0:
        metrics = evaluate_student(
            student=student,
            task_id=args.task_id,
            episodes=args.eval_episodes,
            max_steps=args.eval_max_steps,
            seed=args.seed,
            device=device,
        )
        print(
            f"eval student: success={metrics['success']:.4f}, "
            f"return={metrics['return']:.2f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill one saved RARL teacher actor into a memory-free SimpleAgent."
    )
    parser.add_argument(
        "--teacher",
        required=True,
        help="Path to the RARL teacher directory or model.pt.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Meta-World task id. Inferred from task_* directory names when omitted.",
    )
    parser.add_argument("--save-dir", default="agents/distilled_simple_from_rarl__42")
    parser.add_argument("--collect-episodes", type=int, default=50)
    parser.add_argument("--collect-max-steps", type=int, default=500)
    parser.add_argument(
        "--teacher-action-mode",
        choices=["deterministic", "stochastic"],
        default="deterministic",
    )
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--kl-direction", choices=["forward", "reverse"], default="forward")
    parser.add_argument("--mean-mse-coef", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-deterministic", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dry-run", default=False, action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
