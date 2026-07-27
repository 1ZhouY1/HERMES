#!/usr/bin/env python3
"""
Distill multiple Hybrid RAGRL teachers into one memory-free student policy.

The teachers are expected to be saved as:

    <teacher_dir>/model.pt
    <teacher_dir>/long_term_memory.pt

The student is a SimpleAgent, so the final policy only needs observation input
and does not depend on LongTermMemory at test time.
"""

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import LongTermMemory, SimpleAgent


LOG_STD_MAX = 2
LOG_STD_MIN = -20


def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class LoadedHybridTeacher(nn.Module):
    """Wrap saved HybridRAGRLAgent pieces with the same mean/logstd contract."""

    def __init__(self, net_mean: nn.Module, net_logstd: Optional[nn.Module]):
        super().__init__()
        self.net_mean = net_mean
        self.net_logstd = net_logstd

    def forward(self, x: torch.Tensor, memory_batch: Dict[str, torch.Tensor]):
        if self.net_logstd is None:
            mean = self.net_mean(x, memory_batch=memory_batch)
            if isinstance(mean, (tuple, list)):
                mean = mean[0]
            return mean, torch.zeros_like(mean)

        mean, info = self.net_mean(
            x,
            memory_batch=memory_batch,
            return_info=True,
        )
        log_std = self.net_logstd(info["encoded_state"])
        return mean, log_std


@dataclass
class TeacherBundle:
    path: Path
    task_id: Optional[int]
    model: nn.Module
    memory: LongTermMemory
    n_traj: int


def parse_task_id(path: Path) -> int:
    match = re.search(r"task_(\d+)__", path.name)
    if match is None:
        raise ValueError(f"Cannot infer task id from teacher directory name: {path}")
    return int(match.group(1))


def try_parse_task_id(path: Path) -> Optional[int]:
    try:
        return parse_task_id(path)
    except ValueError:
        return None


def resolve_default_teacher_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    local_agents = script_dir / "agents"
    if local_agents.exists():
        return local_agents
    return Path("experiments/meta-world/agents")


def default_save_dir() -> Path:
    return Path(__file__).resolve().parent / "agents" / "distilled_student__hybrid_teachers__42"


def discover_teacher_dirs(
    teacher_root: Path,
    teacher_glob: str,
    task_ids: Optional[Sequence[int]],
) -> List[Path]:
    if not teacher_root.exists():
        raise FileNotFoundError(f"Teacher root does not exist: {teacher_root}")

    wanted = None if task_ids is None else set(int(t) for t in task_ids)
    dirs = []
    for path in sorted(teacher_root.glob(teacher_glob)):
        if not path.is_dir():
            continue
        if not (path / "model.pt").exists():
            continue
        if not (path / "long_term_memory.pt").exists():
            continue
        task_id = parse_task_id(path)
        if wanted is not None and task_id not in wanted:
            continue
        dirs.append(path)

    if not dirs:
        raise FileNotFoundError(
            f"No teacher directories matching {teacher_glob!r} found under {teacher_root}"
        )
    return dirs


def load_hybrid_teacher(model_dir: Path, device: torch.device) -> Tuple[nn.Module, Dict]:
    payload = safe_torch_load(model_dir / "model.pt", map_location=device)

    if isinstance(payload, dict) and "net_mean" in payload:
        model = LoadedHybridTeacher(
            net_mean=payload["net_mean"],
            net_logstd=payload.get("net_logstd"),
        )
        return model.to(device).eval(), payload.get("meta", {})

    if isinstance(payload, nn.Module):
        return payload.to(device).eval(), {}

    raise TypeError(
        f"Unsupported model checkpoint format in {model_dir / 'model.pt'}: {type(payload)}"
    )


def load_memory(model_dir: Path) -> LongTermMemory:
    data = safe_torch_load(model_dir / "long_term_memory.pt", map_location="cpu")
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


def load_teacher_bundle(
    model_dir: Path,
    device: torch.device,
    task_id: Optional[int] = None,
) -> TeacherBundle:
    model_dir = model_dir.expanduser()
    if not model_dir.exists():
        raise FileNotFoundError(f"Teacher directory does not exist: {model_dir}")
    if not (model_dir / "model.pt").exists():
        raise FileNotFoundError(f"Missing teacher model: {model_dir / 'model.pt'}")
    if not (model_dir / "long_term_memory.pt").exists():
        raise FileNotFoundError(f"Missing teacher memory: {model_dir / 'long_term_memory.pt'}")

    model, meta = load_hybrid_teacher(model_dir, device=device)
    memory = load_memory(model_dir)
    if len(memory) == 0:
        raise ValueError(f"Teacher memory is empty: {model_dir}")

    if task_id is None:
        task_id = try_parse_task_id(model_dir)
    n_traj = int(meta.get("n_traj", min(512, len(memory))))
    n_traj = max(1, min(n_traj, len(memory)))

    return TeacherBundle(
        path=model_dir,
        task_id=task_id,
        model=model,
        memory=memory,
        n_traj=n_traj,
    )


def sample_memory_batch(
    memory: LongTermMemory,
    n_traj: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
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


def sample_observations(
    memory: LongTermMemory,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    traj_idx = torch.randint(0, len(memory), (batch_size,), device=memory.states.device)
    step_idx = torch.randint(0, memory.traj_len, (batch_size,), device=memory.states.device)
    obs = memory.states[traj_idx, step_idx]
    return obs.to(device=device, dtype=torch.float32)


def sac_log_std(raw_log_std: torch.Tensor) -> torch.Tensor:
    log_std = torch.tanh(raw_log_std)
    return LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)


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


def forward_policy(model: nn.Module, obs: torch.Tensor, **kwargs):
    mean, raw_log_std = model(obs, **kwargs)
    return mean, sac_log_std(raw_log_std)


def load_or_create_student(
    obs_dim: int,
    act_dim: int,
    device: torch.device,
    resume_path: Optional[str],
) -> SimpleAgent:
    if resume_path is None:
        return SimpleAgent(obs_dim=obs_dim, act_dim=act_dim).to(device)

    path = Path(resume_path).expanduser()
    if path.is_dir():
        path = path / "model.pt"
    student = safe_torch_load(path, map_location=device)
    if not isinstance(student, SimpleAgent):
        raise TypeError(f"Resume checkpoint should contain a SimpleAgent, got {type(student)}")
    if student.obs_dim != obs_dim or student.act_dim != act_dim:
        raise ValueError(
            f"Resume student shape mismatch: got obs_dim={student.obs_dim}, act_dim={student.act_dim}; "
            f"expected obs_dim={obs_dim}, act_dim={act_dim}."
        )
    return student.to(device)


def save_student(student: SimpleAgent, save_dir: Path, meta: Dict) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    student.save(str(save_dir))
    with open(save_dir / "distill_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


@torch.no_grad()
def validate_distillation_batch(
    student: SimpleAgent,
    teacher: TeacherBundle,
    batch_size: int,
    temperature: float,
    kl_direction: str,
    device: torch.device,
) -> Dict[str, Tuple[int, ...]]:
    obs = sample_observations(
        memory=teacher.memory,
        batch_size=batch_size,
        device=device,
    )
    memory_batch = sample_memory_batch(
        memory=teacher.memory,
        n_traj=teacher.n_traj,
        device=device,
    )
    teacher_mean, teacher_log_std = forward_policy(
        teacher.model,
        obs,
        memory_batch=memory_batch,
    )
    student_mean, student_log_std = forward_policy(student, obs)
    loss = policy_kl(
        teacher_mean=teacher_mean,
        teacher_log_std=teacher_log_std,
        student_mean=student_mean,
        student_log_std=student_log_std,
        temperature=temperature,
        direction=kl_direction,
    )
    if loss.dim() != 0:
        raise ValueError(f"KL loss should be scalar, got shape {tuple(loss.shape)}")
    return {
        "obs": tuple(obs.shape),
        "teacher_mean": tuple(teacher_mean.shape),
        "teacher_log_std": tuple(teacher_log_std.shape),
        "student_mean": tuple(student_mean.shape),
        "student_log_std": tuple(student_log_std.shape),
    }


def choose_teacher(
    teachers: Sequence[TeacherBundle],
    balanced_index: int,
    balanced: bool,
) -> TeacherBundle:
    if balanced:
        return teachers[balanced_index % len(teachers)]
    return random.choice(teachers)


@torch.no_grad()
def evaluate_student(
    student: SimpleAgent,
    task_ids: Sequence[int],
    episodes: int,
    max_steps: int,
    seed: int,
    device: torch.device,
) -> Dict[int, Dict[str, float]]:
    from tasks import get_task

    student.eval()
    results = {}

    for task_id in task_ids:
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
        successes = []
        returns = []

        for ep in range(episodes):
            reset_out = env.reset(seed=seed + ep)
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            ep_return = 0.0
            ep_success = 0.0

            for _ in range(max_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                mean, _ = student(obs_t)
                action = torch.tanh(mean.squeeze(0)) * action_scale + action_bias
                step_out = env.step(action.cpu().numpy().astype(np.float32))

                if len(step_out) == 5:
                    obs, reward, terminated, truncated, info = step_out
                    done = bool(terminated or truncated)
                else:
                    obs, reward, done, info = step_out

                ep_return += float(reward)
                ep_success = max(ep_success, float(info.get("success", 0.0)))
                if done:
                    break

            successes.append(ep_success)
            returns.append(ep_return)

        env.close()
        results[int(task_id)] = {
            "success": float(np.mean(successes)),
            "return": float(np.mean(returns)),
        }

    student.train()
    return results


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(args.device)

    if args.teacher_dirs:
        teacher_dirs = [Path(p).expanduser() for p in args.teacher_dirs]
        if args.teacher_task_ids is not None and len(args.teacher_task_ids) != len(teacher_dirs):
            raise ValueError("--teacher-task-ids must have the same length as --teacher-dirs.")
    else:
        teacher_root = Path(args.teacher_root).expanduser() if args.teacher_root else resolve_default_teacher_root()
        teacher_dirs = discover_teacher_dirs(
            teacher_root=teacher_root,
            teacher_glob=args.teacher_glob,
            task_ids=args.tasks,
        )

    teachers = []
    for i, path in enumerate(teacher_dirs):
        task_id = None
        if args.teacher_task_ids is not None:
            task_id = int(args.teacher_task_ids[i])
        teachers.append(load_teacher_bundle(path, device=device, task_id=task_id))

    obs_dim = teachers[0].memory.obs_dim
    act_dim = teachers[0].memory.action_dim

    for teacher in teachers:
        if teacher.task_id is None and args.eval_episodes > 0:
            raise ValueError(
                f"Cannot evaluate {teacher.path} because its task id is unknown. "
                "Use task_*__... directory names or pass --teacher-task-ids."
            )
        if teacher.memory.obs_dim != obs_dim or teacher.memory.action_dim != act_dim:
            raise ValueError(
                "All teachers must share observation/action dimensions. "
                f"{teacher.path} has obs_dim={teacher.memory.obs_dim}, action_dim={teacher.memory.action_dim}; "
                f"expected obs_dim={obs_dim}, action_dim={act_dim}."
            )

    student = load_or_create_student(
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        resume_path=args.resume_student,
    )
    optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("*** Hybrid teacher distillation ***")
    print(f"device: {device}")
    print(f"student: SimpleAgent(obs_dim={obs_dim}, act_dim={act_dim})")
    for teacher in teachers:
        print(
            f"teacher task {teacher.task_id if teacher.task_id is not None else 'unknown'}: {teacher.path} "
            f"(memory={len(teacher.memory)} trajectories, n_traj={teacher.n_traj})"
        )

    shapes = validate_distillation_batch(
        student=student,
        teacher=teachers[0],
        batch_size=min(args.batch_size, 8),
        temperature=args.temperature,
        kl_direction=args.kl_direction,
        device=device,
    )
    print(f"shape check: {shapes}")
    if args.dry_run:
        print("dry run passed; no training was performed.")
        return

    running_loss = 0.0
    running_kl = 0.0
    running_mse = 0.0
    best_loss = float("inf")

    student.train()
    for step in range(1, args.steps + 1):
        teacher = choose_teacher(
            teachers,
            balanced_index=step - 1,
            balanced=args.balanced_teachers,
        )
        obs = sample_observations(
            memory=teacher.memory,
            batch_size=args.batch_size,
            device=device,
        )
        memory_batch = sample_memory_batch(
            memory=teacher.memory,
            n_traj=teacher.n_traj,
            device=device,
        )

        with torch.no_grad():
            teacher_mean, teacher_log_std = forward_policy(
                teacher.model,
                obs,
                memory_batch=memory_batch,
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
        mse_loss = (student_mean - teacher_mean).pow(2).mean()
        loss = kl_loss + args.mean_mse_coef * mse_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
        optimizer.step()

        running_loss += float(loss.item())
        running_kl += float(kl_loss.item())
        running_mse += float(mse_loss.item())
        best_loss = min(best_loss, float(loss.item()))

        if step % args.log_every == 0 or step == 1:
            denom = args.log_every if step % args.log_every == 0 else 1
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
            checkpoint_meta = build_meta(args, teachers, obs_dim, act_dim, step, best_loss)
            save_student(student, checkpoint_dir, checkpoint_meta)
            print(f"saved checkpoint: {checkpoint_dir / 'model.pt'}")

    save_dir = Path(args.save_dir).expanduser()
    meta = build_meta(args, teachers, obs_dim, act_dim, args.steps, best_loss)
    save_student(student, save_dir, meta)

    print(f"saved student model: {save_dir / 'model.pt'}")
    print(f"saved metadata: {save_dir / 'distill_meta.json'}")

    if args.eval_episodes > 0:
        eval_task_ids = sorted({int(teacher.task_id) for teacher in teachers if teacher.task_id is not None})
        results = evaluate_student(
            student=student,
            task_ids=eval_task_ids,
            episodes=args.eval_episodes,
            max_steps=args.eval_max_steps,
            seed=args.seed,
            device=device,
        )
        print("*** Evaluation ***")
        for task_id, metrics in results.items():
            print(
                f"task {task_id}: "
                f"success={metrics['success']:.4f}, return={metrics['return']:.2f}"
            )


def build_meta(
    args,
    teachers: Sequence[TeacherBundle],
    obs_dim: int,
    act_dim: int,
    step: int,
    best_loss: float,
) -> Dict:
    return {
        "student": "SimpleAgent",
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "step": step,
        "best_recent_loss": best_loss,
        "loss": {
            "type": "gaussian_policy_kl",
            "direction": args.kl_direction,
            "temperature": args.temperature,
            "mean_mse_coef": args.mean_mse_coef,
        },
        "teachers": [
            {
                "path": str(teacher.path),
                "task_id": teacher.task_id,
                "memory_size": len(teacher.memory),
                "n_traj": teacher.n_traj,
            }
            for teacher in teachers
        ],
        "args": vars(args),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Policy-distill Hybrid RAGRL teachers into one memory-free SimpleAgent student."
    )
    parser.add_argument(
        "--teacher-root",
        default=None,
        help="Root containing teacher directories. Default: experiments/meta-world/agents, or ./agents when run inside experiments/meta-world.",
    )
    parser.add_argument(
        "--teacher-glob",
        default="task_*__hybrid__SAC_hybridRAGRL__*",
        help="Glob used under --teacher-root when --teacher-dirs is not provided.",
    )
    parser.add_argument(
        "--teacher-dirs",
        nargs="*",
        default=None,
        help="Explicit teacher directories. Overrides --teacher-root and --teacher-glob.",
    )
    parser.add_argument(
        "--teacher-task-ids",
        type=int,
        nargs="*",
        default=None,
        help="Task ids for --teacher-dirs when ids cannot be inferred from directory names.",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="*",
        default=None,
        help="Optional task ids to keep when discovering teachers.",
    )
    parser.add_argument(
        "--save-dir",
        default=str(default_save_dir()),
        help="Output directory for the distilled student. Only model.pt and metadata are saved.",
    )
    parser.add_argument(
        "--resume-student",
        default=None,
        help="Optional SimpleAgent model.pt or directory to resume distillation from.",
    )
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Gaussian KL temperature. Values below 1 sharpen mean matching.",
    )
    parser.add_argument(
        "--kl-direction",
        choices=["forward", "reverse"],
        default="forward",
        help="forward means KL(teacher || student); reverse means KL(student || teacher).",
    )
    parser.add_argument(
        "--mean-mse-coef",
        type=float,
        default=0.0,
        help="Optional auxiliary MSE on teacher/student Gaussian means.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-deterministic", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--balanced-teachers",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Cycle teachers evenly instead of sampling teacher directories randomly.",
    )
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--eval-max-steps", type=int, default=500)
    parser.add_argument(
        "--dry-run",
        default=False,
        action="store_true",
        help="Load teachers and run one shape/loss check without training.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
