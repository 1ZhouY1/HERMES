#!/usr/bin/env python3
"""
Evaluate one memory-free policy on Meta-World tasks.

The architecture is inferred from model.pt. This supports distilled students
and finetune models saved as a single torch module.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from models import SimpleAgent
from tasks import get_task, get_task_name


@dataclass
class LoadedPolicy:
    model_dir: Path
    model: nn.Module
    meta: Dict
    policy_model_type: str


class DistillMLPAgent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        activation: str = "relu",
        layer_norm: bool = False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.activation = activation
        self.layer_norm = bool(layer_norm)

        if activation == "relu":
            activation_cls = nn.ReLU
        elif activation == "gelu":
            activation_cls = nn.GELU
        elif activation == "tanh":
            activation_cls = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        in_dim = self.obs_dim
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            if self.layer_norm:
                layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(activation_cls())
            in_dim = self.hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(self.hidden_dim, self.act_dim)
        self.fc_logstd = nn.Linear(self.hidden_dim, self.act_dim)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.fc_mean(h), self.fc_logstd(h)


def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def first_output(out):
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def default_model_dir() -> Path:
    return Path(__file__).resolve().parent / "agents" / "student__hybrid_distill__42"


def resolve_model_dir(path: str) -> Path:
    model_path = Path(path).expanduser()
    if model_path.name == "model.pt":
        model_path = model_path.parent
    if not (model_path / "model.pt").exists():
        raise FileNotFoundError(f"Cannot find model.pt in: {model_path}")
    return model_path


def build_distilled_student_from_checkpoint(payload: Dict) -> nn.Module:
    student_meta = payload.get("student", {})
    model_type = student_meta.get("model_type", payload.get("student_model_type", "simple"))
    obs_dim = int(payload["obs_dim"])
    act_dim = int(payload["act_dim"])

    if model_type == "simple":
        model = SimpleAgent(obs_dim=obs_dim, act_dim=act_dim)
    elif model_type == "mlp":
        model = DistillMLPAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=int(student_meta.get("hidden_dim", payload.get("hidden_dim", 512))),
            num_layers=int(student_meta.get("num_layers", payload.get("num_layers", 3))),
            activation=student_meta.get("activation", payload.get("activation", "relu")),
            layer_norm=bool(student_meta.get("layer_norm", payload.get("layer_norm", False))),
        )
    else:
        raise ValueError(f"Unsupported distilled student model_type: {model_type}")

    model.load_state_dict(payload["state_dict"])
    return model


def checkpoint_policy_model_type(payload) -> str:
    if isinstance(payload, dict):
        student_meta = payload.get("student", {})
        return student_meta.get("model_type", payload.get("student_model_type", "student"))
    return payload.__class__.__name__


def load_student_policy(model_dir: Path, device: torch.device) -> LoadedPolicy:
    payload = safe_torch_load(model_dir / "model.pt", map_location=device)
    if isinstance(payload, nn.Module):
        model = payload.to(device).eval()
        return LoadedPolicy(
            model_dir=model_dir,
            model=model,
            meta={},
            policy_model_type=checkpoint_policy_model_type(payload),
        )
    if isinstance(payload, dict) and payload.get("checkpoint_type") == "distilled_student":
        model = build_distilled_student_from_checkpoint(payload)
        return LoadedPolicy(
            model_dir=model_dir,
            model=model.to(device).eval(),
            meta=payload.get("meta", {}),
            policy_model_type=checkpoint_policy_model_type(payload),
        )
    if isinstance(payload, dict) and "state_dict" in payload and "obs_dim" in payload and "act_dim" in payload:
        model = build_distilled_student_from_checkpoint(payload)
        return LoadedPolicy(
            model_dir=model_dir,
            model=model.to(device).eval(),
            meta=payload.get("meta", {}),
            policy_model_type=checkpoint_policy_model_type(payload),
        )
    raise TypeError(f"Unsupported distilled student checkpoint in {model_dir / 'model.pt'}: {type(payload)}")


def reset_env(env, seed=None):
    out = env.reset(seed=seed) if seed is not None else env.reset()
    return out[0] if isinstance(out, tuple) else out


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, reward, done, info
    obs, reward, done, info = out
    return obs, reward, bool(done), info


@torch.no_grad()
def deterministic_action(policy: LoadedPolicy, obs, action_scale, action_bias, device: torch.device) -> np.ndarray:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    mean = first_output(policy.model(obs_t))
    action = torch.tanh(mean.squeeze(0)) * action_scale + action_bias
    return action.cpu().numpy().astype(np.float32)


@torch.no_grad()
def evaluate_task(
    policy: LoadedPolicy,
    task_id: int,
    episodes: int,
    max_steps: int,
    seed: int,
    device: torch.device,
) -> Dict:
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

    successes: List[float] = []
    returns: List[float] = []
    lengths: List[int] = []

    for ep in range(episodes):
        obs = reset_env(env, seed=seed + task_id * 1000 + ep)
        ep_return = 0.0
        ep_success = 0.0
        ep_len = 0

        for step in range(1, max_steps + 1):
            action = deterministic_action(
                policy=policy,
                obs=obs,
                action_scale=action_scale,
                action_bias=action_bias,
                device=device,
            )
            obs, reward, done, info = step_env(env, action)
            ep_return += float(reward)
            ep_success = max(ep_success, float(info.get("success", 0.0)))
            ep_len = step
            if done:
                break

        successes.append(ep_success)
        returns.append(ep_return)
        lengths.append(ep_len)
        print(
            f"task {task_id:02d} episode {ep + 1:02d}: "
            f"success={ep_success:.0f}, return={ep_return:.2f}, length={ep_len}"
        )

    env.close()
    return {
        "task_id": int(task_id),
        "task_name": get_task_name(task_id),
        "policy_model_type": policy.policy_model_type,
        "model_dir": str(policy.model_dir),
        "episodes": int(episodes),
        "success_rate": float(np.mean(successes)),
        "average_return": float(np.mean(returns)),
        "average_length": float(np.mean(lengths)),
        "successes": successes,
        "returns": returns,
        "lengths": lengths,
    }


def evaluate(args):
    if args.per_task_models:
        raise ValueError(
            "eval_distilled_student.py now evaluates one distilled student model. "
            "Pass the student directory with --model."
        )

    device = torch.device(args.device)
    model_dir = resolve_model_dir(args.model)
    policy = load_student_policy(model_dir=model_dir, device=device)

    print("*** Memory-free policy evaluation ***")
    print(f"model: {model_dir}")
    print(f"policy_model_type: {policy.policy_model_type}")
    print(f"device: {device}")
    print(f"tasks: {args.tasks}")
    print(f"episodes per task: {args.episodes}")

    task_results = []
    for task_id in args.tasks:
        result = evaluate_task(
            policy=policy,
            task_id=task_id,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            device=device,
        )
        task_results.append(result)
        print(
            f"task {task_id:02d} ({result['task_name']}): "
            f"success_rate={result['success_rate']:.4f}, "
            f"average_return={result['average_return']:.2f}"
        )

    task_success_rates = [result["success_rate"] for result in task_results]
    all_episode_successes = [
        success
        for result in task_results
        for success in result["successes"]
    ]
    summary = {
        "model": str(model_dir),
        "policy_model_type": policy.policy_model_type,
        "episodes_per_task": args.episodes,
        "tasks": task_results,
        "mean_task_success_rate": float(np.mean(task_success_rates)),
        "overall_episode_success_rate": float(np.mean(all_episode_successes)),
    }

    print("\n*** Summary ***")
    for result in task_results:
        print(
            f"task {result['task_id']:02d} {result['task_name']}: "
            f"{result['success_rate']:.4f}"
        )
    print(f"mean_task_success_rate: {summary['mean_task_success_rate']:.4f}")
    print(f"overall_episode_success_rate: {summary['overall_episode_success_rate']:.4f}")

    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"saved json: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one memory-free policy across Meta-World tasks."
    )
    parser.add_argument(
        "--model",
        default=str(default_model_dir()),
        help="Memory-free policy model directory or model.pt.",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="*",
        default=None,
        help="Task ids to evaluate. Overrides --task-start/--task-end.",
    )
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument(
        "--task-end",
        type=int,
        default=9,
        help="Inclusive task id end when --tasks is omitted.",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save detailed evaluation results as JSON.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--model-type", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--per-task-models", default=False, action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-root", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model-glob", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--n-traj", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.tasks is None:
        if args.task_end < args.task_start:
            raise ValueError("--task-end must be greater than or equal to --task-start")
        args.tasks = list(range(args.task_start, args.task_end + 1))
    return args


if __name__ == "__main__":
    evaluate(parse_args())
