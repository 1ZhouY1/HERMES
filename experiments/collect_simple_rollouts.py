"""Collect state/action/reward trajectories from saved SimpleAgent policies.

Run this script separately for MetaWorld and HalfCheetah. Each task produces one
NPZ file with aligned, ordered transitions and a JSON metadata field.
"""

import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


DOMAIN_DIRS = {
    "meta-world": "meta-world",
    "halfcheetah": "halfcheetah",
}
MODEL_EXPERIMENT_PRIORITY = ("SAC", "SAC_hybridRAGRL", "run_sac", "sac_20k")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=DOMAIN_DIRS)
    parser.add_argument("--tasks", type=int, nargs="+", help="Task IDs; default: all distinct tasks")
    collection = parser.add_mutually_exclusive_group()
    collection.add_argument("--trajectories-per-task", type=int,
                            help="Complete trajectories per task (default: 500)")
    collection.add_argument("--steps-per-task", type=int,
                            help="Legacy mode: total transitions per task, across episodes")
    parser.add_argument("--trajectory-steps", type=int, default=20,
                        help="Transitions in each complete trajectory (default: 20)")
    parser.add_argument("--episode-steps", type=int,
                        help="Maximum steps per rollout (default: MetaWorld 500, HalfCheetah 1000)")
    parser.add_argument("--model-root", type=Path,
                        help="Search recursively under this directory (default: <domain>/agents)")
    parser.add_argument("--model-seed", type=int,
                        help="Training seed in the model directory name; required when a preferred model has multiple seeds")
    parser.add_argument("--model-template", type=str,
                        help="Exact model directory pattern; supports {task_id} and {model_seed}")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: experiments/datasets/simple_trajectories/<domain> in trajectory mode")
    parser.add_argument("--action-mode", choices=("deterministic", "stochastic"),
                        default="stochastic")
    parser.add_argument("--log-interval", type=int, default=5,
                        help="Print progress/SPS every N trajectories (or steps in legacy mode); 0 disables")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.trajectories_per_task is None and args.steps_per_task is None:
        args.trajectories_per_task = 500
    if any(value is not None and value <= 0 for value in
           (args.trajectories_per_task, args.steps_per_task, args.episode_steps)) or args.trajectory_steps <= 0:
        parser.error("Collection counts and lengths must be positive")
    if args.trajectories_per_task is not None and args.episode_steps is not None:
        parser.error("--episode-steps applies only to --steps-per-task mode")
    if args.log_interval < 0:
        parser.error("--log-interval must be non-negative")
    if args.model_template and args.model_root:
        parser.error("Use either --model-template or --model-root")
    return args


def model_details(path, task_id):
    prefix = f"task_{task_id}__simple__"
    _, separator, suffix = path.name.partition(prefix)
    if not separator:
        return None
    experiment, separator, seed = suffix.rpartition("__")
    if not separator or not experiment or not seed.isdecimal():
        return None
    return experiment, int(seed)


def find_model(task_id, args, domain_dir):
    if args.model_template:
        try:
            model_dir = Path(args.model_template.format(
                task_id=task_id, model_seed=args.model_seed))
        except (KeyError, ValueError) as exc:
            raise ValueError("Invalid --model-template format") from exc
        if not (model_dir / "model.pt").is_file():
            raise FileNotFoundError(f"No model.pt found in {model_dir}")
        details = model_details(model_dir, task_id)
        if args.model_seed is not None and details is not None and details[1] != args.model_seed:
            raise ValueError(f"Model directory {model_dir} has seed {details[1]}, "
                             f"not --model-seed {args.model_seed}")
        return model_dir

    root = args.model_root or domain_dir / "agents"
    pattern = f"**/*task_{task_id}__simple__*"
    candidates = sorted(path for path in root.glob(pattern)
                        if path.is_dir() and (path / "model.pt").is_file())
    if not candidates:
        raise FileNotFoundError(f"No SimpleAgent model.pt found for task {task_id}. "
                                "Set --model-root or --model-template.")

    named = [(path, model_details(path, task_id)) for path in candidates]
    if args.model_seed is not None:
        named = [(path, details) for path, details in named
                 if details is not None and details[1] == args.model_seed]
        if not named:
            raise FileNotFoundError(f"No SimpleAgent model.pt for task {task_id} "
                                    f"with model seed {args.model_seed} under {root}")
    for experiment in MODEL_EXPERIMENT_PRIORITY:
        matches = [path for path, details in named
                   if details is not None and details[0] == experiment]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Multiple {experiment} models found for task {task_id}: "
                             f"{matches}. Set --model-seed to select one.")
    raise ValueError(f"No supported simple model found for task {task_id}: "
                     f"{[path for path, _ in named]}. Use --model-template to select one.")


def reset_env(env, seed):
    result = env.reset(seed=seed)
    return result[0] if isinstance(result, tuple) else result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        next_obs, reward, terminated, truncated, _ = result
    else:
        next_obs, reward, done, _ = result
        terminated, truncated = done, False
    return next_obs, float(reward), bool(terminated), bool(truncated)


@torch.no_grad()
def policy_action(model, observation, action_scale, action_bias, mode, device):
    obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).reshape(1, -1)
    mean, raw_log_std = model(obs_tensor)
    if mode == "stochastic":
        log_std = -20.0 + 11.0 * (torch.tanh(raw_log_std) + 1.0)
        pre_tanh = torch.distributions.Normal(mean, log_std.exp()).sample()
    else:
        pre_tanh = mean
    return (torch.tanh(pre_tanh)[0] * action_scale + action_bias).cpu().numpy().astype(np.float32)


def append_transition(data, observation, action, next_obs, reward, terminated,
                      truncated, episode, step_in_episode):
    data["states"].append(np.asarray(observation, dtype=np.float32).reshape(-1).copy())
    data["actions"].append(action.copy())
    data["next_states"].append(np.asarray(next_obs, dtype=np.float32).reshape(-1).copy())
    data["rewards"].append(reward)
    data["terminated"].append(terminated)
    data["truncated"].append(truncated)
    data["episode_id"].append(episode)
    data["step_in_episode"].append(step_in_episode)


def collect_trajectories(env, model, scale, bias, args, device, task_seed,
                         data, task_id, task_name, started_at):
    attempts = 0
    max_attempts = args.trajectories_per_task * 10
    total_steps = args.trajectories_per_task * args.trajectory_steps
    while len(data["states"]) < total_steps:
        if attempts >= max_attempts:
            raise RuntimeError(f"Task {task_id} yielded fewer than {args.trajectories_per_task} "
                               f"complete {args.trajectory_steps}-step trajectories after "
                               f"{max_attempts} attempts; try a shorter --trajectory-steps")
        observation = reset_env(env, task_seed + attempts)
        attempts += 1
        trajectory = {key: [] for key in data}
        trajectory_id = len(data["states"]) // args.trajectory_steps
        for trajectory_step in range(args.trajectory_steps):
            action = policy_action(model, observation, scale, bias, args.action_mode, device)
            next_obs, reward, terminated, truncated = step_env(env, action)
            final_step = trajectory_step == args.trajectory_steps - 1
            append_transition(trajectory, observation, action, next_obs, reward, terminated,
                              truncated or (final_step and not terminated),
                              trajectory_id, trajectory_step)
            if (terminated or truncated) and not final_step:
                break
            observation = next_obs
        if len(trajectory["states"]) != args.trajectory_steps:
            continue
        for key in data:
            data[key].extend(trajectory[key])
        step = len(data["states"])
        if args.log_interval and ((trajectory_id + 1) % args.log_interval == 0 or
                                  trajectory_id + 1 == args.trajectories_per_task):
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            print(f"task={task_id:02d} {task_name} "
                  f"trajectory={trajectory_id + 1}/{args.trajectories_per_task} "
                  f"steps={step}/{total_steps} SPS={step / elapsed:.1f} "
                  f"elapsed={elapsed:.2f}s", flush=True)
    return attempts


def collect_task(task_id, task_name, model_path, get_task, args, device):
    task_seed = args.seed + task_id * 10_000
    random.seed(task_seed)
    np.random.seed(task_seed)
    torch.manual_seed(task_seed)
    env = get_task(task_id)
    try:
        try:
            model = torch.load(model_path / "model.pt", map_location=device, weights_only=False)
        except TypeError:
            model = torch.load(model_path / "model.pt", map_location=device)
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected an nn.Module in {model_path / 'model.pt'}")
        model = model.to(device).eval()
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        if model.obs_dim != obs_dim or model.act_dim != act_dim:
            raise ValueError(f"Model dimensions ({model.obs_dim}, {model.act_dim}) do not "
                             f"match task {task_id} ({obs_dim}, {act_dim})")
        scale = torch.as_tensor((env.action_space.high - env.action_space.low) / 2,
                                dtype=torch.float32, device=device)
        bias = torch.as_tensor((env.action_space.high + env.action_space.low) / 2,
                               dtype=torch.float32, device=device)

        data = {key: [] for key in ("states", "actions", "next_states", "rewards",
                                     "terminated", "truncated", "episode_id", "step_in_episode")}
        started_at = time.perf_counter()
        if args.trajectories_per_task is not None:
            attempts = collect_trajectories(env, model, scale, bias, args, device,
                                            task_seed, data, task_id, task_name, started_at)
        else:
            attempts = None
            episode = 0
            step_in_episode = 0
            observation = reset_env(env, task_seed)
            for step in range(1, args.steps_per_task + 1):
                action = policy_action(model, observation, scale, bias, args.action_mode, device)
                next_obs, reward, terminated, truncated = step_env(env, action)
                step_in_episode += 1
                truncated = truncated or (step_in_episode >= args.episode_steps and not terminated)
                append_transition(data, observation, action, next_obs, reward, terminated,
                                  truncated, episode, step_in_episode - 1)
                if terminated or truncated:
                    episode += 1
                    step_in_episode = 0
                    if len(data["states"]) < args.steps_per_task:
                        observation = reset_env(env, task_seed + episode)
                else:
                    observation = next_obs
                if args.log_interval and (step % args.log_interval == 0 or step == args.steps_per_task):
                    elapsed = max(time.perf_counter() - started_at, 1e-9)
                    print(f"task={task_id:02d} {task_name} "
                          f"step={step}/{args.steps_per_task} "
                          f"SPS={step / elapsed:.1f} elapsed={elapsed:.2f}s", flush=True)

        arrays = {key: np.asarray(values, dtype=np.float32 if key in
                  ("states", "actions", "next_states", "rewards") else
                  np.bool_ if key in ("terminated", "truncated") else np.int32)
                  for key, values in data.items()}
        arrays["metadata"] = np.asarray(json.dumps({
            "domain": args.domain,
            "task_id": task_id,
            "task_name": task_name,
            "model_path": str(model_path.resolve()),
            "model_seed": (model_details(model_path, task_id) or (None, None))[1],
            "steps": len(data["states"]),
            "trajectories_per_task": args.trajectories_per_task,
            "trajectory_steps": args.trajectory_steps if args.trajectories_per_task is not None else None,
            "trajectory_attempts": attempts,
            "episode_steps": args.episode_steps,
            "action_mode": args.action_mode,
            "seed": task_seed,
        }))
        return arrays
    finally:
        env.close()


def main():
    args = parse_args()
    if args.steps_per_task is not None and args.episode_steps is None:
        args.episode_steps = 500 if args.domain == "meta-world" else 1000
    domain_dir = Path(__file__).resolve().parent / DOMAIN_DIRS[args.domain]
    sys.path.insert(0, str(domain_dir))
    tasks_module = importlib.import_module("tasks")
    task_names = (tasks_module.single_tasks if args.domain == "meta-world"
                  else tasks_module.tasks)
    task_ids = args.tasks if args.tasks is not None else list(range(len(task_names)))
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("--tasks contains duplicate task IDs")
    if any(task_id < 0 or task_id >= len(task_names) for task_id in task_ids):
        raise ValueError(f"Task IDs must be in [0, {len(task_names) - 1}]")
    dataset_name = "simple_trajectories" if args.trajectories_per_task is not None else "simple_rollouts"
    output_dir = args.output_dir or Path(__file__).resolve().parent / "datasets" / dataset_name / args.domain
    planned = []
    for task_id in task_ids:
        model_path = find_model(task_id, args, domain_dir)
        output_path = output_dir / f"task_{task_id:02d}_{task_names[task_id]}.npz"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} exists; use --overwrite to replace it")
        planned.append((task_id, model_path, output_path))

    device = torch.device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (task_id, model_path, output_path) in enumerate(planned, start=1):
        print(f"[{index}/{len(planned)}] Collecting task {task_id:02d} "
              f"({task_names[task_id]}) with {model_path}", flush=True)
        arrays = collect_task(task_id, task_names[task_id], model_path,
                              tasks_module.get_task, args, device)
        np.savez_compressed(output_path, **arrays)
        print(f"Saved task {task_id:02d} ({task_names[task_id]}): "
              f"{len(arrays['states'])} transitions -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
