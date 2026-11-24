"""Train a GRU trajectory encoder to predict rewards from state-action sequences.

Shared and per-task models use the same trajectory-level train/validation/test
splits. Run separately for MetaWorld and HalfCheetah because their observation
and action dimensions may differ.
"""

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


EXPECTED_TASK_COUNTS = {"meta-world": 10, "halfcheetah": 11}


@dataclass
class TaskData:
    task_id: int
    task_name: str
    path: Path
    features: np.ndarray  # (trajectories, steps, obs_dim + act_dim)
    rewards: np.ndarray   # (trajectories, steps)
    split: dict


class RewardTrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, head_dim=64):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim), nn.ReLU(), nn.Linear(head_dim, 1)
        )

    def forward(self, sequences):
        hidden_sequence, final_hidden = self.encoder(sequences)
        rewards = self.reward_head(hidden_sequence).squeeze(-1)
        return rewards, final_hidden[-1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=EXPECTED_TASK_COUNTS)
    parser.add_argument("--input-dir", type=Path,
                        help="Default: experiments/datasets/simple_trajectories/<domain>")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: experiments/analysis/simple_trajectory_encoder/<domain>")
    parser.add_argument("--tasks", type=int, nargs="+",
                        help="Selected task IDs; default requires all tasks in the domain")
    parser.add_argument("--mode", choices=("shared", "per-task", "both"), default="both")
    parser.add_argument("--trajectory-steps", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if any(value <= 0 for value in (args.trajectory_steps, args.hidden_dim, args.head_dim,
                                    args.batch_size, args.epochs, args.lr, args.patience)):
        parser.error("Trajectory length, model dimensions, batch size, epochs, lr, and patience must be positive")
    if args.weight_decay < 0 or args.log_every < 0:
        parser.error("--weight-decay and --log-every must be non-negative")
    if not (0 < args.val_ratio < 1 and 0 < args.test_ratio < 1 and
            args.val_ratio + args.test_ratio < 1):
        parser.error("Validation and test ratios must be positive and sum to less than 1")
    if args.tasks is not None and (len(set(args.tasks)) != len(args.tasks) or
                                   any(task < 0 or task >= EXPECTED_TASK_COUNTS[args.domain]
                                       for task in args.tasks)):
        parser.error("--tasks must contain distinct valid task IDs")
    return args


def split_indices(count, val_ratio, test_ratio, rng):
    val_count = max(1, int(count * val_ratio))
    test_count = max(1, int(count * test_ratio))
    if count - val_count - test_count < 1:
        raise ValueError(f"Need at least one train/validation/test trajectory; got {count}")
    indices = rng.permutation(count)
    train_end = count - val_count - test_count
    return {
        "train": indices[:train_end].tolist(),
        "val": indices[train_end:train_end + val_count].tolist(),
        "test": indices[train_end + val_count:].tolist(),
    }


def load_tasks(args, input_dir):
    paths = sorted(input_dir.glob("task_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No task_*.npz files in {input_dir}")
    requested = set(args.tasks) if args.tasks is not None else set(range(EXPECTED_TASK_COUNTS[args.domain]))
    tasks = {}
    rng = np.random.default_rng(args.seed)
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            task_id = int(metadata["task_id"])
            if task_id not in requested:
                continue
            if metadata["domain"] != args.domain:
                raise ValueError(f"{path}: expected domain {args.domain}, got {metadata['domain']}")
            if task_id in tasks:
                raise ValueError(f"Duplicate dataset for task {task_id}")
            length = args.trajectory_steps
            if metadata.get("trajectory_steps") != length:
                raise ValueError(f"{path}: expected fixed {length}-step trajectories; "
                                 "recollect with --trajectories-per-task and --trajectory-steps")
            states = np.asarray(data["states"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.float32)
            rewards = np.asarray(data["rewards"], dtype=np.float32)
            episode_id = np.asarray(data["episode_id"])
            step_in_episode = np.asarray(data["step_in_episode"])
            terminated = np.asarray(data["terminated"])
            truncated = np.asarray(data["truncated"])
        rows = len(rewards)
        if (states.ndim != 2 or actions.ndim != 2 or rewards.ndim != 1 or
                rows == 0 or rows % length != 0 or len(states) != rows or
                len(actions) != rows or any(len(array) != rows for array in
                                            (episode_id, step_in_episode, terminated, truncated))):
            raise ValueError(f"{path}: invalid or misaligned transition arrays")
        count = rows // length
        if (not np.array_equal(episode_id, np.repeat(np.arange(count), length)) or
                not np.array_equal(step_in_episode, np.tile(np.arange(length), count)) or
                np.any(terminated.reshape(count, length)[:, :-1]) or
                np.any(truncated.reshape(count, length)[:, :-1])):
            raise ValueError(f"{path}: trajectories are not complete contiguous {length}-step sequences")
        features = np.concatenate((states, actions), axis=1).reshape(count, length, -1)
        rewards = rewards.reshape(count, length)
        if not np.isfinite(features).all() or not np.isfinite(rewards).all():
            raise ValueError(f"{path}: features or rewards contain NaN or infinity")
        tasks[task_id] = TaskData(task_id, metadata["task_name"], path, features,
                                  rewards, split_indices(count, args.val_ratio,
                                                         args.test_ratio, rng))
    missing = requested - tasks.keys()
    if missing:
        raise FileNotFoundError(f"Missing trajectory datasets for task IDs {sorted(missing)} in {input_dir}")
    tasks = [tasks[task_id] for task_id in sorted(tasks)]
    dimensions = {task.features.shape[-1] for task in tasks}
    if len(dimensions) != 1:
        raise ValueError(f"State-action dimensions differ across tasks: {sorted(dimensions)}; "
                         "train each domain separately")
    return tasks


def make_dataset(tasks, split):
    features = np.concatenate([task.features[task.split[split]] for task in tasks])
    rewards = np.concatenate([task.rewards[task.split[split]] for task in tasks])
    task_ids = np.concatenate([np.full(len(task.split[split]), task.task_id,
                                        dtype=np.int64) for task in tasks])
    return TensorDataset(torch.from_numpy(features), torch.from_numpy(rewards),
                         torch.from_numpy(task_ids))


def train_stats(dataset):
    features = dataset.tensors[0]
    mean = features.mean(dim=(0, 1))
    std = features.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)
    return mean, std


@torch.no_grad()
def evaluate(model, dataset, mean, std, device, batch_size):
    model.eval()
    totals = {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for features, rewards, task_ids in loader:
        prediction, _ = model((features.to(device) - mean) / std)
        errors = prediction - rewards.to(device)
        task_ids = task_ids.to(device)
        for task_id in task_ids.unique().tolist():
            selected = errors[task_ids == task_id]
            row = totals.setdefault(int(task_id), [0.0, 0.0, 0])
            row[0] += selected.square().sum().item()
            row[1] += selected.abs().sum().item()
            row[2] += selected.numel()
    overall = [sum(row[i] for row in totals.values()) for i in range(3)]
    def metrics(row):
        return {"mse": row[0] / row[2], "mae": row[1] / row[2], "steps": row[2]}
    return {"overall": metrics(overall),
            "by_task": {str(task_id): metrics(row) for task_id, row in sorted(totals.items())}}


def train_model(tasks, args, label, device):
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    train_set = make_dataset(tasks, "train")
    val_set = make_dataset(tasks, "val")
    test_set = make_dataset(tasks, "test")
    mean_cpu, std_cpu = train_stats(train_set)
    mean, std = mean_cpu.to(device), std_cpu.to(device)
    model = RewardTrajectoryEncoder(train_set.tensors[0].shape[-1],
                                    args.hidden_dim, args.head_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_squared_error = 0.0
        train_steps = 0
        for features, rewards, _ in loader:
            features, rewards = features.to(device), rewards.to(device)
            prediction, _ = model((features - mean) / std)
            loss = (prediction - rewards).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_squared_error += loss.item() * rewards.numel()
            train_steps += rewards.numel()
        val_metrics = evaluate(model, val_set, mean, std, device,
                               args.batch_size)["overall"]
        val_mse = val_metrics["mse"]
        if val_mse < best_val:
            best_val, best_epoch = val_mse, epoch
            best_state = {name: tensor.detach().cpu().clone()
                          for name, tensor in model.state_dict().items()}
        if args.log_every and (epoch % args.log_every == 0 or epoch == 1 or
                               epoch == args.epochs):
            print(f"{label} epoch={epoch}/{args.epochs} "
                  f"train_mse={train_squared_error / train_steps:.6g} "
                  f"val_mse={val_mse:.6g} best_epoch={best_epoch}", flush=True)
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    results = {
        "best_epoch": best_epoch,
        "train": evaluate(model, train_set, mean, std, device, args.batch_size),
        "val": evaluate(model, val_set, mean, std, device, args.batch_size),
        "test": evaluate(model, test_set, mean, std, device, args.batch_size),
    }
    checkpoint = {
        "encoder_state_dict": {name: tensor.detach().cpu().clone()
                               for name, tensor in model.encoder.state_dict().items()},
        "reward_head_state_dict": {name: tensor.detach().cpu().clone()
                                   for name, tensor in model.reward_head.state_dict().items()},
        "feature_mean": mean_cpu,
        "feature_std": std_cpu,
        "config": {
            "input_dim": train_set.tensors[0].shape[-1],
            "hidden_dim": args.hidden_dim,
            "head_dim": args.head_dim,
            "trajectory_steps": args.trajectory_steps,
            "task_ids": [task.task_id for task in tasks],
            "domain": args.domain,
            "feature_order": "state_action",
        },
        "results": results,
    }
    return checkpoint, results


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    experiment_dir = Path(__file__).resolve().parent
    input_dir = args.input_dir or experiment_dir / "datasets" / "simple_trajectories" / args.domain
    output_dir = args.output_dir or experiment_dir / "analysis" / "simple_trajectory_encoder" / args.domain
    tasks = load_tasks(args, input_dir)
    jobs = []
    if args.mode in ("shared", "both"):
        jobs.append(("shared", tasks, output_dir / "shared.pt"))
    if args.mode in ("per-task", "both"):
        jobs.extend((f"task_{task.task_id:02d}", [task],
                     output_dir / f"task_{task.task_id:02d}.pt") for task in tasks)
    outputs = [path for _, _, path in jobs] + [output_dir / "splits.json", output_dir / "summary.json"]
    if args.mode == "both":
        outputs.append(output_dir / "comparison.csv")
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"{existing[0]} exists; use --overwrite or --output-dir")
    device = torch.device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {str(task.task_id): {"task_name": task.task_name,
                                   "dataset": str(task.path.resolve()),
                                   "indices": task.split} for task in tasks}
    (output_dir / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")
    summary = {"domain": args.domain, "settings": vars(args).copy(), "models": {}}
    summary["settings"]["input_dir"] = str(input_dir)
    summary["settings"]["output_dir"] = str(output_dir)
    for label, model_tasks, path in jobs:
        print(f"Training {label} on tasks {[task.task_id for task in model_tasks]}", flush=True)
        checkpoint, results = train_model(model_tasks, args, label, device)
        torch.save(checkpoint, path)
        summary["models"][label] = {"checkpoint": str(path), **results}
        print(f"Saved {path}; test_mse={results['test']['overall']['mse']:.6g}", flush=True)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.mode == "both":
        with (output_dir / "comparison.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("task_id", "task_name", "test_trajectories",
                             "shared_mse", "per_task_mse", "shared_mae", "per_task_mae"))
            for task in tasks:
                shared = summary["models"]["shared"]["test"]["by_task"][str(task.task_id)]
                own = summary["models"][f"task_{task.task_id:02d}"]["test"]["overall"]
                writer.writerow((task.task_id, task.task_name, len(task.split["test"]),
                                 shared["mse"], own["mse"], shared["mae"], own["mae"]))


if __name__ == "__main__":
    main()
