#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from models import ActionPredictionSummaryModel


def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_dataset(path: Path) -> Dict[str, torch.Tensor]:
    payload = safe_torch_load(path, map_location="cpu")
    required = {"obs", "actions"}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"{path} is missing keys: {sorted(missing)}")

    obs = payload["obs"].float()
    actions = payload["actions"].float()
    if obs.dim() != 3:
        raise ValueError(f"obs should be (N,T+1,obs_dim), got {tuple(obs.shape)}")
    if actions.dim() != 3:
        raise ValueError(f"actions should be (N,T,act_dim), got {tuple(actions.shape)}")
    if obs.size(0) != actions.size(0) or obs.size(1) != actions.size(1) + 1:
        raise ValueError(
            f"shape mismatch: obs={tuple(obs.shape)}, actions={tuple(actions.shape)}"
        )

    return {
        "obs": obs,
        "actions": actions,
        "meta": payload.get("meta", {}),
    }


def compute_obs_stats(obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    flat = obs.reshape(-1, obs.size(-1))
    mean = flat.mean(dim=0)
    std = flat.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


def evaluate(model, loader, obs_mean, obs_std, device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    loss_fn = nn.MSELoss(reduction="sum")

    with torch.no_grad():
        for obs_prefix, actions in loader:
            obs_prefix = obs_prefix.to(device)
            actions = actions.to(device)
            obs_prefix = (obs_prefix - obs_mean.to(device)) / obs_std.to(device)
            pred_actions, _ = model(obs_prefix)
            loss = loss_fn(pred_actions, actions)
            total_loss += float(loss.item())
            total_count += int(actions.numel())

    model.train()
    return total_loss / max(total_count, 1)


def save_checkpoint(
    path: Path,
    model: ActionPredictionSummaryModel,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    args,
    dataset_meta: Dict,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_type": "action_pretrained_summary",
            "summary_state_dict": model.summary.state_dict(),
            "action_head_state_dict": model.action_head.state_dict(),
            "obs_mean": obs_mean.detach().cpu(),
            "obs_std": obs_std.detach().cpu(),
            "meta": {
                "obs_dim": model.obs_dim,
                "action_dim": model.action_dim,
                "summary_dim": model.summary_dim,
                "hidden_dim": model.hidden_dim,
                "objective": "predict_action_from_prefix_summary",
                "loss": "mse(action_head(H_t), a_t)",
                "dataset": dataset_meta,
                "best_val_loss": best_val_loss,
                "args": vars(args),
            },
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretrain a frozen GRU Summary by predicting simple-teacher actions from every prefix H_t."
    )
    parser.add_argument("--dataset", type=Path, default=Path("summary_datasets/simple_teachers_summary_dataset.pt"))
    parser.add_argument("--output", type=Path, default=Path("summary_checkpoints/action_summary.pt"))
    parser.add_argument("--summary-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-every", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    dataset = load_dataset(args.dataset)
    obs = dataset["obs"]
    actions = dataset["actions"]

    obs_prefix = obs[:, :-1, :]
    obs_mean, obs_std = compute_obs_stats(obs_prefix)

    tensor_dataset = TensorDataset(obs_prefix, actions)
    val_size = max(1, int(len(tensor_dataset) * args.val_ratio)) if args.val_ratio > 0 else 0
    train_size = len(tensor_dataset) - val_size
    if train_size <= 0:
        raise ValueError("training split is empty; reduce --val-ratio or collect more episodes.")

    generator = torch.Generator().manual_seed(args.seed)
    if val_size > 0:
        train_dataset, val_dataset = random_split(
            tensor_dataset,
            [train_size, val_size],
            generator=generator,
        )
    else:
        train_dataset = tensor_dataset
        val_dataset = tensor_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = ActionPredictionSummaryModel(
        obs_dim=obs.size(-1),
        action_dim=actions.size(-1),
        summary_dim=args.summary_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()
    best_val_loss = float("inf")

    print("*** Summary action pretraining ***")
    print(f"dataset: {args.dataset}")
    print(f"obs_prefix: {tuple(obs_prefix.shape)}")
    print(f"actions: {tuple(actions.shape)}")
    print(f"device: {device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_batches = 0

        for batch_obs, batch_actions in train_loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)
            batch_obs = (batch_obs - obs_mean.to(device)) / obs_std.to(device)

            pred_actions, _ = model(batch_obs)
            loss = loss_fn(pred_actions, batch_actions)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            running_loss += float(loss.item())
            running_batches += 1

        train_loss = running_loss / max(running_batches, 1)
        val_loss = evaluate(
            model=model,
            loader=val_loader,
            obs_mean=obs_mean,
            obs_std=obs_std,
            device=device,
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            save_checkpoint(
                path=args.output,
                model=model,
                obs_mean=obs_mean,
                obs_std=obs_std,
                args=args,
                dataset_meta=dataset["meta"],
                best_val_loss=best_val_loss,
            )

        print(
            f"epoch {epoch:04d}/{args.epochs}: "
            f"train_mse={train_loss:.6f}, val_mse={val_loss:.6f}, "
            f"best={best_val_loss:.6f}"
        )

        if args.save_every > 0 and epoch % args.save_every == 0:
            checkpoint_path = args.output.with_name(f"{args.output.stem}_epoch_{epoch}{args.output.suffix}")
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                obs_mean=obs_mean,
                obs_std=obs_std,
                args=args,
                dataset_meta=dataset["meta"],
                best_val_loss=best_val_loss,
            )

    with args.output.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "output": str(args.output),
                "best_val_loss": best_val_loss,
                "obs_shape": tuple(obs.shape),
                "actions_shape": tuple(actions.shape),
                "summary_dim": args.summary_dim,
            },
            f,
            indent=2,
        )

    print(f"\nsaved best summary checkpoint: {args.output}")


if __name__ == "__main__":
    main()
