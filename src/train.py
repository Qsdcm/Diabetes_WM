#!/usr/bin/env python3
"""Train, validate, and test the V1 world model or direct GRU baseline."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.data import (
    NormalizationStats,
    WindowDataset,
    compute_train_normalization,
    dataset_summary,
    make_loader,
)
from src.metrics import RolloutMetrics
from src.models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    *,
    amp: bool,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    criterion = nn.MSELoss()
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            prediction = model(batch["history"], batch["future_controls"])
            loss = criterion(prediction, batch["target_cgm"])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        batch_size = batch["history"].shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    normalization: NormalizationStats,
    device: torch.device,
    *,
    amp: bool,
) -> dict[str, Any]:
    model.eval()
    metrics = RolloutMetrics()
    normalized_squared_error = 0.0
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=amp):
            prediction = model(batch["history"], batch["future_controls"])
        target = batch["target_cgm"]
        normalized_squared_error += (prediction - target).square().sum().item()
        count += target.numel()
        metrics.update(
            normalization.denormalize_cgm(prediction),
            normalization.denormalize_cgm(target),
        )
    return {
        "normalized_mse": normalized_squared_error / max(count, 1),
        "metrics": metrics.compute(),
    }


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_rmse: float,
    normalization: NormalizationStats,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_rmse": best_val_rmse,
            "normalization": normalization.to_dict(),
            "model_name": args.model,
            "model_config": {
                "embedding_dim": args.embedding_dim,
                "event_dim": args.event_dim,
                "hidden_dim": args.hidden_dim,
                "num_history_layers": args.num_history_layers,
                "dropout": args.dropout,
            },
        },
        path,
    )


def build_datasets(
    args: argparse.Namespace, normalization: NormalizationStats
) -> tuple[WindowDataset, WindowDataset, WindowDataset]:
    common = {
        "dataset_root": args.dataset_root,
        "normalization": normalization,
        "history_steps": args.history_steps,
        "horizon_steps": args.horizon_steps,
        "validate": True,
    }
    train = WindowDataset(
        index_file=args.index_root / "train_windows.csv",
        sample_stride=args.train_window_stride,
        max_windows=args.max_train_windows,
        seed=args.seed,
        **common,
    )
    val = WindowDataset(
        index_file=args.index_root / "val_windows.csv",
        sample_stride=args.eval_window_stride,
        max_windows=args.max_val_windows,
        seed=args.seed + 1,
        **common,
    )
    test = WindowDataset(
        index_file=args.index_root / "test_windows.csv",
        sample_stride=args.eval_window_stride,
        max_windows=args.max_test_windows,
        seed=args.seed + 2,
        **common,
    )
    return train, val, test


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    amp = args.amp and device.type == "cuda"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    normalization = compute_train_normalization(args.dataset_root, args.index_root)
    train_data, val_data, test_data = build_datasets(args, normalization)
    datasets = {"train": train_data, "val": val_data, "test": test_data}
    print(json.dumps({key: dataset_summary(value) for key, value in datasets.items()}, indent=2))
    print(f"device={device}, amp={amp}")

    loaders = {
        name: make_loader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.num_workers,
            seed=args.seed,
            pin_memory=device.type == "cuda",
        )
        for name, dataset in datasets.items()
    }
    model_config = {
        "embedding_dim": args.embedding_dim,
        "event_dim": args.event_dim,
        "hidden_dim": args.hidden_dim,
        "num_history_layers": args.num_history_layers,
        "dropout": args.dropout,
    }
    model = build_model(args.model, **model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_path = args.output_dir / "best.pt"
    best_val_rmse = float("inf")
    stale_epochs = 0
    history_path = args.output_dir / "history.jsonl"
    history_path.write_text("")

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss = train_epoch(
            model,
            loaders["train"],
            optimizer,
            scaler,
            device,
            amp=amp,
            grad_clip=args.grad_clip,
        )
        validation = evaluate(model, loaders["val"], normalization, device, amp=amp)
        val_rmse = validation["metrics"]["overall"]["rmse_mg_dl"]
        scheduler.step(val_rmse)
        record = {
            "epoch": epoch,
            "train_normalized_mse": train_loss,
            "validation": validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"epoch={epoch:03d} train_mse={train_loss:.6f} "
            f"val_rmse={val_rmse:.3f} mg/dL time={record['seconds']:.1f}s"
        )
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            stale_epochs = 0
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_rmse=best_val_rmse,
                normalization=normalization,
                args=args,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping after {epoch} epochs")
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_result = evaluate(model, loaders["test"], normalization, device, amp=amp)
    result = {
        "model": args.model,
        "best_epoch": checkpoint["epoch"],
        "normalization": normalization.to_dict(),
        "dataset_summary": {key: dataset_summary(value) for key, value in datasets.items()},
        "test": test_result,
    }
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["test"], indent=2))


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=project_root / "Dataset_5min")
    parser.add_argument(
        "--index-root", type=Path, default=project_root / "Dataset_5min/window_index_v1"
    )
    parser.add_argument("--output-dir", type=Path, default=project_root / "outputs/v1_world_model")
    parser.add_argument("--model", choices=("world_model", "baseline"), default="world_model")
    parser.add_argument("--history-steps", type=int, default=24)
    parser.add_argument("--horizon-steps", type=int, default=12)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--event-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-history-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-window-stride", type=int, default=1)
    parser.add_argument("--eval-window-stride", type=int, default=1)
    parser.add_argument("--max-train-windows", type=int)
    parser.add_argument("--max-val-windows", type=int)
    parser.add_argument("--max-test-windows", type=int)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
