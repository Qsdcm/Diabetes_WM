#!/usr/bin/env python3
"""Train the CGM+Time-only baseline with the V1 protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.data import compute_train_normalization, dataset_summary, make_loader
from src.experiment_models import CGMOnlyGRU
from src.train import (
    build_datasets,
    evaluate,
    resolve_device,
    set_seed,
    train_epoch,
)


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    amp = args.amp and device.type == "cuda"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    normalization = compute_train_normalization(args.dataset_root, args.index_root)
    train_data, val_data, test_data = build_datasets(args, normalization)
    datasets = {"train": train_data, "val": val_data, "test": test_data}
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
    print(json.dumps({key: dataset_summary(value) for key, value in datasets.items()}, indent=2))
    print(f"device={device}, amp={amp}")

    model_config = {
        "hidden_dim": args.hidden_dim,
        "num_history_layers": args.num_history_layers,
        "dropout": args.dropout,
    }
    model = CGMOnlyGRU(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_path = args.output_dir / "best.pt"
    history_path = args.output_dir / "history.jsonl"
    history_path.write_text("")
    best_val_rmse = float("inf")
    stale_epochs = 0

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
        val_rmse = validation["metrics"]["micro"]["overall"]["rmse_mg_dl"]
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
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_rmse": float(best_val_rmse),
                    "normalization": normalization.to_dict(),
                    "model_name": "cgm_only",
                    "model_config": model_config,
                },
                best_path,
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
        "model": "cgm_only",
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
    parser.add_argument("--output-dir", type=Path, default=project_root / "outputs/v1_cgm_only")
    parser.add_argument("--history-steps", type=int, default=24)
    parser.add_argument("--horizon-steps", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-history-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    # Unused architecture fields keep build_datasets compatible with train args.
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--event-dim", type=int, default=64)
    parser.add_argument("--model", default="cgm_only")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-window-stride", type=int, default=1)
    parser.add_argument("--eval-window-stride", type=int, default=1)
    parser.add_argument("--max-train-windows", type=int)
    parser.add_argument("--max-val-windows", type=int)
    parser.add_argument("--max-test-windows", type=int)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
