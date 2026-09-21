#!/usr/bin/env python3
"""Evaluate action dependence, perturbations, and event-centered performance."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data import NormalizationStats, WindowDataset, make_loader
from src.experiment_models import CGMOnlyGRU
from src.metrics import HORIZONS, SubjectLevelRolloutMetrics
from src.models import WorldModelV1
from src.train import move_batch, resolve_device, set_seed


EVENT_GROUPS = ("All", "No-event", "Carb-only", "Insulin-only", "Carb+Insulin")
FULL_CONDITIONS = (
    "Full input",
    "Zero future Insulin",
    "Zero future Carb",
    "Zero future Insulin+Carb",
    "Shuffle future Insulin",
    "Shuffle future Carb",
)
EXPERIMENT_SPECS = {
    "full": ("Full world model", "Full input"),
    "zero_insulin": ("Full world model", "Zero future Insulin"),
    "zero_carb": ("Full world model", "Zero future Carb"),
    "zero_both": ("Full world model", "Zero future Insulin+Carb"),
    "shuffle_insulin": ("Full world model", "Shuffle future Insulin"),
    "shuffle_carb": ("Full world model", "Shuffle future Carb"),
    "cgm_only": ("CGM-only", "No actions"),
    "persistence": ("Persistence", "Last observed CGM"),
}


def physical_zero(mean: float, std: float) -> float:
    return (0.0 - mean) / std


def controls_to_physical(
    future_controls: torch.Tensor, normalization: NormalizationStats
) -> tuple[torch.Tensor, torch.Tensor]:
    insulin = torch.expm1(
        future_controls[..., 0] * normalization.insulin_log1p_std
        + normalization.insulin_log1p_mean
    ).clamp_min(0.0)
    carb = torch.expm1(
        future_controls[..., 1] * normalization.carb_log1p_std
        + normalization.carb_log1p_mean
    ).clamp_min(0.0)
    return insulin, carb


def event_labels(
    future_controls: torch.Tensor,
    normalization: NormalizationStats,
    *,
    insulin_threshold: float,
) -> list[str]:
    insulin, carb = controls_to_physical(future_controls, normalization)
    has_insulin = (insulin >= insulin_threshold).any(dim=1)
    # Numerical tolerance prevents normalized physical zero from becoming a
    # tiny positive event after float32 expm1 inversion.
    has_carb = (carb > 1e-6).any(dim=1)
    labels = []
    for insulin_event, carb_event in zip(has_insulin.tolist(), has_carb.tolist()):
        if insulin_event and carb_event:
            labels.append("Carb+Insulin")
        elif insulin_event:
            labels.append("Insulin-only")
        elif carb_event:
            labels.append("Carb-only")
        else:
            labels.append("No-event")
    return labels


def perturb_controls(
    controls: torch.Tensor,
    condition: str,
    normalization: NormalizationStats,
    *,
    donor_controls: torch.Tensor | None = None,
) -> torch.Tensor:
    result = controls.clone()
    if condition in {"Zero future Insulin", "Zero future Insulin+Carb"}:
        result[..., 0] = physical_zero(
            normalization.insulin_log1p_mean, normalization.insulin_log1p_std
        )
    if condition in {"Zero future Carb", "Zero future Insulin+Carb"}:
        result[..., 1] = physical_zero(
            normalization.carb_log1p_mean, normalization.carb_log1p_std
        )
    if condition == "Shuffle future Insulin":
        if donor_controls is None:
            raise ValueError("donor_controls required for shuffle")
        result[..., 0] = donor_controls[..., 0]
    if condition == "Shuffle future Carb":
        if donor_controls is None:
            raise ValueError("donor_controls required for shuffle")
        result[..., 1] = donor_controls[..., 1]
    return result


class GroupedMetrics:
    def __init__(self) -> None:
        self.groups = {group: SubjectLevelRolloutMetrics() for group in EVENT_GROUPS}

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        subject_ids: list[str],
        labels: list[str],
    ) -> None:
        self.groups["All"].update(prediction, target, subject_ids)
        positions: dict[str, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            positions[label].append(index)
        for label, indices in positions.items():
            self.groups[label].update(
                prediction[indices], target[indices], [subject_ids[i] for i in indices]
            )

    def compute(self) -> dict[str, dict[str, object]]:
        return {group: accumulator.compute() for group, accumulator in self.groups.items()}


def collect_control_bank(loader: torch.utils.data.DataLoader) -> torch.Tensor:
    return torch.cat([batch["future_controls"] for batch in loader], dim=0)


@torch.no_grad()
def evaluate_condition(
    model: torch.nn.Module | None,
    loader: torch.utils.data.DataLoader,
    normalization: NormalizationStats,
    device: torch.device,
    *,
    condition: str,
    insulin_threshold: float,
    amp: bool,
    control_bank: torch.Tensor,
    permutation: torch.Tensor,
) -> dict[str, dict[str, object]]:
    if model is not None:
        model.eval()
    grouped = GroupedMetrics()
    offset = 0
    for batch in loader:
        batch_size = batch["history"].shape[0]
        original_controls_cpu = batch["future_controls"]
        labels = event_labels(
            original_controls_cpu, normalization, insulin_threshold=insulin_threshold
        )
        subject_ids = list(batch["subject_id"])
        batch = move_batch(batch, device)
        target = normalization.denormalize_cgm(batch["target_cgm"])
        if condition == "Persistence":
            prediction = normalization.denormalize_cgm(
                batch["history"][:, -1, 0:1].expand(-1, batch["target_cgm"].shape[1])
            )
        else:
            controls = batch["future_controls"]
            donor = None
            if condition.startswith("Shuffle"):
                donor_indices = permutation[offset : offset + batch_size]
                donor = control_bank[donor_indices].to(device, non_blocking=True)
            if condition in FULL_CONDITIONS:
                controls = perturb_controls(
                    controls, condition, normalization, donor_controls=donor
                )
            with torch.cuda.amp.autocast(enabled=amp):
                normalized_prediction = model(batch["history"], controls)
            prediction = normalization.denormalize_cgm(normalized_prediction)
        grouped.update(prediction, target, subject_ids, labels)
        offset += batch_size
    return grouped.compute()


def table_rows(
    model_name: str,
    condition: str,
    grouped: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for event_group in EVENT_GROUPS:
        result = grouped[event_group]
        micro = result["micro"]
        macro = result["subject_macro"]
        per_subject = result["per_subject"]
        accumulator_count = next(
            (
                subject_metrics["5min"]
                for subject_metrics in per_subject.values()
            ),
            None,
        )
        # Every subject metric exists only if at least one window contributed.
        subject_count = len(per_subject)
        for _, horizon in HORIZONS.items():
            # Recover window count from per-subject metric internals is not possible
            # after finalization, so it is attached by the caller below.
            rows.append(
                {
                    "model": model_name,
                    "condition": condition,
                    "model_condition": f"{model_name} / {condition}",
                    "event_group": event_group,
                    "horizon": horizon,
                    "micro_mae_mg_dl": micro.get(horizon, {}).get("mae_mg_dl"),
                    "micro_rmse_mg_dl": micro.get(horizon, {}).get("rmse_mg_dl"),
                    "micro_mard_pct": micro.get(horizon, {}).get("mard_pct"),
                    "subject_macro_mae_mg_dl": macro.get(horizon, {}).get("mae_mg_dl"),
                    "subject_macro_rmse_mg_dl": macro.get(horizon, {}).get("rmse_mg_dl"),
                    "subject_macro_mard_pct": macro.get(horizon, {}).get("mard_pct"),
                    "window_count": 0 if accumulator_count is None else None,
                    "subject_count": subject_count,
                }
            )
    return rows


def count_groups(
    loader: torch.utils.data.DataLoader,
    normalization: NormalizationStats,
    insulin_threshold: float,
) -> dict[str, dict[str, object]]:
    counts = {group: 0 for group in EVENT_GROUPS}
    subjects = {group: set() for group in EVENT_GROUPS}
    for batch in loader:
        labels = event_labels(
            batch["future_controls"], normalization, insulin_threshold=insulin_threshold
        )
        for subject_id, label in zip(batch["subject_id"], labels):
            counts["All"] += 1
            counts[label] += 1
            subjects["All"].add(subject_id)
            subjects[label].add(subject_id)
    return {
        group: {"window_count": counts[group], "subject_count": len(subjects[group])}
        for group in EVENT_GROUPS
    }


def analyze_results(rows: list[dict[str, object]], threshold: float) -> dict[str, object]:
    lookup = {
        (row["model"], row["condition"], row["event_group"], row["horizon"]): row
        for row in rows
    }
    comparisons = []
    warnings = []
    for horizon in ("30min", "60min"):
        full = lookup[("Full world model", "Full input", "All", horizon)]
        base_rmse = float(full["micro_rmse_mg_dl"])
        for condition in FULL_CONDITIONS[1:]:
            row = lookup[("Full world model", condition, "All", horizon)]
            delta = (float(row["micro_rmse_mg_dl"]) - base_rmse) / base_rmse
            comparisons.append(
                {"comparison": condition, "horizon": horizon, "relative_rmse_change": delta}
            )
            if delta < threshold:
                warnings.append(
                    f"{condition} changes {horizon} micro RMSE by only {100*delta:.2f}%"
                )
    for event_group in ("Carb-only", "Insulin-only", "Carb+Insulin"):
        for horizon in ("30min", "60min"):
            full = lookup[("Full world model", "Full input", event_group, horizon)]
            cgm = lookup[("CGM-only", "No actions", event_group, horizon)]
            denominator = float(cgm["micro_rmse_mg_dl"])
            improvement = (
                denominator - float(full["micro_rmse_mg_dl"])
            ) / denominator
            comparisons.append(
                {
                    "comparison": f"Full vs CGM-only ({event_group})",
                    "horizon": horizon,
                    "relative_rmse_improvement": improvement,
                }
            )
            if improvement < threshold:
                warnings.append(
                    f"Full model improves over CGM-only by only {100*improvement:.2f}% "
                    f"for {event_group} at {horizon}"
                )
    conclusion = (
        "Action dependence is weak or unproven; the model may rely mainly on CGM history."
        if warnings
        else "Action perturbations and event windows show material action-conditioned effects."
    )
    return {"threshold": threshold, "comparisons": comparisons, "warnings": warnings, "conclusion": conclusion}


def load_models(
    full_checkpoint_path: Path,
    cgm_checkpoint_path: Path,
    device: torch.device,
) -> tuple[WorldModelV1, CGMOnlyGRU, NormalizationStats]:
    full_checkpoint = torch.load(full_checkpoint_path, map_location=device)
    cgm_checkpoint = torch.load(cgm_checkpoint_path, map_location=device)
    full_normalization = NormalizationStats.from_dict(full_checkpoint["normalization"])
    cgm_normalization = NormalizationStats.from_dict(cgm_checkpoint["normalization"])
    if full_normalization != cgm_normalization:
        raise ValueError("Full and CGM-only checkpoints do not share train normalization")
    full_model = WorldModelV1(**full_checkpoint["model_config"]).to(device)
    full_model.load_state_dict(full_checkpoint["model_state"])
    cgm_model = CGMOnlyGRU(**cgm_checkpoint["model_config"]).to(device)
    cgm_model.load_state_dict(cgm_checkpoint["model_state"])
    return full_model, cgm_model, full_normalization


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    amp = args.amp and device.type == "cuda"
    full_model, cgm_model, normalization = load_models(
        args.full_checkpoint, args.cgm_checkpoint, device
    )
    dataset = WindowDataset(
        args.dataset_root,
        args.index_root / "test_windows.csv",
        normalization,
        history_steps=24,
        horizon_steps=12,
        validate=True,
    )
    loader = make_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )
    group_counts = count_groups(loader, normalization, args.insulin_event_threshold)
    control_bank = collect_control_bank(loader)
    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(dataset), generator=generator)

    all_results: dict[str, dict[str, dict[str, object]]] = {}
    rows: list[dict[str, object]] = []
    for experiment in args.experiments:
        model_name, condition = EXPERIMENT_SPECS[experiment]
        print(f"Evaluating {model_name} / {condition}")
        if model_name == "Full world model":
            model = full_model
            evaluation_condition = condition
        elif model_name == "CGM-only":
            model = cgm_model
            evaluation_condition = condition
        else:
            model = None
            evaluation_condition = "Persistence"
        result = evaluate_condition(
            model,
            loader,
            normalization,
            device,
            condition=evaluation_condition,
            insulin_threshold=args.insulin_event_threshold,
            amp=amp,
            control_bank=control_bank,
            permutation=permutation,
        )
        all_results[f"{model_name} / {condition}"] = result
        rows.extend(table_rows(model_name, condition, result))

    for row in rows:
        row.update(group_counts[str(row["event_group"])])
    analysis = (
        analyze_results(rows, args.minimum_relative_change)
        if set(args.experiments) == set(EXPERIMENT_SPECS)
        else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (args.output_dir / "unified_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "detailed_results.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "insulin_event_threshold_u_5min": args.insulin_event_threshold,
                "event_group_counts": group_counts,
                "results": all_results,
                "analysis": analysis,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if analysis is not None:
        (args.output_dir / "analysis.json").write_text(
            json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"event_group_counts": group_counts, "analysis": analysis}, indent=2))


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=project_root / "Dataset_5min")
    parser.add_argument(
        "--index-root", type=Path, default=project_root / "Dataset_5min/window_index_v1"
    )
    parser.add_argument(
        "--full-checkpoint", type=Path, default=project_root / "outputs/v1_world_model/best.pt"
    )
    parser.add_argument(
        "--cgm-checkpoint", type=Path, default=project_root / "outputs/v1_cgm_only/best.pt"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=project_root / "outputs/action_sensitivity"
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--insulin-event-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-relative-change", type=float, default=0.02)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=tuple(EXPERIMENT_SPECS),
        default=list(EXPERIMENT_SPECS),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
