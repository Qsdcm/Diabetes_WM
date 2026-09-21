#!/usr/bin/env python3
"""Build unified 5-minute world-model sequences from AZT1D and BrisT1D-Open.

The canonical model features are:
    cgm_mg_dl, insulin_u_5min, carb_g_5min, time_sin, time_cos

Rows are emitted only where CGM and insulin are known (or conservatively
filled across a short, configurable gap). Longer missing intervals split a
subject into separate continuous segments instead of being silently zeroed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FIVE_MINUTES = pd.Timedelta(minutes=5)
MMOL_L_TO_MG_DL = 18.0182
MODEL_COLUMNS = [
    "cgm_mg_dl",
    "insulin_u_5min",
    "carb_g_5min",
    "time_sin",
    "time_cos",
]
OUTPUT_COLUMNS = [
    "dataset",
    "subject_id",
    "segment_id",
    "timestamp",
    *MODEL_COLUMNS,
    "cgm_observed",
    "insulin_observed",
]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _sum_min_count(series: pd.Series) -> float:
    return series.sum(min_count=1)


def _last_non_null(series: pd.Series) -> float:
    values = series.dropna()
    return values.iloc[-1] if not values.empty else np.nan


def _repair_az_basal_rate(series: pd.Series) -> tuple[pd.Series, int]:
    """Repair the documented OCR decimal-loss pattern in AZT1D Basal.

    Values such as 825, 1910, and 4127 occur beside 0.825, 1.910, and 4.127.
    A basal rate above 10 U/h is treated as a missing decimal and divided by
    1000. Values still outside [0, 10] U/h after repair are rejected.
    """

    basal = _numeric(series).copy()
    decimal_lost = basal > 10.0
    basal.loc[decimal_lost] = basal.loc[decimal_lost] / 1000.0
    invalid = (basal < 0.0) | (basal > 10.0)
    basal.loc[invalid] = np.nan
    return basal, int(decimal_lost.sum())


def _full_grid(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    index = pd.date_range(frame.index.min(), frame.index.max(), freq="5min")
    index.name = "timestamp"
    return frame.reindex(index)


def _add_time_and_segments(
    frame: pd.DataFrame,
    *,
    dataset: str,
    subject_id: str,
    valid: pd.Series,
) -> pd.DataFrame:
    out = frame.loc[valid].copy()
    if out.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    minute_of_day = out.index.hour * 60 + out.index.minute
    phase = 2.0 * math.pi * minute_of_day / (24.0 * 60.0)
    out["time_sin"] = np.sin(phase)
    out["time_cos"] = np.cos(phase)

    gaps = out.index.to_series().diff().ne(FIVE_MINUTES)
    out["segment_id"] = gaps.cumsum().astype(int) - 1
    out["dataset"] = dataset
    out["subject_id"] = subject_id
    out["timestamp"] = out.index
    return out.reset_index(drop=True)[OUTPUT_COLUMNS]


def process_azt1d(
    path: Path,
    *,
    max_cgm_gap_bins: int = 6,
    max_basal_gap_bins: int = 6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(path, low_memory=False).drop_duplicates()
    cgm_column = "CGM" if "CGM" in raw.columns else "Readings (CGM / BGM)"
    required = {
        "EventDateTime",
        "Basal",
        "TotalBolusInsulinDelivered",
        "CarbSize",
        cgm_column,
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")

    raw["timestamp"] = pd.to_datetime(raw["EventDateTime"], errors="coerce")
    raw = raw.dropna(subset=["timestamp"]).copy()
    raw["cgm_mg_dl"] = _numeric(raw[cgm_column])
    raw["basal_u_h"], repaired_count = _repair_az_basal_rate(raw["Basal"])
    raw["bolus_u"] = _numeric(raw["TotalBolusInsulinDelivered"])
    raw["carb_g"] = _numeric(raw["CarbSize"])
    invalid_bolus = raw["bolus_u"] < 0.0
    invalid_carb = raw["carb_g"] < 0.0
    raw.loc[invalid_bolus, "bolus_u"] = np.nan
    raw.loc[invalid_carb, "carb_g"] = np.nan

    # AZT1D contains merge-generated repeated rows. Collapse each exact event
    # timestamp first so a repeated bolus or meal is counted only once.
    per_event = raw.groupby("timestamp", sort=True).agg(
        cgm_mg_dl=("cgm_mg_dl", "median"),
        basal_u_h=("basal_u_h", "median"),
        bolus_u=("bolus_u", _last_non_null),
        carb_g=("carb_g", _last_non_null),
    )
    per_event["bin"] = per_event.index.floor("5min")
    binned = per_event.groupby("bin", sort=True).agg(
        cgm_mg_dl=("cgm_mg_dl", "median"),
        basal_u_h=("basal_u_h", "median"),
        bolus_u=("bolus_u", _sum_min_count),
        carb_g=("carb_g", _sum_min_count),
    )
    binned.index.name = "timestamp"
    grid = _full_grid(binned)

    grid["cgm_observed"] = grid["cgm_mg_dl"].notna()
    grid["insulin_observed"] = grid["basal_u_h"].notna()
    grid["cgm_mg_dl"] = grid["cgm_mg_dl"].interpolate(
        method="linear", limit=max_cgm_gap_bins, limit_area="inside"
    )
    grid["basal_u_h"] = grid["basal_u_h"].ffill(limit=max_basal_gap_bins)
    grid["bolus_u"] = grid["bolus_u"].fillna(0.0)
    grid["carb_g_5min"] = grid["carb_g"].fillna(0.0)
    grid["insulin_u_5min"] = grid["basal_u_h"] / 12.0 + grid["bolus_u"]

    valid = grid["cgm_mg_dl"].notna() & grid["basal_u_h"].notna()
    subject_id = path.parent.name.replace(" ", "_")
    out = _add_time_and_segments(
        grid,
        dataset="AZT1D",
        subject_id=subject_id,
        valid=valid,
    )
    stats = {
        "dataset": "AZT1D",
        "subject_id": subject_id,
        "source": str(path),
        "raw_rows": int(len(raw)),
        "output_rows": int(len(out)),
        "segments": int(out["segment_id"].nunique()) if not out.empty else 0,
        "basal_ocr_values_repaired": repaired_count,
        "cgm_interpolated_rows": int((valid & ~grid["cgm_observed"]).sum()),
        "basal_forward_filled_rows": int((valid & ~grid["insulin_observed"]).sum()),
        "invalid_insulin_values_rejected": int(invalid_bolus.sum()),
        "invalid_carb_values_rejected": int(invalid_carb.sum()),
    }
    return out, stats


def process_brist1d(
    path: Path,
    *,
    max_cgm_gap_bins: int = 6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(path, low_memory=False).drop_duplicates()
    required = {"timestamp", "bg", "insulin", "carbs"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")

    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw = raw.dropna(subset=["timestamp"]).copy()
    raw["bin"] = raw["timestamp"].dt.floor("5min")
    raw["cgm_mg_dl"] = _numeric(raw["bg"]) * MMOL_L_TO_MG_DL
    raw["insulin_u_5min"] = _numeric(raw["insulin"])
    raw["carb_g_5min"] = _numeric(raw["carbs"])
    invalid_insulin = raw["insulin_u_5min"] < 0.0
    invalid_carb = raw["carb_g_5min"] < 0.0
    raw.loc[invalid_insulin, "insulin_u_5min"] = np.nan
    raw.loc[invalid_carb, "carb_g_5min"] = np.nan

    binned = raw.groupby("bin", sort=True).agg(
        cgm_mg_dl=("cgm_mg_dl", "median"),
        insulin_u_5min=("insulin_u_5min", _sum_min_count),
        carb_g_5min=("carb_g_5min", _sum_min_count),
    )
    binned.index.name = "timestamp"
    grid = _full_grid(binned)
    grid["cgm_observed"] = grid["cgm_mg_dl"].notna()
    grid["insulin_observed"] = grid["insulin_u_5min"].notna()
    grid["cgm_mg_dl"] = grid["cgm_mg_dl"].interpolate(
        method="linear", limit=max_cgm_gap_bins, limit_area="inside"
    )
    grid["carb_g_5min"] = grid["carb_g_5min"].fillna(0.0)

    valid = grid["cgm_mg_dl"].notna() & grid["insulin_u_5min"].notna()
    out = _add_time_and_segments(
        grid,
        dataset="BrisT1D-Open",
        subject_id=path.stem,
        valid=valid,
    )
    stats = {
        "dataset": "BrisT1D-Open",
        "subject_id": path.stem,
        "source": str(path),
        "raw_rows": int(len(raw)),
        "output_rows": int(len(out)),
        "segments": int(out["segment_id"].nunique()) if not out.empty else 0,
        "basal_ocr_values_repaired": 0,
        "cgm_interpolated_rows": int((valid & ~grid["cgm_observed"]).sum()),
        "basal_forward_filled_rows": 0,
        "invalid_insulin_values_rejected": int(invalid_insulin.sum()),
        "invalid_carb_values_rejected": int(invalid_carb.sum()),
    }
    return out, stats


def _write_subject(
    frame: pd.DataFrame,
    output_root: Path,
    *,
    append_combined: bool,
) -> Path:
    dataset = str(frame["dataset"].iloc[0])
    subject_id = str(frame["subject_id"].iloc[0])
    dataset_dir = output_root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{subject_id}.csv"
    frame.to_csv(target, index=False, float_format="%.8g")

    if append_combined:
        combined = output_root / "all_sequences.csv"
        frame.to_csv(
            combined,
            mode="a",
            header=not combined.exists(),
            index=False,
            float_format="%.8g",
        )
    return target


def build_dataset(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    combined = output_root / "all_sequences.csv"
    if args.write_combined and combined.exists():
        combined.unlink()

    jobs: list[tuple[str, Path]] = []
    jobs.extend(("AZT1D", p) for p in args.azt1d_root.glob("Subject */*.csv"))
    jobs.extend(("BrisT1D-Open", p) for p in args.brist1d_root.glob("P*.csv"))
    jobs.sort(key=lambda item: (item[0], str(item[1])))
    if not jobs:
        raise FileNotFoundError("No input subject CSV files were found")

    summaries: list[dict[str, Any]] = []
    for dataset, path in jobs:
        if dataset == "AZT1D":
            frame, stats = process_azt1d(
                path,
                max_cgm_gap_bins=args.max_cgm_gap_bins,
                max_basal_gap_bins=args.max_basal_gap_bins,
            )
        else:
            frame, stats = process_brist1d(
                path,
                max_cgm_gap_bins=args.max_cgm_gap_bins,
            )
        if frame.empty:
            stats["output"] = None
        else:
            stats["output"] = str(
                _write_subject(frame, output_root, append_combined=args.write_combined)
            )
        summaries.append(stats)
        print(
            f"[{dataset}] {stats['subject_id']}: "
            f"{stats['output_rows']} rows, {stats['segments']} segments"
        )

    report = {
        "frequency_minutes": 5,
        "model_columns": MODEL_COLUMNS,
        "max_cgm_gap_minutes": args.max_cgm_gap_bins * 5,
        "max_basal_gap_minutes": args.max_basal_gap_bins * 5,
        "subjects": summaries,
        "total_rows": sum(item["output_rows"] for item in summaries),
    }
    (output_root / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summaries


def parse_args() -> argparse.Namespace:
    data_root = Path("/data/data54/wanghaobo/data/WM_diabets")
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--azt1d-root",
        type=Path,
        default=data_root / "AZT1D/AZT1D 2025/CGM Records",
    )
    parser.add_argument(
        "--brist1d-root",
        type=Path,
        default=(
            data_root
            / "BrisT1D-Open/33z5jc8fa6tob21ptrugzqog08/device_data/processed_state"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "Dataset_5min",
    )
    parser.add_argument("--max-cgm-gap-bins", type=int, default=6)
    parser.add_argument("--max-basal-gap-bins", type=int, default=6)
    parser.add_argument(
        "--write-combined",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
