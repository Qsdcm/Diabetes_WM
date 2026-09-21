#!/usr/bin/env python3
"""Create leakage-safe subject splits and window indices for model training.

The split is performed before window construction. Every subject belongs to
exactly one of train/val/test, and every indexed window stays within one
continuous segment. V1 only admits fully observed CGM and insulin windows.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SubjectFile:
    dataset: str
    subject_id: str
    path: Path

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.subject_id}"


def discover_subjects(dataset_root: Path) -> list[SubjectFile]:
    subjects: list[SubjectFile] = []
    for dataset in ("AZT1D", "BrisT1D-Open"):
        for path in sorted((dataset_root / dataset).glob("*.csv")):
            subjects.append(SubjectFile(dataset, path.stem, path))
    if not subjects:
        raise FileNotFoundError(f"No subject CSV files found below {dataset_root}")
    return subjects


def _allocate_counts(n: int, ratios: tuple[float, float, float]) -> list[int]:
    exact = np.asarray(ratios, dtype=float) * n
    counts = np.floor(exact).astype(int)
    remainder = n - int(counts.sum())
    order = np.argsort(-(exact - counts), kind="stable")
    for index in order[:remainder]:
        counts[index] += 1
    if n >= 3:
        for target in (1, 2):
            if counts[target] == 0:
                donor = int(np.argmax(counts))
                counts[donor] -= 1
                counts[target] += 1
    return counts.tolist()


def split_subjects(
    subjects: Iterable[SubjectFile],
    *,
    seed: int,
    ratios: tuple[float, float, float],
) -> dict[str, list[SubjectFile]]:
    """Deterministically stratify subject identities by source dataset."""

    if any(value <= 0 for value in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("split ratios must be positive and sum to 1")
    result = {name: [] for name in SPLIT_NAMES}
    datasets: dict[str, list[SubjectFile]] = {}
    for subject in subjects:
        datasets.setdefault(subject.dataset, []).append(subject)

    for dataset, members in sorted(datasets.items()):
        members = sorted(members, key=lambda item: item.subject_id)
        random.Random(f"{seed}:{dataset}").shuffle(members)
        counts = _allocate_counts(len(members), ratios)
        offset = 0
        for split, count in zip(SPLIT_NAMES, counts):
            result[split].extend(members[offset : offset + count])
            offset += count
    return result


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"})


def check_subject_eligibility(
    subject: SubjectFile,
    *,
    history_steps: int,
    horizon_steps: int,
) -> tuple[bool, str]:
    """Return whether a subject can produce at least one strict V1 window."""

    total_steps = history_steps + horizon_steps
    frame = pd.read_csv(
        subject.path,
        usecols=["segment_id", "timestamp", "cgm_observed", "insulin_observed"],
        parse_dates=["timestamp"],
    )
    has_long_enough_segment = False
    for segment_id, segment in frame.groupby("segment_id", sort=False):
        segment = segment.sort_values("timestamp")
        if len(segment) < total_steps:
            continue
        has_long_enough_segment = True
        cadence_ok = segment["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=5)).all()
        if not cadence_ok:
            raise ValueError(
                f"{subject.key} segment {segment_id} is not a continuous 5-minute sequence"
            )
        observed = _as_bool(segment["cgm_observed"]) & _as_bool(
            segment["insulin_observed"]
        )
        valid_counts = np.convolve(
            observed.to_numpy(dtype=np.int8),
            np.ones(total_steps, dtype=int),
            mode="valid",
        )
        if np.any(valid_counts == total_steps):
            return True, "eligible"
    if has_long_enough_segment:
        return False, "no_fully_observed_window"
    return False, "no_segment_long_enough"


def build_subject_windows(
    subject: SubjectFile,
    *,
    split: str,
    dataset_root: Path,
    history_steps: int,
    horizon_steps: int,
    stride: int,
    require_fully_observed: bool = True,
) -> pd.DataFrame:
    if history_steps <= 0 or horizon_steps <= 0 or stride <= 0:
        raise ValueError("history_steps, horizon_steps, and stride must be positive")
    frame = pd.read_csv(subject.path, parse_dates=["timestamp"])
    required = {
        "segment_id",
        "timestamp",
        "cgm_observed",
        "insulin_observed",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{subject.path}: missing columns {missing}")

    total_steps = history_steps + horizon_steps
    records: list[dict[str, object]] = []
    relative_path = str(subject.path.relative_to(dataset_root))
    for segment_id, segment in frame.groupby("segment_id", sort=False):
        segment = segment.sort_values("timestamp")
        if len(segment) < total_steps:
            continue
        cadence_ok = segment["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=5)).all()
        if not cadence_ok:
            raise ValueError(
                f"{subject.key} segment {segment_id} is not a continuous 5-minute sequence"
            )

        observed = _as_bool(segment["cgm_observed"]) & _as_bool(
            segment["insulin_observed"]
        )
        observed_values = observed.to_numpy(dtype=np.int8)
        if require_fully_observed:
            eligible = (
                np.convolve(observed_values, np.ones(total_steps, dtype=int), mode="valid")
                == total_steps
            )
        else:
            eligible = np.ones(len(segment) - total_steps + 1, dtype=bool)

        row_positions = segment.index.to_numpy()
        timestamps = segment["timestamp"].to_numpy()
        for local_start in np.flatnonzero(eligible)[::stride]:
            input_end = local_start + history_steps - 1
            target_start = local_start + history_steps
            target_end = local_start + total_steps - 1
            records.append(
                {
                    "split": split,
                    "dataset": subject.dataset,
                    "subject_id": subject.subject_id,
                    "segment_id": int(segment_id),
                    "source_file": relative_path,
                    "start_row": int(row_positions[local_start]),
                    "input_end_row": int(row_positions[input_end]),
                    "target_start_row": int(row_positions[target_start]),
                    "target_end_row": int(row_positions[target_end]),
                    "start_time": timestamps[local_start],
                    "target_start_time": timestamps[target_start],
                    "target_end_time": timestamps[target_end],
                }
            )
    return pd.DataFrame.from_records(records)


def _assert_no_subject_leakage(splits: dict[str, list[SubjectFile]]) -> None:
    key_sets = {name: {subject.key for subject in values} for name, values in splits.items()}
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            overlap = key_sets[left].intersection(key_sets[right])
            if overlap:
                raise AssertionError(f"subject leakage between {left}/{right}: {sorted(overlap)}")


def write_indices(args: argparse.Namespace) -> dict[str, object]:
    subjects = discover_subjects(args.dataset_root)
    eligible_subjects: list[SubjectFile] = []
    excluded_subjects: list[dict[str, str]] = []
    for subject in subjects:
        eligible, reason = check_subject_eligibility(
            subject,
            history_steps=args.history_steps,
            horizon_steps=args.horizon_steps,
        )
        if eligible:
            eligible_subjects.append(subject)
        else:
            excluded_subjects.append({"subject": subject.key, "reason": reason})
            print(f"[exclude] {subject.key}: {reason}")

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    splits = split_subjects(eligible_subjects, seed=args.seed, ratios=ratios)
    _assert_no_subject_leakage(splits)
    args.output_root.mkdir(parents=True, exist_ok=True)

    split_manifest: dict[str, list[str]] = {}
    window_counts: dict[str, int] = {}
    for split in SPLIT_NAMES:
        split_manifest[split] = sorted(subject.key for subject in splits[split])
        pieces = []
        for subject in splits[split]:
            windows = build_subject_windows(
                subject,
                split=split,
                dataset_root=args.dataset_root,
                history_steps=args.history_steps,
                horizon_steps=args.horizon_steps,
                stride=args.stride,
                require_fully_observed=True,
            )
            if not windows.empty:
                pieces.append(windows)
        combined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        combined.to_csv(args.output_root / f"{split}_windows.csv", index=False)
        window_counts[split] = int(len(combined))
        print(f"[{split}] {len(splits[split])} subjects, {len(combined)} windows")

    report: dict[str, object] = {
        "policy": {
            "split_unit": "subject",
            "window_crosses_segment": False,
            "require_cgm_observed": True,
            "require_insulin_observed": True,
            "interpolated_cgm_as_target": False,
        },
        "seed": args.seed,
        "ratios": dict(zip(SPLIT_NAMES, ratios)),
        "history_steps": args.history_steps,
        "horizon_steps": args.horizon_steps,
        "stride": args.stride,
        "subjects": split_manifest,
        "excluded_subjects": excluded_subjects,
        "window_counts": window_counts,
    }
    (args.output_root / "split_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=project_root / "Dataset_5min")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "Dataset_5min/window_index_v1",
    )
    parser.add_argument("--history-steps", type=int, default=24)
    parser.add_argument("--horizon-steps", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


if __name__ == "__main__":
    write_indices(parse_args())
