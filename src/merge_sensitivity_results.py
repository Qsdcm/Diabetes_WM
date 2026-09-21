#!/usr/bin/env python3
"""Merge independently evaluated sensitivity shards into final reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.evaluate_action_sensitivity import analyze_results


FLOAT_COLUMNS = {
    "micro_mae_mg_dl",
    "micro_rmse_mg_dl",
    "micro_mard_pct",
    "subject_macro_mae_mg_dl",
    "subject_macro_rmse_mg_dl",
    "subject_macro_mard_pct",
}
INT_COLUMNS = {"window_count", "subject_count"}


def typed_row(row: dict[str, str]) -> dict[str, object]:
    converted: dict[str, object] = dict(row)
    for column in FLOAT_COLUMNS:
        value = row[column]
        converted[column] = float(value) if value not in {"", "None"} else None
    for column in INT_COLUMNS:
        converted[column] = int(row[column])
    return converted


def main(args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = []
    combined_results: dict[str, object] = {}
    group_counts = None
    seed = None
    threshold = None
    for directory in args.input_dirs:
        with (directory / "unified_results.csv").open(newline="", encoding="utf-8") as handle:
            rows.extend(typed_row(row) for row in csv.DictReader(handle))
        detail = json.loads((directory / "detailed_results.json").read_text())
        combined_results.update(detail["results"])
        if group_counts is None:
            group_counts = detail["event_group_counts"]
            seed = detail["seed"]
            threshold = detail["insulin_event_threshold_u_5min"]
        elif detail["event_group_counts"] != group_counts:
            raise ValueError("Partial result shards disagree on event group counts")

    expected = {
        "Full world model / Full input",
        "Full world model / Zero future Insulin",
        "Full world model / Zero future Carb",
        "Full world model / Zero future Insulin+Carb",
        "Full world model / Shuffle future Insulin",
        "Full world model / Shuffle future Carb",
        "CGM-only / No actions",
        "Persistence / Last observed CGM",
    }
    missing = sorted(expected.difference(combined_results))
    if missing:
        raise ValueError(f"Missing experiment shards: {missing}")
    rows.sort(
        key=lambda row: (
            str(row["model_condition"]),
            str(row["event_group"]),
            str(row["horizon"]),
        )
    )
    analysis = analyze_results(rows, args.minimum_relative_change)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (args.output_dir / "unified_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "detailed_results.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "insulin_event_threshold_u_5min": threshold,
                "event_group_counts": group_counts,
                "results": combined_results,
                "analysis": analysis,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(analysis, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-relative-change", type=float, default=0.02)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
