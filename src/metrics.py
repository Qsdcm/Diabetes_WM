"""Streaming rollout metrics in physical CGM units."""

from __future__ import annotations

import math

import torch


HORIZONS = {1: "5min", 6: "30min", 12: "60min"}


class RolloutMetrics:
    def __init__(self) -> None:
        self.count = 0
        self.absolute_error = 0.0
        self.squared_error = 0.0
        self.absolute_percentage_error = 0.0
        self.horizon_stats = {
            step: {"count": 0, "ae": 0.0, "se": 0.0, "ape": 0.0}
            for step in HORIZONS
        }

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes must match")
        error = prediction.detach().float() - target.detach().float()
        absolute = error.abs()
        squared = error.square()
        percentage = absolute / target.detach().float().abs().clamp_min(1e-6)
        self.count += target.numel()
        self.absolute_error += absolute.sum().item()
        self.squared_error += squared.sum().item()
        self.absolute_percentage_error += percentage.sum().item()
        for step in HORIZONS:
            if prediction.shape[1] < step:
                continue
            stats = self.horizon_stats[step]
            step_absolute = absolute[:, step - 1]
            step_squared = squared[:, step - 1]
            step_percentage = percentage[:, step - 1]
            stats["count"] += prediction.shape[0]
            stats["ae"] += step_absolute.sum().item()
            stats["se"] += step_squared.sum().item()
            stats["ape"] += step_percentage.sum().item()

    @staticmethod
    def _finalize(count: int, ae: float, se: float, ape: float) -> dict[str, float]:
        if count == 0:
            return {"mae_mg_dl": float("nan"), "rmse_mg_dl": float("nan"), "mard_pct": float("nan")}
        return {
            "mae_mg_dl": ae / count,
            "rmse_mg_dl": math.sqrt(se / count),
            "mard_pct": 100.0 * ape / count,
        }

    def compute(self) -> dict[str, dict[str, float]]:
        result = {
            "overall": self._finalize(
                self.count,
                self.absolute_error,
                self.squared_error,
                self.absolute_percentage_error,
            )
        }
        for step, label in HORIZONS.items():
            stats = self.horizon_stats[step]
            result[label] = self._finalize(
                stats["count"], stats["ae"], stats["se"], stats["ape"]
            )
        return result
