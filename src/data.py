"""Leakage-safe PyTorch datasets for continuous CGM rollout."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


VALUE_COLUMNS = ("cgm_mg_dl", "insulin_u_5min", "carb_g_5min")
TIME_COLUMNS = ("time_sin", "time_cos")


@dataclass(frozen=True)
class NormalizationStats:
    cgm_mean: float
    cgm_std: float
    insulin_log1p_mean: float
    insulin_log1p_std: float
    carb_log1p_mean: float
    carb_log1p_std: float

    def to_dict(self) -> dict[str, float]:
        # Keep checkpoints compatible with PyTorch's safe weights-only loader:
        # statistics computed by NumPy must be serialized as Python scalars.
        return {key: float(value) for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> "NormalizationStats":
        return cls(**{key: float(value) for key, value in values.items()})

    def normalize_values(self, values: np.ndarray) -> np.ndarray:
        if values.shape[-1] != 3:
            raise ValueError("values must contain [CGM, Insulin, Carb]")
        if np.any(values[..., 1:3] < 0):
            raise ValueError("Insulin and Carb must be non-negative before log1p")
        transformed = values.astype(np.float32, copy=True)
        transformed[..., 1:3] = np.log1p(transformed[..., 1:3])
        means = np.asarray(
            [self.cgm_mean, self.insulin_log1p_mean, self.carb_log1p_mean],
            dtype=np.float32,
        )
        stds = np.asarray(
            [self.cgm_std, self.insulin_log1p_std, self.carb_log1p_std],
            dtype=np.float32,
        )
        return (transformed - means) / stds

    def denormalize_cgm(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.cgm_std + self.cgm_mean


def _observed_mask(frame: pd.DataFrame) -> np.ndarray:
    def as_bool(column: str) -> np.ndarray:
        series = frame[column]
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False).to_numpy(dtype=bool)
        return (
            series.astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes"})
            .to_numpy(dtype=bool)
        )

    return as_bool("cgm_observed") & as_bool("insulin_observed")


def compute_train_normalization(
    dataset_root: Path, index_root: Path, *, min_std: float = 1e-6
) -> NormalizationStats:
    """Compute statistics from strict rows belonging to train subjects only."""

    manifest = json.loads((index_root / "split_manifest.json").read_text())
    train_subjects = manifest["subjects"]["train"]
    count = 0
    sums = np.zeros(3, dtype=np.float64)
    squared_sums = np.zeros(3, dtype=np.float64)
    for key in train_subjects:
        dataset, subject_id = key.split("/", 1)
        path = dataset_root / dataset / f"{subject_id}.csv"
        frame = pd.read_csv(
            path,
            usecols=[*VALUE_COLUMNS, "cgm_observed", "insulin_observed"],
        )
        values = frame.loc[_observed_mask(frame), list(VALUE_COLUMNS)].to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite training values in {path}")
        if np.any(values[:, 1:3] < 0):
            raise ValueError(f"Negative Insulin/Carb values in {path}")
        values[:, 1:3] = np.log1p(values[:, 1:3])
        count += len(values)
        sums += values.sum(axis=0)
        squared_sums += np.square(values).sum(axis=0)
    if count == 0:
        raise ValueError("No fully observed training rows available for normalization")
    means = sums / count
    variances = np.maximum(squared_sums / count - np.square(means), 0.0)
    stds = np.maximum(np.sqrt(variances), min_std)
    return NormalizationStats(
        cgm_mean=means[0],
        cgm_std=stds[0],
        insulin_log1p_mean=means[1],
        insulin_log1p_std=stds[1],
        carb_log1p_mean=means[2],
        carb_log1p_std=stds[2],
    )


@dataclass
class SubjectArrays:
    values: np.ndarray
    time: np.ndarray
    observed: np.ndarray
    segment_id: np.ndarray


class WindowDataset(Dataset):
    """Lazy window view backed by subject-level arrays cached in memory."""

    def __init__(
        self,
        dataset_root: Path,
        index_file: Path,
        normalization: NormalizationStats,
        *,
        history_steps: int = 24,
        horizon_steps: int = 12,
        sample_stride: int = 1,
        max_windows: int | None = None,
        seed: int = 0,
        validate: bool = True,
    ) -> None:
        if history_steps <= 0 or horizon_steps <= 0 or sample_stride <= 0:
            raise ValueError("history_steps, horizon_steps, and sample_stride must be positive")
        self.dataset_root = Path(dataset_root)
        self.normalization = normalization
        self.history_steps = history_steps
        self.horizon_steps = horizon_steps

        columns = [
            "source_file",
            "segment_id",
            "start_row",
            "input_end_row",
            "target_start_row",
            "target_end_row",
        ]
        manifest = pd.read_csv(index_file, usecols=columns).iloc[::sample_stride].copy()
        if max_windows is not None and len(manifest) > max_windows:
            manifest = manifest.sample(n=max_windows, random_state=seed).sort_index()
        manifest = manifest.reset_index(drop=True)
        categories = pd.Categorical(manifest["source_file"])
        self.source_files = [str(item) for item in categories.categories]
        self.subject_keys = [
            Path(relative_path).with_suffix("").as_posix()
            for relative_path in self.source_files
        ]
        self.source_codes = categories.codes.astype(np.int16, copy=False)
        self.segment_ids = manifest["segment_id"].to_numpy(dtype=np.int32)
        self.starts = manifest["start_row"].to_numpy(dtype=np.int32)
        self.input_ends = manifest["input_end_row"].to_numpy(dtype=np.int32)
        self.target_starts = manifest["target_start_row"].to_numpy(dtype=np.int32)
        self.target_ends = manifest["target_end_row"].to_numpy(dtype=np.int32)
        self.subjects = self._load_subjects()
        if validate:
            self._validate_manifest()

    def _load_subjects(self) -> list[SubjectArrays]:
        subjects = []
        for relative_path in self.source_files:
            frame = pd.read_csv(
                self.dataset_root / relative_path,
                usecols=[
                    *VALUE_COLUMNS,
                    *TIME_COLUMNS,
                    "cgm_observed",
                    "insulin_observed",
                    "segment_id",
                ],
            )
            values = frame.loc[:, VALUE_COLUMNS].to_numpy(dtype=np.float32)
            normalized = self.normalization.normalize_values(values).astype(
                np.float32, copy=False
            )
            subjects.append(
                SubjectArrays(
                    values=normalized,
                    time=frame.loc[:, TIME_COLUMNS].to_numpy(dtype=np.float32),
                    observed=_observed_mask(frame),
                    segment_id=frame["segment_id"].to_numpy(dtype=np.int32),
                )
            )
        return subjects

    def _validate_manifest(self) -> None:
        if not np.all(self.input_ends - self.starts + 1 == self.history_steps):
            raise ValueError("Window index history length does not match configuration")
        if not np.all(self.target_starts == self.input_ends + 1):
            raise ValueError("Target must begin immediately after history")
        if not np.all(self.target_ends - self.target_starts + 1 == self.horizon_steps):
            raise ValueError("Window index horizon length does not match configuration")
        for source_code, arrays in enumerate(self.subjects):
            selected = np.flatnonzero(self.source_codes == source_code)
            starts = self.starts[selected]
            ends = self.target_ends[selected] + 1
            if len(ends) and ends.max() > len(arrays.values):
                raise ValueError(f"Window exceeds source length: {self.source_files[source_code]}")
            if np.any(arrays.segment_id[starts] != arrays.segment_id[ends - 1]):
                raise ValueError(f"Window crosses segment: {self.source_files[source_code]}")
            invalid_prefix = np.r_[0, np.cumsum(~arrays.observed)]
            if np.any(invalid_prefix[ends] - invalid_prefix[starts]):
                raise ValueError(
                    f"Window contains interpolated/missing values: {self.source_files[source_code]}"
                )

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        source_code = int(self.source_codes[index])
        arrays = self.subjects[source_code]
        start = int(self.starts[index])
        target_start = int(self.target_starts[index])
        end = int(self.target_ends[index]) + 1

        history = np.concatenate(
            [arrays.values[start:target_start], arrays.time[start:target_start]], axis=-1
        )
        # The future tensor intentionally has no CGM channel.
        future_controls = np.concatenate(
            [arrays.values[target_start:end, 1:3], arrays.time[target_start:end]], axis=-1
        )
        target_cgm = arrays.values[target_start:end, 0]
        return {
            "history": torch.from_numpy(history),
            "future_controls": torch.from_numpy(future_controls),
            "target_cgm": torch.from_numpy(target_cgm),
            # Identifier is metadata for evaluation only and never a model input.
            "subject_id": self.subject_keys[source_code],
        }


def make_loader(
    dataset: WindowDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
        drop_last=False,
    )


def dataset_summary(dataset: WindowDataset) -> dict[str, Any]:
    return {
        "windows": len(dataset),
        "source_files": len(dataset.source_files),
        "history_steps": dataset.history_steps,
        "horizon_steps": dataset.horizon_steps,
    }
