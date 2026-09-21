import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import NormalizationStats, WindowDataset
from src.metrics import RolloutMetrics
from src.models import DirectGRUBaseline, WorldModelV1


class ModelTests(unittest.TestCase):
    def test_world_model_rollout_shape_and_control_only_future(self):
        model = WorldModelV1(embedding_dim=4, event_dim=8, hidden_dim=16)
        history = torch.randn(3, 24, 5)
        future_controls = torch.randn(3, 12, 4)
        output = model(history, future_controls)
        self.assertEqual(output.shape, (3, 12))
        with self.assertRaises(ValueError):
            model(history, torch.randn(3, 12, 5))

    def test_baseline_rollout_shape(self):
        model = DirectGRUBaseline(hidden_dim=16)
        output = model(torch.randn(2, 24, 5), torch.randn(2, 12, 4))
        self.assertEqual(output.shape, (2, 12))

    def test_rollout_metrics_at_requested_horizons(self):
        target = torch.full((2, 12), 100.0)
        prediction = target + 10.0
        metrics = RolloutMetrics()
        metrics.update(prediction, target)
        result = metrics.compute()
        for key in ("overall", "5min", "30min", "60min"):
            self.assertAlmostEqual(result[key]["mae_mg_dl"], 10.0)
            self.assertAlmostEqual(result[key]["rmse_mg_dl"], 10.0)
            self.assertAlmostEqual(result[key]["mard_pct"], 10.0, places=5)


class DatasetTests(unittest.TestCase):
    def test_dataset_returns_history_controls_and_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject_path = root / "AZT1D" / "Subject_1.csv"
            subject_path.parent.mkdir()
            rows = 40
            frame = pd.DataFrame(
                {
                    "segment_id": [0] * rows,
                    "timestamp": pd.date_range("2024-01-01", periods=rows, freq="5min"),
                    "cgm_mg_dl": np.arange(rows) + 100,
                    "insulin_u_5min": np.full(rows, 0.1),
                    "carb_g_5min": np.zeros(rows),
                    "time_sin": np.zeros(rows),
                    "time_cos": np.ones(rows),
                    "cgm_observed": [True] * rows,
                    "insulin_observed": [True] * rows,
                }
            )
            frame.to_csv(subject_path, index=False)
            index_path = root / "train_windows.csv"
            pd.DataFrame(
                {
                    "source_file": ["AZT1D/Subject_1.csv"],
                    "segment_id": [0],
                    "start_row": [0],
                    "input_end_row": [23],
                    "target_start_row": [24],
                    "target_end_row": [35],
                }
            ).to_csv(index_path, index=False)
            stats = NormalizationStats(100, 10, 0, 1, 0, 1)
            dataset = WindowDataset(root, index_path, stats)
            sample = dataset[0]

        self.assertEqual(sample["history"].shape, (24, 5))
        self.assertEqual(sample["future_controls"].shape, (12, 4))
        self.assertEqual(sample["target_cgm"].shape, (12,))
        self.assertAlmostEqual(sample["target_cgm"][0].item(), 2.4, places=5)


if __name__ == "__main__":
    unittest.main()
