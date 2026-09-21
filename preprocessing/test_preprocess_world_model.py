import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.preprocess_world_model import (
    MMOL_L_TO_MG_DL,
    process_azt1d,
    process_brist1d,
)


class PreprocessingTests(unittest.TestCase):
    def test_azt1d_units_duplicate_events_and_ocr_basal(self):
        frame = pd.DataFrame(
            {
                "EventDateTime": [
                    "2024-01-01 00:04:00",
                    "2024-01-01 00:09:00",
                    "2024-01-01 00:09:00",
                    "2024-01-01 00:14:00",
                ],
                "CGM": [100, 110, 110, 120],
                "Basal": [1200, 1200, 1.2, 1.2],
                "TotalBolusInsulinDelivered": [np.nan, 2.0, 2.0, np.nan],
                "CarbSize": [np.nan, 30.0, 30.0, np.nan],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Subject 1" / "Subject 1.csv"
            path.parent.mkdir()
            frame.to_csv(path, index=False)
            out, stats = process_azt1d(path)

        self.assertEqual(len(out), 3)
        self.assertAlmostEqual(out.loc[0, "insulin_u_5min"], 0.1)
        self.assertAlmostEqual(out.loc[1, "insulin_u_5min"], 2.1)
        self.assertAlmostEqual(out.loc[1, "carb_g_5min"], 30.0)
        self.assertEqual(stats["basal_ocr_values_repaired"], 2)
        self.assertTrue((out["timestamp"].diff().dropna() == pd.Timedelta(minutes=5)).all())

    def test_brist1d_conversion_and_short_cgm_interpolation(self):
        frame = pd.DataFrame(
            {
                "timestamp": [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:05:00",
                    "2024-01-01 00:10:00",
                ],
                "bg": [5.0, np.nan, 7.0],
                "insulin": [0.1, 0.1, 0.1],
                "carbs": [np.nan, 20.0, np.nan],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "P01.csv"
            frame.to_csv(path, index=False)
            out, stats = process_brist1d(path)

        self.assertEqual(len(out), 3)
        self.assertAlmostEqual(out.loc[0, "cgm_mg_dl"], 5.0 * MMOL_L_TO_MG_DL)
        self.assertAlmostEqual(out.loc[1, "cgm_mg_dl"], 6.0 * MMOL_L_TO_MG_DL)
        self.assertAlmostEqual(out.loc[1, "carb_g_5min"], 20.0)
        self.assertEqual(stats["cgm_interpolated_rows"], 1)
        self.assertEqual(out["segment_id"].nunique(), 1)

    def test_long_missing_insulin_splits_segments(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=4, freq="5min"),
                "bg": [5.0, 5.5, 6.0, 6.5],
                "insulin": [0.1, np.nan, np.nan, 0.1],
                "carbs": [np.nan] * 4,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "P01.csv"
            frame.to_csv(path, index=False)
            out, _ = process_brist1d(path)

        self.assertEqual(len(out), 2)
        self.assertEqual(out["segment_id"].tolist(), [0, 1])

    def test_negative_insulin_is_rejected(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="5min"),
                "bg": [5.0, 5.5, 6.0],
                "insulin": [0.1, -1.0, 0.1],
                "carbs": [np.nan] * 3,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "P01.csv"
            frame.to_csv(path, index=False)
            out, stats = process_brist1d(path)

        self.assertEqual(len(out), 2)
        self.assertEqual(stats["invalid_insulin_values_rejected"], 1)
        self.assertTrue((out["insulin_u_5min"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
