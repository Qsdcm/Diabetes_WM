import tempfile
import unittest
from pathlib import Path

import pandas as pd

from preprocessing.build_window_index import (
    SubjectFile,
    _assert_no_subject_leakage,
    build_subject_windows,
    check_subject_eligibility,
    split_subjects,
)


class WindowIndexTests(unittest.TestCase):
    def _write_subject(self, root: Path) -> SubjectFile:
        path = root / "AZT1D" / "Subject_1.csv"
        path.parent.mkdir(parents=True)
        frame = pd.DataFrame(
            {
                "segment_id": [0] * 5 + [1] * 5,
                "timestamp": list(pd.date_range("2024-01-01", periods=5, freq="5min"))
                + list(pd.date_range("2024-01-02", periods=5, freq="5min")),
                "cgm_observed": [True, True, False, True, True] + [True] * 5,
                "insulin_observed": [True] * 10,
            }
        )
        frame.to_csv(path, index=False)
        return SubjectFile("AZT1D", "Subject_1", path)

    def test_windows_do_not_cross_segments_and_require_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = self._write_subject(root)
            windows = build_subject_windows(
                subject,
                split="train",
                dataset_root=root,
                history_steps=2,
                horizon_steps=1,
                stride=1,
            )

        self.assertEqual(len(windows), 3)
        self.assertEqual(set(windows["segment_id"]), {1})
        self.assertTrue((windows["target_end_row"] - windows["start_row"] == 2).all())

    def test_subject_split_is_disjoint_and_deterministic(self):
        subjects = [
            SubjectFile("AZT1D", f"S{i}", Path(f"S{i}.csv")) for i in range(10)
        ] + [
            SubjectFile("BrisT1D-Open", f"P{i}", Path(f"P{i}.csv")) for i in range(10)
        ]
        first = split_subjects(subjects, seed=7, ratios=(0.7, 0.15, 0.15))
        second = split_subjects(subjects, seed=7, ratios=(0.7, 0.15, 0.15))
        _assert_no_subject_leakage(first)

        self.assertEqual(
            {name: [item.key for item in values] for name, values in first.items()},
            {name: [item.key for item in values] for name, values in second.items()},
        )
        for split in first:
            self.assertEqual({item.dataset for item in first[split]}, {"AZT1D", "BrisT1D-Open"})

    def test_subject_without_a_strict_window_is_ineligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = self._write_subject(root)
            eligible, reason = check_subject_eligibility(
                subject,
                history_steps=2,
                horizon_steps=1,
            )
            frame = pd.read_csv(subject.path)
            frame.loc[[2, 7], "cgm_observed"] = False
            frame.to_csv(subject.path, index=False)
            too_strict, strict_reason = check_subject_eligibility(
                subject,
                history_steps=2,
                horizon_steps=1,
            )

        self.assertTrue(eligible)
        self.assertEqual(reason, "eligible")
        self.assertFalse(too_strict)
        self.assertEqual(strict_reason, "no_fully_observed_window")


if __name__ == "__main__":
    unittest.main()
