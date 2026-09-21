import unittest

import torch

from src.data import NormalizationStats
from src.evaluate_action_sensitivity import (
    controls_to_physical,
    event_labels,
    perturb_controls,
)
from src.experiment_models import CGMOnlyGRU


class ActionSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.stats = NormalizationStats(100.0, 20.0, 0.2, 0.5, 0.3, 0.7)

    def test_cgm_only_is_invariant_to_insulin_and_carb(self):
        torch.manual_seed(1)
        model = CGMOnlyGRU(hidden_dim=16).eval()
        history = torch.randn(2, 24, 5)
        controls = torch.randn(2, 12, 4)
        changed_history = history.clone()
        changed_history[..., 1:3] += 100.0
        changed_controls = controls.clone()
        changed_controls[..., 0:2] -= 100.0
        with torch.no_grad():
            original = model(history, controls)
            changed = model(changed_history, changed_controls)
        self.assertTrue(torch.equal(original, changed))

    def test_physical_zero_uses_normalized_zero(self):
        controls = torch.randn(3, 12, 4)
        zeroed = perturb_controls(
            controls, "Zero future Insulin+Carb", self.stats
        )
        insulin, carb = controls_to_physical(zeroed, self.stats)
        self.assertTrue(torch.allclose(insulin, torch.zeros_like(insulin), atol=1e-6))
        self.assertTrue(torch.allclose(carb, torch.zeros_like(carb), atol=1e-6))

    def test_event_groups(self):
        zero_insulin = -self.stats.insulin_log1p_mean / self.stats.insulin_log1p_std
        zero_carb = -self.stats.carb_log1p_mean / self.stats.carb_log1p_std
        controls = torch.zeros(4, 12, 4)
        controls[..., 0] = zero_insulin
        controls[..., 1] = zero_carb
        insulin_event = (torch.log1p(torch.tensor(1.0)) - self.stats.insulin_log1p_mean) / self.stats.insulin_log1p_std
        carb_event = (torch.log1p(torch.tensor(20.0)) - self.stats.carb_log1p_mean) / self.stats.carb_log1p_std
        controls[1, 0, 1] = carb_event
        controls[2, 0, 0] = insulin_event
        controls[3, 0, 0] = insulin_event
        controls[3, 0, 1] = carb_event
        self.assertEqual(
            event_labels(controls, self.stats, insulin_threshold=0.5),
            ["No-event", "Carb-only", "Insulin-only", "Carb+Insulin"],
        )

    def test_shuffle_preserves_selected_channel_values(self):
        controls = torch.arange(4 * 3 * 4, dtype=torch.float32).reshape(4, 3, 4)
        donor = controls[torch.tensor([2, 0, 3, 1])]
        shuffled = perturb_controls(
            controls, "Shuffle future Insulin", self.stats, donor_controls=donor
        )
        self.assertTrue(torch.equal(shuffled[..., 0], donor[..., 0]))
        self.assertTrue(torch.equal(shuffled[..., 1:], controls[..., 1:]))


if __name__ == "__main__":
    unittest.main()
