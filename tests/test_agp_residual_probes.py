import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from agp_residual_probes import (  # noqa: E402
    FixedUnseenProbeConfig,
    fixed_unseen_metrics,
    select_fixed_unseen_probes,
)
from agp_holdout_feedback import plot_fixed_unseen_probes  # noqa: E402
from agp_holdout_study import (  # noqa: E402
    feedback_threshold_decision,
    fixed_unseen_plot_series,
)


class AGPResidualProbeTests(unittest.TestCase):
    def test_fixed_unseen_selection_is_disjoint_and_deterministic(self):
        labels = ["XIII", "YIII", "ZIII", "IXII", "IYII", "IZII"]
        rms = np.asarray([2.0, 0.0, 1.0, 1.0e-15, 3.0, 0.0])
        config = FixedUnseenProbeConfig(
            enabled=True,
            active_terms=2,
            null_terms=2,
            reference_rms_threshold=1.0e-12,
            seed=11,
            candidate_multiplier=4,
        )

        first = select_fixed_unseen_probes(
            labels,
            rms,
            excluded_labels={"XIII", "IZII"},
            config=config,
        )
        second = select_fixed_unseen_probes(
            labels,
            rms,
            excluded_labels={"XIII", "IZII"},
            config=config,
        )

        self.assertEqual(first, second)
        self.assertEqual(set(first["active_labels"]), {"ZIII", "IYII"})
        self.assertEqual(set(first["null_labels"]), {"YIII", "IXII"})
        self.assertFalse(set(first["active_labels"]) & set(first["null_labels"]))

    def test_fixed_unseen_metrics_separate_active_ratio_from_null_leakage(self):
        residual = torch.tensor([[2.0, 1.0, 3.0, 4.0]])
        reference = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
        metrics = fixed_unseen_metrics(
            residual=residual,
            reference=reference,
            active_indices=[0, 1],
            null_indices=[2, 3],
            reference_floor=1.0e-12,
        )

        self.assertAlmostEqual(metrics["active_relative"], 1.0)
        self.assertEqual(metrics["active_status"]["reason"], "finite_reference")
        self.assertAlmostEqual(metrics["null_absolute_per_term"], 12.5)
        self.assertAlmostEqual(metrics["null_scaled"], 5.0)

    def test_fixed_unseen_metrics_never_invents_zero_reference_ratio(self):
        metrics = fixed_unseen_metrics(
            residual=torch.ones((1, 2)),
            reference=torch.zeros((1, 2)),
            active_indices=[],
            null_indices=[0, 1],
            reference_floor=1.0e-12,
        )

        self.assertIsNone(metrics["active_relative"])
        self.assertEqual(metrics["active_status"]["reason"], "empty_subset")
        self.assertTrue(np.isfinite(metrics["null_absolute_per_term"]))
        self.assertIsNone(metrics["null_scaled"])

    def test_fixed_unseen_metrics_reports_zero_reference_for_nonempty_active_subset(self):
        metrics = fixed_unseen_metrics(
            residual=torch.ones((1, 2)),
            reference=torch.zeros((1, 2)),
            active_indices=[0, 1],
            null_indices=[],
            reference_floor=1.0e-12,
        )

        self.assertIsNone(metrics["active_relative"])
        self.assertEqual(metrics["active_status"]["reason"], "zero_reference")

    def test_fixed_active_probe_is_the_stable_unseen_gate(self):
        row = {
            "feedback_round": 7,
            "holdout_relative_residual": 0.05,
            "unseen_relative_residual": None,
            "fixed_unseen_active_relative": 0.8,
            "fixed_unseen_active_status": {"valid": True, "reason": "finite_reference"},
        }

        decision = feedback_threshold_decision(
            [row],
            holdout_threshold=0.1,
            unseen_threshold=1.0,
        )

        self.assertEqual(decision["status"], "found_feedback_round")
        self.assertEqual(decision["unseen_gate_source"], "fixed_unseen_active")

    def test_fixed_active_gate_is_not_tested_when_probe_is_incomplete_or_reference_is_zero(self):
        row = {
            "feedback_round": 3,
            "holdout_relative_residual": 0.01,
            "fixed_unseen_enabled": True,
            "fixed_unseen_probe_status": "insufficient_candidates",
            "fixed_unseen_active_relative": None,
            "fixed_unseen_active_status": {"valid": False, "reason": "zero_reference"},
        }

        decision = feedback_threshold_decision(
            [row],
            holdout_threshold=0.1,
            unseen_threshold=1.0,
        )

        self.assertEqual(decision["status"], "not_found_in_feedback_run")
        self.assertEqual(decision["unseen_gate"]["status"], "not_tested")
        self.assertEqual(decision["unseen_gate"]["reason"], "insufficient_candidates")

    def test_plot_series_keeps_null_leakage_when_moving_ratio_is_undefined(self):
        series = fixed_unseen_plot_series([{
            "feedback_round": 7,
            "fixed_unseen_active_relative": 0.4,
            "fixed_unseen_null_absolute_per_term": 0.03,
            "fixed_unseen_null_scaled": 0.02,
            "unseen_relative_residual": None,
        }])

        self.assertEqual(series["rounds"].tolist(), [7.0])
        self.assertEqual(series["active_relative"].tolist(), [0.4])
        self.assertEqual(series["null_absolute_per_term"].tolist(), [0.03])
        self.assertEqual(series["null_scaled"].tolist(), [0.02])
        self.assertTrue(np.isnan(series["moving_unseen_relative"][0]))

    def test_fixed_unseen_plot_writes_separate_active_and_null_panels(self):
        rows = [
            {
                "feedback_round": 0,
                "fixed_unseen_active_relative": 0.5,
                "fixed_unseen_null_absolute_per_term": 0.03,
                "fixed_unseen_null_scaled": 0.04,
            },
            {
                "feedback_round": 1,
                "fixed_unseen_active_relative": None,
                "fixed_unseen_null_absolute_per_term": 0.02,
                "fixed_unseen_null_scaled": None,
            },
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixed_unseen.pdf"
            plot_fixed_unseen_probes(rows, path, unseen_threshold=1.0)

            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
