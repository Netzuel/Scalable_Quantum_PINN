import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
