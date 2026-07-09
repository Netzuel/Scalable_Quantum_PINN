import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from projected_sparse_training_common import (  # noqa: E402
    plan_fixed_k_support_swap,
    remap_trainable_state_for_agp_labels,
)
from agp_holdout_feedback import (  # noqa: E402
    adaptive_temporal_refinement_settings_from_feedback,
    compact_support_swap_plan,
    make_adaptive_tau_grid,
    support_swap_settings_from_feedback,
    temporal_refinement_settings_from_feedback,
)
from utils import SparsePauliOperator  # noqa: E402


class AGPSupportSwapTests(unittest.TestCase):
    def test_support_swap_settings_are_read_from_holdout_feedback_config(self):
        settings = support_swap_settings_from_feedback(
            {
                "support_swap": {
                    "enabled": True,
                    "terms_per_iteration": 128,
                    "start_round": 2,
                    "candidate_pool_multiplier": 12,
                    "protect_top_fraction": 0.05,
                    "new_gate_logit": 2.0,
                }
            }
        )

        self.assertTrue(settings.enabled)
        self.assertEqual(settings.terms_per_iteration, 128)
        self.assertEqual(settings.start_round, 2)
        self.assertEqual(settings.candidate_pool_multiplier, 12)
        self.assertEqual(settings.protect_top_fraction, 0.05)
        self.assertEqual(settings.new_gate_logit, 2.0)

    def test_temporal_refinement_settings_are_read_from_holdout_feedback_config(self):
        settings = temporal_refinement_settings_from_feedback(
            {
                "temporal_refinement": {
                    "enabled": True,
                    "epochs": 2500,
                    "num_points": 64,
                    "lr": 2.5e-6,
                    "optimizer": "AdamW",
                    "run_dir": "temporal_refinement",
                }
            }
        )

        self.assertTrue(settings.enabled)
        self.assertEqual(settings.epochs, 2500)
        self.assertEqual(settings.num_points, 64)
        self.assertEqual(settings.lr, 2.5e-6)
        self.assertEqual(settings.optimizer, "AdamW")
        self.assertEqual(settings.run_dir, "temporal_refinement")

    def test_adaptive_temporal_refinement_settings_are_read_from_holdout_feedback_config(self):
        settings = adaptive_temporal_refinement_settings_from_feedback(
            {
                "adaptive_temporal_refinement": {
                    "enabled": True,
                    "epochs": 1200,
                    "dense_points": 257,
                    "num_points": 65,
                    "lr": 1.5e-6,
                    "optimizer": "AdamW",
                    "run_dir": "adaptive_temporal_refinement",
                    "weight_power": 0.75,
                    "min_weight": 0.2,
                    "max_weight": 5.0,
                    "difficulty": "residual_x_cd_norm",
                }
            }
        )

        self.assertTrue(settings.enabled)
        self.assertEqual(settings.epochs, 1200)
        self.assertEqual(settings.dense_points, 257)
        self.assertEqual(settings.num_points, 65)
        self.assertEqual(settings.lr, 1.5e-6)
        self.assertEqual(settings.optimizer, "AdamW")
        self.assertEqual(settings.run_dir, "adaptive_temporal_refinement")
        self.assertEqual(settings.weight_power, 0.75)
        self.assertEqual(settings.min_weight, 0.2)
        self.assertEqual(settings.max_weight, 5.0)
        self.assertEqual(settings.difficulty, "residual_x_cd_norm")

    def test_adaptive_tau_grid_concentrates_points_near_hard_times(self):
        dense_tau = torch.linspace(0.0, 1.0, 101)
        difficulty = torch.full_like(dense_tau, 0.05)
        difficulty[45:56] = 10.0

        tau, metadata = make_adaptive_tau_grid(
            dense_tau,
            difficulty,
            num_points=21,
            weight_power=1.0,
            min_weight=0.1,
            max_weight=8.0,
        )

        self.assertEqual(tau.shape, (21, 1))
        self.assertAlmostEqual(float(tau[0, 0]), 0.0)
        self.assertAlmostEqual(float(tau[-1, 0]), 1.0)
        self.assertTrue(torch.all(tau[1:, 0] >= tau[:-1, 0]))
        focused_count = int(((tau[:, 0] >= 0.45) & (tau[:, 0] <= 0.55)).sum().item())
        self.assertGreaterEqual(focused_count, 5)
        self.assertEqual(metadata["num_points"], 21)
        self.assertEqual(metadata["dense_points"], 101)
        self.assertGreater(metadata["max_weight"], metadata["min_weight"])

    def test_compact_support_swap_plan_omits_full_support_lists(self):
        compact = compact_support_swap_plan(
            {
                "enabled": True,
                "swap_count": 2,
                "new_agp_labels": ["A", "B", "C"],
                "candidate_labels": ["D", "E", "F"],
                "removed_labels": ["A", "B"],
                "added_labels": ["D", "E"],
                "candidate_rows": [
                    {"label": "D", "score": 3.0},
                    {"label": "E", "score": 2.0},
                    {"label": "F", "score": 1.0},
                ],
                "reason": "planned",
            },
            preview_terms=2,
        )

        self.assertNotIn("new_agp_labels", compact)
        self.assertNotIn("candidate_labels", compact)
        self.assertEqual(compact["swap_count"], 2)
        self.assertEqual(compact["removed_labels"], ["A", "B"])
        self.assertEqual(compact["added_labels"], ["D", "E"])
        self.assertEqual([row["label"] for row in compact["candidate_rows"]], ["D", "E"])

    def test_fixed_k_support_swap_replaces_weak_terms_with_hard_residual_candidates(self):
        h0 = SparsePauliOperator({"XI": -1.0, "IX": -1.0})
        h1 = SparsePauliOperator({"ZI": 0.7, "IZ": -0.3, "ZZ": 1.1})
        current = ["XI", "YI", "IY", "YY"]
        importance_rows = [
            {"label": "YI", "rms": 1.0},
            {"label": "YY", "rms": 0.5},
            {"label": "XI", "rms": 1.0e-7},
            {"label": "IY", "rms": 1.0e-8},
        ]
        residual_spectrum = [
            {"label": "ZX", "residual_rms": 9.0},
            {"label": "XZ", "residual_rms": 6.0},
            {"label": "YY", "residual_rms": 5.0},
        ]

        plan = plan_fixed_k_support_swap(
            current_agp_labels=current,
            coefficient_importance=importance_rows,
            residual_spectrum=residual_spectrum,
            h0=h0,
            h1=h1,
            max_swaps=2,
            candidate_pool_size=16,
            protect_top_fraction=0.25,
        )

        self.assertEqual(plan["swap_count"], 2)
        self.assertEqual(len(plan["new_agp_labels"]), len(current))
        self.assertEqual(len(set(plan["new_agp_labels"])), len(current))
        self.assertTrue(set(plan["removed_labels"]).issubset({"XI", "IY"}))
        self.assertTrue(set(plan["added_labels"]).isdisjoint(current))
        self.assertTrue(set(plan["added_labels"]).issubset(set(plan["candidate_labels"])))

    def test_trainable_state_remap_preserves_retained_output_rows_and_initializes_new_terms(self):
        old_labels = ["AA", "BB", "CC"]
        new_labels = ["BB", "CC", "DD"]
        state = {
            "body": {
                "layers.2.linear.weight": torch.tensor([[1.0, 1.1], [2.0, 2.2], [3.0, 3.3]]),
                "layers.2.linear.bias": torch.tensor([10.0, 20.0, 30.0]),
                "layers.0.linear.weight": torch.tensor([[5.0], [6.0]]),
            },
            "calibration": {
                "gate_logits": torch.tensor([-8.0, 4.0, 1.5]),
                "agp_labels": old_labels,
                "log_gamma": torch.tensor(0.0),
                "gate_temperature": 1.0,
                "target_active_terms": 2,
            },
            "agp_labels": old_labels,
        }

        remapped = remap_trainable_state_for_agp_labels(
            state,
            old_labels=old_labels,
            new_labels=new_labels,
            removed_labels=["AA"],
            added_labels=["DD"],
            new_gate_logit=2.5,
        )

        body = remapped["body"]
        self.assertTrue(torch.equal(body["layers.2.linear.weight"][0], torch.tensor([2.0, 2.2])))
        self.assertTrue(torch.equal(body["layers.2.linear.weight"][1], torch.tensor([3.0, 3.3])))
        self.assertTrue(torch.equal(body["layers.2.linear.weight"][2], torch.tensor([1.0, 1.1])))
        self.assertTrue(torch.equal(body["layers.0.linear.weight"], state["body"]["layers.0.linear.weight"]))
        self.assertTrue(torch.equal(remapped["calibration"]["gate_logits"], torch.tensor([4.0, 1.5, 2.5])))
        self.assertEqual(remapped["calibration"]["agp_labels"], new_labels)
        self.assertEqual(remapped["agp_labels"], new_labels)


if __name__ == "__main__":
    unittest.main()
