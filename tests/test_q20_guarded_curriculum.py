import sys
import unittest
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
Q20_DIR = ROOT / "q20" / "sweep_test"
TESTS_DIR = ROOT / "tests"
for path in (Q20_DIR, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from coupled_curriculum_training import merge_agp_candidate_additions, step_gate_decision
import certify_sparse_agp
import prune_support
from restart_folders import configured_paths
from support_refinement import (
    fixed_budget_swap_labels,
    resolve_active_agp_budget,
    resolve_exploratory_agp_budget,
)


class Q20GuardedCurriculumTests(unittest.TestCase):
    def test_q20_summary_paths_are_run_scoped(self):
        payload = json.loads((Q20_DIR / "config.json").read_text(encoding="utf-8"))
        summary = payload["summary"]

        for key in ("path_images", "path_data"):
            configured = str(summary[key]).strip("/")
            self.assertTrue(configured.startswith("runs/"), configured)
            self.assertNotIn(configured, {"Images", "Models_Data"})

    def test_restart_folders_does_not_recreate_root_scratch_dirs(self):
        configured = {path.relative_to(Q20_DIR).as_posix().rstrip("/") for path in configured_paths()}

        self.assertIn("runs", configured)
        self.assertNotIn("Images", configured)
        self.assertNotIn("Models_Data", configured)

    def test_postprocessing_discovers_configured_coupled_output_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_q20 = Path(tmp)
            (fake_q20 / "config.json").write_text(
                json.dumps(
                    {
                        "coupled_curriculum": {
                            "output_root": "runs/coupled_curriculum_probe_robust_v2",
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary_path = (
                fake_q20
                / "runs"
                / "coupled_curriculum_probe_robust_v2"
                / "run_a"
                / "Models_Data"
                / "coupled_curriculum_summary_residual_9728.json"
            )
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(json.dumps({"rows": [{"agp_terms": 1}]}) + "\n", encoding="utf-8")

            prune_run_dir = prune_support.RUN_DIR
            certify_run_dir = certify_sparse_agp.RUN_DIR
            try:
                prune_support.RUN_DIR = fake_q20
                certify_sparse_agp.RUN_DIR = fake_q20

                self.assertEqual(prune_support.latest_coupled_run(), summary_path.parents[1])
                self.assertEqual(certify_sparse_agp.latest_summary(), summary_path)
            finally:
                prune_support.RUN_DIR = prune_run_dir
                certify_sparse_agp.RUN_DIR = certify_run_dir

    def test_q20_fixed_budget_uses_q7_output_cap(self):
        self.assertEqual(resolve_active_agp_budget(n_qubits=6, cap_qubits=7), 4**6)
        self.assertEqual(resolve_active_agp_budget(n_qubits=7, cap_qubits=7), 4**7)
        self.assertEqual(resolve_active_agp_budget(n_qubits=20, cap_qubits=7), 4**7)

        self.assertEqual(
            resolve_exploratory_agp_budget(n_qubits=20, active_cap_qubits=7, exploratory_cap_qubits=8),
            4**8,
        )

    def test_q20_config_uses_fixed_budget_support_refinement(self):
        payload = json.loads((Q20_DIR / "config.json").read_text(encoding="utf-8"))
        coupled = payload["coupled_curriculum"]
        refinement = coupled["support_refinement"]

        self.assertEqual(coupled["base_agp_terms"], "auto")
        self.assertEqual(coupled["max_agp_terms"], 4**7)
        self.assertEqual(refinement["mode"], "fixed_budget_swap")
        self.assertEqual(refinement["active_cap_qubits"], 7)
        self.assertEqual(refinement["active_agp_terms"], "auto")
        self.assertEqual(refinement["exploratory_cap_qubits"], 8)

    def test_fixed_budget_swap_keeps_k_and_removes_low_importance_terms(self):
        result = fixed_budget_swap_labels(
            current_agp_labels=["IIIX", "IIIY", "IIIZ", "IIXX"],
            candidate_additions=[
                {"label": "IXXX", "score": 10.0},
                {"label": "IYYY", "score": 9.0},
                {"label": "IIIX", "score": 100.0},
            ],
            active_importance_terms=[
                {"label": "IIIX", "importance": 0.9},
                {"label": "IIIY", "importance": 0.8},
                {"label": "IIIZ", "importance": 0.01},
                {"label": "IIXX", "importance": 0.02},
            ],
            swap_terms=2,
            protected_fraction=0.0,
        )

        self.assertEqual(len(result["agp_labels"]), 4)
        self.assertEqual([row["label"] for row in result["added_agp_terms"]], ["IXXX", "IYYY"])
        self.assertEqual([row["label"] for row in result["removed_agp_terms"]], ["IIIZ", "IIXX"])
        self.assertEqual(set(result["agp_labels"]), {"IIIX", "IIIY", "IXXX", "IYYY"})

    def test_probe_watch_candidates_satisfy_required_probe_support(self):
        selected = merge_agp_candidate_additions(
            feedback_candidates=[
                {"label": "XXXX", "score": 100.0, "order": 4},
                {"label": "YYYY", "score": 2.0, "order": 4},
            ],
            probe_candidates=[
                {"label": "ZZZZ", "score": 5.0, "order": 4},
            ],
            probe_watch_candidates=[
                {"label": "XXYY", "score": 4.0, "order": 4},
            ],
            current_agp_labels=set(),
            add_terms=2,
            probe_score_weight=1.0,
            require_probe_support=True,
            source_diversity_bonus=0.0,
            max_terms_per_order=None,
        )

        labels = [str(row["label"]) for row in selected]
        self.assertEqual(labels, ["ZZZZ", "XXYY"])
        self.assertGreater(float(selected[1]["probe_watch_score"]), 0.0)
        self.assertNotIn("XXXX", labels)

    def test_step_gate_rejects_probe_watch_worsening(self):
        previous = {
            "holdout_relative_residual": 0.05,
            "probe_gate_total_residual": 0.50,
            "probe_gate_reference_residual": 1.0,
            "probe_watch_total_residual": 0.50,
            "probe_watch_reference_residual": 1.0,
        }
        candidate = {
            "holdout_relative_residual": 0.049,
            "probe_gate_total_residual": 0.49,
            "probe_gate_reference_residual": 1.0,
            "probe_watch_total_residual": 1.00,
            "probe_watch_reference_residual": 1.0,
        }

        gate = step_gate_decision(
            previous_feedback_row=previous,
            candidate_feedback_row=candidate,
            residual_candidate_count=128,
            agp_candidate_count=16,
            attempt_kind="residual_agp",
            probe_max_worsening_factor=1.02,
            probe_max_worsening_delta=0.0,
            probe_absolute_max_worsening_factor=1.02,
            probe_absolute_max_worsening_delta=0.0,
            feedback_max_worsening_factor=1.02,
            probe_improvement_target=1.0,
            probe_min_improvement_factor=0.99,
            feedback_min_improvement_factor=1.0,
            reference_floor=1e-12,
            validation_probe_prefixes=("probe_gate", "probe_watch"),
            primary_probe_prefix="probe_gate",
        )

        self.assertFalse(gate["accepted"])
        self.assertTrue(gate["probe_pass"])
        self.assertFalse(gate["validation_probes"]["probe_watch"]["probe_pass"])
        self.assertFalse(gate["validation_probes"]["probe_watch"]["probe_total_pass"])


if __name__ == "__main__":
    unittest.main()
