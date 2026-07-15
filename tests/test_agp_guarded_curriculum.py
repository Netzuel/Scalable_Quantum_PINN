import sys
import unittest
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
Q20_DIR = (
    ROOT
    / "tests"
    / "sparse_agp_curriculum"
    / "transverse_field_diagonal_ising"
    / "q20"
    / "sweep_test"
)
SCRIPTS_DIR = ROOT / "scripts"
DIAGNOSTICS_DIR = SCRIPTS_DIR / "diagnostics"
TESTS_DIR = ROOT / "tests"
for path in (DIAGNOSTICS_DIR, SCRIPTS_DIR, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agp_coupled_curriculum import (
    build_fixed_probe_residual_labels,
    load_existing_fixed_unseen_probe_labels,
    merge_agp_candidate_additions,
    step_gate_decision,
)
import agp_restart
import agp_certify_coupled
import agp_certify_support
import agp_prune_support
from agp_support_refinement import (
    fixed_budget_swap_labels,
    resolve_active_agp_budget,
    resolve_exploratory_agp_budget,
)
from utils import SparsePauliOperator


class AGPGuardedCurriculumTests(unittest.TestCase):
    def test_coupled_probe_builder_excludes_existing_fixed_unseen_manifest_labels(self):
        h0 = SparsePauliOperator({"XI": -1.0, "IX": -1.0})
        h1 = SparsePauliOperator({"ZI": 0.7, "IZ": -0.3, "ZZ": 1.1})
        unrestricted, _ = build_fixed_probe_residual_labels(
            h0=h0,
            h1=h1,
            feedback_residual_labels=[],
            probe_agp_terms=4,
            probe_residual_terms=1,
            intermediate_top_k=8,
            probe_name="probe_gate",
        )
        fixed_label = unrestricted[0]
        labels, metadata = build_fixed_probe_residual_labels(
            h0=h0,
            h1=h1,
            feedback_residual_labels=[],
            extra_excluded_labels=[fixed_label],
            probe_agp_terms=4,
            probe_residual_terms=1,
            intermediate_top_k=8,
            probe_name="probe_gate",
        )

        self.assertNotIn(fixed_label, labels)
        self.assertEqual(metadata["probe_name"], "probe_gate")

    def test_coupled_loader_collects_existing_fixed_unseen_manifest_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "run" / "Models_Data" / "fixed_unseen_probe_labels.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps({"active_labels": ["XI"], "null_labels": ["YI"]}) + "\n",
                encoding="utf-8",
            )

            labels, paths = load_existing_fixed_unseen_probe_labels([root])

        self.assertEqual(labels, {"XI", "YI"})
        self.assertEqual(paths, [manifest])

    def test_loaded_fixed_unseen_labels_exclude_a_coupled_certification_probe(self):
        h0 = SparsePauliOperator({"XI": -1.0, "IX": -1.0})
        h1 = SparsePauliOperator({"ZI": 0.7, "IZ": -0.3, "ZZ": 1.1})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unrestricted, _ = build_fixed_probe_residual_labels(
                h0=h0,
                h1=h1,
                feedback_residual_labels=[],
                probe_agp_terms=4,
                probe_residual_terms=1,
                intermediate_top_k=8,
                probe_name="probe_test",
            )
            manifest = root / "Models_Data" / "fixed_unseen_probe_labels.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps({"active_labels": unrestricted, "null_labels": []}) + "\n",
                encoding="utf-8",
            )
            fixed_labels, _ = load_existing_fixed_unseen_probe_labels([root])
            labels, _ = build_fixed_probe_residual_labels(
                h0=h0,
                h1=h1,
                feedback_residual_labels=[],
                extra_excluded_labels=sorted(fixed_labels),
                probe_agp_terms=4,
                probe_residual_terms=1,
                intermediate_top_k=8,
                probe_name="probe_test",
            )

        self.assertTrue(fixed_labels.isdisjoint(labels))

    def test_q20_summary_paths_are_run_scoped(self):
        payload = json.loads((Q20_DIR / "config.json").read_text(encoding="utf-8"))
        summary = payload["summary"]

        for key in ("path_images", "path_data"):
            configured = str(summary[key]).strip("/")
            self.assertTrue(configured.startswith("runs/"), configured)
            self.assertNotIn(configured, {"Images", "Models_Data"})

    def test_restart_folders_does_not_recreate_root_scratch_dirs(self):
        config_path = Q20_DIR / "config.json"
        agp_restart.configure_run_dir(config_path)
        configured = {
            path.relative_to(Q20_DIR).as_posix().rstrip("/")
            for path in agp_restart.configured_paths(config_path)
        }

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

            prune_run_dir = agp_prune_support.RUN_DIR
            certify_run_dir = agp_certify_coupled.RUN_DIR
            try:
                agp_prune_support.RUN_DIR = fake_q20
                agp_certify_coupled.RUN_DIR = fake_q20

                self.assertEqual(agp_prune_support.latest_coupled_run(), summary_path.parents[1])
                self.assertEqual(agp_certify_coupled.latest_summary(), summary_path)
            finally:
                agp_prune_support.RUN_DIR = prune_run_dir
                agp_certify_coupled.RUN_DIR = certify_run_dir

    def test_q20_fixed_budget_uses_q7_output_cap(self):
        self.assertEqual(resolve_active_agp_budget(n_qubits=6, cap_qubits=7), 4**6)
        self.assertEqual(resolve_active_agp_budget(n_qubits=7, cap_qubits=7), 4**7)
        self.assertEqual(resolve_active_agp_budget(n_qubits=20, cap_qubits=7), 4**7)

        self.assertEqual(
            resolve_exploratory_agp_budget(n_qubits=20, active_cap_qubits=7, exploratory_cap_qubits=8),
            4**8,
        )

    def test_q20_config_uses_q15_parity_fixed_k_support_swaps(self):
        payload = json.loads((Q20_DIR / "config.json").read_text(encoding="utf-8"))
        feedback = payload["holdout_feedback"]
        support_swap = feedback["support_swap"]

        self.assertNotIn("coupled_curriculum", payload)
        self.assertEqual(feedback["base_agp_terms"], 32768)
        self.assertEqual(feedback["holdout_residual_top_k"], 81920)
        self.assertEqual(feedback["iterations"], 20)
        self.assertEqual(feedback["add_residual_terms_per_iteration"], 3072)
        self.assertTrue(support_swap["enabled"])
        self.assertEqual(support_swap["terms_per_iteration"], 256)
        self.assertEqual(support_swap["start_round"], 2)

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

    def test_stratified_certification_probe_selection_is_bounded_and_disjoint(self):
        scored = [
            ("XXII", 100.0),
            ("YYII", 90.0),
            ("XIII", 80.0),
            ("IXII", 70.0),
            ("ZZZZ", 60.0),
            ("IIII", 50.0),
            ("XYZI", 40.0),
        ]

        selected = agp_certify_support.select_order_stratified_labels(
            scored,
            count=4,
            excluded={"XXII", "IIII"},
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(len(set(selected)), 4)
        self.assertNotIn("XXII", selected)
        self.assertNotIn("IIII", selected)
        self.assertIn("YYII", selected)
        self.assertTrue(any(agp_certify_support.pauli_weight(label) == 1 for label in selected))
        self.assertTrue(any(agp_certify_support.pauli_weight(label) == 4 for label in selected))

    def test_support_certification_claim_requires_no_failures_or_gaps_for_certified(self):
        checks = {
            "training_residual": {"status": "pass"},
            "holdout_residual": {"status": "pass"},
            "unseen_residual": {"status": "pass"},
            "multi_holdout": {"status": "pass"},
            "q_sweep_plateau": {"status": "pass"},
            "omitted_term_pressure": {"status": "pass"},
            "physical_validation": {"status": "not tested"},
        }

        self.assertEqual(
            agp_certify_support.classify_claim_level(checks),
            "candidate_robust_sparse_agp",
        )
        checks["physical_validation"] = {"status": "pass"}
        self.assertEqual(
            agp_certify_support.classify_claim_level(checks),
            "certified_sparse_agp_for_this_path_and_tolerance",
        )
        checks["omitted_term_pressure"] = {"status": "fail"}
        self.assertEqual(
            agp_certify_support.classify_claim_level(checks),
            "projected_sparse_agp_experiment",
        )


if __name__ == "__main__":
    unittest.main()
