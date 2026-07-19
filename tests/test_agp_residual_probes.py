import json
import hashlib
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
    partition_fixed_unseen_candidates,
    select_fixed_unseen_probes,
)
from agp_holdout_feedback import (  # noqa: E402
    build_fixed_unseen_probe_manifest,
    feedback_unseen_plot_values,
    fixed_unseen_probe_settings_from_feedback,
    load_or_validate_fixed_unseen_probe,
    plot_fixed_unseen_probes,
    reserve_and_fit_feedback_candidates,
    save_fixed_unseen_probe,
    write_feedback_summary,
)
from agp_holdout_study import (  # noqa: E402
    Thresholds,
    feedback_threshold_decision,
    fixed_unseen_plot_series,
)


class AGPResidualProbeTests(unittest.TestCase):
    def test_feedback_reservoir_can_be_geometry_stratified(self):
        labels = [
            "XIIIIIII",
            "IXIIIIII",
            "IIIIIIXI",
            "IIIIIIIX",
            "XXIIIIII",
            "IXXIIIII",
            "IIIIIXXI",
            "IIIIIIXX",
        ]
        reference_rms = np.asarray([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0])

        probe, common, top_k, add_terms, fitted = reserve_and_fit_feedback_candidates(
            candidate_labels=labels,
            reference_rms=reference_rms,
            excluded_labels=set(),
            probe_settings=FixedUnseenProbeConfig(
                enabled=False,
                reservation_mode="pre_feedback_global",
            ),
            residual_top_k=4,
            add_residual_terms=0,
            residual_budget={},
            initial_residual_terms=0,
            rounds=1,
            unseen_batches_after_final_iteration=0,
            q=8,
            residual_stratification={
                "enabled": True,
                "locality_quotas": {"1": 2, "2": 2},
                "spatial_bins": 2,
                "seed": 11,
            },
        )

        orders = [sum(symbol != "I" for symbol in label) for label in common]
        self.assertEqual(probe["reserved_terms"], 0)
        self.assertEqual(top_k, 4)
        self.assertEqual(add_terms, 0)
        self.assertEqual(orders.count(1), 2)
        self.assertEqual(orders.count(2), 2)
        self.assertTrue(fitted["candidate_stratification"]["enabled"])
        self.assertEqual(fitted["candidate_stratification"]["selected_terms"], 4)

    def test_fixed_probe_is_reserved_before_feedback_budgeting(self):
        labels = ["XI", "YI", "ZI", "IX", "IY", "IZ"]
        reference_rms = np.asarray([4.0, 0.0, 3.0, 0.0, 2.0, 0.0])
        config = FixedUnseenProbeConfig(
            enabled=True,
            active_terms=2,
            null_terms=2,
            reference_rms_threshold=1.0e-12,
            seed=11,
        )

        probe, feedback_labels = partition_fixed_unseen_candidates(
            labels,
            reference_rms,
            excluded_labels=set(),
            config=config,
        )

        reserved = set(probe["active_labels"]) | set(probe["null_labels"])
        self.assertEqual(len(probe["active_labels"]), 2)
        self.assertEqual(len(probe["null_labels"]), 2)
        self.assertFalse(reserved & set(feedback_labels))
        self.assertEqual(
            feedback_labels,
            [label for label in labels if label not in reserved],
        )
        self.assertEqual(probe["reservation_mode"], "pre_feedback_global")

    def test_formal_probes_are_removed_without_dropping_existing_training_labels(self):
        labels = ["FORMAL", "TRAIN", "ACTIVE", "NULL"]
        reference_rms = np.asarray([5.0, 4.0, 3.0, 0.0])
        config = FixedUnseenProbeConfig(
            enabled=True,
            active_terms=1,
            null_terms=1,
            reference_rms_threshold=1.0e-12,
        )

        probe, feedback_labels = partition_fixed_unseen_candidates(
            labels,
            reference_rms,
            excluded_labels={"FORMAL", "TRAIN"},
            feedback_excluded_labels={"FORMAL"},
            config=config,
        )

        self.assertNotIn("FORMAL", feedback_labels)
        self.assertIn("TRAIN", feedback_labels)
        self.assertNotIn("FORMAL", probe["active_labels"] + probe["null_labels"])
        self.assertNotIn("TRAIN", probe["active_labels"] + probe["null_labels"])

    def test_fixed_probe_counts_accept_q_aware_resource_policies(self):
        settings = fixed_unseen_probe_settings_from_feedback(
            {
                "fixed_unseen_probes": {
                    "enabled": True,
                    "active_terms": {
                        "mode": "per_qubit",
                        "per_qubit": 0.4,
                        "minimum": 4,
                        "maximum": 64,
                    },
                    "null_terms": {
                        "mode": "per_qubit",
                        "per_qubit": 16.0,
                        "minimum": 64,
                        "maximum": 4096,
                    },
                }
            },
            q=156,
            capacity=4**156,
        )

        self.assertEqual(settings.active_terms, 62)
        self.assertEqual(settings.null_terms, 2496)
        self.assertEqual(settings.active_resource_budget["mode"], "per_qubit")
        self.assertEqual(settings.null_resource_budget["mode"], "per_qubit")

    def test_fixed_probe_reservation_mode_defaults_legacy_and_allows_global_opt_in(self):
        legacy = fixed_unseen_probe_settings_from_feedback(
            {"fixed_unseen_probes": {"enabled": True}}
        )
        global_mode = fixed_unseen_probe_settings_from_feedback(
            {
                "fixed_unseen_probes": {
                    "enabled": True,
                    "reservation_mode": "pre_feedback_global",
                }
            }
        )

        self.assertEqual(legacy.reservation_mode, "post_holdout_tail")
        self.assertEqual(global_mode.reservation_mode, "pre_feedback_global")

    def test_probe_reservation_precedes_feedback_round_fit(self):
        labels = ["XI", "YI", "ZI", "IX", "IY", "IZ", "XX", "YY", "ZZ", "XY"]
        reference_rms = np.asarray([4.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.5, 0.0])
        settings = FixedUnseenProbeConfig(
            enabled=True,
            active_terms=1,
            null_terms=1,
            reference_rms_threshold=1.0e-12,
        )

        probe, common, residual_top_k, add_terms, fitted = reserve_and_fit_feedback_candidates(
            candidate_labels=labels,
            reference_rms=reference_rms,
            excluded_labels=set(),
            probe_settings=settings,
            residual_top_k=10,
            add_residual_terms=3,
            residual_budget={"mode": "auto"},
            initial_residual_terms=2,
            rounds=2,
            unseen_batches_after_final_iteration=1,
        )

        reserved = set(probe["active_labels"]) | set(probe["null_labels"])
        self.assertEqual(residual_top_k, 8)
        self.assertEqual(add_terms, 2)
        self.assertEqual(len(common), 8)
        self.assertFalse(reserved & set(common))
        self.assertEqual(fitted["reserved_fixed_unseen_probe_terms"], 2)

    def test_legacy_tail_reservation_preserves_the_first_q_feedback_labels(self):
        labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
        reference_rms = np.asarray([9.0, 0.0, 8.0, 0.0, 7.0, 0.0, 6.0, 0.0])
        settings = fixed_unseen_probe_settings_from_feedback(
            {
                "fixed_unseen_probes": {
                    "enabled": True,
                    "active_terms": 1,
                    "null_terms": 1,
                }
            }
        )

        probe, common, residual_top_k, _, _ = reserve_and_fit_feedback_candidates(
            candidate_labels=labels,
            reference_rms=reference_rms,
            excluded_labels=set(),
            probe_settings=settings,
            residual_top_k=4,
            add_residual_terms=0,
            residual_budget={"mode": "explicit"},
            initial_residual_terms=2,
            rounds=0,
            unseen_batches_after_final_iteration=0,
        )

        self.assertEqual(settings.reservation_mode, "post_holdout_tail")
        self.assertEqual(residual_top_k, 4)
        self.assertEqual(common, labels[:4])
        self.assertTrue(set(probe["active_labels"] + probe["null_labels"]) <= set(labels[4:]))

    def test_main_residual_plot_uses_fixed_probe_without_hiding_moving_gaps(self):
        fixed, moving = feedback_unseen_plot_values(
            [
                {
                    "fixed_unseen_active_relative": 0.4,
                    "unseen_relative_residual": None,
                    "unseen_residual_terms": 5,
                },
                {
                    "fixed_unseen_active_relative": 0.3,
                    "unseen_relative_residual": 2.0,
                    "unseen_residual_terms": 5,
                },
            ]
        )

        self.assertEqual(fixed, [0.4, 0.3])
        self.assertTrue(np.isnan(moving[0]))
        self.assertEqual(moving[1], 2.0)

    def test_current_manifest_loader_rejects_missing_hash_and_invalid_provenance(self):
        base = {
            "schema_version": 2,
            "enabled": True,
            "status": "complete",
            "active_labels": ["XI"],
            "null_labels": ["YI"],
            "reference_rms_metadata": {"selected": {}},
            "certification_eligible": True,
            "provenance": "pre_training_fixed_probe",
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixed_unseen_probe_labels.json"
            save_fixed_unseen_probe(path, base)
            with self.assertRaisesRegex(ValueError, "missing_manifest_sha256"):
                load_or_validate_fixed_unseen_probe(path, expected_excluded_labels=set())

            invalid = build_fixed_unseen_probe_manifest(
                base,
                certification_eligible=True,
                provenance="diagnostic_backfill",
            )
            save_fixed_unseen_probe(path, invalid)
            with self.assertRaisesRegex(ValueError, "provenance"):
                load_or_validate_fixed_unseen_probe(path, expected_excluded_labels=set())

    def test_certification_manifest_requires_pretraining_lifecycle_evidence(self):
        manifest = build_fixed_unseen_probe_manifest(
            {
                "schema_version": 2,
                "enabled": True,
                "status": "complete",
                "active_labels": ["XI"],
                "null_labels": ["YI"],
                "reference_rms_metadata": {"selected": {}},
            },
            certification_eligible=True,
            provenance="pre_training_fixed_probe",
        )
        self.assertEqual(
            manifest["training_lifecycle"],
            {
                "probe_selection_phase": "before_optimizer_step",
                "baseline_checkpoint_present": False,
            },
        )

        manifest.pop("training_lifecycle")
        unhashed = {
            key: value
            for key, value in manifest.items()
            if key != "manifest_sha256"
        }
        manifest["manifest_sha256"] = hashlib.sha256(
            json.dumps(
                unhashed,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        decision = feedback_threshold_decision(
            [
                {
                    "feedback_round": 1,
                    "holdout_relative_residual": 0.01,
                    "fixed_unseen_active_relative": 0.2,
                    "fixed_unseen_active_status": {
                        "valid": True,
                        "reason": "finite_reference",
                    },
                }
            ],
            holdout_threshold=0.1,
            unseen_threshold=1.0,
            fixed_unseen_probe=manifest,
        )
        self.assertEqual(decision["unseen_gate"]["status"], "not_tested")
        self.assertEqual(
            decision["unseen_gate"]["reason"],
            "missing_pretraining_lifecycle",
        )

    def test_current_manifest_gate_fails_closed_on_inconsistent_provenance(self):
        row = {
            "feedback_round": 1,
            "holdout_relative_residual": 0.01,
            "fixed_unseen_active_relative": 0.2,
            "fixed_unseen_active_status": {"valid": True, "reason": "finite_reference"},
        }
        manifest = build_fixed_unseen_probe_manifest(
            {
                "schema_version": 2,
                "enabled": True,
                "status": "complete",
                "active_labels": ["XI"],
                "null_labels": ["YI"],
                "reference_rms_metadata": {"selected": {}},
            },
            certification_eligible=True,
            provenance="diagnostic_backfill",
        )

        decision = feedback_threshold_decision(
            [row],
            holdout_threshold=0.1,
            unseen_threshold=1.0,
            fixed_unseen_probe=manifest,
        )

        self.assertEqual(decision["unseen_gate"]["status"], "not_tested")
        self.assertEqual(decision["unseen_gate"]["reason"], "invalid_fixed_unseen_provenance")

    def test_current_manifest_gate_fails_closed_on_invalid_hash_and_legacy_schema(self):
        row = {
            "feedback_round": 1,
            "holdout_relative_residual": 0.01,
            "fixed_unseen_active_relative": 0.2,
            "fixed_unseen_active_status": {"valid": True, "reason": "finite_reference"},
        }
        current_manifest = build_fixed_unseen_probe_manifest(
            {
                "schema_version": 2,
                "enabled": True,
                "status": "complete",
                "active_labels": ["XI"],
                "null_labels": ["YI"],
                "reference_rms_metadata": {"selected": {}},
            },
            certification_eligible=True,
            provenance="pre_training_fixed_probe",
        )
        current_manifest["manifest_sha256"] = "tampered"

        invalid_hash = feedback_threshold_decision(
            [row],
            holdout_threshold=0.1,
            unseen_threshold=1.0,
            fixed_unseen_probe=current_manifest,
        )
        self.assertEqual(invalid_hash["unseen_gate"]["status"], "not_tested")
        self.assertEqual(
            invalid_hash["unseen_gate"]["reason"],
            "invalid_fixed_unseen_manifest_hash",
        )

        legacy = feedback_threshold_decision(
            [row],
            holdout_threshold=0.1,
            unseen_threshold=1.0,
            fixed_unseen_probe={
                "schema_version": 1,
                "enabled": True,
                "status": "complete",
                "certification_eligible": True,
                "provenance": "pre_training_fixed_probe",
            },
        )
        self.assertEqual(legacy["unseen_gate"]["status"], "not_tested")
        self.assertEqual(legacy["unseen_gate"]["reason"], "legacy_fixed_unseen_manifest")

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

    def test_valid_fixed_active_row_lifecycle_without_manifest_is_not_tested(self):
        row = {
            "feedback_round": 7,
            "holdout_relative_residual": 0.05,
            "unseen_relative_residual": None,
            "fixed_unseen_enabled": True,
            "fixed_unseen_probe_status": "complete",
            "fixed_unseen_active_relative": 0.8,
            "fixed_unseen_active_status": {"valid": True, "reason": "finite_reference"},
        }

        decision = feedback_threshold_decision(
            [row],
            holdout_threshold=0.1,
            unseen_threshold=1.0,
        )

        self.assertEqual(decision["status"], "not_found_in_feedback_run")
        self.assertEqual(decision["unseen_gate_source"], "fixed_unseen_active")
        self.assertEqual(decision["unseen_gate"]["status"], "not_tested")
        self.assertEqual(decision["unseen_gate"]["reason"], "missing_fixed_unseen_manifest")

    def test_explicit_enabled_complete_manifest_allows_fixed_active_gate(self):
        row = {
            "feedback_round": 7,
            "holdout_relative_residual": 0.05,
            "fixed_unseen_active_relative": 0.8,
            "fixed_unseen_active_status": {"valid": True, "reason": "finite_reference"},
        }
        manifest = build_fixed_unseen_probe_manifest(
            {"enabled": True, "status": "complete", "schema_version": 2},
            certification_eligible=True,
            provenance="pre_training_fixed_probe",
        )

        decision = feedback_threshold_decision(
            [row],
            holdout_threshold=0.1,
            unseen_threshold=1.0,
            fixed_unseen_probe=manifest,
        )

        self.assertEqual(decision["status"], "found_feedback_round")
        self.assertEqual(decision["unseen_gate"]["status"], "pass")

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
            fixed_unseen_probe=build_fixed_unseen_probe_manifest(
                {"enabled": True, "status": "insufficient_candidates", "schema_version": 2},
                certification_eligible=True,
                provenance="pre_training_fixed_probe",
            ),
        )

        self.assertEqual(decision["status"], "not_found_in_feedback_run")
        self.assertEqual(decision["unseen_gate"]["status"], "not_tested")
        self.assertEqual(decision["unseen_gate"]["reason"], "insufficient_candidates")

    def test_diagnostic_backfill_manifest_cannot_certify_fixed_active_gate(self):
        row = {
            "feedback_round": 1,
            "holdout_relative_residual": 0.01,
            "fixed_unseen_active_relative": 0.2,
            "fixed_unseen_active_status": {"valid": True, "reason": "finite_reference"},
        }

        decision = feedback_threshold_decision(
            [row],
            holdout_threshold=0.1,
            unseen_threshold=1.0,
            fixed_unseen_probe=build_fixed_unseen_probe_manifest(
                {"enabled": True, "status": "complete", "schema_version": 2},
                certification_eligible=False,
                provenance="diagnostic_backfill",
            ),
        )

        self.assertEqual(decision["status"], "not_found_in_feedback_run")
        self.assertEqual(decision["unseen_gate"]["status"], "not_tested")
        self.assertEqual(decision["unseen_gate"]["reason"], "historical_diagnostic_backfill")

    def test_plot_series_preserves_nan_gaps_and_active_null_labels(self):
        series = fixed_unseen_plot_series([
            {
                "feedback_round": 7,
                "fixed_unseen_active_relative": 0.4,
                "fixed_unseen_null_absolute_per_term": 0.03,
                "fixed_unseen_null_scaled": 0.02,
                "unseen_relative_residual": None,
            },
            {
                "feedback_round": 8,
                "fixed_unseen_active_relative": None,
                "fixed_unseen_null_absolute_per_term": None,
                "fixed_unseen_null_scaled": 0.04,
                "unseen_relative_residual": 2.0,
            },
        ])

        self.assertEqual(series["rounds"].tolist(), [7.0, 8.0])
        self.assertEqual(series["active_relative"][0], 0.4)
        self.assertTrue(np.isnan(series["active_relative"][1]))
        self.assertEqual(series["null_absolute_per_term"][0], 0.03)
        self.assertTrue(np.isnan(series["null_absolute_per_term"][1]))
        self.assertEqual(series["null_scaled"].tolist(), [0.02, 0.04])
        self.assertTrue(np.isnan(series["moving_unseen_relative"][0]))
        self.assertEqual(series["moving_unseen_relative"][1], 2.0)
        self.assertEqual(series["labels"]["active_relative"], "fixed active quotient")
        self.assertEqual(series["labels"]["null_absolute_per_term"], "null absolute / term")
        self.assertEqual(series["labels"]["null_scaled"], "null scaled")

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

    def test_feedback_summary_writes_fixed_manifest_decision_and_diagnostics(self):
        row = {
            "feedback_round": 0,
            "n_qubits": 2,
            "training_final_relative_residual": 0.01,
            "holdout_relative_residual": 0.05,
            "unseen_relative_residual": None,
            "unseen_relative_residual_status": {"valid": False, "reason": "zero_reference"},
            "unseen_residual_terms": 0,
            "seen_residual": 0.01,
            "unseen_residual": 0.0,
            "seen_relative_residual": 0.01,
            "fixed_unseen_active_relative": 0.5,
            "fixed_unseen_active_status": {"valid": True, "reason": "finite_reference"},
            "fixed_unseen_null_absolute_per_term": 0.03,
            "fixed_unseen_null_scaled": 0.04,
        }
        manifest = build_fixed_unseen_probe_manifest(
            {
                "schema_version": 2,
                "enabled": True,
                "status": "complete",
                "candidate_universe": {"count": 4, "sha256": "manifest-identity"},
            },
            certification_eligible=True,
            provenance="pre_training_fixed_probe",
        )
        thresholds = Thresholds(
            plateau=1.0,
            holdout=0.1,
            unseen=1.0,
            top_stability=0.0,
            top_fraction=0.1,
        )

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_feedback_summary(
                output_dir=output_dir,
                rows=[row],
                spectra={0: []},
                round_rows=[],
                residual_top_k=4,
                thresholds=thresholds,
                residual_budget={},
                fixed_unseen_probe=manifest,
            )

            with (output_dir / "Models_Data" / "holdout_feedback_summary_residual_4.json").open(
                encoding="utf-8"
            ) as handle:
                payload = json.load(handle)

        self.assertEqual(payload["fixed_unseen_probe"]["candidate_universe"]["sha256"], "manifest-identity")
        self.assertEqual(payload["fixed_unseen_probe"]["status"], "complete")
        self.assertEqual(payload["decision"]["unseen_gate_source"], "fixed_unseen_active")
        self.assertEqual(payload["decision"]["unseen_gate"]["status"], "pass")
        self.assertEqual(payload["moving_unseen_diagnostic"][0]["status"]["reason"], "zero_reference")
        self.assertEqual(payload["rows"][0]["fixed_unseen_active_relative"], 0.5)
        self.assertEqual(payload["rows"][0]["fixed_unseen_null_scaled"], 0.04)

    def test_feedback_summary_rejects_failed_refinement_and_selects_metric_champion(self):
        row = {
            "feedback_round": 0,
            "n_qubits": 2,
            "training_final_relative_residual": 0.01,
            "holdout_relative_residual": 0.05,
            "unseen_relative_residual": 0.5,
            "unseen_relative_residual_status": {"valid": True, "reason": "finite_reference"},
            "unseen_residual_terms": 1,
            "seen_residual": 0.01,
            "unseen_residual": 0.02,
            "seen_relative_residual": 0.01,
            "fixed_unseen_active_relative": 0.5,
            "fixed_unseen_active_status": {"valid": True, "reason": "finite_reference"},
            "fixed_unseen_null_absolute_per_term": 0.0,
            "fixed_unseen_null_scaled": 0.0,
            "run_dir": "baseline",
        }
        manifest = build_fixed_unseen_probe_manifest(
            {"schema_version": 2, "enabled": True, "status": "complete"},
            certification_eligible=True,
            provenance="pre_training_fixed_probe",
        )
        thresholds = Thresholds(
            plateau=1.0,
            holdout=0.1,
            unseen=1.0,
            top_stability=0.0,
            top_fraction=0.1,
        )
        temporal = {
            "enabled": True,
            "run_dir": "temporal_refinement",
            "holdout_relative_residual": 0.04,
            "fixed_unseen_active_relative": 0.8,
            "fixed_unseen_active_status": {"valid": True, "reason": "finite_reference"},
        }
        adaptive = {
            "enabled": True,
            "run_dir": "adaptive_temporal_refinement",
            "holdout_relative_residual": 0.03,
            "fixed_unseen_active_relative": 1.2,
            "fixed_unseen_active_status": {"valid": True, "reason": "finite_reference"},
        }

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_feedback_summary(
                output_dir=output_dir,
                rows=[row],
                spectra={0: []},
                round_rows=[],
                residual_top_k=4,
                thresholds=thresholds,
                residual_budget={},
                fixed_unseen_probe=manifest,
                temporal_refinement=temporal,
                adaptive_temporal_refinement=adaptive,
            )
            payload = json.loads(
                (output_dir / "Models_Data" / "holdout_feedback_summary_residual_4.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(payload["temporal_refinement"]["accepted"])
        self.assertFalse(payload["adaptive_temporal_refinement"]["accepted"])
        self.assertEqual(payload["selected_run"]["status"], "accepted")
        self.assertEqual(payload["selected_run"]["source"], "feedback_round")
        self.assertEqual(payload["selected_run"]["run_dir"], "baseline")

    def test_feedback_summary_ranks_accepted_refinements_by_frozen_metrics(self):
        row = {
            "feedback_round": 0,
            "n_qubits": 2,
            "training_final_relative_residual": 0.01,
            "holdout_relative_residual": 0.05,
            "unseen_relative_residual": 0.5,
            "unseen_relative_residual_status": {"valid": True, "reason": "finite_reference"},
            "unseen_residual_terms": 1,
            "seen_residual": 0.01,
            "unseen_residual": 0.02,
            "seen_relative_residual": 0.01,
            "fixed_unseen_active_relative": 0.5,
            "fixed_unseen_active_status": {"valid": True, "reason": "finite_reference"},
            "fixed_unseen_null_absolute_per_term": 0.0,
            "fixed_unseen_null_scaled": 0.05,
            "run_dir": "rounds/round_00",
        }
        manifest = build_fixed_unseen_probe_manifest(
            {"schema_version": 2, "enabled": True, "status": "complete"},
            certification_eligible=True,
            provenance="pre_training_fixed_probe",
        )
        thresholds = Thresholds(
            plateau=1.0,
            holdout=0.1,
            unseen=1.0,
            top_stability=0.0,
            top_fraction=0.1,
        )
        temporal = {
            "enabled": True,
            "run_dir": "temporal_refinement",
            "holdout_relative_residual": 0.04,
            "fixed_unseen_active_relative": 0.2,
            "fixed_unseen_active_status": {"valid": True, "reason": "finite_reference"},
            "fixed_unseen_null_scaled": 0.2,
        }
        adaptive = {
            "enabled": True,
            "run_dir": "adaptive_temporal_refinement",
            "holdout_relative_residual": 0.03,
            "fixed_unseen_active_relative": 0.4,
            "fixed_unseen_active_status": {"valid": True, "reason": "finite_reference"},
            "fixed_unseen_null_scaled": 0.1,
        }

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_feedback_summary(
                output_dir=output_dir,
                rows=[row],
                spectra={0: []},
                round_rows=[],
                residual_top_k=4,
                thresholds=thresholds,
                residual_budget={},
                fixed_unseen_probe=manifest,
                temporal_refinement=temporal,
                adaptive_temporal_refinement=adaptive,
            )
            payload = json.loads(
                (output_dir / "Models_Data" / "holdout_feedback_summary_residual_4.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(payload["temporal_refinement"]["accepted"])
        self.assertTrue(payload["adaptive_temporal_refinement"]["accepted"])
        self.assertEqual(payload["selected_run"]["source"], "temporal_refinement")
        self.assertEqual(
            payload["selected_run"]["selection_metric"],
            {
                "fixed_unseen_active_relative": 0.2,
                "holdout_relative_residual": 0.04,
                "fixed_unseen_null_scaled": 0.2,
            },
        )
        candidates = payload["selected_run"]["selection_candidates"]
        self.assertEqual(
            [candidate["source"] for candidate in candidates],
            ["temporal_refinement", "adaptive_temporal_refinement", "feedback_round"],
        )
        self.assertEqual(
            [candidate["selection_metric"] for candidate in candidates],
            [
                {
                    "fixed_unseen_active_relative": 0.2,
                    "holdout_relative_residual": 0.04,
                    "fixed_unseen_null_scaled": 0.2,
                },
                {
                    "fixed_unseen_active_relative": 0.4,
                    "holdout_relative_residual": 0.03,
                    "fixed_unseen_null_scaled": 0.1,
                },
                {
                    "fixed_unseen_active_relative": 0.5,
                    "holdout_relative_residual": 0.05,
                    "fixed_unseen_null_scaled": 0.05,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
