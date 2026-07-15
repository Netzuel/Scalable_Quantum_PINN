import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
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
    build_expanding_fixed_unseen_probe,
    compact_support_swap_plan,
    configured_generated_run_roots,
    FIXED_UNSEEN_ROW_FIELDS,
    feedback_refinements_complete,
    fixed_unseen_probe_candidate_cap,
    fixed_unseen_probe_manifest_identity,
    fixed_unseen_probe_settings_from_feedback,
    fixed_unseen_reference_rms,
    assert_fixed_unseen_manifest_lifecycle,
    load_existing_certification_probe_labels,
    load_or_validate_fixed_unseen_probe,
    make_adaptive_tau_grid,
    merge_fixed_unseen_probe_metrics,
    pau_transfer_stability_settings_from_feedback,
    save_fixed_unseen_probe,
    support_swap_settings_from_feedback,
    temporal_refinement_settings_from_feedback,
)
import agp_baseline_train  # noqa: E402
import agp_holdout_feedback  # noqa: E402
from utils import SparsePauliOperator  # noqa: E402


class AGPSupportSwapTests(unittest.TestCase):
    def test_uncapped_fixed_unseen_expansion_doubles_until_complete(self):
        settings = fixed_unseen_probe_settings_from_feedback(
            {"fixed_unseen_probes": {"enabled": True, "active_terms": 1, "null_terms": 1}}
        )
        requests: list[int] = []

        def generate(request: int) -> list[str]:
            requests.append(request)
            return ["M", "A", "B", "C", "D", "E", "F", "G"][:request]

        probe, _ = build_expanding_fixed_unseen_probe(
            generate_candidates=generate,
            reference_rms_for_labels=lambda labels: np.asarray(
                [0.0 if label == "G" else 1.0 for label in labels]
            ),
            settings=settings,
            moving_holdout_terms=1,
            excluded_labels=set(),
            initial_request=2,
            resource_cap=None,
        )

        self.assertEqual(requests, [2, 4, 8])
        self.assertEqual(probe["status"], "complete")
        self.assertIsNone(probe["resource_cap"])

    def test_fixed_unseen_expansion_respects_exact_configured_cap(self):
        settings = fixed_unseen_probe_settings_from_feedback(
            {"fixed_unseen_probes": {"enabled": True, "active_terms": 1, "null_terms": 1}}
        )
        requests: list[int] = []
        probe, _ = build_expanding_fixed_unseen_probe(
            generate_candidates=lambda request: requests.append(request) or ["M", "A"][:request],
            reference_rms_for_labels=lambda labels: np.ones(len(labels)),
            settings=settings,
            moving_holdout_terms=1,
            excluded_labels=set(),
            initial_request=2,
            resource_cap=2,
        )

        self.assertEqual(requests, [2])
        self.assertEqual(probe["resource_cap"], 2)
        self.assertEqual(probe["insufficiency_reason"], "resource_cap_reached")

    def test_fixed_unseen_candidate_cap_is_optional_and_rejects_below_holdout(self):
        self.assertIsNone(
            fixed_unseen_probe_candidate_cap({}, moving_holdout_terms=2)
        )
        self.assertIsNone(
            fixed_unseen_probe_candidate_cap({}, initial_request=2)
        )
        self.assertEqual(
            fixed_unseen_probe_candidate_cap(
                {"fixed_unseen_probes": {"max_candidate_terms": 2}},
                moving_holdout_terms=2,
            ),
            2,
        )
        with self.assertRaisesRegex(ValueError, "moving holdout.*2"):
            fixed_unseen_probe_candidate_cap(
                {"fixed_unseen_probes": {"max_candidate_terms": 1}},
                moving_holdout_terms=2,
            )

    def test_certification_manifests_are_discovered_from_configured_run_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            scenario = Path(temporary)
            coupled_root = scenario / "runs" / "coupled"
            manifests = []
            for filename, label in (
                ("probe_gate_residual_labels.json", "ZI"),
                ("probe_watch_residual_labels.json", "IZ"),
                ("probe_test_residual_labels.json", "ZZ"),
            ):
                path = coupled_root / "generated" / "Models_Data" / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"labels": [label]}) + "\n", encoding="utf-8")
                manifests.append(path.resolve())
            payload = {"coupled_curriculum": {"output_root": "runs/coupled"}}

            roots = configured_generated_run_roots(payload, scenario_root=scenario)
            labels, paths = load_existing_certification_probe_labels(roots)

        self.assertEqual(labels, {"ZI", "IZ", "ZZ"})
        self.assertEqual(paths, sorted(manifests, key=str))

    def test_normal_runner_rejects_historical_state_without_fixed_probe_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for marker_name in ("summary", "round_checkpoint", "refinement_checkpoint"):
                with self.subTest(marker=marker_name):
                    output_dir = root / marker_name
                    data_dir = output_dir / "Models_Data"
                    if marker_name == "summary":
                        summary = data_dir / "holdout_feedback_summary_residual_2.json"
                        summary.parent.mkdir(parents=True)
                        summary.write_text(json.dumps({"rows": [{"feedback_round": 0}]}), encoding="utf-8")
                    elif marker_name == "round_checkpoint":
                        checkpoint = output_dir / "rounds" / "round_01" / "Models_Data" / "training_checkpoint.pt"
                        checkpoint.parent.mkdir(parents=True)
                        checkpoint.touch()
                    else:
                        checkpoint = output_dir / "adaptive_temporal_refinement" / "Models_Data" / "training_checkpoint.pt"
                        checkpoint.parent.mkdir(parents=True)
                        checkpoint.touch()

                    with self.assertRaisesRegex(
                        RuntimeError,
                        "new run root.*diagnostics-only.*certification-ineligible",
                    ):
                        assert_fixed_unseen_manifest_lifecycle(
                            output_dir=output_dir,
                            data_dir=data_dir,
                            residual_top_k=2,
                        )

    def test_main_persists_fixed_unseen_lifecycle_and_resumes_with_same_manifest(self):
        row_fields = {
            "training_final_relative_residual": 0.5,
            "holdout_relative_residual": 0.5,
            "unseen_relative_residual": 0.5,
            "unseen_relative_residual_status": {"valid": True, "reason": "finite_reference"},
            "seen_residual": 1.0,
            "seen_relative_residual": 0.5,
            "unseen_residual": 1.0,
            "unseen_reference_residual": 2.0,
            "unseen_residual_per_term": 1.0,
            "unseen_residual_terms": 1,
        }
        fixed_metrics = {
            "fixed_unseen_active_terms": 1,
            "fixed_unseen_active_residual": 1.0,
            "fixed_unseen_active_reference_residual": 2.0,
            "fixed_unseen_active_relative": 0.5,
            "fixed_unseen_active_status": {"valid": True, "reason": "finite_reference"},
            "fixed_unseen_null_terms": 1,
            "fixed_unseen_null_absolute_per_term": 0.25,
            "fixed_unseen_null_scaled": 0.125,
        }

        with tempfile.TemporaryDirectory() as temporary:
            scenario = Path(temporary)
            config_path = scenario / "config.json"
            payload = {
                "physical": {"parameters": {"num_qubits": 2, "hamiltonian_source": "unused.json"}},
                "support_sweep": {"agp_terms": [1], "intermediate_top_k": 8, "residual_top_k": 1},
                "holdout_feedback": {
                    "base_agp_terms": 1,
                    "holdout_residual_top_k": 2,
                    "iterations": 1,
                    "add_residual_terms_per_iteration": 1,
                    "unseen_residual_batches_after_final_iteration": 0,
                    "epochs_per_iteration": 1,
                    "baseline_root": "runs/baselines",
                    "output_root": "runs/feedback",
                    "fixed_unseen_probes": {
                        "enabled": True,
                        "active_terms": 1,
                        "null_terms": 1,
                        "candidate_multiplier": 1,
                        "max_candidate_terms": 8,
                    },
                    "temporal_refinement": {
                        "enabled": True,
                        "epochs": 1,
                        "num_points": 2,
                        "lr": 1.0e-4,
                        "run_dir": "temporal_refinement",
                    },
                },
                "coupled_curriculum": {"output_root": "runs/coupled"},
                "training": {"parameters": {"epochs": 1, "num_points": 2, "lr": 1.0e-4}},
            }
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            baseline_checkpoint = scenario / "runs" / "baselines" / "agp_1" / "Models_Data" / "training_checkpoint.pt"
            baseline_checkpoint.parent.mkdir(parents=True)
            torch.save(
                {
                    "model_state_dict": {},
                    "agp_labels": ["XI"],
                    "residual_labels": ["XX"],
                    "config": {},
                },
                baseline_checkpoint,
            )
            certification_path = (
                scenario
                / "runs"
                / "coupled"
                / "prior"
                / "Models_Data"
                / "probe_gate_residual_labels.json"
            )
            certification_path.parent.mkdir(parents=True)
            certification_path.write_text(json.dumps({"labels": ["ZI"]}) + "\n", encoding="utf-8")
            candidates = ["XY", "YX", "ZI", "IZ", "YY", "ZZ", "II", "ZX"]
            train_calls: list[Path] = []

            def fake_train_feedback_round(**kwargs):
                run_dir = kwargs["run_dir"]
                train_calls.append(run_dir)
                checkpoint = run_dir / "Models_Data" / "training_checkpoint.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": {},
                        "agp_labels": kwargs["agp_labels"],
                        "residual_labels": kwargs["residual_labels"],
                        "config": {},
                    },
                    checkpoint,
                )
                metadata = {
                    "first_commutator_nnz": 1,
                    "second_commutator_nnz": 1,
                    "final_intermediate_terms": 1,
                    "final_residual_terms": len(kwargs["residual_labels"]),
                    "support_swap": {"enabled": False, "swap_count": 0},
                }
                return kwargs["trainable_state"], {"relative_residual": 0.5}, metadata

            def fake_evaluate_one_run(**kwargs):
                labels = list(kwargs["common_residual_labels"])
                row = {
                    **row_fields,
                    "agp_terms": 1,
                    "holdout_residual_terms": len(labels),
                }
                spectrum = [
                    {"label": label, "residual_rms": float(len(labels) - index)}
                    for index, label in enumerate(labels)
                ]
                return row, spectrum

            old_feedback_run_dir = agp_holdout_feedback.RUN_DIR
            old_baseline_run_dir = agp_baseline_train.RUN_DIR
            argv = ["agp_holdout_feedback.py", "--config", str(config_path)]
            try:
                with patch.object(sys, "argv", argv), patch(
                    "agp_holdout_feedback.build_common_holdout_residual_labels",
                    side_effect=lambda **kwargs: (candidates[: kwargs["residual_top_k"]], 1),
                ), patch(
                    "agp_holdout_feedback.fixed_unseen_reference_rms",
                    side_effect=lambda **kwargs: np.asarray(
                        [0.0 if label in {"II", "ZZ"} else 1.0 for label in kwargs["candidate_labels"]]
                    ),
                ), patch(
                    "agp_holdout_feedback.load_pauli_hamiltonian_pair",
                    return_value=(None, None),
                ), patch(
                    "agp_holdout_feedback.evaluate_one_run",
                    side_effect=fake_evaluate_one_run,
                ), patch(
                    "agp_holdout_feedback.evaluate_fixed_unseen_probe",
                    return_value=fixed_metrics,
                ), patch(
                    "agp_holdout_feedback.train_feedback_round",
                    side_effect=fake_train_feedback_round,
                ):
                    agp_holdout_feedback.main()
                    output_dir = scenario / "runs" / "feedback" / "agp_1_residual_2_add_1_rounds_1"
                    manifest_path = output_dir / "Models_Data" / "fixed_unseen_probe_labels.json"
                    summary_path = output_dir / "Models_Data" / "holdout_feedback_summary_residual_2.json"
                    first_manifest = manifest_path.read_bytes()
                    first_summary = json.loads(summary_path.read_text(encoding="utf-8"))

                    agp_holdout_feedback.main()
                    resumed_manifest = manifest_path.read_bytes()
                    resumed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            finally:
                agp_holdout_feedback.RUN_DIR = old_feedback_run_dir
                agp_baseline_train.RUN_DIR = old_baseline_run_dir

        manifest = json.loads(first_manifest)
        self.assertNotIn("ZI", manifest["active_labels"] + manifest["null_labels"])
        self.assertIn(str(certification_path.resolve()), manifest["certification_probe_manifest_paths"])
        self.assertEqual(first_manifest, resumed_manifest)
        self.assertEqual(len(train_calls), 2)
        for summary in (first_summary, resumed_summary):
            self.assertEqual(len(summary["rows"]), 2)
            for row in summary["rows"]:
                self.assertEqual(
                    {key for key in row if key.startswith("fixed_unseen_")},
                    set(FIXED_UNSEEN_ROW_FIELDS),
                )
            self.assertEqual(
                {key for key in summary["rounds"][0] if key.startswith("fixed_unseen_")},
                set(FIXED_UNSEEN_ROW_FIELDS),
            )
            self.assertEqual(
                {key for key in summary["temporal_refinement"] if key.startswith("fixed_unseen_")},
                set(FIXED_UNSEEN_ROW_FIELDS),
            )

    def test_fixed_unseen_evaluator_returns_exact_prefixed_row_schema(self):
        probe = {
            "active_labels": ["XI"],
            "null_labels": ["YI"],
            "reference_rms_threshold": 1.0e-12,
        }
        rows = iter(
            [
                {"holdout_total_residual": 2.0, "holdout_reference_residual": 4.0},
                {"holdout_total_residual": 3.0, "holdout_reference_residual": 0.0},
            ]
        )
        with patch("agp_holdout_feedback.load_checkpoint_labels", return_value=([], ["ZI"])), patch(
            "agp_holdout_feedback.evaluate_one_run",
            side_effect=lambda **_: (next(rows), []),
        ):
            from agp_holdout_feedback import evaluate_fixed_unseen_probe

            result = evaluate_fixed_unseen_probe(
                run_dir=Path("unused"),
                config_payload={},
                probe_metadata=probe,
                intermediate_top_k=2,
                device=torch.device("cpu"),
            )

        self.assertEqual(set(result), set(FIXED_UNSEEN_ROW_FIELDS))
        self.assertEqual(result["fixed_unseen_active_terms"], 1)
        self.assertAlmostEqual(result["fixed_unseen_active_relative"], 0.5)
        self.assertEqual(result["fixed_unseen_null_terms"], 1)
        self.assertAlmostEqual(result["fixed_unseen_null_absolute_per_term"], 3.0)

    def test_fixed_unseen_manifest_rejects_candidate_identity_mismatch(self):
        settings = fixed_unseen_probe_settings_from_feedback(
            {
                "fixed_unseen_probes": {
                    "enabled": True,
                    "active_terms": 1,
                    "null_terms": 1,
                    "reference_rms_threshold": 1.0e-12,
                    "seed": 7,
                    "candidate_multiplier": 2,
                }
            }
        )
        identity = fixed_unseen_probe_manifest_identity(
            settings=settings,
            candidate_universe_labels=["XI", "YI", "ZI"],
            excluded_labels={"II"},
            requested_candidate_terms=3,
        )
        payload = {
            **identity,
            "active_labels": ["XI"],
            "null_labels": ["YI"],
            "active_reference_rms": [2.0],
            "null_reference_rms": [0.0],
            "reference_rms_metadata": {"selected_hash": "placeholder"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixed_unseen_probe_labels.json"
            save_fixed_unseen_probe(path, payload)
            changed = fixed_unseen_probe_manifest_identity(
                settings=settings,
                candidate_universe_labels=["XI", "YI", "ZZ"],
                excluded_labels={"II"},
                requested_candidate_terms=3,
            )
            with self.assertRaisesRegex(ValueError, "candidate universe identity"):
                load_or_validate_fixed_unseen_probe(
                    path,
                    expected_excluded_labels={"II"},
                    expected_identity=changed,
                )

    def test_fixed_unseen_expansion_retries_until_both_partitions_fill(self):
        settings = fixed_unseen_probe_settings_from_feedback(
            {
                "fixed_unseen_probes": {
                    "enabled": True,
                    "active_terms": 1,
                    "null_terms": 1,
                    "candidate_multiplier": 1,
                }
            }
        )
        requests: list[int] = []

        def generate(request: int) -> list[str]:
            requests.append(request)
            return ["XI", "YI"] if request == 2 else ["XI", "YI", "ZI", "II"]

        probe, universe = build_expanding_fixed_unseen_probe(
            generate_candidates=generate,
            reference_rms_for_labels=lambda labels: np.asarray(
                [1.0 if label in {"XI", "ZI"} else 0.0 for label in labels]
            ),
            settings=settings,
            moving_holdout_terms=1,
            excluded_labels={"XI"},
            initial_request=2,
            resource_cap=4,
        )

        self.assertEqual(requests, [2, 4])
        self.assertEqual(universe, ["XI", "YI", "ZI", "II"])
        self.assertEqual(probe["status"], "complete")
        self.assertEqual(probe["expansion_history"][-1]["realized_candidate_terms"], 4)
        self.assertEqual(probe["active_labels"], ["ZI"])
        self.assertEqual(probe["null_labels"], ["II"])

    def test_fixed_unseen_expansion_records_generator_saturation(self):
        settings = fixed_unseen_probe_settings_from_feedback(
            {"fixed_unseen_probes": {"enabled": True, "active_terms": 2, "null_terms": 1}}
        )
        requests: list[int] = []

        def generate(request: int) -> list[str]:
            requests.append(request)
            return ["XI", "YI"]

        probe, _ = build_expanding_fixed_unseen_probe(
            generate_candidates=generate,
            reference_rms_for_labels=lambda labels: np.asarray([1.0 for _ in labels]),
            settings=settings,
            moving_holdout_terms=1,
            excluded_labels=set(),
            initial_request=2,
            resource_cap=8,
        )

        self.assertEqual(requests, [2, 4])
        self.assertEqual(probe["status"], "insufficient_candidates")
        self.assertEqual(probe["insufficiency_reason"], "generator_saturated")
        self.assertEqual(probe["realized_tail_terms"], 1)

    def test_fixed_unseen_manifest_rejects_selected_label_and_reference_metadata_mismatch(self):
        settings = fixed_unseen_probe_settings_from_feedback(
            {"fixed_unseen_probes": {"enabled": True, "active_terms": 1, "null_terms": 1}}
        )
        probe, _ = build_expanding_fixed_unseen_probe(
            generate_candidates=lambda _: ["XI", "YI", "ZI"],
            reference_rms_for_labels=lambda labels: np.asarray(
                [1.0 if label == "YI" else 0.0 for label in labels]
            ),
            settings=settings,
            moving_holdout_terms=1,
            excluded_labels=set(),
            initial_request=3,
            resource_cap=6,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixed_unseen_probe_labels.json"
            save_fixed_unseen_probe(path, probe)
            with self.assertRaisesRegex(ValueError, "selected active_labels"):
                load_or_validate_fixed_unseen_probe(
                    path,
                    expected_excluded_labels={"XI"},
                    expected_identity=probe,
                    expected_reference_rms_metadata=probe["reference_rms_metadata"],
                    expected_selected_labels={"active": ["ZI"], "null": probe["null_labels"]},
                )
            stale_metadata = dict(probe["reference_rms_metadata"])
            stale_metadata["candidate_sha256"] = "stale"
            with self.assertRaisesRegex(ValueError, "reference RMS metadata"):
                load_or_validate_fixed_unseen_probe(
                    path,
                    expected_excluded_labels={"XI"},
                    expected_identity=probe,
                    expected_reference_rms_metadata=stale_metadata,
                    expected_selected_labels={"active": probe["active_labels"], "null": probe["null_labels"]},
                )

    def test_stage_metric_merges_keep_the_exact_fixed_unseen_schema(self):
        metrics = {key: index for index, key in enumerate(FIXED_UNSEEN_ROW_FIELDS)}
        for stage in ("baseline", "curriculum", "temporal_refinement", "adaptive_temporal_refinement"):
            row = merge_fixed_unseen_probe_metrics({"stage": stage}, metrics)
            self.assertEqual({key for key in row if key.startswith("fixed_unseen_")}, set(FIXED_UNSEEN_ROW_FIELDS))

    def test_fixed_unseen_settings_are_read_from_feedback_config(self):
        settings = fixed_unseen_probe_settings_from_feedback(
            {
                "fixed_unseen_probes": {
                    "enabled": True,
                    "active_terms": 4096,
                    "null_terms": 4096,
                    "reference_rms_threshold": 1.0e-12,
                    "seed": 11,
                    "candidate_multiplier": 8,
                }
            }
        )

        self.assertTrue(settings.enabled)
        self.assertEqual(settings.active_terms, 4096)
        self.assertEqual(settings.null_terms, 4096)

    def test_persisted_fixed_probe_rejects_changed_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixed_unseen_probe_labels.json"
            save_fixed_unseen_probe(path, {"active_labels": ["XI"], "null_labels": ["YI"]})
            with self.assertRaisesRegex(ValueError, "immutable fixed unseen probe"):
                load_or_validate_fixed_unseen_probe(path, expected_excluded_labels={"XI"})

    def test_empty_fixed_probe_candidate_tail_needs_no_model(self):
        values = fixed_unseen_reference_rms(
            h0=None,
            h1=None,
            settings=None,
            agp_labels=[],
            candidate_labels=[],
        )

        self.assertEqual(values.shape, (0,))

    def test_pau_transfer_stability_settings_are_read_from_config(self):
        settings = pau_transfer_stability_settings_from_feedback(
            {
                "pau_transfer_stability": {
                    "enabled": True,
                    "max_initial_relative_residual": 1.0e8,
                    "fallback": "silu_rational_fit",
                }
            }
        )

        self.assertTrue(settings.enabled)
        self.assertEqual(settings.max_initial_relative_residual, 1.0e8)
        self.assertEqual(settings.fallback, "silu_rational_fit")

    def test_feedback_refinements_require_summary_entries_and_checkpoints(self):
        temporal = temporal_refinement_settings_from_feedback(
            {"temporal_refinement": {"enabled": True, "run_dir": "temporal_refinement"}}
        )
        adaptive = adaptive_temporal_refinement_settings_from_feedback(
            {"adaptive_temporal_refinement": {"enabled": True, "run_dir": "adaptive_temporal_refinement"}}
        )
        summary = {
            "temporal_refinement": {"enabled": True, "run_dir": "temporal_refinement"},
            "adaptive_temporal_refinement": {"enabled": True, "run_dir": "adaptive_temporal_refinement"},
        }

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            self.assertFalse(feedback_refinements_complete(summary, output_dir, temporal, adaptive))
            for run_dir in ("temporal_refinement", "adaptive_temporal_refinement"):
                checkpoint = output_dir / run_dir / "Models_Data" / "training_checkpoint.pt"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.touch()
            self.assertTrue(feedback_refinements_complete(summary, output_dir, temporal, adaptive))

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
