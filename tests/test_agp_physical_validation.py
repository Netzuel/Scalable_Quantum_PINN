import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
FRAMEWORK_SCRIPTS_DIR = ROOT / "tests" / "sparse_agp_curriculum" / "scripts"
TESTS_DIR = ROOT / "tests"
for path in (SCRIPTS_DIR, FRAMEWORK_SCRIPTS_DIR, TESTS_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agp_support import KrylovSupportConfig, select_krylov_agp_labels
from agp_holdout_feedback import fit_residual_budget_to_available
from agp_holdout_study import relative_metric_with_reference_status
from agp_plot_annotations import (
    find_physical_summary_for_images_dir,
    hcd_context_lines_for_images_dir,
    physical_comparison_payload_for_images_dir,
    physical_comparison_rows,
    physical_validation_note,
    plot_physical_comparison_table,
)
from agp_physical_validation import (
    apply_pauli_sum,
    build_action_cache,
    build_learned_variant_specs,
    final_run_from_summary,
    select_best_learned_variant,
    variational_l1_agp,
)
from agp_regenerate_hcd_summaries import find_coefficient_export
import agp_physical_validation
from utils import SparsePauliOperator, transverse_field_ising_problem


class AGPPhysicalValidationTests(unittest.TestCase):
    def test_hcd_context_lines_include_hamiltonians_and_energies(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)

            q15_study = root / "q15" / "sweep_test"
            q15_images = q15_study / "runs" / "retained" / "Images"
            q15_data = q15_images.parent / "Models_Data"
            q15_images.mkdir(parents=True)
            q15_data.mkdir(parents=True)
            (q15_study / "config.json").write_text(
                __import__("json").dumps(
                    {
                        "physical": {
                            "parameters": {
                                "system": "TransverseIsingDriverProblem",
                                "num_qubits": 15,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (q15_data / "physical_validation_summary.json").write_text(
                __import__("json").dumps(
                    {
                        "ground_energy": -19.25,
                        "results": {
                            "no_cd": {
                                "final_energy": -8.0,
                                "ground_state_fidelity": 0.01,
                            },
                            "kipu_dqfm_l1": {
                                "final_energy": -14.5,
                                "ground_state_fidelity": 0.25,
                            },
                            "learned_sparse_agp": {
                                "final_energy": -19.062659369221652,
                                "ground_state_fidelity": 0.95,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            q20_study = root / "q20" / "sweep_test"
            q20_images = q20_study / "runs" / "retained" / "Images"
            q20_images.mkdir(parents=True)
            (q20_study / "config.json").write_text(
                __import__("json").dumps(
                    {
                        "physical": {
                            "parameters": {
                                "system": "Hidrogen",
                                "num_qubits": 20,
                            }
                        },
                        "scalable_validation": {
                            "ground_energy": -1.1400734808760409,
                            "pinn_final_energy_status": "not_tested",
                        },
                    }
                ),
                encoding="utf-8",
            )

            q15_lines = hcd_context_lines_for_images_dir(q15_images)
            q20_lines = hcd_context_lines_for_images_dir(q20_images)

            self.assertEqual(len(q15_lines), 5)
            self.assertIn(r"H_{\mathrm{initial}}=-\sum", q15_lines[0])
            self.assertIn(r"H_{\mathrm{final}}=-\sum", q15_lines[1])
            self.assertIn("-19.25", q15_lines[2])
            self.assertIn("no CD", q15_lines[3])
            self.assertIn("-8", q15_lines[3])
            self.assertIn("nested", q15_lines[3])
            self.assertIn("0.25", q15_lines[3])
            self.assertIn("PINN", q15_lines[4])
            self.assertIn("-19.0627", q15_lines[4])
            self.assertIn("0.95", q15_lines[4])

            self.assertEqual(len(q20_lines), 5)
            self.assertIn(r"\Pi_{\{I,Z\}}", q20_lines[0])
            self.assertIn(r"\mathcal{S}_{H_2}", q20_lines[1])
            self.assertIn("-1.14007", q20_lines[2])
            self.assertIn("not computed", q20_lines[3])
            self.assertIn("not computed", q20_lines[4])

    def test_plot_payload_uses_tracked_scalable_pinn_energy_fallback(self):
        with TemporaryDirectory() as tmp:
            study_dir = Path(tmp) / "q20" / "sweep_test"
            images_dir = study_dir / "runs" / "retained" / "Images"
            images_dir.mkdir(parents=True)
            (study_dir / "config.json").write_text(
                __import__("json").dumps(
                    {
                        "physical": {
                            "parameters": {
                                "system": "Hidrogen",
                                "num_qubits": 20,
                            }
                        },
                        "scalable_validation": {
                            "ground_energy": -1.1400734808760409,
                            "pinn_final_energy_status": "performed",
                            "pinn_final_energy": -1.12,
                        },
                    }
                ),
                encoding="utf-8",
            )

            payload = physical_comparison_payload_for_images_dir(images_dir)

            self.assertEqual(
                payload["results"]["learned_sparse_agp"]["final_energy"],
                -1.12,
            )

    def test_aggregate_hcd_uses_checkpoint_named_by_mps_summary(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            round_dir = run_dir / "rounds" / "round_20"
            round_coefficients = round_dir / "Models_Data" / "final_agp_coefficients.pt"
            adaptive_coefficients = run_dir / "adaptive_temporal_refinement" / "Models_Data" / "final_agp_coefficients.pt"
            summary_path = round_dir / "mps_validation" / "Models_Data" / "mps_physical_validation_summary.json"
            for path in (round_coefficients, adaptive_coefficients):
                path.parent.mkdir(parents=True)
                path.write_bytes(b"checkpoint")
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                __import__("json").dumps({"trained_run": str(round_dir.resolve())}),
                encoding="utf-8",
            )

            selected = find_coefficient_export(run_dir)

            self.assertEqual(selected.resolve(), round_coefficients.resolve())

    def test_plot_payload_discovers_nested_mps_physical_summary(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            images_dir = run_dir / "Images"
            summary_path = run_dir / "mps_validation" / "Models_Data" / "mps_physical_validation_summary.json"
            images_dir.mkdir(parents=True)
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                __import__("json").dumps(
                    {
                        "n_qubits": 156,
                        "ground_energy": -209.6,
                        "results": {
                            "learned_sparse_agp": {
                                "final_energy": -188.15,
                                "ground_state_fidelity": 0.0207,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            payload = physical_comparison_payload_for_images_dir(images_dir)

            self.assertEqual(payload["results"]["learned_sparse_agp"]["final_energy"], -188.15)

    def test_plot_payload_discovers_hydrogen_physical_summary(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            images_dir = run_dir / "Images"
            summary_path = run_dir / "Models_Data" / "hydrogen_physical_validation_summary.json"
            images_dir.mkdir(parents=True)
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                __import__("json").dumps(
                    {
                        "n_qubits": 20,
                        "ground_energy": -1.1400734808760409,
                        "results": {
                            "learned_sparse_agp": {
                                "final_energy": -1.12,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            payload = physical_comparison_payload_for_images_dir(images_dir)

            self.assertEqual(payload["results"]["learned_sparse_agp"]["final_energy"], -1.12)

    def test_checkpoint_local_plot_rejects_sibling_mps_summary(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            trained_run = run_dir / "rounds" / "round_20"
            images_dir = run_dir / "adaptive_temporal_refinement" / "Images"
            summary_path = trained_run / "mps_validation" / "Models_Data" / "mps_physical_validation_summary.json"
            images_dir.mkdir(parents=True)
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                __import__("json").dumps(
                    {
                        "trained_run": str(trained_run.resolve()),
                        "ground_energy": -209.6,
                        "results": {"learned_sparse_agp": {"final_energy": -188.15}},
                    }
                ),
                encoding="utf-8",
            )

            payload = physical_comparison_payload_for_images_dir(images_dir)

            self.assertEqual(payload["results"], {})


    def test_physical_comparison_rows_include_exact_nested_and_pinn(self):
        payload = {
            "n_qubits": 15,
            "ground_energy": -20.0,
            "results": {
                "no_cd": {
                    "final_energy": -10.0,
                    "ground_state_fidelity": 0.05,
                },
                "kipu_dqfm_l1": {
                    "final_energy": -18.5,
                    "ground_state_fidelity": 0.75,
                },
                "learned_sparse_agp": {
                    "final_energy": -19.8,
                    "ground_state_fidelity": 0.95,
                },
            },
        }

        rows = physical_comparison_rows(payload)

        self.assertEqual(
            [row["method"] for row in rows],
            ["Exact ground state", "No counterdiabatic term", "Nested commutator l=1", "PINN sparse AGP"],
        )
        self.assertEqual(rows[0]["final_energy"], -20.0)
        self.assertEqual(rows[0]["energy_error"], 0.0)
        self.assertEqual(rows[0]["ground_state_fidelity"], 1.0)
        self.assertAlmostEqual(rows[1]["energy_error"], 10.0)
        self.assertAlmostEqual(rows[2]["energy_error"], 1.5)
        self.assertAlmostEqual(rows[3]["energy_error"], 0.2)

    def test_physical_comparison_rows_accept_canonical_nested_l1_key(self):
        rows = physical_comparison_rows(
            {
                "ground_energy": -2.0,
                "results": {
                    "nested_l1": {
                        "final_energy": -1.5,
                        "ground_state_fidelity": 0.75,
                    }
                },
            }
        )

        self.assertEqual(rows[2]["method"], "Nested commutator l=1")
        self.assertEqual(rows[2]["final_energy"], -1.5)
        self.assertEqual(rows[2]["ground_state_fidelity"], 0.75)

    def test_unconverged_mps_table_is_labeled_as_diagnostic(self):
        note = physical_validation_note(
            {
                "backend": "quimb_mps",
                "convergence": {
                    "status": "not_tested",
                    "reason": "Only one resolution was run.",
                },
                "certification": {"status": "not_tested"},
            }
        )

        self.assertIn("MPS convergence: not tested", note)
        self.assertIn("diagnostic only", note)

    def test_tenpy_table_note_reports_backend_support_and_uncertified_status(self):
        note = physical_validation_note(
            {
                "backend": "tenpy_tdvp_mpo",
                "full_learned_terms": 32768,
                "convergence": {"status": "not_feasible"},
                "certification": {"status": "not_tested"},
            }
        )

        self.assertIn("tenpy_tdvp_mpo", note)
        self.assertIn("32768", note)
        self.assertIn("not feasible", note)
        self.assertIn("diagnostic only", note)

    def test_table_hides_partial_mpo_final_time_metrics_and_labels_ablation(self):
        payload = {
            "backend": "tenpy_tdvp_mpo",
            "convergence": {"status": "not_tested"},
            "certification": {"status": "not_tested"},
            "resolution_results": [{"ablation": True, "learned_terms": 4, "full_learned_terms": 8}],
            "ground_energy": -1.0,
            "results": {
                "no_cd": {
                    "final_energy": -0.9,
                    "ground_state_fidelity": 0.8,
                    "mps_diagnostics": {"status": "not_feasible", "completed_steps": 1, "steps": 4},
                }
            },
        }

        rows = physical_comparison_rows(payload)
        note = physical_validation_note(payload)

        self.assertIsNone(rows[1]["final_energy"])
        self.assertIsNone(rows[1]["ground_state_fidelity"])
        self.assertIn("partial/not feasible", note)
        self.assertIn("ablation", note)
        self.assertIn("4/8", note)

    def test_legacy_metrics_render_without_completed_step_diagnostics(self):
        rows = physical_comparison_rows(
            {
                "backend": "quimb_product_formula",
                "ground_energy": -1.0,
                "results": {
                    "no_cd": {
                        "final_energy": -0.9,
                        "ground_state_fidelity": 0.8,
                        "mps_diagnostics": {"status": "ok"},
                    }
                },
            }
        )

        self.assertEqual(rows[1]["final_energy"], -0.9)
        self.assertEqual(rows[1]["ground_state_fidelity"], 0.8)

    def test_certified_mpo_note_reports_ladder_convergence_not_protocol_status(self):
        note = physical_validation_note(
            {
                "backend": "tenpy_tdvp_mpo",
                "full_learned_terms": 8,
                "convergence": {"status": "pass"},
                "certification": {"status": "pass"},
                "results": {
                    "no_cd": {
                        "mps_diagnostics": {"status": "unresolved_error", "completed_steps": 0, "steps": 4}
                    }
                },
            }
        )

        self.assertIn("convergence: pass", note)
        self.assertIn("certification: pass", note)

    def test_q156_comparison_table_marks_dynamical_metrics_unavailable(self):
        with TemporaryDirectory() as tmp:
            study_dir = Path(tmp) / "sweep_test"
            run_dir = study_dir / "runs" / "feedback" / "agp_32768"
            images_dir = run_dir / "Images"
            data_dir = run_dir / "Models_Data"
            images_dir.mkdir(parents=True)
            data_dir.mkdir(parents=True)
            config = {
                "physical": {"parameters": {"system": "TransverseIsingDriverProblem", "num_qubits": 156}},
                "scalable_validation": {
                    "ground_energy": -209.6,
                    "statevector_validation": "not_tested",
                    "reason": "A full 2**156 statevector cannot be represented.",
                },
            }
            (study_dir / "config.json").write_text(__import__("json").dumps(config), encoding="utf-8")
            derived_config = {
                "physical": {
                    "system": "TransverseIsingDriverProblem",
                    "n_qubits": 156,
                }
            }
            (data_dir / "config.json").write_text(
                __import__("json").dumps(derived_config),
                encoding="utf-8",
            )

            payload = physical_comparison_payload_for_images_dir(images_dir)
            output = plot_physical_comparison_table(images_dir)
            rows = physical_comparison_rows(payload)

            self.assertEqual(payload["ground_energy"], -209.6)
            self.assertIsNone(rows[1]["final_energy"])
            self.assertIsNone(rows[2]["ground_state_fidelity"])
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)

    def test_physical_summary_search_tolerates_vanished_directories(self):
        calls = 0

        def flaky_glob(_path, _pattern):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise FileNotFoundError("directory disappeared during traversal")
            return iter(())

        with patch.object(Path, "glob", flaky_glob):
            summary = find_physical_summary_for_images_dir(Path("/tmp/run/Images"))

        self.assertIsNone(summary)

    def test_pauli_action_matches_single_qubit_matrices(self):
        ket_zero = np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)

        for label, expected in (
            ("X", np.asarray([0.0 + 0.0j, 1.0 + 0.0j])),
            ("Y", np.asarray([0.0 + 0.0j, 0.0 + 1.0j])),
            ("Z", np.asarray([1.0 + 0.0j, 0.0 + 0.0j])),
        ):
            actions = build_action_cache([label])
            observed = apply_pauli_sum(ket_zero, {label: 1.0}, actions)
            np.testing.assert_allclose(observed, expected)

    def test_variational_l1_agp_matches_two_level_direction(self):
        h0 = SparsePauliOperator({"Z": 1.0}, n_qubits=1)
        h1 = SparsePauliOperator({"X": 1.0}, n_qubits=1)

        agp = variational_l1_agp(h0, h1, 0.5)

        self.assertAlmostEqual(agp.coefficient("Y").real, 1.0, places=12)
        self.assertAlmostEqual(agp.coefficient("Y").imag, 0.0, places=12)

    def test_krylov_support_is_bounded_and_nonempty(self):
        h0, h1 = transverse_field_ising_problem(5, field=1.0, coupling=1.0)

        labels, metadata = select_krylov_agp_labels(
            h0,
            h1,
            KrylovSupportConfig(target_terms=100, max_depth=2, max_frontier=16),
        )

        self.assertEqual(len(labels), 100)
        self.assertEqual(metadata["agp_support_strategy"], "nested_commutator_krylov_pool")
        self.assertGreater(metadata["locality_completion_terms"], 0)
        self.assertNotIn("IIIII", labels)

    def test_feedback_budget_auto_reduces_additions_to_preserve_all_rounds_and_unseen_batch(self):
        budget = {
            "mode": "auto",
            "resolved_holdout_residual_top_k": 13312,
            "initial_residual_terms": 2048,
            "feedback_iterations": 10,
            "add_residual_terms_per_iteration": 1024,
            "unseen_batches_after_final_iteration": 1,
            "minimum_budget_before_final_unseen_exhaustion": 12288,
            "final_round_expected_unseen_terms": 1024,
        }

        residual_top_k, add_terms, fitted = fit_residual_budget_to_available(
            residual_top_k=13312,
            add_residual_terms=1024,
            residual_budget=budget,
            available_residual_terms=6737,
            initial_residual_terms=2048,
            rounds=10,
            unseen_batches_after_final_iteration=1,
        )

        self.assertEqual(residual_top_k, 6737)
        self.assertEqual(add_terms, 426)
        self.assertEqual(fitted["effective_add_residual_terms_per_iteration"], 426)
        self.assertEqual(fitted["final_round_expected_unseen_terms"], 429)
        self.assertEqual(fitted["residual_budget_fit_status"], "auto_reduced_additions_to_preserve_rounds")

    def test_relative_metric_marks_zero_reference_as_invalid(self):
        value, status = relative_metric_with_reference_status(
            residual=42.0,
            reference=0.0,
            eps=1e-7,
        )

        self.assertIsNone(value)
        self.assertFalse(status["valid"])
        self.assertEqual(status["reason"], "zero_reference")

    def test_final_run_prefers_temporal_refinement_recorded_in_feedback_summary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_run_dir = agp_physical_validation.RUN_DIR
            agp_physical_validation.RUN_DIR = root
            try:
                output_dir = (
                    root
                    / "runs"
                    / "refined"
                    / "agp_8_residual_16_add_2_rounds_3"
                )
                refined = output_dir / "temporal_refinement"
                refined.mkdir(parents=True)
                data_dir = output_dir / "Models_Data"
                data_dir.mkdir(parents=True)
                summary = {
                    "temporal_refinement": {
                        "enabled": True,
                        "run_dir": "temporal_refinement",
                    }
                }
                (data_dir / "holdout_feedback_summary_residual_16.json").write_text(
                    __import__("json").dumps(summary),
                    encoding="utf-8",
                )
                config = {
                    "support_sweep": {"residual_top_k": 10},
                    "holdout_feedback": {
                        "base_agp_terms": 8,
                        "iterations": 3,
                        "add_residual_terms_per_iteration": 2,
                        "holdout_residual_top_k": 16,
                        "output_root": "runs/refined",
                    },
                }

                self.assertEqual(final_run_from_summary(config), refined)
            finally:
                agp_physical_validation.RUN_DIR = old_run_dir

    def test_final_run_prefers_adaptive_temporal_refinement_over_temporal_refinement(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_run_dir = agp_physical_validation.RUN_DIR
            agp_physical_validation.RUN_DIR = root
            try:
                output_dir = (
                    root
                    / "runs"
                    / "refined"
                    / "agp_8_residual_16_add_2_rounds_3"
                )
                temporal = output_dir / "temporal_refinement"
                adaptive = output_dir / "adaptive_temporal_refinement"
                temporal.mkdir(parents=True)
                adaptive.mkdir(parents=True)
                data_dir = output_dir / "Models_Data"
                data_dir.mkdir(parents=True)
                summary = {
                    "temporal_refinement": {
                        "enabled": True,
                        "run_dir": "temporal_refinement",
                    },
                    "adaptive_temporal_refinement": {
                        "enabled": True,
                        "run_dir": "adaptive_temporal_refinement",
                    },
                }
                (data_dir / "holdout_feedback_summary_residual_16.json").write_text(
                    __import__("json").dumps(summary),
                    encoding="utf-8",
                )
                config = {
                    "support_sweep": {"residual_top_k": 10},
                    "holdout_feedback": {
                        "base_agp_terms": 8,
                        "iterations": 3,
                        "add_residual_terms_per_iteration": 2,
                        "holdout_residual_top_k": 16,
                        "output_root": "runs/refined",
                    },
                }

                self.assertEqual(final_run_from_summary(config), adaptive)
            finally:
                agp_physical_validation.RUN_DIR = old_run_dir

    def test_final_run_can_select_residual_champion_when_configured(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_run_dir = agp_physical_validation.RUN_DIR
            agp_physical_validation.RUN_DIR = root
            try:
                output_dir = (
                    root
                    / "runs"
                    / "refined"
                    / "agp_8_residual_16_add_2_rounds_3"
                )
                baseline = root / "runs" / "baselines" / "agp_8"
                temporal = output_dir / "temporal_refinement"
                adaptive = output_dir / "adaptive_temporal_refinement"
                for run_dir in (baseline, temporal, adaptive):
                    data_dir = run_dir / "Models_Data"
                    data_dir.mkdir(parents=True)
                    (data_dir / "final_agp_coefficients.pt").write_bytes(b"placeholder")
                data_dir = output_dir / "Models_Data"
                data_dir.mkdir(parents=True)
                summary = {
                    "rows": [
                        {
                            "run_dir": str(baseline),
                            "holdout_relative_residual": 0.20,
                        }
                    ],
                    "temporal_refinement": {
                        "enabled": True,
                        "run_dir": "temporal_refinement",
                        "holdout_relative_residual": 0.03,
                    },
                    "adaptive_temporal_refinement": {
                        "enabled": True,
                        "run_dir": "adaptive_temporal_refinement",
                        "holdout_relative_residual": 0.07,
                    },
                }
                (data_dir / "holdout_feedback_summary_residual_16.json").write_text(
                    __import__("json").dumps(summary),
                    encoding="utf-8",
                )
                config = {
                    "physical_validation": {"trained_run_selection": "best_holdout_residual"},
                    "support_sweep": {"residual_top_k": 10},
                    "holdout_feedback": {
                        "base_agp_terms": 8,
                        "iterations": 3,
                        "add_residual_terms_per_iteration": 2,
                        "holdout_residual_top_k": 16,
                        "output_root": "runs/refined",
                    },
                }

                self.assertEqual(final_run_from_summary(config), temporal)
            finally:
                agp_physical_validation.RUN_DIR = old_run_dir

    def test_learned_variant_specs_expand_term_and_scale_sweeps(self):
        specs = build_learned_variant_specs(
            {
                "learned_top_terms": 1024,
                "learned_top_terms_sweep": [512, 1024],
                "learned_scale_sweep": [0.75, 1.0],
            },
            max_terms_override=None,
            term_sweep_override=None,
            scale_sweep_override=None,
        )

        self.assertEqual(
            [(spec.name, spec.max_terms, spec.scale, spec.is_default) for spec in specs],
            [
                ("learned_sparse_agp_terms_512_scale_0p75", 512, 0.75, False),
                ("learned_sparse_agp_terms_512_scale_1", 512, 1.0, False),
                ("learned_sparse_agp", 1024, 1.0, True),
                ("learned_sparse_agp_terms_1024_scale_0p75", 1024, 0.75, False),
            ],
        )

    def test_best_learned_variant_uses_energy_then_fidelity(self):
        results = {
            "learned_sparse_agp_terms_512_scale_1": {
                "energy_error": 3.0,
                "ground_state_fidelity": 0.7,
                "excitation_probability": 0.3,
                "learned_terms": 512,
                "learned_scale": 1.0,
                "retained_rms_norm_fraction": 0.8,
            },
            "learned_sparse_agp_terms_1024_scale_1": {
                "energy_error": 2.0,
                "ground_state_fidelity": 0.5,
                "excitation_probability": 0.5,
                "learned_terms": 1024,
                "learned_scale": 1.0,
                "retained_rms_norm_fraction": 0.9,
            },
            "learned_sparse_agp_terms_2048_scale_1": {
                "energy_error": 2.0,
                "ground_state_fidelity": 0.8,
                "excitation_probability": 0.2,
                "learned_terms": 2048,
                "learned_scale": 1.0,
                "retained_rms_norm_fraction": 0.95,
            },
        }

        best = select_best_learned_variant(results, metric="energy_error")

        self.assertEqual(best["name"], "learned_sparse_agp_terms_2048_scale_1")
        self.assertEqual(best["selection_metric"], "energy_error")

    def test_statevector_cli_runs_from_repo_root_without_pythonpath(self):
        """Exercise the CLI import and its post-summary plotting refresh."""

        with TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            hamiltonian_path = temp_root / "tiny_pair.json"
            hamiltonian_path.write_text(
                json.dumps(
                    {
                        "format": "pauli_hamiltonian_pair_v1",
                        "system": "CliImportTest",
                        "n_qubits": 1,
                        "distance": "1_0",
                        "hamiltonians": {
                            "initial": {"terms": {"X": [1.0, 0.0]}},
                            "final": {"terms": {"Z": [1.0, 0.0]}},
                        },
                    }
                ),
                encoding="utf-8",
            )
            trained_run = temp_root / "trained_run"
            coefficient_path = trained_run / "Models_Data" / "final_agp_coefficients.pt"
            coefficient_path.parent.mkdir(parents=True)
            torch.save(
                {
                    "pauli_labels": ["Y"],
                    "counterdiabatic_coefficients": torch.tensor([[0.1], [0.1]]),
                    "tau": torch.tensor([0.0, 1.0]),
                    "lambda": torch.tensor([0.0, 1.0]),
                    "d_lambda_dt": torch.tensor([1.0, 1.0]),
                },
                coefficient_path,
            )
            config_path = temp_root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "physical": {
                            "parameters": {
                                "system": "CliImportTest",
                                "num_qubits": 1,
                                "distance": "1_0",
                                "T": 0.01,
                                "hamiltonian_source": str(hamiltonian_path),
                            }
                        },
                        "physical_validation": {
                            "evolution_steps": 2,
                            "learned_top_terms": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            output_dir = temp_root / "validation"
            environment = dict(os.environ)
            environment.pop("PYTHONPATH", None)
            result = subprocess.run(
                [
                    sys.executable,
                    str(FRAMEWORK_SCRIPTS_DIR / "agp_physical_validation.py"),
                    "--config",
                    str(config_path),
                    "--trained-run",
                    str(trained_run),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertIn('"validation_identity"', result.stdout)
            payload = json.loads(
                (output_dir / "Models_Data" / "physical_validation_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["validation_identity"]["n_qubits"], 1)
            self.assertTrue((output_dir / "Images" / "hcd_connection_summary.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
