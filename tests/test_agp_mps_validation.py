import json
import sys
import unittest
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_SCRIPTS_DIR = ROOT / "tests" / "sparse_agp_curriculum" / "scripts"
if str(FRAMEWORK_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_SCRIPTS_DIR))

import agp_mps_validation as validation_module
from agp_mps_validation import (
    apply_grouped_hamiltonian_rotation_mps,
    assess_mps_convergence,
    assess_independent_mpo_convergence,
    assess_statevector_agreement,
    apply_pauli_rotation_mps,
    diagonal_ising_mps_metrics,
    evolve_protocol_mps,
    make_product_mps,
    pauli_rotation_matrix,
    run_mps_case,
    run_mpo_case,
    assess_mpo_compression,
    statevector_results_for_learned_terms,
    validation_certification,
    cached_protocol_result,
    resolve_validation_backend,
    require_full_learned_support,
    group_hamiltonian_terms_by_support,
)
from utils import SparsePauliOperator


class AGPMPSValidationTests(unittest.TestCase):
    def test_backend_accepts_direct_time_full_support_representation(self):
        backend = resolve_validation_backend(
            {
                "mpo_backend": {
                    "name": "tenpy_tdvp_mpo",
                    "representation": "direct_time_full_support",
                }
            }
        )

        self.assertEqual(backend["representation"], "direct_time_full_support")

    def test_backend_accepts_joint_time_full_support_representation(self):
        backend = resolve_validation_backend(
            {
                "mpo_backend": {
                    "name": "tenpy_tdvp_mpo",
                    "representation": "joint_time_full_support",
                    "coefficient_error_max": 2.0e-4,
                    "time_window_size": 3,
                    "adaptive_time_windows": True,
                    "time_axis_position": 7,
                }
            }
        )

        self.assertEqual(backend["representation"], "joint_time_full_support")
        self.assertEqual(backend["coefficient_error_max"], 2.0e-4)
        self.assertEqual(backend["time_window_size"], 3)
        self.assertTrue(backend["adaptive_time_windows"])
        self.assertEqual(backend["time_axis_position"], 7)

    def test_backend_rejects_unknown_operator_representation(self):
        with self.assertRaisesRegex(ValueError, "representation"):
            resolve_validation_backend(
                {"mpo_backend": {"name": "tenpy_tdvp_mpo", "representation": "magic"}}
            )

    def test_preflight_cli_selects_only_named_preflight_cases(self):
        configured = [
            {"name": "speed_preflight", "preflight_only": True, "steps": 1},
            {"name": "coarse", "steps": 24},
        ]

        selected = validation_module.select_validation_cases(
            configured,
            preflight_only=True,
        )

        self.assertEqual(selected, [configured[0]])

    def test_preflight_cli_rejects_config_without_preflight_case(self):
        with self.assertRaisesRegex(ValueError, "preflight_only"):
            validation_module.select_validation_cases(
                [{"name": "coarse", "steps": 24}],
                preflight_only=True,
            )

    def test_normal_validation_excludes_preflight_cases(self):
        configured = [
            {"name": "speed_preflight", "preflight_only": True, "steps": 1},
            {"name": "coarse", "steps": 24},
        ]

        selected = validation_module.select_validation_cases(
            configured,
            preflight_only=False,
        )

        self.assertEqual(selected, [configured[1]])

    def test_preflight_marker_must_be_a_boolean(self):
        with self.assertRaisesRegex(ValueError, "must be a Boolean"):
            validation_module.select_validation_cases(
                [{"name": "speed_preflight", "preflight_only": "false"}],
                preflight_only=False,
            )

    def test_preflight_keeps_selected_case_when_cli_overrides_are_present(self):
        selected = [{"name": "speed_preflight", "preflight_only": True, "mpo_max_bond": 64}]

        resolved = validation_module.apply_case_override_mode(
            selected,
            preflight_only=True,
            override_requested=True,
        )

        self.assertEqual(resolved, selected)

    def test_preflight_uses_an_isolated_output_namespace(self):
        base = Path("mpo_validation")

        self.assertEqual(
            validation_module.execution_output_dir(base, preflight_only=True),
            base / "preflight",
        )
        self.assertEqual(
            validation_module.execution_output_dir(base, preflight_only=False),
            base,
        )

    def test_explicit_cli_ablation_marks_a_reduced_support_case(self):
        self.assertTrue(
            validation_module.resolve_case_ablation(
                cli_ablation=True,
                case_ablation=False,
                backend_ablation=False,
            )
        )
        self.assertFalse(
            validation_module.resolve_case_ablation(
                cli_ablation=False,
                case_ablation=False,
                backend_ablation=False,
            )
        )

    def test_statevector_calibration_uses_final_ablation_results_without_a_gate_resolution(self):
        final_results = {
            "no_cd": {"final_energy": -1.0},
            "kipu_dqfm_l1": {"final_energy": -2.0},
            "learned_sparse_agp": {"final_energy": -3.0},
        }

        selected = validation_module.statevector_comparison_results(
            gate_resolution=None,
            final_results=final_results,
        )

        self.assertEqual(
            set(selected),
            {"no_cd", "nested_l1", "learned_sparse_agp"},
        )
        self.assertEqual(selected["nested_l1"]["final_energy"], -2.0)

    def test_cli_accepts_independent_mpo_and_mps_numerical_overrides(self):
        with patch(
            "sys.argv",
            [
                "agp_mps_validation.py",
                "--config",
                "config.json",
                "--mpo-max-bond",
                "512",
                "--mps-max-bond",
                "128",
                "--mpo-cutoff",
                "0",
                "--mps-cutoff",
                "1e-12",
            ],
        ):
            args = validation_module.parse_args()

        self.assertEqual(args.mpo_max_bond, 512)
        self.assertEqual(args.mps_max_bond, 128)
        self.assertEqual(args.mpo_cutoff, 0.0)
        self.assertEqual(args.mps_cutoff, 1.0e-12)

    def test_failed_preflight_exports_canonical_not_tested_status(self):
        payload = {
            "backend": "tenpy_tdvp_mpo",
            "n_qubits": 24,
            "trained_run": "/tmp/run",
            "coefficient_path": "/tmp/run/final.pt",
            "ground_energy": -28.1,
            "ground_bitstring": "0" * 24,
            "protocols": ["no_cd", "kipu_dqfm_l1", "learned_sparse_agp"],
            "full_learned_terms": 32768,
            "results": {
                "learned_sparse_agp": {
                    "mps_diagnostics": {
                        "mpo_action_error": 0.941,
                        "mpo_action_status": "not_feasible",
                    }
                }
            },
        }

        status = validation_module.preflight_gate_status_payload(
            payload,
            action_error_max=1.0e-3,
            preflight_summary_path=Path("preflight/Models_Data/summary.json"),
        )

        self.assertEqual(status["execution_mode"], "validation_status")
        self.assertEqual(status["certification"]["status"], "not_tested")
        self.assertEqual(status["preflight_gate"]["status"], "fail")
        self.assertEqual(status["results"], {})
        self.assertIn("0.941", status["availability_note"])

    def test_successful_preflight_reports_pass_but_remains_uncertified(self):
        payload = {
            "results": {
                "learned_sparse_agp": {
                    "mps_diagnostics": {
                        "mpo_action_error": 1.0e-6,
                        "mpo_action_status": "measured",
                    }
                }
            }
        }

        status = validation_module.preflight_gate_status_payload(
            payload,
            action_error_max=1.0e-3,
            preflight_summary_path=Path("preflight/summary.json"),
        )

        self.assertEqual(status["preflight_gate"]["status"], "pass")
        self.assertEqual(status["certification"]["status"], "not_tested")

    def test_preflight_status_does_not_overwrite_canonical_validation(self):
        with TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "mps_physical_validation_summary.json"
            summary_path.write_text(
                json.dumps({"execution_mode": "validation", "certification": {"status": "pass"}}),
                encoding="utf-8",
            )

            self.assertFalse(validation_module.should_publish_preflight_status(summary_path))

            summary_path.write_text(
                json.dumps({"execution_mode": "validation_status"}),
                encoding="utf-8",
            )
            self.assertTrue(validation_module.should_publish_preflight_status(summary_path))

    def test_cached_payload_requires_matching_execution_mode(self):
        payload = {
            "n_qubits": 24,
            "coefficient_path": "/tmp/final.pt",
            "execution_mode": "preflight_only",
        }

        self.assertFalse(
            validation_module.previous_payload_matches_execution(
                payload,
                n_qubits=24,
                coefficient_path=Path("/tmp/final.pt"),
                execution_mode="validation",
            )
        )
        self.assertTrue(
            validation_module.previous_payload_matches_execution(
                payload,
                n_qubits=24,
                coefficient_path=Path("/tmp/final.pt"),
                execution_mode="preflight_only",
            )
        )
        self.assertFalse(
            validation_module.previous_payload_matches_execution(
                {key: value for key, value in payload.items() if key != "execution_mode"},
                n_qubits=24,
                coefficient_path=Path("/tmp/final.pt"),
                execution_mode="validation",
            )
        )

    def test_parse_args_accepts_preflight_only(self):
        argv = ["agp_mps_validation.py", "--config", "config.json", "--preflight-only"]
        with patch.object(sys, "argv", argv):
            args = validation_module.parse_args()

        self.assertTrue(args.preflight_only)

    def test_group_hamiltonian_terms_preserves_every_pauli_coefficient(self):
        grouped = group_hamiltonian_terms_by_support(
            [("XI", 0.25), ("YI", -0.5), ("XZ", 0.75)]
        )

        x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
        z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        np.testing.assert_allclose(grouped[(0,)], 0.25 * x - 0.5 * y, atol=1.0e-12)
        np.testing.assert_allclose(grouped[(0, 1)], 0.75 * np.kron(x, z), atol=1.0e-12)

    def test_grouped_hamiltonian_rotation_matches_dense_exponential(self):
        state = make_product_mps("00")
        terms = [("XI", 0.25), ("YI", -0.5), ("XZ", 0.75)]
        angle = 0.31

        diagnostics = apply_grouped_hamiltonian_rotation_mps(
            state,
            terms,
            angle,
            cutoff=1.0e-14,
            max_bond=16,
        )

        x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
        z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        identity = np.eye(2, dtype=np.complex128)
        local_hamiltonian = 0.25 * x - 0.5 * y
        eigenvalues, eigenvectors = np.linalg.eigh(local_hamiltonian)
        local_gate = (eigenvectors * np.exp(-1.0j * angle * eigenvalues)) @ eigenvectors.conj().T
        nonlocal_gate = np.cos(0.75 * angle) * np.eye(4) - 1.0j * np.sin(0.75 * angle) * np.kron(x, z)
        expected = nonlocal_gate @ np.kron(local_gate, identity) @ np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.complex128
        )

        np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1), expected, atol=1.0e-11)
        self.assertEqual(diagnostics["pauli_terms"], 3)
        self.assertEqual(diagnostics["support_groups"], 2)

    def test_singleton_support_group_uses_exact_pauli_rotation_without_eigh(self):
        state = make_product_mps("000")
        angle = 0.31
        coefficient = 0.75

        with patch("agp_mps_validation.np.linalg.eigh", side_effect=AssertionError("unexpected eigh")):
            diagnostics = apply_grouped_hamiltonian_rotation_mps(
                state,
                [("XIZ", coefficient)],
                angle,
                cutoff=1.0e-14,
                max_bond=16,
            )

        expected = pauli_rotation_matrix("XIZ", angle * coefficient) @ np.array(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.complex128
        )
        np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1), expected, atol=1.0e-11)
        self.assertEqual(diagnostics, {"pauli_terms": 1, "support_groups": 1})

    def test_protocol_cache_requires_identical_resolution_settings(self):
        previous = [
            {
                "settings": {
                    "steps": 96,
                    "max_bond": 128,
                    "learned_terms": 1024,
                    "cutoff": 1.0e-12,
                    "coefficient_threshold": 1.0e-10,
                    "operator_grouping": "support",
                },
                "results": {"learned_sparse_agp": {"final_energy": -18.99}},
            }
        ]
        settings = dict(previous[0]["settings"])

        cached = cached_protocol_result(previous, settings=settings, protocol="learned_sparse_agp")

        self.assertEqual(cached["final_energy"], -18.99)
        self.assertIsNone(cached_protocol_result(previous, settings={**settings, "steps": 48}, protocol="learned_sparse_agp"))
        self.assertIsNone(
            cached_protocol_result(
                previous,
                settings={**settings, "operator_grouping": "pauli_term"},
                protocol="learned_sparse_agp",
            )
        )

    def test_joint_time_cache_migrates_operator_gate_to_action_diagnostics(self):
        settings = {
            "operator_representation": "joint_time_full_support",
            "steps": 1,
        }
        previous = [
            {
                "settings": settings,
                "results": {
                    "learned_sparse_agp": {
                        "final_energy": -1.0,
                        "mps_diagnostics": {
                            "representation": "joint_time_full_support",
                            "operator_certificate": {
                                "status": "pass",
                                "max_relative_action_error_upper_bound": 5.0e-3,
                            },
                            "mpo_action_status": "not_tested",
                        },
                    }
                },
            }
        ]

        cached = cached_protocol_result(
            previous,
            settings=settings,
            protocol="learned_sparse_agp",
        )

        self.assertEqual(cached["mps_diagnostics"]["mpo_action_status"], "pass")
        self.assertEqual(cached["mps_diagnostics"]["mpo_action_error"], 5.0e-3)
        self.assertEqual(
            cached["mps_diagnostics"]["mpo_action_diagnostics"]["status"], "pass"
        )

    def test_certification_requires_only_configured_validation_gates(self):
        certification = validation_certification(
            convergence={"status": "not_tested"},
            statevector_agreement={"status": "pass"},
            require_convergence=False,
            require_statevector=True,
        )

        self.assertEqual(certification["status"], "pass")
        self.assertEqual(certification["required_gates"], ["statevector_agreement"])

    def test_mpo_cache_key_includes_every_numerical_axis_and_checkpoint_identity(self):
        settings = {
            "backend": "tenpy_tdvp_mpo",
            "integrator": "tdvp",
            "steps": 24,
            "temporal_grid_points": 257,
            "temporal_retained_norm": 0.9999,
            "mpo_max_bond": 64,
            "mpo_cutoff": 1.0e-10,
            "mps_max_bond": 32,
            "mps_cutoff": 1.0e-9,
            "lanczos_max": 20,
            "mpo_workspace_cap_bytes": 1024,
            "resource_caps": {"max_build_seconds": 3600, "max_peak_memory_gb": 24},
            "qubit_order_candidates": ["native", "spectral"],
            "checkpoint_identity": {"path": "/tmp/final.pt", "size": 123, "mtime_ns": 456},
            "learned_scale": 1.0,
            "hamiltonian_identity": "h0h1-sha256",
            "ground_reference_identity": "ground-sha256",
            "ground_bitstring": "000000000000000",
            "schedule_identity": "learned-schedule-sha256",
            "learned_terms": 32768,
        }
        previous = [{"settings": dict(settings), "results": {"learned_sparse_agp": {"final_energy": -1.0}}}]

        self.assertIsNotNone(cached_protocol_result(previous, settings=settings, protocol="learned_sparse_agp"))
        for key, value in (
            ("temporal_retained_norm", 0.999),
            ("mpo_max_bond", 128),
            ("mps_cutoff", 1.0e-8),
            ("steps", 48),
            ("integrator", "expm_mpo"),
            ("checkpoint_identity", {"path": "/tmp/final.pt", "size": 124, "mtime_ns": 456}),
            ("learned_scale", 0.5),
            ("hamiltonian_identity", "different-h0h1-sha256"),
            ("ground_reference_identity", "different-ground-sha256"),
            ("ground_bitstring", "100000000000000"),
            ("schedule_identity", "different-schedule-sha256"),
        ):
            with self.subTest(key=key):
                self.assertIsNone(
                    cached_protocol_result(
                        previous,
                        settings={**settings, key: value},
                        protocol="learned_sparse_agp",
                    )
                )

    def test_certification_requires_temporal_mpo_mps_and_timestep_gates(self):
        certification = validation_certification(
            convergence={"status": "pass"},
            compression={"status": "pass"},
            statevector_agreement={"status": "not_tested"},
            require_convergence=True,
            require_compression=True,
            state_convergence={"status": "pass"},
            require_state_convergence=True,
            require_statevector=False,
        )

        self.assertEqual(certification["status"], "pass")
        self.assertIn("mpo_compression", certification["required_gates"])
        self.assertIn("state_convergence", certification["required_gates"])

    def test_independent_mpo_convergence_uses_named_time_and_state_pairs(self):
        def case(name, *, steps, mps_bond, energy_shift=0.0):
            results = {
                protocol: {
                    "final_energy": -2.0 + energy_shift,
                    "ground_state_fidelity": 0.4 + energy_shift,
                }
                for protocol in ("no_cd", "nested_l1", "learned_sparse_agp")
            }
            return {
                "name": name,
                "settings": {
                    "steps": steps,
                    "timestep": 1.0 / steps,
                    "mps_max_bond": mps_bond,
                    "mps_cutoff": 1.0e-10,
                    "mpo_max_bond": 64,
                    "mpo_cutoff": 1.0e-12,
                    "time_window_size": 1,
                    "time_axis_position": 2,
                },
                "results": results,
            }

        assessment = assess_independent_mpo_convergence(
            [
                case("time_coarse", steps=8, mps_bond=64, energy_shift=0.01),
                case("state_coarse", steps=16, mps_bond=32, energy_shift=0.005),
                case("fine", steps=16, mps_bond=64),
            ],
            convergence_pairs={
                "timestep": ("time_coarse", "fine"),
                "state": ("state_coarse", "fine"),
            },
            energy_atol=0.05,
            fidelity_atol=0.02,
        )

        self.assertEqual(assessment["status"], "pass", assessment)
        self.assertEqual(assessment["timestep"]["status"], "pass")
        self.assertEqual(assessment["state"]["status"], "pass")

    def test_independent_mpo_convergence_rejects_confounded_pair(self):
        common_results = {
            protocol: {"final_energy": -2.0, "ground_state_fidelity": 0.4}
            for protocol in ("no_cd", "nested_l1", "learned_sparse_agp")
        }
        resolutions = [
            {
                "name": "coarse",
                "settings": {
                    "steps": 8,
                    "timestep": 0.125,
                    "mps_max_bond": 32,
                    "mps_cutoff": 1.0e-9,
                    "mpo_max_bond": 64,
                    "mpo_cutoff": 1.0e-12,
                    "time_window_size": 1,
                    "time_axis_position": 2,
                },
                "results": common_results,
            },
            {
                "name": "fine",
                "settings": {
                    "steps": 16,
                    "timestep": 0.0625,
                    "mps_max_bond": 64,
                    "mps_cutoff": 1.0e-10,
                    "mpo_max_bond": 64,
                    "mpo_cutoff": 1.0e-12,
                    "time_window_size": 1,
                    "time_axis_position": 2,
                },
                "results": common_results,
            },
        ]

        assessment = assess_independent_mpo_convergence(
            resolutions,
            convergence_pairs={
                "timestep": ("coarse", "fine"),
                "state": ("coarse", "fine"),
            },
            energy_atol=0.05,
            fidelity_atol=0.02,
        )

        self.assertEqual(assessment["status"], "not_tested")
        self.assertEqual(assessment["timestep"]["status"], "not_comparable")
        self.assertEqual(assessment["state"]["status"], "not_comparable")

    def test_direct_full_support_compression_gate_does_not_require_temporal_modes(self):
        direct_diagnostics = {
            "status": "ok",
            "representation": "direct_time_full_support",
            "operator_gate_status": "pass",
            "mpo_action_status": "pass",
            "mpo_action_error": 0.0,
            "resource_statuses": {"dynamic_mpo_assembly": "ok"},
            "operator_certificates": [
                {
                    "source_completeness_status": "pass",
                    "operator_gate_status": "pass",
                    "max_relative_action_error_upper_bound": 0.0,
                }
            ],
        }
        baseline_diagnostics = {
            "status": "ok",
            "representation": "temporal_mode_block_sum",
            "temporal_rank": 1,
            "temporal_retained_norm": 1.0,
            "static_mpo_compression": {"h0": {"status": "ok"}},
            "resource_statuses": {"dynamic_mpo_assembly": "ok"},
            "mpo_action_status": "measured",
            "mpo_action_error": 0.0,
        }

        result = assess_mpo_compression(
            {
                "no_cd": {"mps_diagnostics": baseline_diagnostics},
                "nested_l1": {"mps_diagnostics": baseline_diagnostics},
                "learned_sparse_agp": {"mps_diagnostics": direct_diagnostics},
            },
            action_error_max=1.0e-3,
        )

        self.assertEqual(result["status"], "pass", result)
        self.assertEqual(result["protocols"]["learned_sparse_agp"]["temporal"], "not_applicable")
        self.assertEqual(result["protocols"]["learned_sparse_agp"]["source_completeness"], "pass")

    def test_joint_time_full_support_gate_uses_joint_operator_certificate(self):
        joint_diagnostics = {
            "status": "ok",
            "representation": "joint_time_full_support",
            "operator_gate_status": "pass",
            "source_completeness_status": "pass",
            "operator_certificate": {
                "status": "pass",
                "coefficient_status": "pass",
                "action_status": "pass",
                "max_relative_action_error_upper_bound": 4.0e-4,
            },
            "resource_statuses": {
                "joint_time_pauli_compression": "ok",
                "dynamic_mpo_slicing": "ok",
            },
        }
        baseline_diagnostics = {
            "status": "ok",
            "representation": "temporal_mode_block_sum",
            "temporal_rank": 1,
            "temporal_retained_norm": 1.0,
            "static_mpo_compression": {"h0": {"status": "ok"}},
            "resource_statuses": {"dynamic_mpo_assembly": "ok"},
            "mpo_action_status": "measured",
            "mpo_action_error": 0.0,
        }

        result = assess_mpo_compression(
            {
                "no_cd": {"mps_diagnostics": baseline_diagnostics},
                "nested_l1": {"mps_diagnostics": baseline_diagnostics},
                "learned_sparse_agp": {"mps_diagnostics": joint_diagnostics},
            },
            action_error_max=1.0e-3,
        )

        self.assertEqual(result["status"], "pass", result)
        learned = result["protocols"]["learned_sparse_agp"]
        self.assertEqual(learned["temporal"], "not_applicable")
        self.assertEqual(learned["dynamic_mpo"], "pass")
        self.assertEqual(learned["mpo_action"], "pass")

    def test_finite_action_upper_bound_can_certify_numerically_unresolved_probe(self):
        diagnostics = {
            "status": "ok",
            "representation": "temporal_mode_block_sum",
            "temporal_rank": 1,
            "temporal_retained_norm": 1.0,
            "static_mpo_compression": {"h0": {"status": "ok"}},
            "resource_statuses": {"dynamic_mpo_assembly": "ok"},
            "mpo_action_status": "numerically_unresolved",
            "mpo_action_error": 9.0e-5,
            "mpo_action_diagnostics": {
                "status": "numerically_unresolved",
                "max_relative_action_error_upper_bound": 9.0e-5,
            },
        }

        passed = assess_mpo_compression(
            {"no_cd": {"mps_diagnostics": diagnostics}},
            action_error_max=1.0e-3,
        )
        failed = assess_mpo_compression(
            {"no_cd": {"mps_diagnostics": diagnostics}},
            action_error_max=1.0e-5,
        )

        self.assertEqual(passed["protocols"]["no_cd"]["mpo_action"], "pass")
        self.assertEqual(failed["protocols"]["no_cd"]["mpo_action"], "fail")

    def test_certification_rejects_ablation_and_incomplete_mpo_resolution_ladders(self):
        common = {
            "convergence": {"status": "pass"},
            "compression": {"status": "pass"},
            "statevector_agreement": {"status": "pass"},
            "require_convergence": True,
            "require_compression": True,
            "require_statevector": True,
        }
        self.assertEqual(
            validation_certification(**common, ablation=True, completed_comparable_resolutions=2)["status"],
            "not_tested",
        )
        self.assertEqual(
            validation_certification(**common, ablation=False, completed_comparable_resolutions=1)["status"],
            "not_tested",
        )

    def test_mpo_eligibility_rejects_no_cd_only_resolution(self):
        import agp_mps_validation

        case = {
            "ablation": False,
            "learned_support": "full_support",
            "settings": {
                "backend": "tenpy_tdvp_mpo",
                "integrator": "tdvp",
                "learned_terms": 8,
                "full_learned_terms": 8,
                "learned_scale": 1.0,
                "hamiltonian_identity": "hamiltonian",
                "ground_reference_identity": "ground",
                "ground_bitstring": "00",
                "schedule_identity": "schedule",
                "checkpoint_identity": "checkpoint",
                "coefficient_identity": "coefficient",
                "total_time": 1.0,
                "initial_state": "+ +",
            },
            "results": {
                "no_cd": {
                    "mps_diagnostics": {"status": "ok", "completed_steps": 8, "steps": 8}
                }
            },
        }

        self.assertEqual(agp_mps_validation._completed_comparable_mpo_resolution_count([case]), 0)

    def test_mpo_eligibility_skips_interleaved_ablation_for_all_gates(self):
        import agp_mps_validation

        def resolution(name, *, ablation=False):
            return {
                "name": name,
                "ablation": ablation,
                "learned_support": "ablation" if ablation else "full_support",
                "settings": {
                    "backend": "tenpy_tdvp_mpo",
                    "integrator": "tdvp",
                    "n_qubits": 2,
                    "learned_terms": 4 if ablation else 8,
                    "full_learned_terms": 8,
                    "learned_scale": 1.0,
                    "hamiltonian_identity": "hamiltonian",
                    "ground_reference_identity": "ground",
                    "ground_bitstring": "00",
                    "schedule_identity": "schedule",
                    "schedule_parameters_identity": "schedule-parameters",
                    "checkpoint_identity": "checkpoint",
                    "coefficient_identity": "coefficient",
                    "total_time": 1.0,
                    "initial_state": "+ +",
                },
                "results": {
                    protocol: {
                        "final_energy": -1.0,
                        "ground_state_fidelity": 1.0,
                        "mps_diagnostics": {"status": "ok", "completed_steps": 8, "steps": 8},
                    }
                    for protocol in ("no_cd", "kipu_dqfm_l1", "learned_sparse_agp")
                },
            }

        eligible = agp_mps_validation.eligible_mpo_resolution_ladder(
            [resolution("coarse"), resolution("ablation", ablation=True), resolution("fine")]
        )

        self.assertEqual([row["name"] for row in eligible], ["coarse", "fine"])
        self.assertEqual(agp_mps_validation._completed_comparable_mpo_resolution_count(eligible), 2)

    def test_final_eligible_resolution_owns_top_level_metrics_after_trailing_ablation(self):
        import agp_mps_validation

        def resolution(name, *, ablation=False, final_energy=-1.0):
            return {
                "name": name,
                "ablation": ablation,
                "learned_support": "ablation" if ablation else "full_support",
                "settings": {
                    "backend": "tenpy_tdvp_mpo",
                    "integrator": "tdvp",
                    "n_qubits": 2,
                    "learned_terms": 4 if ablation else 8,
                    "full_learned_terms": 8,
                    "learned_scale": 1.0,
                    "hamiltonian_identity": "hamiltonian",
                    "ground_reference_identity": "ground",
                    "ground_bitstring": "00",
                    "schedule_identity": "schedule",
                    "schedule_parameters_identity": "schedule-parameters",
                    "checkpoint_identity": "checkpoint",
                    "coefficient_identity": "coefficient",
                    "total_time": 1.0,
                    "initial_state": "++",
                    "steps": 8,
                    "statevector_integrator": "rk4_renormalized",
                },
                "results": {
                    protocol: {
                        "final_energy": final_energy,
                        "ground_state_fidelity": 1.0,
                        "mps_diagnostics": {"status": "ok", "completed_steps": 8, "steps": 8},
                    }
                    for protocol in ("no_cd", "kipu_dqfm_l1", "learned_sparse_agp")
                },
            }

        payload = {"results": {"learned_sparse_agp": {"final_energy": -0.5}}}
        selected = agp_mps_validation.publish_final_eligible_mpo_results(
            payload,
            [resolution("coarse", final_energy=-1.1), resolution("fine", final_energy=-1.0), resolution("ablation", ablation=True, final_energy=-0.5)]
        )

        self.assertEqual(selected["name"], "fine")
        self.assertEqual(payload["results"]["learned_sparse_agp"]["final_energy"], -1.0)
        self.assertEqual(payload["certification_resolution"]["name"], "fine")

    def test_mpo_compression_uses_measured_upper_bound_and_unresolved_statuses(self):
        base_diagnostics = {
            "status": "ok",
            "temporal_retained_norm": 1.0,
            "temporal_rank": 1,
            "static_mpo_compression": {"h0": {}},
            "resource_statuses": {"dynamic_mpo_assembly": "ok"},
        }
        failed = assess_mpo_compression(
            {
                "learned_sparse_agp": {
                    "mps_diagnostics": {
                        **base_diagnostics,
                        "mpo_action_diagnostics": {
                            "status": "pass",
                            "max_relative_action_error_upper_bound": 2.0e-3,
                        },
                    }
                }
            },
            action_error_max=1.0e-3,
        )
        unresolved = assess_mpo_compression(
            {
                "learned_sparse_agp": {
                    "mps_diagnostics": {
                        **base_diagnostics,
                        "mpo_action_diagnostics": {"status": "numerically_unresolved"},
                    }
                }
            },
            action_error_max=1.0e-3,
        )
        not_feasible = assess_mpo_compression(
            {
                "learned_sparse_agp": {
                    "mps_diagnostics": {
                        **base_diagnostics,
                        "mpo_action_diagnostics": {"status": "not_feasible"},
                    }
                }
            },
            action_error_max=1.0e-3,
        )
        incomplete_evolution = assess_mpo_compression(
            {
                "learned_sparse_agp": {
                    "mps_diagnostics": {
                        **base_diagnostics,
                        "status": "not_feasible",
                        "mpo_action_diagnostics": {"status": "pass", "max_relative_action_error_upper_bound": 0.0},
                    }
                }
            },
            action_error_max=1.0e-3,
        )

        self.assertEqual(failed["status"], "fail")
        self.assertEqual(unresolved["status"], "not_tested")
        self.assertEqual(not_feasible["status"], "not_tested")
        self.assertEqual(incomplete_evolution["status"], "not_tested")

    def test_quotients_tolerate_unavailable_metrics(self):
        import agp_mps_validation

        results = {
            "no_cd": {"energy_error": 1.0, "excitation_probability": None, "z_rmse": None, "nearest_neighbor_zz_rmse": None},
            "learned_sparse_agp": {"energy_error": 0.5, "excitation_probability": None, "z_rmse": None, "nearest_neighbor_zz_rmse": None},
        }

        agp_mps_validation._add_baseline_quotients(results)

        self.assertEqual(results["learned_sparse_agp"]["energy_error_quotient_vs_no_cd"], 0.5)
        self.assertIsNone(results["learned_sparse_agp"]["z_rmse_quotient_vs_no_cd"])

    @unittest.skipUnless(find_spec("tenpy") is not None, "TeNPy tensor-network extra is not installed")
    def test_mpo_resolution_records_complete_metrics_and_action_probes(self):
        h0 = SparsePauliOperator({"X": -1.0}, n_qubits=1)
        h1 = SparsePauliOperator({"Z": -1.0}, n_qubits=1)
        settings = {
            "integrator": "tdvp",
            "steps": 4,
            "timestep": 0.25,
            "temporal_grid_points": 5,
            "temporal_retained_norm": 0.9999,
            "mpo_max_bond": 8,
            "mpo_cutoff": 1.0e-12,
            "mps_max_bond": 8,
            "mps_cutoff": 1.0e-12,
            "lanczos_max": 12,
            "mpo_workspace_cap_bytes": 8 * 1024 * 1024,
            "action_probe_seed": 11,
            "action_probe_product_states": 1,
            "action_probe_random_mps": 0,
            "action_probe_exact_work_cap": 100_000,
            "action_error_max": 1.0e-8,
        }
        backend = {"qubit_order_candidates": ("native",), **settings}

        result = run_mpo_case(
            h0=h0,
            h1=h1,
            learned=None,
            exact_ground_energy=-1.0,
            ground_bitstring="0",
            protocols=("no_cd",),
            total_time=1.0,
            settings=settings,
            backend=backend,
        )["no_cd"]

        self.assertIn("excitation_probability", result)
        self.assertIn("z_rmse", result)
        self.assertIn("nearest_neighbor_zz_rmse", result)
        self.assertEqual(result["z_observables_status"], "ok")
        self.assertEqual(result["nearest_neighbor_zz_status"], "not_applicable")
        self.assertEqual(result["mps_diagnostics"]["completed_steps"], 4)
        self.assertIn("static_mpo_action_probes", result["mps_diagnostics"])
        self.assertIn("dynamic_mpo_action_probes", result["mps_diagnostics"])
        action_diagnostics = result["mps_diagnostics"]["mpo_action_diagnostics"]
        self.assertIn("max_relative_action_error_upper_bound", action_diagnostics)
        self.assertTrue(action_diagnostics["finite_error_intervals"])

    def test_tenpy_backend_is_explicit_and_learned_certification_requires_full_support(self):
        backend = resolve_validation_backend(
            {"mpo_backend": {"name": "tenpy_tdvp_mpo", "qubit_order_candidates": ["native", "spectral"]}}
        )

        self.assertEqual(backend["name"], "tenpy_tdvp_mpo")
        self.assertEqual(backend["qubit_order_candidates"], ("native", "spectral"))
        self.assertEqual(resolve_validation_backend({})["name"], "quimb_product_formula")
        with self.assertRaisesRegex(ValueError, "full learned support"):
            require_full_learned_support(selected_terms=1024, available_terms=32768, ablation=False)
        self.assertEqual(
            require_full_learned_support(selected_terms=1024, available_terms=32768, ablation=True),
            "ablation",
        )

    def test_statevector_reference_selects_matching_learned_support(self):
        payload = {
            "results": {
                "no_cd": {"final_energy": -2.0},
                "learned_sparse_agp": {"final_energy": -4.0, "learned_terms": 2048},
            },
            "learned_variant_results": {
                "learned_sparse_agp_terms_1024_scale_1": {
                    "final_energy": -3.5,
                    "learned_terms": 1024,
                    "learned_scale": 1.0,
                }
            },
        }

        selected = statevector_results_for_learned_terms(payload, learned_terms=1024, learned_scale=1.0)

        self.assertEqual(selected["no_cd"]["final_energy"], -2.0)
        self.assertEqual(selected["learned_sparse_agp"]["final_energy"], -3.5)

    def test_statevector_reference_requires_matching_full_physics_identity(self):
        payload = {
            "validation_identity": {
                "n_qubits": 2,
                "hamiltonian_identity": "wrong-hamiltonian",
                "schedule_identity": "schedule",
                "total_time": 1.0,
                "checkpoint_identity": "checkpoint",
                "coefficient_identity": "coefficient",
                "learned_terms": 8,
                "full_learned_terms": 8,
                "learned_scale": 1.0,
                "initial_state": "+ +",
                "ground_reference_identity": "ground",
                "ground_bitstring": "00",
                "steps": 16,
                "integrator": "tdvp",
            },
            "results": {
                "no_cd": {"final_energy": -1.0},
                "kipu_dqfm_l1": {"final_energy": -1.0},
                "learned_sparse_agp": {"final_energy": -1.0, "learned_terms": 8},
            },
        }
        required_identity = {**payload["validation_identity"], "hamiltonian_identity": "expected-hamiltonian"}

        selected = statevector_results_for_learned_terms(
            payload,
            learned_terms=8,
            learned_scale=1.0,
            require_matching_learned_terms=True,
            required_identity=required_identity,
        )

        self.assertEqual(selected, {})

    def test_statevector_cli_identity_matches_and_rejects_mpo_identity_mismatch(self):
        import agp_physical_validation

        h0 = SparsePauliOperator({"XI": -1.0}, n_qubits=2)
        h1 = SparsePauliOperator({"ZZ": -1.0}, n_qubits=2)
        learned = {
            "tau": np.array([0.0, 1.0]),
            "lambda": np.array([0.0, 1.0]),
            "d_lambda_dt": np.array([0.0, 0.0]),
            "schedule_source": "exported",
            "selected_terms": 1,
            "available_terms": 1,
        }
        with __import__("tempfile").TemporaryDirectory() as tmp:
            coefficient_path = Path(tmp) / "final_agp_coefficients.pt"
            coefficient_path.write_bytes(b"identity")
            identity = agp_physical_validation.build_statevector_validation_identity(
                h0=h0,
                h1=h1,
                learned=learned,
                coefficient_path=coefficient_path,
                ground_energy=-1.0,
                ground_bitstring="00",
                total_time=1.0,
                steps=16,
                learned_scale=1.0,
                mps_integrator="tdvp",
            )

        payload = {
            "validation_identity": identity,
            "results": {
                "no_cd": {"final_energy": -0.5},
                "kipu_dqfm_l1": {"final_energy": -0.75},
                "learned_sparse_agp": {"final_energy": -1.0, "learned_terms": 1},
            },
        }
        matched = statevector_results_for_learned_terms(
            payload,
            learned_terms=1,
            learned_scale=1.0,
            require_matching_learned_terms=True,
            required_identity=identity,
        )
        mismatched = statevector_results_for_learned_terms(
            payload,
            learned_terms=1,
            learned_scale=1.0,
            require_matching_learned_terms=True,
            required_identity={**identity, "total_time": 2.0},
        )

        self.assertEqual(set(matched), {"no_cd", "kipu_dqfm_l1", "learned_sparse_agp"})
        self.assertEqual(mismatched, {})

    def test_mpo_backend_exception_preserves_complete_result_schema(self):
        h0 = SparsePauliOperator({"X": -1.0}, n_qubits=1)
        h1 = SparsePauliOperator({"Z": -1.0}, n_qubits=1)
        settings = {
            "integrator": "tdvp",
            "steps": 4,
            "timestep": 0.25,
            "temporal_grid_points": 5,
            "temporal_retained_norm": 0.9999,
            "mpo_max_bond": 8,
            "mpo_cutoff": 1.0e-12,
            "mps_max_bond": 8,
            "mps_cutoff": 1.0e-12,
            "lanczos_max": 12,
            "mpo_workspace_cap_bytes": 8 * 1024 * 1024,
        }
        with patch("scripts.agp_mpo_backend.evolve_protocol_tdvp", side_effect=RuntimeError("backend failure")):
            row = run_mpo_case(
                h0=h0,
                h1=h1,
                learned=None,
                exact_ground_energy=-1.0,
                ground_bitstring="0",
                protocols=("no_cd",),
                total_time=1.0,
                settings=settings,
                backend={"qubit_order_candidates": ("native",), **settings},
            )["no_cd"]

        self.assertEqual(
            set(row),
            {
                "final_energy",
                "ground_energy",
                "ground_state_fidelity",
                "energy_error",
                "excitation_probability",
                "z_rmse",
                "z_observables_status",
                "nearest_neighbor_zz_rmse",
                "nearest_neighbor_zz_status",
                "mps_diagnostics",
            },
        )
        self.assertEqual(row["z_observables_status"], "not_tested")
        self.assertEqual(row["mps_diagnostics"]["status"], "unresolved_error")

    def test_statevector_agreement_gate_compares_same_protocols(self):
        mps = {
            "no_cd": {"final_energy": -2.001, "ground_state_fidelity": 0.201},
            "learned_sparse_agp": {"final_energy": -4.004, "ground_state_fidelity": 0.804},
        }
        statevector = {
            "no_cd": {"final_energy": -2.0, "ground_state_fidelity": 0.20},
            "learned_sparse_agp": {"final_energy": -4.0, "ground_state_fidelity": 0.80},
        }

        assessment = assess_statevector_agreement(
            mps,
            statevector,
            energy_atol=0.005,
            fidelity_atol=0.005,
        )

        self.assertEqual(assessment["status"], "pass")
        self.assertEqual(set(assessment["protocols"]), set(mps))

    def test_validation_case_emits_physical_metrics_and_mps_diagnostics(self):
        h0 = SparsePauliOperator({"X": -1.0}, n_qubits=1)
        h1 = SparsePauliOperator({"Z": -1.0}, n_qubits=1)

        result = run_mps_case(
            h0=h0,
            h1=h1,
            learned=None,
            exact_ground_energy=-1.0,
            ground_bitstring="0",
            protocols=("no_cd",),
            total_time=1.0,
            steps=8,
            cutoff=1.0e-12,
            max_bond=4,
            coefficient_threshold=0.0,
        )

        self.assertEqual(set(result), {"no_cd"})
        self.assertIn("final_energy", result["no_cd"])
        self.assertIn("ground_state_fidelity", result["no_cd"])
        self.assertEqual(result["no_cd"]["mps_diagnostics"]["steps"], 8)

    def test_convergence_requires_all_protocol_deltas_within_tolerance(self):
        coarse = {
            "no_cd": {"final_energy": -2.0, "ground_state_fidelity": 0.20},
            "kipu_dqfm_l1": {"final_energy": -3.0, "ground_state_fidelity": 0.30},
            "learned_sparse_agp": {"final_energy": -4.0, "ground_state_fidelity": 0.80},
        }
        fine = {
            "no_cd": {"final_energy": -2.005, "ground_state_fidelity": 0.201},
            "kipu_dqfm_l1": {"final_energy": -3.008, "ground_state_fidelity": 0.302},
            "learned_sparse_agp": {"final_energy": -4.009, "ground_state_fidelity": 0.804},
        }

        assessment = assess_mps_convergence(
            coarse,
            fine,
            energy_atol=0.01,
            fidelity_atol=0.005,
        )

        self.assertEqual(assessment["status"], "pass")
        self.assertTrue(all(row["pass"] for row in assessment["protocols"].values()))

    def test_no_cd_evolution_matches_independent_rk4_reference(self):
        h0 = SparsePauliOperator({"X": -1.0}, n_qubits=1)
        h1 = SparsePauliOperator({"Z": -1.0}, n_qubits=1)
        total_time = 1.0
        state, diagnostics = evolve_protocol_mps(
            protocol="no_cd",
            h0=h0,
            h1=h1,
            learned=None,
            total_time=total_time,
            steps=256,
            cutoff=1.0e-14,
            max_bond=8,
        )

        x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        psi = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
        dt = total_time / 4096

        def rhs(t, vector):
            lam = np.sin(0.5 * np.pi * t / total_time) ** 2
            hamiltonian = -(1.0 - lam) * x - lam * z
            return -1.0j * hamiltonian @ vector

        for step in range(4096):
            t = step * dt
            k1 = rhs(t, psi)
            k2 = rhs(t + 0.5 * dt, psi + 0.5 * dt * k1)
            k3 = rhs(t + 0.5 * dt, psi + 0.5 * dt * k2)
            k4 = rhs(t + dt, psi + dt * k3)
            psi += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            psi /= np.linalg.norm(psi)

        mps_vector = np.asarray(state.to_dense()).reshape(-1)
        overlap = abs(np.vdot(psi, mps_vector)) ** 2
        self.assertGreater(overlap, 1.0 - 2.0e-8)
        self.assertEqual(diagnostics["gate_count"], 1024)

    def test_support_grouped_evolution_records_full_term_and_group_counts(self):
        h0 = SparsePauliOperator({"XI": -1.0, "IX": -1.0}, n_qubits=2)
        h1 = SparsePauliOperator({"ZI": -0.4, "IZ": -0.5, "ZZ": -1.0}, n_qubits=2)

        _, diagnostics = evolve_protocol_mps(
            protocol="no_cd",
            h0=h0,
            h1=h1,
            learned=None,
            total_time=1.0,
            steps=4,
            cutoff=1.0e-12,
            max_bond=8,
            operator_grouping="support",
        )

        self.assertEqual(diagnostics["operator_grouping"], "support")
        self.assertEqual(diagnostics["pauli_term_applications"], 40)
        self.assertEqual(diagnostics["gate_count"], 24)

    def test_support_grouping_converges_to_legacy_pauli_term_evolution(self):
        h0 = SparsePauliOperator({"XI": -1.0, "IX": -1.0}, n_qubits=2)
        h1 = SparsePauliOperator({"ZI": -0.4, "IZ": -0.5, "ZZ": -1.0}, n_qubits=2)
        common = {
            "protocol": "no_cd",
            "h0": h0,
            "h1": h1,
            "learned": None,
            "total_time": 1.0,
            "steps": 128,
            "cutoff": 1.0e-14,
            "max_bond": 16,
        }

        legacy, _ = evolve_protocol_mps(**common, operator_grouping="pauli_term")
        grouped, _ = evolve_protocol_mps(**common, operator_grouping="support")

        legacy_vector = np.asarray(legacy.to_dense()).reshape(-1)
        grouped_vector = np.asarray(grouped.to_dense()).reshape(-1)
        self.assertGreater(abs(np.vdot(legacy_vector, grouped_vector)) ** 2, 1.0 - 1.0e-8)

    def test_pauli_rotation_matches_dense_definition(self):
        angle = 0.371
        x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
        pauli = np.kron(x, y)
        expected = np.cos(angle) * np.eye(4) - 1.0j * np.sin(angle) * pauli

        np.testing.assert_allclose(pauli_rotation_matrix("XY", angle), expected, atol=1.0e-12)

    def test_pauli_rotation_updates_product_mps(self):
        state = make_product_mps("00")

        apply_pauli_rotation_mps(state, "XI", np.pi / 2.0, cutoff=1.0e-14, max_bond=8)

        self.assertAlmostEqual(abs(complex(state.amplitude("10"))) ** 2, 1.0, places=12)
        self.assertAlmostEqual(abs(complex(state.amplitude("00"))) ** 2, 0.0, places=12)

    def test_diagonal_metrics_recover_known_ground_product_state(self):
        state = make_product_mps("000")
        terms = {
            "ZII": -0.3,
            "IZI": -0.4,
            "IIZ": -0.5,
            "ZZI": -1.0,
            "IZZ": -1.1,
        }

        metrics = diagonal_ising_mps_metrics(state, terms, exact_ground_energy=-3.3)

        self.assertAlmostEqual(metrics["final_energy"], -3.3, places=12)
        self.assertAlmostEqual(metrics["energy_error"], 0.0, places=12)
        self.assertAlmostEqual(metrics["ground_fidelity"], 1.0, places=12)
        self.assertAlmostEqual(metrics["state_norm"], 1.0, places=12)

    def test_diagonal_metrics_use_configured_ground_bitstring_observable_targets(self):
        state = make_product_mps("101")
        terms = {"ZII": 1.0, "IZI": -1.0, "IIZ": 1.0}

        metrics = diagonal_ising_mps_metrics(
            state,
            terms,
            exact_ground_energy=-3.0,
            ground_bitstring="101",
        )

        self.assertAlmostEqual(metrics["final_energy"], -3.0, places=12)
        self.assertAlmostEqual(metrics["ground_fidelity"], 1.0, places=12)
        self.assertAlmostEqual(metrics["z_rmse"], 0.0, places=12)
        self.assertAlmostEqual(metrics["nearest_neighbor_zz_rmse"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
