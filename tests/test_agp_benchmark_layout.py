import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
FRAMEWORK_SCRIPTS_DIR = ROOT / "tests" / "sparse_agp_curriculum" / "scripts"
ISING_SCENARIO_DIR = (
    ROOT / "tests" / "sparse_agp_curriculum" / "transverse_field_diagonal_ising"
)
for path in (SCRIPTS_DIR, FRAMEWORK_SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agp_holdout_feedback import payload_with_feedback_baseline_neural
from agp_baseline_train import settings_for_support
from agp_qubit_grid_benchmark import (
    DEFAULT_GRID_ROOT,
    grid_config,
    run_command,
    validate_physical_summary_contract,
)
from models import PadeActivation


class AGPBenchmarkLayoutTests(unittest.TestCase):
    @staticmethod
    def _grid_payload(q: int):
        return grid_config(
            q,
            max_agp_terms=32768,
            max_initial_residual_terms=4096,
            holdout_residual_top_k=65536,
            add_residual_terms=3072,
            iterations=15,
            epochs=5000,
            epochs_per_iteration=1000,
            temporal_epochs=2500,
            adaptive_temporal_epochs=1500,
            evolution_steps=96,
            max_learned_terms=2048,
            learned_action_cache_size=128,
        )

    def test_pade_stability_fallback_approximates_silu(self):
        activation = PadeActivation()
        activation.reset_to_silu_rational_fit()
        x = torch.linspace(-20.0, 20.0, 4001)

        max_error = torch.max(torch.abs(activation(x) - torch.nn.functional.silu(x)))

        self.assertLess(float(max_error.detach()), 0.12)

    def test_grid_subprocesses_use_stable_python_hash_seed(self):
        with patch("agp_qubit_grid_benchmark.subprocess.run") as mocked_run:
            run_command(["python", "example.py"])

        environment = mocked_run.call_args.kwargs["env"]
        self.assertEqual(environment["PYTHONHASHSEED"], "0")

    def test_grid_defaults_inside_ising_scenario(self):
        self.assertEqual(
            DEFAULT_GRID_ROOT,
            Path("tests/sparse_agp_curriculum/transverse_field_diagonal_ising/grid"),
        )

    def test_common_scripts_exist(self):
        expected_common = {
            "agp_baseline_train.py",
            "agp_holdout_feedback.py",
            "agp_holdout_study.py",
            "agp_evaluate_holdout.py",
            "agp_residual_calibration.py",
            "agp_restart.py",
            "agp_support.py",
            "full_pauli_training_common.py",
            "projected_sparse_training_common.py",
            "agp_plot_annotations.py",
        }
        expected_framework = {
            "agp_mps_validation.py",
            "agp_physical_validation.py",
            "agp_qubit_grid_benchmark.py",
            "build_driver_problem_hamiltonian.py",
            "agp_regenerate_hcd_summaries.py",
        }

        self.assertTrue(SCRIPTS_DIR.is_dir())
        self.assertTrue(expected_common.issubset({path.name for path in SCRIPTS_DIR.glob("*.py")}))
        self.assertTrue(FRAMEWORK_SCRIPTS_DIR.is_dir())
        self.assertEqual(expected_framework, {path.name for path in FRAMEWORK_SCRIPTS_DIR.glob("*.py")})

    def test_tests_tree_contains_only_unit_tests_and_benchmark_configs(self):
        allowed_python = {
            "tests/__init__.py",
            "tests/test_agp_benchmark_layout.py",
            "tests/test_agp_guarded_curriculum.py",
            "tests/test_agp_physical_validation.py",
            "tests/test_full_pauli_pinn.py",
            "tests/test_ising_ground_state_solver.py",
            "tests/test_agp_mps_validation.py",
            "tests/test_agp_joint_calibration.py",
            "tests/test_agp_support_swap.py",
            "tests/test_qiskit_hamiltonian_generator.py",
            "tests/test_sparse_pauli.py",
            "tests/sparse_agp_curriculum/scripts/agp_physical_validation.py",
            "tests/sparse_agp_curriculum/scripts/agp_mps_validation.py",
            "tests/sparse_agp_curriculum/scripts/agp_qubit_grid_benchmark.py",
            "tests/sparse_agp_curriculum/scripts/agp_regenerate_hcd_summaries.py",
            "tests/sparse_agp_curriculum/scripts/build_driver_problem_hamiltonian.py",
        }
        python_files = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "tests").rglob("*.py")
            if "__pycache__" not in path.parts
        }

        self.assertEqual(python_files, allowed_python)

    def test_legacy_experiment_folders_are_removed_from_tests(self):
        obsolete = {
            "full_pauli_2_qubits",
            "full_pauli_4_qubits",
            "full_pauli_6_qubits",
            "sparse_agp_20_qubits",
            "sparse_agp_156_qubits",
        }
        present = {path.name for path in (ROOT / "tests").iterdir() if path.is_dir()}

        self.assertTrue(obsolete.isdisjoint(present))

    def test_retained_ising_studies_share_one_scenario_folder(self):
        expected = {"q15", "q20", "q156"}
        present = {
            path.name
            for path in ISING_SCENARIO_DIR.iterdir()
            if path.is_dir() and path.name.startswith("q")
        }

        self.assertEqual(present, expected)
        self.assertTrue((ISING_SCENARIO_DIR / "README.md").is_file())
        for study in expected:
            self.assertFalse(
                (ROOT / "tests" / "sparse_agp_curriculum" / study).exists()
            )

    def test_documentation_lives_under_docs(self):
        self.assertTrue((ROOT / "docs").is_dir())
        self.assertFalse((ROOT / "documentation").exists())

    def test_retained_benchmark_configs_define_their_physical_systems(self):
        expected_systems = {
            "q15": "TransverseIsingDriverProblem",
            "q20": "TransverseIsingDriverProblem",
            "q156": "TransverseIsingDriverProblem",
        }
        for study, expected_system in expected_systems.items():
            config_path = ISING_SCENARIO_DIR / study / "sweep_test" / "config.json"
            self.assertTrue(config_path.is_file(), config_path)
            self.assertTrue(config_path.with_name("README.md").is_file())
            payload = json.loads(config_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["default_pipeline"]["entrypoint"], "scripts/agp_holdout_feedback.py")
            self.assertEqual(payload["physical"]["parameters"]["system"], expected_system)
            physical_validation = payload.get("physical_validation")
            if isinstance(physical_validation, dict):
                self.assertEqual(
                    physical_validation["entrypoint"],
                    "tests/sparse_agp_curriculum/scripts/agp_physical_validation.py",
                )
            study_python = [
                path
                for path in config_path.parent.rglob("*.py")
                if "__pycache__" not in path.parts
            ]
            self.assertEqual(study_python, [])

    def test_q20_matches_q15_ising_lineage_for_twenty_rounds(self):
        q15_path = ISING_SCENARIO_DIR / "q15/sweep_test/config.json"
        q20_path = ISING_SCENARIO_DIR / "q20/sweep_test/config.json"
        q15 = json.loads(q15_path.read_text(encoding="utf-8"))
        q20 = json.loads(q20_path.read_text(encoding="utf-8"))

        self.assertEqual(q20["physical"]["parameters"]["system"], "TransverseIsingDriverProblem")
        self.assertEqual(q20["physical"]["parameters"]["num_qubits"], 20)
        self.assertEqual(q20["default_pipeline"]["agp_terms"], 32768)
        self.assertEqual(q20["holdout_feedback"]["base_agp_terms"], 32768)
        self.assertEqual(q20["holdout_feedback"]["holdout_residual_top_k"], 81920)
        self.assertEqual(q20["holdout_feedback"]["iterations"], 20)
        self.assertEqual(q20["holdout_feedback"]["add_residual_terms_per_iteration"], 3072)
        self.assertEqual(q20["holdout_feedback"]["unseen_residual_batches_after_final_iteration"], 1)

        for key in (
            "neural",
            "support_sweep",
            "agp_calibration",
            "training",
            "schedule_optimization",
        ):
            self.assertEqual(q20[key], q15[key])
        for key in (
            "baseline_neural",
            "support_swap",
            "temporal_refinement",
            "adaptive_temporal_refinement",
        ):
            q15_section = {
                name: value
                for name, value in q15["holdout_feedback"][key].items()
                if name != "description"
            }
            q20_section = {
                name: value
                for name, value in q20["holdout_feedback"][key].items()
                if name != "description"
            }
            self.assertEqual(q20_section, q15_section)

        validation = q20["physical_validation"]
        self.assertEqual(validation["statevector_qubits"], 20)
        self.assertEqual(validation["evolution_steps"], 96)
        self.assertEqual(validation["learned_top_terms"], 2048)
        self.assertLessEqual(validation["learned_action_cache_size"], 16)
        tensor_validation = q20["tensor_network_validation"]
        self.assertTrue(tensor_validation["enabled"])
        self.assertEqual(tensor_validation["operator_grouping"], "support")
        self.assertEqual(tensor_validation["coefficient_threshold"], 0.0)
        self.assertEqual(
            [case["learned_terms"] for case in tensor_validation["resolutions"]],
            [32768, 32768],
        )
        self.assertNotIn("scalable_validation", q20)
        self.assertNotIn("hydrogen_energy_validation", q20)

    def test_q156_config_uses_scalable_twenty_round_curriculum(self):
        config_path = ISING_SCENARIO_DIR / "q156/sweep_test/config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["physical"]["parameters"]["num_qubits"], 156)
        self.assertEqual(payload["default_pipeline"]["agp_terms"], 32768)
        self.assertEqual(payload["holdout_feedback"]["base_agp_terms"], 32768)
        self.assertEqual(payload["holdout_feedback"]["holdout_residual_top_k"], 81920)
        self.assertEqual(payload["holdout_feedback"]["iterations"], 20)
        self.assertEqual(payload["holdout_feedback"]["add_residual_terms_per_iteration"], 3072)
        self.assertEqual(payload["holdout_feedback"]["unseen_residual_batches_after_final_iteration"], 1)
        self.assertNotIn("physical_validation", payload)
        exact_reference = ROOT / payload["scalable_validation"]["exact_final_ground_reference"]
        self.assertTrue(exact_reference.is_file(), exact_reference)
        self.assertEqual(len(payload["scalable_validation"]["ground_bitstring"]), 156)

    def test_q15_support_selection_is_configured_not_script_specific(self):
        config_path = ISING_SCENARIO_DIR / "q15/sweep_test/config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["support_sweep"]["agp_support_selection"]["strategy"],
            "nested_commutator_krylov_pool",
        )

    def test_holdout_feedback_can_override_baseline_neural_config(self):
        payload = {
            "neural": {
                "general": {
                    "n_hidden": 4,
                    "n_neurons": 96,
                    "activation": "pau",
                    "layer_type": "quadratic",
                }
            },
            "holdout_feedback": {
                "baseline_neural": {
                    "activation": "silu",
                }
            },
        }

        baseline_payload = payload_with_feedback_baseline_neural(payload)

        self.assertEqual(payload["neural"]["general"]["activation"], "pau")
        self.assertEqual(baseline_payload["neural"]["general"]["activation"], "silu")
        self.assertEqual(baseline_payload["neural"]["general"]["n_neurons"], 96)
        self.assertEqual(settings_for_support(baseline_payload, 8).model.activation, "silu")
        self.assertEqual(settings_for_support(payload, 8).model.activation, "pau")

    def test_baseline_training_respects_low_q_projected_opt_in(self):
        payload = {
            "physical": {
                "parameters": {
                    "system": "TransverseIsingDriverProblem",
                    "num_qubits": 3,
                    "distance": "1_0",
                    "hamiltonian_source": "Hamiltonians_to_use/pauli_decompositions/index.json",
                }
            },
            "neural": {
                "model": "ProjectedSparseAGPPINN",
                "general": {
                    "n_hidden": 2,
                    "n_neurons": 8,
                    "activation": "silu",
                    "layer_type": "linear",
                },
            },
            "support": {"allow_low_q_projected": True},
            "support_sweep": {"intermediate_top_k": 16, "residual_top_k": 16},
            "training": {"parameters": {"epochs": 1}},
        }

        settings = settings_for_support(payload, 16)

        self.assertTrue(settings.allow_low_q_projected)

    def test_qubit_grid_config_preserves_the_common_methodology(self):
        q3_payload = self._grid_payload(3)
        self.assertEqual(q3_payload["physical"]["parameters"]["num_qubits"], 3)
        self.assertEqual(q3_payload["default_pipeline"]["agp_terms"], 64)
        self.assertTrue(q3_payload["support"]["allow_low_q_projected"])
        self.assertEqual(q3_payload["support_sweep"]["agp_support_selection"]["strategy"], "full_pauli_basis")
        self.assertFalse(q3_payload["holdout_feedback"]["support_swap"]["enabled"])
        self.assertEqual(q3_payload["holdout_feedback"]["iterations"], 1)
        self.assertEqual(q3_payload["holdout_feedback"]["add_residual_terms_per_iteration"], 0)
        self.assertEqual(q3_payload["holdout_feedback"]["baseline_neural"]["activation"], "silu")
        self.assertEqual(q3_payload["physical_validation"]["learned_top_terms"], 64)

        q6_payload = self._grid_payload(6)
        self.assertEqual(q6_payload["agp_calibration"]["target_active_terms"], 4096)

        q9_payload = self._grid_payload(9)
        self.assertEqual(q9_payload["agp_calibration"]["target_active_terms"], 2048)
        self.assertEqual(q9_payload["holdout_feedback"]["baseline_neural"]["activation"], "silu")
        self.assertEqual(q9_payload["physical_validation"]["learned_top_terms_sweep"], [1024, 2048])
        self.assertNotIn("trained_run_selection", q9_payload["physical_validation"])
        self.assertEqual(q3_payload["default_pipeline"]["entrypoint"], "scripts/agp_holdout_feedback.py")
        self.assertEqual(q3_payload["physical_validation"]["entrypoint"], "tests/sparse_agp_curriculum/scripts/agp_physical_validation.py")
        self.assertEqual(q3_payload["summary"]["runs_dir"], "runs/")

    def test_grid_summary_contract_rejects_mismatched_deployment(self):
        payload = {
            "n_qubits": 15,
            "steps": 96,
            "trained_run": "runs/example/adaptive_temporal_refinement",
            "learned_agp_truncation": {"selected_terms": 256},
            "results": {
                "no_cd": {},
                "kipu_dqfm_l1": {},
                "learned_sparse_agp": {"learned_terms": 256},
            },
        }

        with self.assertRaisesRegex(ValueError, "expected 2048 learned terms"):
            validate_physical_summary_contract(15, payload)

    def test_grid_summary_contract_accepts_retained_q15_methodology(self):
        payload = {
            "n_qubits": 15,
            "steps": 96,
            "trained_run": "runs/example/adaptive_temporal_refinement",
            "learned_agp_truncation": {"selected_terms": 2048},
            "results": {
                "no_cd": {},
                "kipu_dqfm_l1": {},
                "learned_sparse_agp": {"learned_terms": 2048},
            },
        }

        validate_physical_summary_contract(15, payload)

if __name__ == "__main__":
    unittest.main()
