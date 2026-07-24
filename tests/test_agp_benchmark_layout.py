import json
import sys
import tempfile
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
SPIN_HUBO_SCENARIO_DIR = (
    ROOT / "tests" / "sparse_agp_curriculum" / "transverse_field_spin_hubo"
)
for path in (SCRIPTS_DIR, FRAMEWORK_SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agp_holdout_feedback import payload_with_feedback_baseline_neural
from agp_baseline_train import settings_for_support
from agp_size_intensive_study import (
    assess_acceptance,
    execution_flags_for_q,
    result_row as size_study_result_row,
    run_command as run_size_study_command,
    validate_config as validate_size_study_config,
)
from agp_qubit_grid_benchmark import (
    DEFAULT_GRID_ROOT,
    grid_config,
    run_command,
    validate_physical_summary_contract,
)
from models import PadeActivation


class AGPBenchmarkLayoutTests(unittest.TestCase):
    def test_size_intensive_acceptance_requires_threshold_and_no_size_drop(self):
        passing = assess_acceptance(
            [
                {"q": 15, "ground_state_fidelity": 0.970},
                {"q": 20, "ground_state_fidelity": 0.965},
                {"q": 25, "ground_state_fidelity": 0.960},
            ],
            minimum_fidelity=0.95,
            maximum_adjacent_drop=0.01,
        )
        failing = assess_acceptance(
            [
                {"q": 15, "ground_state_fidelity": 0.970},
                {"q": 20, "ground_state_fidelity": 0.955},
                {"q": 25, "ground_state_fidelity": 0.949},
            ],
            minimum_fidelity=0.95,
            maximum_adjacent_drop=0.01,
        )

        self.assertEqual(passing["status"], "pass")
        self.assertEqual(failing["status"], "fail")

    def test_size_intensive_manifest_marks_v6_as_current_benchmark(self):
        manifest = json.loads(
            (
                ISING_SCENARIO_DIR / "size_intensive_pinn_study.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(
            manifest["methodology"],
            "size_extensive_normalized_variational_action_conventional_pinn_v6",
        )
        self.assertEqual(manifest["benchmark_status"], "retained_current")
        self.assertEqual(manifest["promotion_decision"]["status"], "promoted")
        self.assertEqual(
            manifest["training_constraints"]["variational_action_weight"],
            0.1,
        )

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

    def test_size_study_subprocesses_use_stable_python_hash_seed(self):
        with patch("agp_size_intensive_study.subprocess.run") as mocked_run:
            run_size_study_command(["python", "example.py"], cwd=ROOT)

        self.assertEqual(mocked_run.call_args.kwargs["env"]["PYTHONHASHSEED"], "0")

    def test_size_study_never_cleans_or_retrains_declared_retained_anchor(self):
        manifest = {"retained_anchor_q": 15}

        self.assertEqual(
            execution_flags_for_q(manifest, 15, clean=True, train=True),
            {"clean": False, "train": False},
        )
        self.assertEqual(
            execution_flags_for_q(manifest, 20, clean=True, train=True),
            {"clean": True, "train": True},
        )

    def test_size_study_uses_exact_full_support_result_for_q15(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            trained_run = root / "runs" / "candidate"
            summary_path = trained_run / "Models_Data" / "physical_validation_summary.json"
            summary_path.parent.mkdir(parents=True)
            config_path.write_text(
                json.dumps(
                    {
                        "physical": {"parameters": {"num_qubits": 15}},
                        "holdout_feedback": {"base_agp_terms": 8},
                        "physical_validation": {"trained_run": "runs/candidate"},
                        "tensor_network_validation": {
                            "trained_run": "runs/candidate",
                            "output_dir": "mpo_validation",
                        },
                    }
                ),
                encoding="utf-8",
            )
            summary_path.write_text(
                json.dumps(
                    {
                        "results": {
                            "learned_sparse_agp": {
                                "learned_terms": 8,
                                "final_energy": -2.9,
                                "ground_energy": -3.0,
                                "energy_error": 0.1,
                                "ground_state_fidelity": 0.97,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            row = size_study_result_row(config_path)

        self.assertEqual(row["certification"], "exact_statevector")
        self.assertTrue(row["full_support"])
        self.assertEqual(row["ground_state_fidelity"], 0.97)

    def test_size_study_rejects_nonunit_physical_duration(self):
        source = (
            ISING_SCENARIO_DIR
            / "q20"
            / "sweep_test"
            / "size_intensive_pinn"
            / "config.json"
        )
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["physical"]["parameters"]["T"] = 2.0
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "config.json"
            candidate.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fixed physical duration T=1"):
                validate_size_study_config(candidate)

    def test_projected_settings_propagate_variational_action_weight(self):
        source = (
            ISING_SCENARIO_DIR
            / "q20"
            / "sweep_test"
            / "size_intensive_pinn"
            / "config.json"
        )
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["training"]["loss"]["variational_action"] = 0.125

        explicit = settings_for_support(payload, 32)
        del payload["training"]["loss"]["variational_action"]
        default = settings_for_support(payload, 32)

        self.assertEqual(explicit.variational_action_weight, 0.125)
        self.assertEqual(default.variational_action_weight, 0.0)

    def test_q20_size_candidate_uses_isolated_variational_action_run(self):
        config_path = (
            ISING_SCENARIO_DIR
            / "q20"
            / "sweep_test"
            / "size_intensive_pinn"
            / "config.json"
        )
        payload = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["physical"]["parameters"]["T"], 1.0)
        self.assertGreater(payload["training"]["loss"]["variational_action"], 0.0)
        self.assertEqual(payload["size_intensive_scaling"]["version"], "v6_action")
        self.assertIn(
            "variational_action_v6",
            payload["holdout_feedback"]["output_root"],
        )
        self.assertIn(
            payload["holdout_feedback"]["output_root"],
            payload["tensor_network_validation"]["trained_run"],
        )
        payload["training"]["loss"]["variational_action"] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "config.json"
            invalid.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "positive variational-action"):
                validate_size_study_config(invalid)

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
            "agp_validation_identity.py",
            "agp_qubit_grid_benchmark.py",
            "build_driver_problem_hamiltonian.py",
            "agp_regenerate_hcd_summaries.py",
            "spin_hubo_benchmark.py",
        }

        self.assertTrue(SCRIPTS_DIR.is_dir())
        self.assertTrue(expected_common.issubset({path.name for path in SCRIPTS_DIR.glob("*.py")}))
        self.assertTrue(FRAMEWORK_SCRIPTS_DIR.is_dir())
        self.assertEqual(expected_framework, {path.name for path in FRAMEWORK_SCRIPTS_DIR.glob("*.py")})

    def test_q15_q20_graph_candidates_are_isolated_and_methodology_matched(self):
        for q, residual_terms, rounds in ((15, 65536, 15), (20, 81920, 20)):
            candidate = ISING_SCENARIO_DIR / f"q{q}" / "sweep_test" / "hamiltonian_pauli_graph"
            payload = json.loads((candidate / "config.json").read_text(encoding="utf-8"))
            general = payload["neural"]["general"]
            feedback = payload["holdout_feedback"]

            self.assertEqual(general["coefficient_architecture"], "hamiltonian_pauli_graph")
            self.assertEqual(feedback["base_agp_terms"], 32768)
            self.assertEqual(feedback["holdout_residual_top_k"], residual_terms)
            self.assertEqual(feedback["iterations"], rounds)
            self.assertFalse(feedback["allow_legacy_baseline_reuse"])
            self.assertEqual(feedback["baseline_root"], "runs/baselines")
            self.assertTrue(str(feedback["output_root"]).startswith("runs/"))

    def test_q15_q20_factor_graph_candidates_are_isolated_and_methodology_matched(self):
        for q, residual_terms, rounds in ((15, 65536, 15), (20, 81920, 20)):
            candidate = (
                ISING_SCENARIO_DIR
                / f"q{q}"
                / "sweep_test"
                / "hamiltonian_pauli_factor_graph"
            )
            payload = json.loads((candidate / "config.json").read_text(encoding="utf-8"))
            general = payload["neural"]["general"]
            feedback = payload["holdout_feedback"]
            acceptance = payload["candidate_acceptance"]

            self.assertEqual(
                general["coefficient_architecture"], "hamiltonian_pauli_factor_graph"
            )
            self.assertGreaterEqual(general["graph_latent_rank"], 96)
            self.assertGreaterEqual(general["graph_term_width"], 128)
            self.assertEqual(feedback["base_agp_terms"], 32768)
            self.assertEqual(feedback["holdout_residual_top_k"], residual_terms)
            self.assertEqual(feedback["iterations"], rounds)
            self.assertFalse(feedback["allow_legacy_baseline_reuse"])
            self.assertTrue(str(feedback["output_root"]).startswith("runs/"))
            self.assertEqual(acceptance["minimum_q15_ground_fidelity"], 0.95)
            self.assertEqual(acceptance["minimum_q20_ground_fidelity"], 0.95)
            self.assertFalse(acceptance["uses_ground_truth_during_training_or_selection"])

    def test_tests_tree_contains_only_unit_tests_and_benchmark_configs(self):
        allowed_python = {
            "tests/__init__.py",
            "tests/test_agp_benchmark_layout.py",
            "tests/test_agp_residual_probes.py",
            "tests/test_agp_guarded_curriculum.py",
            "tests/test_agp_physical_validation.py",
            "tests/test_agp_resample_checkpoint.py",
            "tests/test_agp_resource_policy.py",
            "tests/test_full_pauli_pinn.py",
            "tests/test_ising_ground_state_solver.py",
            "tests/test_agp_mps_validation.py",
            "tests/test_agp_mpo_backend.py",
            "tests/test_agp_tn_regression_matrix.py",
            "tests/test_agp_tn_router.py",
            "tests/test_agp_joint_calibration.py",
            "tests/test_agp_support_swap.py",
            "tests/test_agp_stratified_support.py",
            "tests/test_qiskit_hamiltonian_generator.py",
            "tests/test_sparse_pauli.py",
            "tests/test_spin_hubo_benchmark.py",
            "tests/sparse_agp_curriculum/scripts/agp_physical_validation.py",
            "tests/sparse_agp_curriculum/scripts/agp_mps_validation.py",
            "tests/sparse_agp_curriculum/scripts/agp_validation_identity.py",
            "tests/sparse_agp_curriculum/scripts/agp_qubit_grid_benchmark.py",
            "tests/sparse_agp_curriculum/scripts/agp_regenerate_hcd_summaries.py",
            "tests/sparse_agp_curriculum/scripts/build_driver_problem_hamiltonian.py",
            "tests/sparse_agp_curriculum/scripts/spin_hubo_benchmark.py",
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
        expected = {"q15", "q20", "q25", "q156"}
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

    def test_size_extensive_active_support_configs_follow_declared_scaling_law(self):
        expected = {
            15: {
                "K": 32768, "Q": 65536, "rounds": 15, "active": 2048,
                "add": 3072, "swap": 256, "probe": 4096, "T": 1.0, "steps": 48,
            },
            20: {
                "K": 58368, "Q": 116736, "rounds": 20, "active": 3840,
                "add": 4096, "swap": 512, "probe": 5632, "T": 1.0, "steps": 48,
            },
            25: {
                "K": 91136, "Q": 182272, "rounds": 25, "active": 5888,
                "add": 5120, "swap": 512, "probe": 7168, "T": 1.0, "steps": 48,
            },
        }
        for q, values in expected.items():
            candidate = ISING_SCENARIO_DIR / f"q{q}" / "sweep_test" / "size_intensive_pinn"
            payload = json.loads((candidate / "config.json").read_text(encoding="utf-8"))
            feedback = payload["holdout_feedback"]
            validation = payload["tensor_network_validation"]

            self.assertTrue((candidate / "README.md").is_file())
            self.assertEqual(payload["physical"]["parameters"]["num_qubits"], q)
            self.assertEqual(payload["physical"]["parameters"]["T"], values["T"])
            self.assertEqual(
                payload["neural"]["general"]["coefficient_architecture"],
                "independent_outputs",
            )
            self.assertEqual(feedback["base_agp_terms"], values["K"])
            self.assertEqual(feedback["holdout_residual_top_k"], values["Q"])
            self.assertEqual(feedback["iterations"], values["rounds"])
            self.assertEqual(feedback["add_residual_terms_per_iteration"], values["add"])
            self.assertFalse(feedback["allow_legacy_baseline_reuse"])
            self.assertEqual(feedback["fixed_unseen_probes"]["active_terms"], values["probe"])
            self.assertEqual(feedback["fixed_unseen_probes"]["null_terms"], values["probe"])
            self.assertEqual(feedback["support_swap"]["terms_per_iteration"], values["swap"])
            self.assertEqual(feedback["support_swap"]["locality_penalty_power"], 0.0)
            self.assertEqual(payload["training"]["loss"]["residual_objective"], "absolute")
            self.assertNotIn("residual_block_normalization", payload["training"]["loss"])
            self.assertEqual(payload["agp_calibration"]["target_active_terms"], values["active"])
            self.assertEqual(validation["protocols"], ["learned_sparse_agp"])
            self.assertEqual(validation["resolutions"][-1]["steps"], values["steps"])
            self.assertTrue(
                all(row["learned_terms"] == values["K"] for row in validation["resolutions"])
            )
            self.assertTrue(payload["size_intensive_scaling"]["no_cross_system_initialization"])

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

    def test_retained_sweeps_configure_stable_unseen_probes(self):
        config_paths = (
            ISING_SCENARIO_DIR / "q15/sweep_test/config.json",
            ISING_SCENARIO_DIR / "q20/sweep_test/config.json",
            ISING_SCENARIO_DIR / "q156/sweep_test/config.json",
            SPIN_HUBO_SCENARIO_DIR
            / "run_002_hamiltonian_341/q24/sweep_test/config.json",
        )

        for config_path in config_paths:
            with self.subTest(config=config_path):
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                probes = payload["holdout_feedback"]["fixed_unseen_probes"]
                self.assertTrue(probes["enabled"])
                self.assertEqual(probes["active_terms"], 4096)
                self.assertEqual(probes["null_terms"], 4096)
                self.assertEqual(probes["reference_rms_threshold"], 1e-12)
                self.assertEqual(probes["seed"], 11)
                self.assertEqual(probes["candidate_multiplier"], 8)

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
            [32768, 32768, 32768, 32768],
        )
        self.assertNotIn("scalable_validation", q20)
        self.assertNotIn("hydrogen_energy_validation", q20)

    def test_q24_spin_hubo_matches_q20_training_methodology(self):
        q20_path = ISING_SCENARIO_DIR / "q20/sweep_test/config.json"
        q24_path = (
            SPIN_HUBO_SCENARIO_DIR
            / "run_002_hamiltonian_341/q24/sweep_test/config.json"
        )
        q20 = json.loads(q20_path.read_text(encoding="utf-8"))
        q24 = json.loads(q24_path.read_text(encoding="utf-8"))

        physical = q24["physical"]["parameters"]
        self.assertEqual(physical["system"], "TransverseFieldSpinHUBO")
        self.assertEqual(physical["num_qubits"], 24)
        self.assertEqual(q24["default_pipeline"]["agp_terms"], 32768)
        self.assertEqual(q24["holdout_feedback"]["base_agp_terms"], 32768)
        self.assertEqual(q24["holdout_feedback"]["holdout_residual_top_k"], 81920)
        self.assertEqual(q24["holdout_feedback"]["iterations"], 20)
        self.assertEqual(q24["holdout_feedback"]["add_residual_terms_per_iteration"], 3072)

        for key in (
            "neural",
            "support_sweep",
            "agp_calibration",
            "training",
            "schedule_optimization",
        ):
            self.assertEqual(q24[key], q20[key])
        for key in (
            "support_swap",
            "temporal_refinement",
            "adaptive_temporal_refinement",
        ):
            q20_section = {
                name: value
                for name, value in q20["holdout_feedback"][key].items()
                if name != "description"
            }
            q24_section = {
                name: value
                for name, value in q24["holdout_feedback"][key].items()
                if name != "description"
            }
            self.assertEqual(q24_section, q20_section)

        tensor_validation = q24["tensor_network_validation"]
        self.assertTrue(tensor_validation["enabled"])
        self.assertEqual(tensor_validation["operator_grouping"], "support")
        self.assertEqual(tensor_validation["coefficient_threshold"], 0.0)
        validation_ladder = [
            case
            for case in tensor_validation["resolutions"]
            if not case.get("preflight_only", False)
        ]
        self.assertEqual(
            [case["learned_terms"] for case in validation_ladder],
            [32768, 32768, 32768],
        )
        preflight = [
            case
            for case in tensor_validation["resolutions"]
            if case.get("preflight_only", False)
        ]
        self.assertEqual([case["learned_terms"] for case in preflight], [32768])

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

    def test_diagonal_ising_tensor_network_configs_use_joint_full_support_ladders(self):
        for q in (15, 20, 156):
            with self.subTest(q=q):
                config_path = ISING_SCENARIO_DIR / f"q{q}/sweep_test/config.json"
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                validation = payload["tensor_network_validation"]
                backend = validation["mpo_backend"]
                self.assertTrue(validation["enabled"])
                self.assertEqual(validation["output_dir"], "mpo_validation")
                self.assertEqual(backend["name"], "tenpy_tdvp_mpo")
                self.assertEqual(backend["integrator"], "tdvp")
                self.assertEqual(backend["representation"], "joint_time_full_support")
                self.assertEqual(validation["coefficient_threshold"], 0.0)
                preflight = [
                    case
                    for case in validation["resolutions"]
                    if case.get("preflight_only", False)
                ]
                canonical = [
                    case
                    for case in validation["resolutions"]
                    if not case.get("preflight_only", False)
                ]
                self.assertEqual(len(preflight), 1)
                self.assertEqual(len(canonical), 3)
                self.assertTrue(
                    all(case["learned_terms"] == 32768 for case in validation["resolutions"])
                )
                self.assertEqual(
                    set(validation["convergence_pairs"]),
                    {"timestep", "state"},
                )

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
