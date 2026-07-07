import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class AGPBenchmarkLayoutTests(unittest.TestCase):
    def test_common_scripts_exist(self):
        expected = {
            "agp_baseline_train.py",
            "agp_holdout_feedback.py",
            "agp_holdout_study.py",
            "agp_evaluate_holdout.py",
            "agp_residual_calibration.py",
            "agp_restart.py",
            "agp_support.py",
            "full_pauli_training_common.py",
            "projected_sparse_training_common.py",
            "agp_physical_validation.py",
            "build_driver_problem_hamiltonian.py",
        }

        self.assertTrue(SCRIPTS_DIR.is_dir())
        self.assertTrue(expected.issubset({path.name for path in SCRIPTS_DIR.glob("*.py")}))

    def test_tests_tree_contains_only_unit_tests_and_benchmark_configs(self):
        allowed_python = {
            "tests/__init__.py",
            "tests/test_agp_benchmark_layout.py",
            "tests/test_full_pauli_pinn.py",
            "tests/test_q15_physical_validation.py",
            "tests/test_agp_joint_calibration.py",
            "tests/test_agp_residual_calibration.py",
            "tests/test_q20_guarded_curriculum.py",
            "tests/test_qiskit_hamiltonian_generator.py",
            "tests/test_sparse_pauli.py",
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

    def test_documentation_lives_under_docs(self):
        self.assertTrue((ROOT / "docs").is_dir())
        self.assertFalse((ROOT / "documentation").exists())

    def test_q15_and_q20_configs_live_under_tests_and_point_to_common_scripts(self):
        for study in ("q15", "q20"):
            config_path = ROOT / "tests" / study / "sweep_test" / "config.json"
            self.assertTrue(config_path.is_file(), config_path)
            payload = json.loads(config_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["default_pipeline"]["entrypoint"], "scripts/agp_holdout_feedback.py")
            study_python = [
                path
                for path in config_path.parent.rglob("*.py")
                if "__pycache__" not in path.parts
            ]
            self.assertEqual(study_python, [])

    def test_q15_support_selection_is_configured_not_script_specific(self):
        payload = json.loads((ROOT / "tests/q15/sweep_test/config.json").read_text(encoding="utf-8"))

        self.assertEqual(
            payload["support_sweep"]["agp_support_selection"]["strategy"],
            "nested_commutator_krylov_pool",
        )


if __name__ == "__main__":
    unittest.main()
