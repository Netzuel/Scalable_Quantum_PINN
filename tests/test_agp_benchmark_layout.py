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
            "agp_restart.py",
            "agp_support.py",
            "agp_physical_validation.py",
            "build_driver_problem_hamiltonian.py",
        }

        self.assertTrue(SCRIPTS_DIR.is_dir())
        self.assertTrue(expected.issubset({path.name for path in SCRIPTS_DIR.glob("*.py")}))

    def test_q15_and_q20_configs_live_under_tests_and_point_to_common_scripts(self):
        for study in ("q15", "q20"):
            config_path = ROOT / "tests" / study / "sweep_test" / "config.json"
            self.assertTrue(config_path.is_file(), config_path)
            payload = json.loads(config_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["default_pipeline"]["entrypoint"], "scripts/agp_holdout_feedback.py")

    def test_q15_support_selection_is_configured_not_script_specific(self):
        payload = json.loads((ROOT / "tests/q15/sweep_test/config.json").read_text(encoding="utf-8"))

        self.assertEqual(
            payload["support_sweep"]["agp_support_selection"]["strategy"],
            "nested_commutator_krylov_pool",
        )


if __name__ == "__main__":
    unittest.main()
