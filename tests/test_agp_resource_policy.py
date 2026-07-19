import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from agp_resource_policy import resolve_resource_budget


class AGPResourcePolicyTests(unittest.TestCase):
    def test_legacy_integer_budget_is_unchanged(self):
        budget = resolve_resource_budget(2048, q=156, capacity=32768, name="active_terms")

        self.assertEqual(budget.mode, "fixed")
        self.assertEqual(budget.requested, 2048)
        self.assertEqual(budget.realized, 2048)
        self.assertEqual(budget.clipping_reasons, ())

    def test_per_qubit_budget_uses_deterministic_half_up_rounding(self):
        budget = resolve_resource_budget(
            {"mode": "per_qubit", "per_qubit": 102.4},
            q=156,
            capacity=32768,
            name="active_terms",
        )

        self.assertEqual(budget.requested, 15974)
        self.assertEqual(budget.realized, 15974)
        self.assertEqual(budget.to_dict()["per_qubit"], 102.4)

    def test_minimum_maximum_and_capacity_are_recorded(self):
        minimum = resolve_resource_budget(
            {"mode": "per_qubit", "per_qubit": 1.0, "minimum": 4096},
            q=20,
            capacity=10000,
            name="residual_terms",
        )
        maximum = resolve_resource_budget(
            {"mode": "per_qubit", "per_qubit": 1000.0, "maximum": 5000},
            q=20,
            capacity=10000,
            name="residual_terms",
        )
        capacity = resolve_resource_budget(
            {"mode": "per_qubit", "per_qubit": 1000.0},
            q=20,
            capacity=4500,
            name="residual_terms",
        )

        self.assertEqual((minimum.realized, minimum.clipping_reasons), (4096, ("minimum",)))
        self.assertEqual((maximum.realized, maximum.clipping_reasons), (5000, ("maximum",)))
        self.assertEqual((capacity.realized, capacity.clipping_reasons), (4500, ("capacity",)))

    def test_invalid_resource_specs_fail_closed(self):
        invalid_specs = (
            -1,
            {"mode": "per_qubit", "per_qubit": -0.1},
            {"mode": "per_qubit", "per_qubit": math.inf},
            {"mode": "unknown", "per_qubit": 1.0},
            {"mode": "per_qubit", "per_qubit": 1.0, "minimum": 5, "maximum": 4},
        )

        for spec in invalid_specs:
            with self.subTest(spec=spec):
                with self.assertRaises((TypeError, ValueError)):
                    resolve_resource_budget(spec, q=20, capacity=32768, name="test")

        with self.assertRaisesRegex(ValueError, "q must be positive"):
            resolve_resource_budget(1, q=0, capacity=10, name="test")
        with self.assertRaisesRegex(ValueError, "capacity"):
            resolve_resource_budget(1, q=2, capacity=-1, name="test")


if __name__ == "__main__":
    unittest.main()
