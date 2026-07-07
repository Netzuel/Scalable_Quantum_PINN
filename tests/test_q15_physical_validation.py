import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
TESTS_DIR = ROOT / "tests"
for path in (SCRIPTS_DIR, TESTS_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agp_support import KrylovSupportConfig, select_krylov_agp_labels
from agp_holdout_feedback import fit_residual_budget_to_available
from agp_holdout_study import relative_metric_with_reference_status
from agp_physical_validation import apply_pauli_sum, build_action_cache, variational_l1_agp
from utils import SparsePauliOperator, transverse_field_ising_problem


class Q15PhysicalValidationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
