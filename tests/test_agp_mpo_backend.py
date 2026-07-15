"""Tests for the dependency-free preparation layer of MPO evaluation."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from scripts.agp_mpo_backend import (
    factor_direct_cd_coefficients,
    permute_pauli_label,
    select_qubit_order,
    unpermute_pauli_label,
)


class OptionalDependencyTests(unittest.TestCase):
    def test_tensor_network_extra_pins_mpo_dependencies(self) -> None:
        pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('quimb==1.11.2', pyproject)
        self.assertIn('physics-tenpy==1.1.0', pyproject)


class TemporalFactorizationTests(unittest.TestCase):
    def test_temporal_factorization_uses_all_terms_and_meets_norm_target(self) -> None:
        tau = np.linspace(0.0, 1.0, 9)
        factors = np.stack([np.sin(np.pi * tau), np.sin(2.0 * np.pi * tau)], axis=1)
        modes = np.asarray([[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 0.25, 2.0]])
        direct = factors @ modes

        result = factor_direct_cd_coefficients(tau, direct, retained_norm=0.999999)

        self.assertEqual(result.rank, 2)
        self.assertEqual(result.static_modes.shape, (2, 4))
        self.assertGreaterEqual(result.retained_norm_fraction, 0.999999)
        np.testing.assert_allclose(result.reconstruct(), direct, atol=1.0e-11)

    def test_factorization_chooses_the_smallest_rank_meeting_squared_norm_target(self) -> None:
        tau = np.linspace(0.0, 1.0, 4)
        direct = np.asarray([[0.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

        rank_one = factor_direct_cd_coefficients(tau, direct, retained_norm=0.9)
        rank_two = factor_direct_cd_coefficients(tau, direct, retained_norm=0.900001)

        self.assertEqual(rank_one.rank, 1)
        self.assertAlmostEqual(rank_one.retained_norm_fraction, 0.9)
        self.assertEqual(rank_two.rank, 2)

    def test_factorization_preserves_zero_direct_cd_endpoints(self) -> None:
        tau = np.linspace(0.0, 1.0, 5)
        direct = np.asarray(
            [[0.0, 0.0], [1.0, -2.0], [3.0, 4.0], [1.0, 2.0], [0.0, 0.0]]
        )

        result = factor_direct_cd_coefficients(tau, direct, retained_norm=1.0)

        np.testing.assert_array_equal(result.reconstruct()[0], np.zeros(2))
        np.testing.assert_array_equal(result.reconstruct()[-1], np.zeros(2))
        self.assertEqual(result.endpoint_max_abs_error, 0.0)

    def test_factorization_rejects_invalid_shapes_nonfinite_values_and_retained_norm(self) -> None:
        tau = np.linspace(0.0, 1.0, 3)
        direct = np.zeros((3, 2))

        with self.assertRaisesRegex(ValueError, "shape"):
            factor_direct_cd_coefficients(tau[:-1], direct)
        with self.assertRaisesRegex(ValueError, "finite"):
            factor_direct_cd_coefficients(tau, np.asarray([[0.0, 0.0], [np.nan, 0.0], [0.0, 0.0]]))
        with self.assertRaisesRegex(ValueError, "retained_norm"):
            factor_direct_cd_coefficients(tau, direct, retained_norm=0.0)


class QubitOrderingTests(unittest.TestCase):
    def test_qubit_order_is_deterministic_and_permutation_safe(self) -> None:
        terms = [("XIIX", 4.0), ("IXXI", 1.0), ("ZIZI", 2.0)]

        first = select_qubit_order(terms, n_qubits=4, candidates=("native", "spectral"))
        second = select_qubit_order(terms, n_qubits=4, candidates=("native", "spectral"))

        self.assertEqual(first.order, second.order)
        self.assertEqual(sorted(first.order), [0, 1, 2, 3])
        self.assertEqual(
            unpermute_pauli_label(permute_pauli_label("XYZI", first.order), first.order),
            "XYZI",
        )

    def test_default_candidates_include_scoring_metadata_and_select_by_requested_score(self) -> None:
        terms = [("XIIX", 3.0), ("IXXI", -2.0), ("ZIIZ", 1.0)]

        result = select_qubit_order(terms, n_qubits=4)

        scores = result.candidate_scores
        self.assertEqual({score.candidate for score in scores}, {"native", "reversed", "spectral"})
        self.assertEqual(scores, tuple(sorted(scores, key=lambda score: score.candidate)))
        self.assertEqual(
            (result.max_cut_terms, result.mean_cut_terms, result.candidate),
            min((score.max_cut_terms, score.mean_cut_terms, score.candidate) for score in scores),
        )

    def test_ordering_rejects_invalid_labels_and_candidate_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "Pauli"):
            select_qubit_order([("XQA", 1.0)], n_qubits=3)
        with self.assertRaisesRegex(ValueError, "candidate"):
            select_qubit_order([("XII", 1.0)], n_qubits=3, candidates=("unknown",))
        with self.assertRaisesRegex(ValueError, "permutation"):
            permute_pauli_label("XYZ", (0, 0, 2))


if __name__ == "__main__":
    unittest.main()
