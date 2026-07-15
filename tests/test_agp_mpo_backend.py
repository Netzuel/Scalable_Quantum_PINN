"""Tests for the dependency-free preparation layer of MPO evaluation."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 in the required torch-mps environment.
    from pip._vendor import tomli as tomllib

from scripts.agp_mpo_backend import (
    factor_direct_cd_coefficients,
    permute_pauli_label,
    select_qubit_order,
    unpermute_pauli_label,
)


class OptionalDependencyTests(unittest.TestCase):
    def test_tensor_network_extra_pins_mpo_dependencies(self) -> None:
        pyproject = tomllib.loads(
            (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
        )

        self.assertEqual(
            pyproject["project"]["optional-dependencies"]["tensor-network"],
            ["quimb==1.11.2", "physics-tenpy==1.1.0"],
        )


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

    def test_factorization_keeps_global_rank_when_it_preserves_every_term(self) -> None:
        tau = np.linspace(0.0, 1.0, 4)
        direct = np.asarray([[0.0, 0.0], [3.0, 6.0], [0.0, 0.0], [0.0, 0.0]])

        result = factor_direct_cd_coefficients(tau, direct, retained_norm=0.9)

        self.assertEqual(result.rank_for_retained_norm, 1)
        self.assertEqual(result.rank, 1)
        self.assertFalse(result.rank_increased_for_term_preservation)
        self.assertAlmostEqual(result.retained_norm_fraction, 1.0)

    def test_low_energy_column_forces_rank_increase_for_per_column_retained_energy(self) -> None:
        tau = np.linspace(0.0, 1.0, 5)
        direct = np.asarray([[0.0, 0.0], [10.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])

        result = factor_direct_cd_coefficients(tau, direct, retained_norm=0.99)

        self.assertEqual(result.rank_for_retained_norm, 1)
        self.assertEqual(result.rank, 2)
        self.assertTrue(result.rank_increased_for_term_preservation)
        self.assertIn("squared temporal norm", result.rank_increase_reason)
        self.assertGreaterEqual(
            result.minimum_column_retained_energy_fraction,
            0.99 - 1.0e-12,
        )
        np.testing.assert_allclose(result.column_retained_energy_fractions, np.ones(2), atol=1.0e-12)

    def test_underflow_scale_column_remains_nonzero_and_forces_energy_retention(self) -> None:
        tau = np.linspace(0.0, 1.0, 5)
        direct = np.asarray(
            [[0.0, 0.0], [1.0e-100, 0.0], [0.0, 1.0e-200], [0.0, 0.0], [0.0, 0.0]]
        )

        result = factor_direct_cd_coefficients(tau, direct, retained_norm=0.99)

        self.assertEqual(result.rank_for_retained_norm, 1)
        self.assertEqual(result.rank, 2)
        self.assertTrue(result.rank_increased_for_term_preservation)
        self.assertGreaterEqual(result.minimum_column_retained_energy_fraction, 0.99 - 1.0e-12)
        self.assertTrue(np.all(np.isfinite(result.column_retained_energy_fractions)))
        self.assertTrue(np.isfinite(result.retained_norm_fraction))
        np.testing.assert_allclose(result.column_retained_energy_fractions, np.ones(2), atol=1.0e-12)

    def test_large_finite_coefficients_keep_energy_diagnostics_finite(self) -> None:
        tau = np.linspace(0.0, 1.0, 5)
        direct = np.asarray(
            [[0.0, 0.0], [1.0e200, 0.0], [0.0, 5.0e199], [0.0, 0.0], [0.0, 0.0]]
        )

        result = factor_direct_cd_coefficients(tau, direct, retained_norm=0.99)

        self.assertTrue(np.isfinite(result.retained_norm_fraction))
        self.assertTrue(np.all(np.isfinite(result.column_retained_energy_fractions)))
        self.assertGreaterEqual(result.minimum_column_retained_energy_fraction, 0.99 - 1.0e-12)

    def test_factorization_preserves_zero_direct_cd_endpoints(self) -> None:
        tau = np.linspace(0.0, 1.0, 5)
        direct = np.asarray(
            [[0.0, 0.0], [1.0, -2.0], [3.0, 4.0], [1.0, 2.0], [0.0, 0.0]]
        )

        result = factor_direct_cd_coefficients(tau, direct, retained_norm=1.0)

        np.testing.assert_allclose(result.reconstruct()[0], np.zeros(2), atol=1.0e-12)
        np.testing.assert_allclose(result.reconstruct()[-1], np.zeros(2), atol=1.0e-12)
        self.assertLessEqual(result.endpoint_max_abs_error, 1.0e-12)

    def test_factorization_rejects_invalid_shapes_nonfinite_values_and_retained_norm(self) -> None:
        tau = np.linspace(0.0, 1.0, 3)
        direct = np.zeros((3, 2))

        with self.assertRaisesRegex(ValueError, "shape"):
            factor_direct_cd_coefficients(tau[:-1], direct)
        with self.assertRaisesRegex(ValueError, "finite"):
            factor_direct_cd_coefficients(tau, np.asarray([[0.0, 0.0], [np.nan, 0.0], [0.0, 0.0]]))
        with self.assertRaisesRegex(ValueError, "retained_norm"):
            factor_direct_cd_coefficients(tau, direct, retained_norm=0.0)

    def test_factorization_requires_normalized_strictly_increasing_tau(self) -> None:
        direct = np.zeros((4, 2))

        with self.assertRaisesRegex(ValueError, "start at 0"):
            factor_direct_cd_coefficients(np.asarray([0.1, 0.4, 0.7, 1.0]), direct)
        with self.assertRaisesRegex(ValueError, "end at 1"):
            factor_direct_cd_coefficients(np.asarray([0.0, 0.3, 0.6, 0.9]), direct)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            factor_direct_cd_coefficients(np.asarray([0.0, 0.5, 0.5, 1.0]), direct)

    def test_factorization_rejects_nonzero_direct_cd_endpoints(self) -> None:
        tau = np.linspace(0.0, 1.0, 3)
        direct = np.asarray([[1.0e-4, 0.0], [1.0, -2.0], [0.0, 0.0]])

        with self.assertRaisesRegex(ValueError, "endpoint"):
            factor_direct_cd_coefficients(tau, direct)

    def test_factorization_diagnostics_describe_the_returned_reconstruction(self) -> None:
        tau = np.linspace(0.0, 1.0, 5)
        direct = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])

        result = factor_direct_cd_coefficients(tau, direct, retained_norm=0.8)
        reconstructed = result.reconstruct()

        self.assertEqual(result.max_abs_error, float(np.max(np.abs(reconstructed - direct))))
        self.assertEqual(
            result.endpoint_max_abs_error,
            float(np.max(np.abs(reconstructed[[0, -1]] - direct[[0, -1]]))),
        )


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

    def test_spectral_order_falls_back_deterministically_for_degenerate_graphs(self) -> None:
        terms = [("XIII", 1.0), ("IYII", 2.0), ("IIZI", 3.0), ("IIIX", 4.0)]

        first = select_qubit_order(terms, n_qubits=4, candidates=("spectral",))
        second = select_qubit_order(tuple(reversed(terms)), n_qubits=4, candidates=("spectral",))

        self.assertEqual(first.order, (0, 1, 2, 3))
        self.assertEqual(first, second)

    def test_spectral_order_falls_back_for_symmetric_fiedler_eigenspaces(self) -> None:
        terms = [
            ("XXII", 1.0),
            ("XIXI", 1.0),
            ("XIIX", 1.0),
            ("IXXI", 1.0),
            ("IXIX", 1.0),
            ("IIXX", 1.0),
        ]

        result = select_qubit_order(terms, n_qubits=4, candidates=("spectral",))

        self.assertEqual(result.order, (0, 1, 2, 3))

    def test_spectral_order_is_invariant_to_uniform_coefficient_rescaling(self) -> None:
        terms = [("XIXI", 1.0), ("IIXX", 1.0), ("IXIX", 1.0)]
        scaled_terms = [(label, coefficient * 1.0e-15) for label, coefficient in terms]

        baseline = select_qubit_order(terms, n_qubits=4, candidates=("spectral",))
        scaled = select_qubit_order(scaled_terms, n_qubits=4, candidates=("spectral",))

        self.assertNotEqual(baseline.order, (0, 1, 2, 3))
        self.assertEqual(baseline, scaled)

    def test_ordering_rejects_fractional_and_boolean_indices(self) -> None:
        terms = [("XII", 1.0)]

        for invalid_n_qubits in (3.0, True):
            with self.assertRaisesRegex(ValueError, "n_qubits"):
                select_qubit_order(terms, n_qubits=invalid_n_qubits)
        for invalid_order in ((0, 1.0, 2), (0, False, 2)):
            with self.assertRaisesRegex(ValueError, "integer"):
                permute_pauli_label("XYZ", invalid_order)


if __name__ == "__main__":
    unittest.main()
