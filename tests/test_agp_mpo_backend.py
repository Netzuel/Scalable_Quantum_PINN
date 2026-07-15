"""Tests for the dependency-free preparation layer of MPO evaluation."""

from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
import subprocess
import sys
import tracemalloc
import unittest
from unittest import mock
import warnings

import numpy as np

import scripts.agp_mpo_backend as mpo_backend

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 in the required torch-mps environment.
    from pip._vendor import tomli as tomllib

from scripts.agp_mpo_backend import (
    build_exact_pauli_mpo,
    compress_mpo_hilbert_schmidt,
    dense_pauli_sum,
    factor_direct_cd_coefficients,
    mpo_to_dense,
    permute_pauli_label,
    probe_mpo_compression,
    select_qubit_order,
    unpermute_pauli_label,
)


_TENPY_AVAILABLE = importlib.util.find_spec("tenpy") is not None


def _base4_pauli_label(index: int, n_qubits: int) -> str:
    symbols = "IXYZ"
    encoded = []
    for _ in range(n_qubits):
        encoded.append(symbols[index % 4])
        index //= 4
    return "".join(reversed(encoded))


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


@unittest.skipUnless(_TENPY_AVAILABLE, "TeNPy tensor-network extra is not installed")
class ExactPauliMPOTests(unittest.TestCase):
    def test_exact_pauli_mpo_contains_every_nonzero_combined_label(self) -> None:
        terms = [("XI", 0.3), ("YZ", -0.7), ("ZZ", 0.2), ("II", 0.1)]

        mpo, metadata = build_exact_pauli_mpo(terms, n_qubits=2, order=(0, 1))

        np.testing.assert_allclose(mpo_to_dense(mpo), dense_pauli_sum(terms), atol=1.0e-12)
        self.assertEqual(metadata["input_terms"], 4)
        self.assertEqual(metadata["unique_labels"], 4)
        self.assertEqual(metadata["included_terms"], 4)
        self.assertEqual(metadata["included_labels"], ["II", "XI", "YZ", "ZZ"])
        self.assertEqual(metadata["dropped_terms"], 0)
        self.assertEqual(metadata["dropped_labels"], [])

    def test_duplicate_labels_are_combined_before_cancellation_is_reported(self) -> None:
        terms = [("XI", 0.3), ("YZ", 0.25), ("XI", -0.3), ("YZ", 0.125)]

        mpo, metadata = build_exact_pauli_mpo(terms, n_qubits=2, order=(0, 1))

        np.testing.assert_allclose(
            mpo_to_dense(mpo), dense_pauli_sum([("YZ", 0.375)]), atol=1.0e-12
        )
        self.assertEqual(metadata["input_terms"], 4)
        self.assertEqual(metadata["unique_labels"], 2)
        self.assertEqual(metadata["duplicate_labels"], ["XI", "YZ"])
        self.assertEqual(metadata["dropped_terms"], 1)
        self.assertEqual(metadata["dropped_labels"], ["XI"])
        self.assertEqual(metadata["dropped_coefficients"], {"XI": 0j})
        self.assertEqual(metadata["combined_coefficients"], {"XI": 0j, "YZ": 0.375 + 0j})

    def test_arithmetic_zero_tolerance_is_explicit_and_defaults_to_exact_zero(self) -> None:
        terms = [("X", 1.0), ("X", -1.0 + 1.0e-15)]

        retained, retained_metadata = build_exact_pauli_mpo(
            terms, n_qubits=1, order=(0,)
        )
        dropped, dropped_metadata = build_exact_pauli_mpo(
            terms,
            n_qubits=1,
            order=(0,),
            arithmetic_zero_tolerance=1.0e-12,
        )

        self.assertEqual(retained_metadata["dropped_terms"], 0)
        self.assertGreater(np.linalg.norm(mpo_to_dense(retained)), 0.0)
        self.assertEqual(dropped_metadata["dropped_labels"], ["X"])
        self.assertEqual(dropped_metadata["arithmetic_zero_tolerance"], 1.0e-12)
        np.testing.assert_array_equal(mpo_to_dense(dropped), np.zeros((2, 2)))

    def test_non_native_order_matches_explicitly_permuted_pauli_sum(self) -> None:
        terms = [("XYZ", 0.4), ("ZIX", -0.2), ("III", 0.05)]
        order = (2, 0, 1)

        mpo, metadata = build_exact_pauli_mpo(terms, n_qubits=3, order=order)
        permuted_terms = [(permute_pauli_label(label, order), value) for label, value in terms]

        np.testing.assert_allclose(
            mpo_to_dense(mpo), dense_pauli_sum(permuted_terms), atol=1.0e-12
        )
        self.assertEqual(metadata["order"], order)
        self.assertEqual(
            metadata["included_chain_labels"],
            sorted(label for label, _ in permuted_terms),
        )

    def test_builder_handles_large_qubit_count_without_dense_helpers(self) -> None:
        terms = [("X" + "I" * 11, 0.25), ("I" * 10 + "YZ", -0.5)]

        mpo, metadata = build_exact_pauli_mpo(terms, n_qubits=12, order=tuple(range(12)))

        self.assertEqual(mpo.L, 12)
        self.assertEqual(metadata["included_terms"], 2)
        with self.assertRaisesRegex(ValueError, "q <= 4"):
            mpo_to_dense(mpo)
        with self.assertRaisesRegex(ValueError, "q <= 4"):
            dense_pauli_sum(terms)


@unittest.skipUnless(_TENPY_AVAILABLE, "TeNPy tensor-network extra is not installed")
class MPOCompressionTests(unittest.TestCase):
    def test_q24_compression_owns_retained_cores_within_hard_traced_cap(self) -> None:
        from tenpy.networks.site import SpinHalfSite

        n_qubits = 24
        terms = tuple(
            (
                _base4_pauli_label(
                    (index * 0x9E3779B97F4A7C15) % (4**n_qubits), n_qubits
                ),
                complex((index % 29) + 1) / 29.0,
            )
            for index in range(2048)
        )

        class PauliSource:
            L = n_qubits
            bc = "finite"
            sites = [SpinHalfSite(conserve=None) for _ in range(n_qubits)]
            chi = [1] * (n_qubits + 1)
            _agp_pauli_terms = terms

        workspace_cap = 33 * 1024 * 1024
        ownership: list[bool] = []
        original_builder = mpo_backend._mpo_from_pauli_cores

        with mock.patch.object(
            mpo_backend.np,
            "fromiter",
            side_effect=AssertionError("coefficient buffer was allocated before preflight"),
        ) as fromiter:
            blocked, blocked_diagnostics = compress_mpo_hilbert_schmidt(
                PauliSource(),
                max_bond=64,
                cutoff=1.0e-12,
                workspace_cap_bytes=1,
            )
        fromiter.assert_not_called()
        self.assertIsNone(blocked)
        self.assertEqual(blocked_diagnostics["status"], "not_feasible")
        self.assertGreater(blocked_diagnostics["required_workspace_bytes"], 1)

        def checked_builder(*args: object, **kwargs: object) -> object:
            cores = kwargs["cores"]
            ownership.extend(core.flags.owndata and core.base is None for core in cores)
            return original_builder(*args, **kwargs)

        gc.collect()
        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]
        try:
            with mock.patch.object(
                mpo_backend, "_mpo_from_pauli_cores", side_effect=checked_builder
            ):
                compressed, diagnostics = compress_mpo_hilbert_schmidt(
                    PauliSource(),
                    max_bond=64,
                    cutoff=1.0e-12,
                    workspace_cap_bytes=workspace_cap,
                )
            traced_peak = tracemalloc.get_traced_memory()[1] - baseline
        finally:
            tracemalloc.stop()

        self.assertEqual(diagnostics["status"], "ok")
        self.assertIsNotNone(compressed)
        self.assertTrue(ownership)
        self.assertTrue(all(ownership))
        self.assertLessEqual(traced_peak, workspace_cap)
        self.assertLessEqual(diagnostics["peak_workspace_bytes"], workspace_cap)
        self.assertLessEqual(diagnostics["required_workspace_bytes"], workspace_cap)

    def test_adversarial_full_support_compression_never_densifies_exact_mpo(self) -> None:
        terms = [
            (
                _base4_pauli_label((index * 0x9E3779B1) % (4**12), 12),
                complex((index % 17) + 1) / 17.0,
            )
            for index in range(512)
        ]
        exact, _ = build_exact_pauli_mpo(
            terms, n_qubits=12, order=tuple(range(12))
        )
        workspace_cap = 8 * 1024 * 1024

        with mock.patch(
            "tenpy.linalg.np_conserved.Array.to_ndarray",
            side_effect=AssertionError("exact MPO tensor was densified"),
        ):
            compressed, diagnostics = compress_mpo_hilbert_schmidt(
                exact,
                max_bond=8,
                cutoff=1.0e-12,
                workspace_cap_bytes=workspace_cap,
            )

        self.assertEqual(diagnostics["status"], "ok")
        self.assertIsNotNone(compressed)
        self.assertEqual(diagnostics["input_terms"], 512)
        self.assertLessEqual(diagnostics["peak_workspace_bytes"], workspace_cap)
        self.assertTrue(all(bond <= 8 for bond in compressed.chi[1:-1]))

    def test_compression_returns_not_feasible_before_exceeding_workspace_cap(self) -> None:
        terms = [
            (_base4_pauli_label(index, 8), complex(index + 1))
            for index in range(64)
        ]
        exact, _ = build_exact_pauli_mpo(terms, n_qubits=8, order=tuple(range(8)))

        compressed, diagnostics = compress_mpo_hilbert_schmidt(
            exact,
            max_bond=16,
            cutoff=0.0,
            workspace_cap_bytes=1024,
        )

        self.assertIsNone(compressed)
        self.assertEqual(diagnostics["status"], "not_feasible")
        self.assertEqual(diagnostics["workspace_cap_bytes"], 1024)
        self.assertGreater(diagnostics["required_workspace_bytes"], 1024)
        self.assertLessEqual(diagnostics["peak_workspace_bytes"], 1024)

    def test_compressed_mpo_reports_and_respects_operator_error(self) -> None:
        terms = [("XII", 0.3), ("IYZ", -0.7), ("ZZI", 0.2), ("XYZ", 0.1)]
        exact, _ = build_exact_pauli_mpo(terms, n_qubits=3, order=(0, 1, 2))

        compressed, diagnostics = compress_mpo_hilbert_schmidt(
            exact, max_bond=16, cutoff=1.0e-13
        )

        relative = np.linalg.norm(mpo_to_dense(exact) - mpo_to_dense(compressed)) / np.linalg.norm(
            mpo_to_dense(exact)
        )
        self.assertLess(relative, 1.0e-11)
        self.assertGreaterEqual(diagnostics["discarded_weight"], 0.0)
        self.assertEqual(
            diagnostics["discarded_weight"],
            sum(diagnostics["per_bond_discarded_weights"]),
        )
        self.assertEqual(len(diagnostics["per_bond_discarded_weights"]), exact.L - 1)
        self.assertEqual(diagnostics["post_bonds"], list(compressed.chi))
        self.assertTrue(all(diagnostics["cutoff_satisfied_by_bond"]))

    def test_hard_max_bond_is_enforced_and_reports_cutoff_violation(self) -> None:
        terms = [
            ("XXII", 1.0),
            ("YYII", 0.9),
            ("IXXI", -0.8),
            ("IYYI", 0.7),
            ("IIXX", 0.6),
            ("IIYY", -0.5),
        ]
        exact, _ = build_exact_pauli_mpo(terms, n_qubits=4, order=(0, 1, 2, 3))

        compressed, diagnostics = compress_mpo_hilbert_schmidt(
            exact, max_bond=1, cutoff=0.0
        )

        self.assertTrue(all(bond <= 1 for bond in compressed.chi[1:-1]))
        self.assertGreater(diagnostics["discarded_weight"], 0.0)
        self.assertIn(False, diagnostics["cutoff_satisfied_by_bond"])
        self.assertEqual(diagnostics["max_bond"], 1)
        self.assertEqual(diagnostics["cutoff"], 0.0)
        dense_error_squared = float(
            np.linalg.norm(mpo_to_dense(exact) - mpo_to_dense(compressed)) ** 2
        )
        self.assertAlmostEqual(
            diagnostics["total_discarded_squared_norm"],
            dense_error_squared,
            places=11,
        )
        source_norm_squared = float(np.linalg.norm(mpo_to_dense(exact)) ** 2)
        self.assertAlmostEqual(
            diagnostics["discarded_weight"],
            dense_error_squared / source_norm_squared,
            places=11,
        )

    def test_action_probes_are_seeded_deterministic_and_detect_compression_error(self) -> None:
        terms = [
            ("XXI", 1.0),
            ("YYI", -0.8),
            ("IXX", 0.6),
            ("IYY", 0.4),
            ("XYZ", -0.3),
        ]
        exact, _ = build_exact_pauli_mpo(terms, n_qubits=3, order=(0, 1, 2))
        compressed, _ = compress_mpo_hilbert_schmidt(exact, max_bond=1, cutoff=0.0)
        settings = {
            "product_states": (("up", "up", "up"), ("up", "down", "up")),
            "random_state_count": 2,
            "random_bond": 3,
            "seed": 1729,
        }

        with mock.patch.object(
            exact,
            "apply_naively",
            side_effect=AssertionError("exact MPO action was formed"),
        ):
            first = probe_mpo_compression(exact, compressed, **settings)
            second = probe_mpo_compression(exact, compressed, **settings)

        self.assertEqual(first, second)
        self.assertEqual(
            first["tested_probes"] + first["numerically_unresolved_probes"], 4
        )
        self.assertEqual(first["not_tested_probes"], 0)
        self.assertGreater(first["max_relative_action_error"], 0.0)
        self.assertTrue(
            all(
                probe["status"] in ("tested", "numerically_unresolved")
                for probe in first["probes"]
            )
        )
        product = np.zeros(2**3, dtype=np.complex128)
        product[2] = 1.0  # |up, down, up> in the native computational order.
        exact_dense = mpo_to_dense(exact)
        compressed_dense = mpo_to_dense(compressed)
        expected_error = np.linalg.norm((exact_dense - compressed_dense) @ product) / np.linalg.norm(
            exact_dense @ product
        )
        product_probe = first["probes"][1]
        if product_probe["status"] == "tested":
            self.assertAlmostEqual(product_probe["relative_action_error"], expected_error)
        else:
            self.assertIsNone(product_probe["relative_action_error"])
            self.assertGreaterEqual(
                product_probe["relative_action_error_upper_bound"], expected_error
            )
        from tenpy.networks.mps import MPS

        random_state = np.random.get_state()
        try:
            np.random.seed(settings["seed"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                random_mps = MPS.from_random_unitary_evolution(
                    exact.sites,
                    chi=2,
                    p_state=["up"] * exact.L,
                    bc="finite",
                    dtype=np.complex128,
                )
        finally:
            np.random.set_state(random_state)
        random_dense = random_mps.get_theta(0, exact.L).to_ndarray().reshape(-1)
        expected_random_error = np.linalg.norm(
            (exact_dense - compressed_dense) @ random_dense
        ) / np.linalg.norm(exact_dense @ random_dense)
        random_probe = first["probes"][2]
        if random_probe["status"] == "tested":
            self.assertAlmostEqual(
                random_probe["relative_action_error"], expected_random_error
            )
        else:
            self.assertIsNone(random_probe["relative_action_error"])
            self.assertGreaterEqual(
                random_probe["relative_action_error_upper_bound"],
                expected_random_error,
            )

    def test_random_probe_reports_not_feasible_above_exact_work_cap(self) -> None:
        terms = [
            (_base4_pauli_label(index, 4), complex(index + 1) / 16.0)
            for index in range(16)
        ]
        exact, _ = build_exact_pauli_mpo(terms, n_qubits=4, order=(0, 1, 2, 3))
        compressed, compression = compress_mpo_hilbert_schmidt(
            exact, max_bond=4, cutoff=1.0e-12
        )
        self.assertEqual(compression["status"], "ok")

        with mock.patch.object(
            exact,
            "apply_naively",
            side_effect=AssertionError("exact MPO action was formed"),
        ):
            diagnostics = probe_mpo_compression(
                exact,
                compressed,
                product_states=(),
                random_state_count=1,
                random_bond=4,
                seed=7,
                exact_work_cap=1,
            )

        self.assertEqual(diagnostics["tested_probes"], 0)
        self.assertEqual(diagnostics["not_feasible_probes"], 1)
        self.assertEqual(diagnostics["probes"][0]["status"], "not_feasible")
        self.assertGreater(diagnostics["probes"][0]["estimated_exact_work"], 1)

    def test_random_probe_preflights_workspace_before_constructing_state(self) -> None:
        from tenpy.networks.mps import MPS

        exact, _ = build_exact_pauli_mpo(
            [("XX", 1.0), ("YZ", -0.25)], n_qubits=2, order=(0, 1)
        )
        compressed, compression = compress_mpo_hilbert_schmidt(
            exact, max_bond=4, cutoff=0.0
        )
        self.assertEqual(compression["status"], "ok")

        with mock.patch.object(
            MPS,
            "from_random_unitary_evolution",
            side_effect=AssertionError("random MPS constructor was called"),
        ) as constructor:
            diagnostics = probe_mpo_compression(
                exact,
                compressed,
                product_states=(),
                random_state_count=1,
                random_bond=2,
                workspace_cap_bytes=1,
            )

        constructor.assert_not_called()
        self.assertEqual(diagnostics["tested_probes"], 0)
        self.assertEqual(diagnostics["not_feasible_probes"], 1)
        self.assertEqual(diagnostics["probes"][0]["status"], "not_feasible")
        self.assertGreater(diagnostics["probes"][0]["required_workspace_bytes"], 1)

    def test_action_error_zeros_roundoff_but_detects_small_real_difference(self) -> None:
        terms = [("XX", 0.7), ("YZ", -0.3), ("II", 0.2)]
        exact, _ = build_exact_pauli_mpo(terms, n_qubits=2, order=(0, 1))
        identical, compression = compress_mpo_hilbert_schmidt(
            exact, max_bond=16, cutoff=0.0
        )
        self.assertEqual(compression["status"], "ok")

        identical_diagnostics = probe_mpo_compression(
            exact,
            identical,
            product_states=(("up", "down"),),
            random_state_count=1,
            random_bond=2,
            seed=19,
        )
        self.assertEqual(identical_diagnostics["tested_probes"], 2)
        self.assertTrue(
            all(
                probe["relative_action_error"] == 0.0
                for probe in identical_diagnostics["probes"]
            )
        )

        relative_perturbation = 1.0e-10
        perturbed_terms = [
            (label, coefficient * (1.0 + relative_perturbation))
            for label, coefficient in terms
        ]
        perturbed, _ = build_exact_pauli_mpo(
            perturbed_terms, n_qubits=2, order=(0, 1)
        )
        perturbed_diagnostics = probe_mpo_compression(
            exact,
            perturbed,
            product_states=(("up", "down"),),
            random_state_count=0,
        )
        perturbed_probe = perturbed_diagnostics["probes"][0]
        if perturbed_probe["status"] == "tested":
            measured = perturbed_probe["relative_action_error"]
            self.assertGreater(measured, 0.0)
            self.assertAlmostEqual(measured, relative_perturbation, delta=1.0e-13)
        else:
            self.assertEqual(perturbed_probe["status"], "numerically_unresolved")
            self.assertIsNone(perturbed_probe["relative_action_error"])
            self.assertGreater(perturbed_probe["squared_difference_estimate"], 0.0)
            self.assertAlmostEqual(
                np.sqrt(
                    perturbed_probe["squared_difference_estimate"]
                    / perturbed_probe["action_norm"] ** 2
                ),
                relative_perturbation,
                delta=1.0e-13,
            )
            self.assertLessEqual(
                perturbed_probe["relative_action_error_lower_bound"],
                relative_perturbation,
            )
            self.assertGreaterEqual(
                perturbed_probe["relative_action_error_upper_bound"],
                relative_perturbation,
            )

    def test_product_probe_keeps_unit_relative_error_after_small_scale_rescaling(self) -> None:
        for scale in (1.0, 1.0e-10):
            exact, _ = build_exact_pauli_mpo(
                [("I", scale)], n_qubits=1, order=(0,)
            )
            perturbed, _ = build_exact_pauli_mpo(
                [("I", scale), ("X", scale)], n_qubits=1, order=(0,)
            )

            diagnostics = probe_mpo_compression(
                exact,
                perturbed,
                product_states=(("up",),),
                random_state_count=0,
            )

            probe = diagnostics["probes"][0]
            self.assertEqual(probe["status"], "tested")
            self.assertAlmostEqual(probe["relative_action_error"], 1.0, places=12)

    def test_product_probe_marks_cancellation_limited_leakage_unresolved(self) -> None:
        exact, _ = build_exact_pauli_mpo([("I", 1.0)], n_qubits=1, order=(0,))
        perturbed, _ = build_exact_pauli_mpo(
            [("I", 1.0), ("X", 1.0e-8)], n_qubits=1, order=(0,)
        )

        diagnostics = probe_mpo_compression(
            exact,
            perturbed,
            product_states=(("up",),),
            random_state_count=0,
        )

        probe = diagnostics["probes"][0]
        self.assertNotEqual(probe["relative_action_error"], 0.0)
        if probe["status"] == "tested":
            self.assertAlmostEqual(probe["relative_action_error"], 1.0e-8, places=14)
        else:
            self.assertEqual(probe["status"], "numerically_unresolved")
            self.assertIsNone(probe["relative_action_error"])
            self.assertGreaterEqual(probe["relative_action_error_upper_bound"], 1.0e-8)

    def test_small_scale_random_probe_never_reports_substantial_error_as_zero(self) -> None:
        scale = 1.0e-10
        exact, _ = build_exact_pauli_mpo(
            [("II", scale)], n_qubits=2, order=(0, 1)
        )
        perturbed, _ = build_exact_pauli_mpo(
            [("II", scale), ("XI", scale)], n_qubits=2, order=(0, 1)
        )

        diagnostics = probe_mpo_compression(
            exact,
            perturbed,
            product_states=(),
            random_state_count=1,
            random_bond=2,
            seed=31,
        )

        probe = diagnostics["probes"][0]
        self.assertNotEqual(probe["relative_action_error"], 0.0)
        if probe["status"] == "tested":
            self.assertGreater(probe["relative_action_error"], 0.5)
        else:
            self.assertEqual(probe["status"], "numerically_unresolved")
            self.assertIsNone(probe["relative_action_error"])
            self.assertGreater(probe["relative_action_error_upper_bound"], 0.5)

    def test_single_site_random_probe_uses_seeded_local_state_without_unitary_evolution(self) -> None:
        from tenpy.networks.mps import MPS

        exact, _ = build_exact_pauli_mpo([("I", 0.7)], n_qubits=1, order=(0,))
        identical, compression = compress_mpo_hilbert_schmidt(
            exact, max_bond=2, cutoff=0.0
        )
        self.assertEqual(compression["status"], "ok")

        with mock.patch.object(
            MPS,
            "from_random_unitary_evolution",
            side_effect=AssertionError("single-site random-unitary evolution was called"),
        ) as unitary_evolution:
            diagnostics = probe_mpo_compression(
                exact,
                identical,
                product_states=(),
                random_state_count=1,
                random_bond=4,
                seed=23,
            )

        unitary_evolution.assert_not_called()
        probe = diagnostics["probes"][0]
        self.assertEqual(probe["status"], "tested")
        self.assertEqual(probe["relative_action_error"], 0.0)

    def test_mutated_exact_identity_certificate_never_reports_zero(self) -> None:
        exact, _ = build_exact_pauli_mpo(
            [("XI", 0.7), ("IZ", -0.2)], n_qubits=2, order=(0, 1)
        )
        compressed, compression = compress_mpo_hilbert_schmidt(
            exact, max_bond=8, cutoff=0.0
        )
        self.assertEqual(compression["status"], "ok")
        self.assertEqual(
            probe_mpo_compression(
                exact,
                compressed,
                product_states=(("up", "down"),),
                random_state_count=0,
            )["probes"][0]["relative_action_error"],
            0.0,
        )

        mutated_tensor = compressed.get_W(0).copy()
        mutated_tensor *= 2.0
        compressed.set_W(0, mutated_tensor)
        diagnostics = probe_mpo_compression(
            exact,
            compressed,
            product_states=(("up", "down"),),
            random_state_count=0,
        )

        probe = diagnostics["probes"][0]
        self.assertNotEqual(probe["relative_action_error"], 0.0)
        self.assertEqual(diagnostics["exact_identity_certificate_status"], "invalidated")
        if probe["status"] == "tested":
            self.assertAlmostEqual(probe["relative_action_error"], 1.0, places=12)
        else:
            self.assertEqual(probe["status"], "numerically_unresolved")
            self.assertGreaterEqual(probe["relative_action_error_upper_bound"], 1.0)

    def test_product_unresolved_bounds_cover_adversarial_dense_error(self) -> None:
        exact, _ = build_exact_pauli_mpo([("I", 1.0)], n_qubits=1, order=(0,))
        perturbed, _ = build_exact_pauli_mpo(
            [("I", 1.0), ("X", 1.0e-8)], n_qubits=1, order=(0,)
        )
        dense_error = np.linalg.norm(
            (mpo_to_dense(perturbed) - mpo_to_dense(exact)) @ np.asarray([1.0, 0.0])
        )

        probe = probe_mpo_compression(
            exact,
            perturbed,
            product_states=(("up",),),
            random_state_count=0,
        )["probes"][0]

        self.assertEqual(probe["status"], "numerically_unresolved")
        self.assertLessEqual(probe["relative_action_error_lower_bound"], dense_error)
        self.assertGreaterEqual(probe["relative_action_error_upper_bound"], dense_error)
        self.assertGreater(probe["roundoff_operation_estimate"], 0)
        self.assertGreater(probe["squared_difference_arithmetic_uncertainty"], 0.0)

    def test_product_probe_stably_aggregates_conditioned_pauli_paths(self) -> None:
        for scale in (1.0, 1.0e-10, 1.0e10):
            exact_terms = [
                ("II", 1.0e16 * scale),
                ("IZ", scale),
                ("ZI", -1.0e16 * scale),
                ("ZZ", scale),
            ]
            candidate_terms = exact_terms + [("XI", 1.0e-8 * scale)]
            exact, _ = build_exact_pauli_mpo(exact_terms, n_qubits=2, order=(0, 1))
            candidate, _ = build_exact_pauli_mpo(
                candidate_terms, n_qubits=2, order=(0, 1)
            )
            input_state = np.asarray([1.0, 0.0, 0.0, 0.0])
            exact_dense = mpo_to_dense(exact)
            dense_relative_error = np.linalg.norm(
                (mpo_to_dense(candidate) - exact_dense) @ input_state
            ) / np.linalg.norm(exact_dense @ input_state)
            self.assertAlmostEqual(dense_relative_error, 5.0e-9, places=20)

            probe = probe_mpo_compression(
                exact,
                candidate,
                product_states=(("up", "up"),),
                random_state_count=0,
            )["probes"][0]

            if probe["status"] == "tested":
                self.assertAlmostEqual(
                    probe["relative_action_error"], dense_relative_error, places=12
                )
            else:
                self.assertEqual(probe["status"], "numerically_unresolved")
                self.assertLessEqual(
                    probe["relative_action_error_lower_bound"], dense_relative_error
                )
                self.assertGreaterEqual(
                    probe["relative_action_error_upper_bound"], dense_relative_error
                )
                self.assertLess(probe["relative_action_error_upper_bound"], 1.0e-5)
            self.assertEqual(probe["exact_action_aggregation_method"], "math.fsum_components")
            self.assertGreater(probe["exact_action_aggregation_condition_number"], 1.0e15)
            self.assertGreater(probe["exact_action_aggregation_absolute_uncertainty"], 0.0)

    def test_random_unresolved_bounds_cover_adversarial_dense_error(self) -> None:
        from tenpy.networks.mps import MPS

        exact, _ = build_exact_pauli_mpo([("II", 1.0)], n_qubits=2, order=(0, 1))
        perturbed, _ = build_exact_pauli_mpo(
            [("II", 1.0), ("XI", 1.0e-8)], n_qubits=2, order=(0, 1)
        )
        random_state = np.random.get_state()
        try:
            np.random.seed(37)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                state = MPS.from_random_unitary_evolution(
                    exact.sites,
                    chi=2,
                    p_state=["up"] * exact.L,
                    bc="finite",
                    dtype=np.complex128,
                )
        finally:
            np.random.set_state(random_state)
        state_vector = state.get_theta(0, exact.L).to_ndarray().reshape(-1)
        dense_error = np.linalg.norm(
            (mpo_to_dense(perturbed) - mpo_to_dense(exact)) @ state_vector
        )

        probe = probe_mpo_compression(
            exact,
            perturbed,
            product_states=(),
            random_state_count=1,
            random_bond=2,
            seed=37,
        )["probes"][0]

        self.assertEqual(probe["status"], "numerically_unresolved")
        self.assertLessEqual(probe["relative_action_error_lower_bound"], dense_error)
        self.assertGreaterEqual(probe["relative_action_error_upper_bound"], dense_error)
        self.assertGreater(probe["roundoff_operation_estimate"], 0)
        self.assertGreater(probe["squared_difference_arithmetic_uncertainty"], 0.0)

    def test_zero_action_denominator_is_not_tested(self) -> None:
        zero, _ = build_exact_pauli_mpo(
            [("X", 1.0), ("X", -1.0)], n_qubits=1, order=(0,)
        )
        compressed, _ = compress_mpo_hilbert_schmidt(zero, max_bond=2, cutoff=0.0)

        diagnostics = probe_mpo_compression(
            zero,
            compressed,
            product_states=(("up",),),
            random_state_count=0,
            seed=11,
        )

        self.assertEqual(diagnostics["tested_probes"], 0)
        self.assertEqual(diagnostics["not_tested_probes"], 1)
        self.assertIsNone(diagnostics["max_relative_action_error"])
        self.assertEqual(diagnostics["probes"][0]["status"], "not_tested")
        self.assertIsNone(diagnostics["probes"][0]["relative_action_error"])

    def test_cancellation_conditioned_zero_product_action_is_not_tested(self) -> None:
        exact, _ = build_exact_pauli_mpo(
            [("I", 1.0), ("Z", -1.0)], n_qubits=1, order=(0,)
        )

        probe = probe_mpo_compression(
            exact,
            exact,
            product_states=(("up",),),
            random_state_count=0,
        )["probes"][0]

        self.assertEqual(probe["status"], "not_tested")
        self.assertEqual(probe["reason"], "zero_action_denominator")
        self.assertIsNone(probe["relative_action_error"])
        self.assertIsNone(probe["relative_action_error_lower_bound"])
        self.assertIsNone(probe["relative_action_error_upper_bound"])
        self.assertIsNone(probe["relative_action_error_numerical_floor"])

    def test_compression_rejects_invalid_resource_limits(self) -> None:
        exact, _ = build_exact_pauli_mpo([("X", 1.0)], n_qubits=1, order=(0,))

        for max_bond in (0, True, 1.5):
            with self.assertRaisesRegex(ValueError, "max_bond"):
                compress_mpo_hilbert_schmidt(exact, max_bond=max_bond, cutoff=0.0)
        for cutoff in (-1.0, 1.0, np.nan):
            with self.assertRaisesRegex(ValueError, "cutoff"):
                compress_mpo_hilbert_schmidt(exact, max_bond=2, cutoff=cutoff)


@unittest.skipUnless(_TENPY_AVAILABLE, "TeNPy tensor-network extra is not installed")
class TDVPEvolutionTests(unittest.TestCase):
    def _single_qubit_settings(self) -> dict[str, object]:
        return {
            "h0_terms": [("X", -1.0)],
            "h1_terms": [("Z", -1.0)],
            "cd_factorization": None,
            "total_time": 1.0,
            "steps": 64,
            "mps_max_bond": 8,
            "mps_cutoff": 1.0e-13,
            "mpo_max_bond": 16,
            "mpo_cutoff": 1.0e-13,
        }

    def test_tdvp_no_cd_matches_dense_midpoint_evolution(self) -> None:
        settings = self._single_qubit_settings()
        state, diagnostics = mpo_backend.evolve_protocol_tdvp(**settings)
        reference = mpo_backend.dense_midpoint_evolution(
            settings["h0_terms"],
            settings["h1_terms"],
            total_time=1.0,
            steps=4096,
        )

        self.assertGreater(
            abs(mpo_backend.state_overlap_dense(state, reference)) ** 2,
            1.0 - 1.0e-7,
        )
        self.assertEqual(diagnostics["steps"], 64)
        self.assertEqual(diagnostics["completed_steps"], 64)
        self.assertEqual(diagnostics["integrator"], "tdvp")
        self.assertEqual(diagnostics["tdvp_engine"], "single_site_tdvp_l1")
        self.assertEqual(diagnostics["status"], "ok")
        for key in (
            "norm_drift",
            "truncation_error",
            "peak_mps_bond",
            "final_mps_bond",
            "static_mpo_bonds",
            "dynamic_mpo_peak_bond",
            "operator_build_seconds",
            "evolution_seconds",
            "resource_statuses",
        ):
            self.assertIn(key, diagnostics)

    def test_prepare_tdvp_operators_includes_every_mode_and_term(self) -> None:
        result = mpo_backend.prepare_tdvp_operators(
            labels=("XI", "YI", "XZ"),
            static_modes=np.asarray([[1.0, 2.0, 3.0], [0.5, 0.25, -0.1]]),
            temporal_factors=np.ones((5, 2)),
            n_qubits=2,
            order=(0, 1),
            mpo_max_bond=16,
            mpo_cutoff=1.0e-13,
        )

        self.assertEqual(result.diagnostics["learned_input_terms"], 3)
        self.assertEqual(result.diagnostics["temporal_rank"], 2)
        self.assertEqual(result.diagnostics["support_fraction"], 1.0)
        self.assertEqual(result.diagnostics["full_support_status"], "pass")
        self.assertEqual(len(result.cd_mode_mpos), 2)
        self.assertEqual(
            len(result.diagnostics["static_mpo_compression"]["cd_modes"]),
            2,
        )
        self.assertTrue(
            all(
                item["status"] == "ok"
                for item in result.diagnostics["static_mpo_compression"]["cd_modes"]
            )
        )

    def test_expm_mpo_matches_tdvp_on_one_and_two_qubits(self) -> None:
        cases = (
            self._single_qubit_settings(),
            {
                "h0_terms": [("XI", -1.0), ("IX", -1.0)],
                "h1_terms": [("ZI", -0.7), ("IZ", -1.1), ("ZZ", 0.2)],
                "cd_factorization": None,
                "total_time": 0.5,
                "steps": 48,
                "mps_max_bond": 8,
                "mps_cutoff": 1.0e-13,
                "mpo_max_bond": 16,
                "mpo_cutoff": 1.0e-13,
            },
        )

        for settings in cases:
            with self.subTest(n_qubits=len(settings["h0_terms"][0][0])):
                tdvp_state, tdvp_diagnostics = mpo_backend.evolve_protocol_tdvp(
                    **settings
                )
                expm_state, expm_diagnostics = mpo_backend.evolve_protocol_expm_mpo(
                    **settings
                )
                self.assertGreater(
                    abs(tdvp_state.overlap(expm_state)) ** 2,
                    1.0 - 2.0e-6,
                )
                self.assertEqual(tdvp_diagnostics["status"], "ok")
                self.assertEqual(expm_diagnostics["status"], "ok")
                self.assertEqual(expm_diagnostics["integrator"], "expm_mpo")

    def test_learned_and_nested_l1_paths_match_dense_midpoint_evolution(self) -> None:
        h0 = [("XI", -1.0), ("IX", -0.8)]
        h1 = [("ZI", -0.7), ("IZ", -1.1), ("ZZ", 0.15)]
        tau = np.linspace(0.0, 1.0, 65)
        direct_cd = np.stack(
            [0.18 * np.sin(np.pi * tau), -0.11 * np.sin(2.0 * np.pi * tau)],
            axis=1,
        )
        factorization = factor_direct_cd_coefficients(
            tau,
            direct_cd,
            retained_norm=1.0 - 1.0e-14,
        )
        common = {
            "h0_terms": h0,
            "h1_terms": h1,
            "total_time": 0.6,
            "steps": 64,
            "mps_max_bond": 8,
            "mps_cutoff": 1.0e-13,
            "mpo_max_bond": 32,
            "mpo_cutoff": 1.0e-13,
        }

        learned_state, learned_diagnostics = mpo_backend.evolve_protocol_tdvp(
            **common,
            cd_labels=("YI", "IY"),
            cd_factorization=factorization,
            protocol="learned",
        )
        learned_reference = mpo_backend.dense_midpoint_evolution(
            h0,
            h1,
            total_time=0.6,
            steps=4096,
            cd_labels=("YI", "IY"),
            cd_factorization=factorization,
            protocol="learned",
        )
        nested_state, nested_diagnostics = mpo_backend.evolve_protocol_tdvp(
            **common,
            cd_factorization=None,
            protocol="nested_l1",
        )
        nested_reference = mpo_backend.dense_midpoint_evolution(
            h0,
            h1,
            total_time=0.6,
            steps=4096,
            protocol="nested_l1",
        )

        self.assertGreater(
            abs(mpo_backend.state_overlap_dense(learned_state, learned_reference)) ** 2,
            1.0 - 2.0e-6,
        )
        self.assertGreater(
            abs(mpo_backend.state_overlap_dense(nested_state, nested_reference)) ** 2,
            1.0 - 2.0e-6,
        )
        self.assertEqual(learned_diagnostics["evaluated_cd_terms"], 2)
        self.assertEqual(learned_diagnostics["temporal_rank"], 2)
        self.assertEqual(nested_diagnostics["protocol"], "nested_l1")
        self.assertGreater(nested_diagnostics["evaluated_cd_terms"], 0)

    def test_schedule_factors_initial_state_and_ground_bits_use_midpoint_order(self) -> None:
        schedule_calls: list[float] = []

        def linear_schedule(tau: float, total_time: float) -> tuple[float, float]:
            schedule_calls.append(tau)
            return tau, 1.0 / total_time

        state, diagnostics = mpo_backend.evolve_protocol_tdvp(
            h0_terms=[("XI", -1.0), ("IZ", -0.25)],
            h1_terms=[("ZI", -0.8), ("IZ", 0.4)],
            cd_factorization=None,
            protocol="no_cd",
            schedule=linear_schedule,
            total_time=0.2,
            steps=4,
            order=(1, 0),
            initial_state=("0", "+"),
            ground_bitstring="10",
            mps_max_bond=8,
            mps_cutoff=1.0e-13,
            mpo_max_bond=16,
            mpo_cutoff=1.0e-13,
        )

        self.assertEqual(schedule_calls, [0.125, 0.375, 0.625, 0.875])
        self.assertEqual(diagnostics["midpoint_lambdas"], schedule_calls)
        self.assertEqual(diagnostics["initial_state_original"], ["0", "+"])
        self.assertEqual(diagnostics["initial_state_chain"], ["+", "0"])
        self.assertEqual(diagnostics["ground_bitstring_original"], "10")
        self.assertEqual(diagnostics["ground_bitstring_chain"], "01")
        self.assertEqual(getattr(state, "_agp_qubit_order"), (1, 0))
        self.assertEqual(diagnostics["ground_fidelity_status"], "ok")

    def test_resource_failures_are_explicit_and_do_not_fallback(self) -> None:
        state, diagnostics = mpo_backend.evolve_protocol_tdvp(
            **self._single_qubit_settings(),
            mpo_workspace_cap_bytes=1,
        )

        self.assertEqual(diagnostics["status"], "not_feasible")
        self.assertEqual(diagnostics["completed_steps"], 0)
        self.assertEqual(
            diagnostics["resource_statuses"]["static_mpo_compression"],
            "not_feasible",
        )
        self.assertIsNotNone(state)

    def test_dense_evolution_helpers_reject_more_than_four_qubits(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most four qubits"):
            mpo_backend.dense_midpoint_evolution(
                [("XIIII", -1.0)],
                [("ZIIII", -1.0)],
                steps=2,
            )


class LazyOptionalImportTests(unittest.TestCase):
    def test_module_imports_without_tenpy_and_mpo_call_fails_with_install_hint(self) -> None:
        root = Path(__file__).resolve().parents[1]
        code = """
import builtins
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'tenpy' or name.startswith('tenpy.'):
        raise ModuleNotFoundError("blocked optional dependency")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import scripts.agp_mpo_backend as backend
try:
    backend.build_exact_pauli_mpo([('X', 1.0)], n_qubits=1, order=(0,))
except ModuleNotFoundError as error:
    assert 'tensor-network' in str(error)
else:
    raise AssertionError('MPO construction did not require the optional dependency')
"""

        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
