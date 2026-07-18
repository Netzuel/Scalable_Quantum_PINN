"""Problem-independent regression matrix for full-support TN validation."""

from __future__ import annotations

from functools import lru_cache
from itertools import product
import importlib.util
import unittest

import numpy as np

from scripts.agp_mpo_backend import (
    build_full_support_identity,
    build_time_pauli_tensor_train,
    evolve_protocol_time_tensor_tdvp,
    mpo_to_dense,
    slice_time_pauli_mpo,
)


_TENPY_AVAILABLE = importlib.util.find_spec("tenpy") is not None
_PAULI = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.asarray(((0.0, 1.0), (1.0, 0.0)), dtype=np.complex128),
    "Y": np.asarray(((0.0, -1.0j), (1.0j, 0.0)), dtype=np.complex128),
    "Z": np.asarray(((1.0, 0.0), (0.0, -1.0)), dtype=np.complex128),
}


@lru_cache(maxsize=None)
def _dense_pauli(label: str) -> np.ndarray:
    matrix = np.asarray([[1.0]], dtype=np.complex128)
    for symbol in label:
        matrix = np.kron(matrix, _PAULI[symbol])
    return matrix


def _dense_sum(terms: tuple[tuple[str, complex], ...], order: tuple[int, ...]) -> np.ndarray:
    size = 2 ** len(order)
    matrix = np.zeros((size, size), dtype=np.complex128)
    for label, coefficient in terms:
        matrix += complex(coefficient) * _dense_pauli(
            "".join(label[index] for index in order)
        )
    return matrix


def _plus_state(n_qubits: int) -> np.ndarray:
    state = np.asarray([1.0], dtype=np.complex128)
    plus = np.asarray((1.0, 1.0), dtype=np.complex128) / np.sqrt(2.0)
    for _ in range(n_qubits):
        state = np.kron(state, plus)
    return state


def _mps_vector(state: object) -> np.ndarray:
    theta = state.get_theta(0, int(state.L)).to_ndarray()
    return complex(state.norm) * np.asarray(theta)[0, ..., 0].reshape(-1)


def _dense_midpoint_evolution(
    *,
    h0_terms: tuple[tuple[str, complex], ...],
    h1_terms: tuple[tuple[str, complex], ...],
    labels: tuple[str, ...],
    tau_grid: np.ndarray,
    coefficients: np.ndarray,
    total_time: float,
    steps: int,
    order: tuple[int, ...],
) -> np.ndarray:
    state = _plus_state(len(order))
    h0 = _dense_sum(h0_terms, order)
    h1 = _dense_sum(h1_terms, order)
    ordered_paulis = tuple(
        _dense_pauli("".join(label[index] for index in order)) for label in labels
    )
    for step in range(steps):
        tau = (step + 0.5) / steps
        lam = float(np.sin(0.5 * np.pi * tau) ** 2)
        hamiltonian = (1.0 - lam) * h0 + lam * h1
        interpolated = np.asarray(
            [np.interp(tau, tau_grid, coefficients[:, index]) for index in range(len(labels))]
        )
        for coefficient, pauli in zip(interpolated, ordered_paulis, strict=True):
            hamiltonian += coefficient * pauli
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        state = (eigenvectors * np.exp(-1.0j * total_time / steps * eigenvalues)) @ (
            eigenvectors.conj().T @ state
        )
    return state


def _coefficient_samples(tau: np.ndarray, count: int) -> np.ndarray:
    index = np.arange(count, dtype=np.float64)
    first = np.where((index.astype(np.int64) % 2) == 0, 1.0, -1.0)
    first *= (1.0 + (7.0 * index) % 29.0) / 31.0
    second = np.where((index.astype(np.int64) % 3) == 0, -1.0, 1.0)
    second *= (1.0 + (11.0 * index) % 23.0) / 37.0
    return (
        np.sin(np.pi * tau)[:, None] * first[None, :]
        + 0.17 * np.sin(2.0 * np.pi * tau)[:, None] * second[None, :]
    )


def _z_dense_terms(n_qubits: int) -> tuple[tuple[str, complex], ...]:
    rows = []
    for mask in range(1, 2**n_qubits):
        label = "".join(
            "Z" if mask & (1 << (n_qubits - site - 1)) else "I"
            for site in range(n_qubits)
        )
        rows.append((label, (-1.0 if mask % 2 else 1.0) / (1.0 + mask)))
    return tuple(rows)


@unittest.skipUnless(_TENPY_AVAILABLE, "physics-tenpy is required")
class FullSupportTensorNetworkRegressionMatrix(unittest.TestCase):
    def test_exact_small_system_dynamics_across_topology_and_global_scale(self) -> None:
        dense_labels = tuple("".join(symbols) for symbols in product("IXYZ", repeat=4))
        cases = (
            {
                "name": "local_q2",
                "n_qubits": 2,
                "h0": (("XI", -1.0), ("IX", -0.8)),
                "h1": (("ZI", -0.7), ("IZ", -1.1), ("ZZ", -0.4)),
                "labels": ("YI", "IY", "YZ"),
                "order": (0, 1),
            },
            {
                "name": "long_range_q3",
                "n_qubits": 3,
                "h0": (("XII", -1.0), ("IXI", -0.9), ("IIX", -1.1)),
                "h1": (("ZII", -0.6), ("IIZ", -0.8), ("ZIZ", -0.5), ("ZZZ", 0.3)),
                "labels": ("YII", "IIY", "YZY", "XYZ"),
                "order": (2, 0, 1),
            },
            {
                "name": "dense_q4",
                "n_qubits": 4,
                "h0": tuple(
                    ("I" * site + "X" + "I" * (3 - site), -1.0)
                    for site in range(4)
                ),
                "h1": _z_dense_terms(4),
                "labels": dense_labels,
                "order": (3, 1, 0, 2),
            },
        )
        tau = np.linspace(0.0, 1.0, 9)
        steps = 32

        for case in cases:
            base_coefficients = _coefficient_samples(tau, len(case["labels"]))
            for scale in (1.0e-8, 1.0, 1.0e8):
                with self.subTest(case=case["name"], scale=scale):
                    h0 = tuple((label, scale * value) for label, value in case["h0"])
                    h1 = tuple((label, scale * value) for label, value in case["h1"])
                    coefficients = scale * base_coefficients
                    total_time = 0.3 / scale
                    state, diagnostics = evolve_protocol_time_tensor_tdvp(
                        h0_terms=h0,
                        h1_terms=h1,
                        learned_tau=tau,
                        learned_direct_cd_coefficients=coefficients,
                        learned_labels=case["labels"],
                        full_support_identity=build_full_support_identity(
                            case["labels"], coefficients
                        ),
                        total_time=total_time,
                        steps=steps,
                        order=case["order"],
                        initial_state=("+",) * case["n_qubits"],
                        mps_max_bond=16,
                        mps_cutoff=1.0e-13,
                        mpo_max_bond=64,
                        mpo_cutoff=0.0,
                        lanczos_max=30,
                        mpo_workspace_cap_bytes=128 * 1024**2,
                        coefficient_error_max=1.0e-10,
                        action_error_max=1.0e-9,
                        action_probe_product_states=2,
                        action_probe_time_samples=2,
                        action_probe_seed=17,
                        time_window_size=steps,
                        adaptive_time_windows=True,
                    )
                    reference = _dense_midpoint_evolution(
                        h0_terms=h0,
                        h1_terms=h1,
                        labels=case["labels"],
                        tau_grid=tau,
                        coefficients=coefficients,
                        total_time=total_time,
                        steps=steps,
                        order=case["order"],
                    )
                    actual = _mps_vector(state)
                    fidelity = abs(np.vdot(actual, reference)) ** 2
                    fidelity /= np.vdot(actual, actual).real * np.vdot(reference, reference).real

                    self.assertEqual(diagnostics["status"], "ok")
                    self.assertEqual(diagnostics["operator_gate_status"], "pass")
                    self.assertEqual(diagnostics["evaluated_cd_terms"], len(case["labels"]))
                    self.assertGreater(fidelity, 1.0 - 1.0e-6)

    def test_joint_operator_handles_large_internal_coefficient_dynamic_range(self) -> None:
        labels = tuple("".join(symbols) for symbols in product("IXYZ", repeat=3))
        tau = np.asarray((0.2, 0.5, 0.8))
        magnitudes = np.logspace(-8, 8, len(labels))
        coefficients = np.sin(np.pi * tau)[:, None] * magnitudes[None, :]
        train, diagnostics = build_time_pauli_tensor_train(
            h0_terms=(("XII", -1.0),),
            h1_terms=(("ZZZ", 0.7),),
            learned_labels=labels,
            direct_cd_coefficients=coefficients,
            lambda_samples=np.asarray((0.1, 0.5, 0.9)),
            n_qubits=3,
            order=(2, 0, 1),
            max_bond=64,
            cutoff=0.0,
            workspace_cap_bytes=64 * 1024**2,
            full_support_identity=build_full_support_identity(labels, coefficients),
            identity_coefficient_samples=coefficients,
        )

        self.assertIsNotNone(train)
        self.assertEqual(diagnostics["source_completeness_status"], "pass")
        for sample in range(tau.size):
            expected_terms = [
                ("XII", -(1.0 - (0.1 + 0.4 * sample))),
                ("ZZZ", 0.7 * (0.1 + 0.4 * sample)),
                *zip(labels, coefficients[sample], strict=True),
            ]
            expected = _dense_sum(tuple(expected_terms), (2, 0, 1))
            actual = mpo_to_dense(slice_time_pauli_mpo(train, sample))
            relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
            self.assertLess(relative_error, 1.0e-9)

    def test_centered_time_axis_avoids_time_first_rank_inflation(self) -> None:
        labels = tuple("".join(symbols) for symbols in product("IXYZ", repeat=4))
        rng = np.random.default_rng(91)
        coefficients = rng.normal(size=(16, len(labels)))
        identity = build_full_support_identity(labels, coefficients)
        common = {
            "h0_terms": (("IIII", 0.0),),
            "h1_terms": (("IIII", 0.0),),
            "learned_labels": labels,
            "direct_cd_coefficients": coefficients,
            "lambda_samples": np.linspace(0.0, 1.0, 16),
            "n_qubits": 4,
            "order": (0, 1, 2, 3),
            "max_bond": 16,
            "cutoff": 0.0,
            "workspace_cap_bytes": 128 * 1024**2,
            "full_support_identity": identity,
            "identity_coefficient_samples": coefficients,
        }

        _, time_first = build_time_pauli_tensor_train(
            **common,
            time_axis_position=0,
        )
        centered, centered_diagnostics = build_time_pauli_tensor_train(
            **common,
            time_axis_position=2,
        )

        self.assertGreater(
            time_first["compression"]["max_relative_coefficient_error_upper_bound"],
            1.0e-3,
        )
        self.assertIsNotNone(centered)
        self.assertEqual(centered_diagnostics["time_axis_position"], 2)
        self.assertLess(
            centered_diagnostics["compression"][
                "max_relative_coefficient_error_upper_bound"
            ],
            1.0e-10,
        )
        for sample in (0, 7, 15):
            expected = _dense_sum(
                tuple(zip(labels, coefficients[sample], strict=True)),
                (0, 1, 2, 3),
            )
            np.testing.assert_allclose(
                mpo_to_dense(slice_time_pauli_mpo(centered, sample)),
                expected,
                rtol=1.0e-10,
                atol=1.0e-10,
            )

    def test_rank_cap_failure_is_fail_closed(self) -> None:
        labels = tuple("".join(symbols) for symbols in product("IXYZ", repeat=4))
        tau = np.linspace(0.0, 1.0, 5)
        coefficients = _coefficient_samples(tau, len(labels))
        _, diagnostics = evolve_protocol_time_tensor_tdvp(
            h0_terms=(("XIII", -1.0),),
            h1_terms=(("ZZZZ", -1.0),),
            learned_tau=tau,
            learned_direct_cd_coefficients=coefficients,
            learned_labels=labels,
            full_support_identity=build_full_support_identity(labels, coefficients),
            steps=2,
            mpo_max_bond=1,
            mpo_cutoff=0.0,
            mps_max_bond=4,
            mps_cutoff=1.0e-12,
            mpo_workspace_cap_bytes=64 * 1024**2,
            coefficient_error_max=1.0e-14,
            action_error_max=1.0e-14,
            action_probe_product_states=1,
            action_probe_time_samples=1,
            time_window_size=2,
            adaptive_time_windows=True,
        )

        self.assertEqual(diagnostics["status"], "not_feasible")
        self.assertEqual(diagnostics["operator_gate_status"], "fail")
        self.assertLess(diagnostics["completed_steps"], 2)
        self.assertIsNone(diagnostics["final_energy"])
        self.assertIsNone(diagnostics["ground_fidelity"])


if __name__ == "__main__":
    unittest.main()
