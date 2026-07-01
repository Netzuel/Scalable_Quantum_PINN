"""Symbolic Pauli utilities recovered for full-coefficient AGP training.

This module intentionally exposes coefficient-space Pauli helpers only. Dense
``2**q x 2**q`` operators from the older QFI code are not included here.
"""

from __future__ import annotations

from utils import (
    PAULI_ALPHABET,
    SparseRightCommutator,
    all_local_pauli_labels,
    all_pauli_labels,
    commutator_pauli_labels,
    fixed_sinusoidal_schedule,
    multiply_pauli_labels,
    pauli_weight,
)


def build_full_pauli_basis(num_qubits: int) -> list[str]:
    """Return all ``4**num_qubits`` Pauli strings, including identity."""

    return all_pauli_labels(num_qubits)


def build_k_local_pauli_basis(num_qubits: int, max_weight: int) -> list[str]:
    """Return the k-local Pauli basis, including the identity string."""

    return all_local_pauli_labels(num_qubits, max_weight=max_weight, include_identity=True)
