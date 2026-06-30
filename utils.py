"""Sparse Pauli algebra utilities for scalable AGP PINNs.

The original Quantum_PINN repository learned dense matrices for the
counterdiabatic operator. This module keeps the same physics target but works
with sparse Pauli-coordinate operators, so memory and output size scale with
the selected ansatz support rather than with the full Hilbert-space matrix.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterable, Mapping, Sequence

import torch

PAULI_ALPHABET = ("I", "X", "Y", "Z")

_MUL_TABLE: dict[tuple[str, str], tuple[complex, str]] = {
    ("I", "I"): (1.0 + 0.0j, "I"),
    ("I", "X"): (1.0 + 0.0j, "X"),
    ("I", "Y"): (1.0 + 0.0j, "Y"),
    ("I", "Z"): (1.0 + 0.0j, "Z"),
    ("X", "I"): (1.0 + 0.0j, "X"),
    ("Y", "I"): (1.0 + 0.0j, "Y"),
    ("Z", "I"): (1.0 + 0.0j, "Z"),
    ("X", "X"): (1.0 + 0.0j, "I"),
    ("Y", "Y"): (1.0 + 0.0j, "I"),
    ("Z", "Z"): (1.0 + 0.0j, "I"),
    ("X", "Y"): (0.0 + 1.0j, "Z"),
    ("Y", "Z"): (0.0 + 1.0j, "X"),
    ("Z", "X"): (0.0 + 1.0j, "Y"),
    ("Y", "X"): (0.0 - 1.0j, "Z"),
    ("Z", "Y"): (0.0 - 1.0j, "X"),
    ("X", "Z"): (0.0 - 1.0j, "Y"),
}


def validate_pauli_label(label: str, n_qubits: int | None = None) -> str:
    """Validate and normalize a Pauli-string label such as ``IXYZ``."""

    if not isinstance(label, str) or not label:
        raise ValueError("Pauli labels must be non-empty strings.")
    label = label.upper()
    invalid = sorted(set(label) - set(PAULI_ALPHABET))
    if invalid:
        raise ValueError(f"Invalid Pauli symbols {invalid} in label {label!r}.")
    if n_qubits is not None and len(label) != n_qubits:
        raise ValueError(f"Expected {n_qubits} qubits, got label {label!r}.")
    return label


def pauli_weight(label: str) -> int:
    """Return the number of non-identity factors in a Pauli label."""

    return sum(symbol != "I" for symbol in validate_pauli_label(label))


def infer_n_qubits(labels: Iterable[str]) -> int:
    """Infer a common qubit count from labels."""

    labels = [validate_pauli_label(label) for label in labels]
    if not labels:
        raise ValueError("Cannot infer qubit count from an empty label set.")
    lengths = {len(label) for label in labels}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent Pauli-label lengths: {sorted(lengths)}")
    return lengths.pop()


def multiply_pauli_labels(left: str, right: str) -> tuple[complex, str]:
    """Multiply two Pauli strings and return ``(phase, label)``."""

    left = validate_pauli_label(left)
    right = validate_pauli_label(right, len(left))
    phase = 1.0 + 0.0j
    out: list[str] = []
    for l_symbol, r_symbol in zip(left, right):
        local_phase, local_label = _MUL_TABLE[(l_symbol, r_symbol)]
        phase *= local_phase
        out.append(local_label)
    return phase, "".join(out)


def commutator_pauli_labels(left: str, right: str) -> tuple[complex, str] | None:
    """Return the Pauli string and phase for ``[left, right]``.

    ``None`` means the two strings commute.
    """

    phase_lr, label_lr = multiply_pauli_labels(left, right)
    phase_rl, label_rl = multiply_pauli_labels(right, left)
    if label_lr != label_rl:
        raise RuntimeError("Pauli multiplication produced inconsistent labels.")
    phase = phase_lr - phase_rl
    if abs(phase) < 1e-14:
        return None
    return phase, label_lr


def sort_pauli_labels(labels: Iterable[str]) -> list[str]:
    """Sort labels by locality first, then lexicographically."""

    return sorted({validate_pauli_label(label) for label in labels}, key=lambda x: (pauli_weight(x), x))


@dataclass(frozen=True)
class SparsePauliOperator:
    """Sparse Pauli-coordinate operator.

    The coefficients may be complex. Hermitian Hamiltonians normally use real
    coefficients; commutators naturally generate imaginary coefficients.
    """

    terms: Mapping[str, complex]
    n_qubits: int | None = None
    drop_tol: float = 1e-14

    def __post_init__(self) -> None:
        labels = list(self.terms)
        n_qubits = self.n_qubits if self.n_qubits is not None else infer_n_qubits(labels)
        clean_terms: dict[str, complex] = {}
        for label, coeff in self.terms.items():
            label = validate_pauli_label(label, n_qubits)
            coeff = complex(coeff)
            if abs(coeff) > self.drop_tol:
                clean_terms[label] = coeff
        object.__setattr__(self, "terms", clean_terms)
        object.__setattr__(self, "n_qubits", n_qubits)

    @classmethod
    def zero(cls, n_qubits: int) -> "SparsePauliOperator":
        return cls({}, n_qubits=n_qubits)

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, complex]]) -> "SparsePauliOperator":
        accumulator: defaultdict[str, complex] = defaultdict(complex)
        labels: list[str] = []
        for label, coeff in pairs:
            label = validate_pauli_label(label)
            labels.append(label)
            accumulator[label] += complex(coeff)
        return cls(dict(accumulator), n_qubits=infer_n_qubits(labels))

    @property
    def labels(self) -> list[str]:
        return sort_pauli_labels(self.terms)

    def coefficient(self, label: str) -> complex:
        return self.terms.get(validate_pauli_label(label, self.n_qubits), 0.0 + 0.0j)

    def scale(self, scalar: complex) -> "SparsePauliOperator":
        return SparsePauliOperator(
            {label: scalar * coeff for label, coeff in self.terms.items()},
            n_qubits=self.n_qubits,
        )

    def add(self, other: "SparsePauliOperator", scale: complex = 1.0) -> "SparsePauliOperator":
        if self.n_qubits != other.n_qubits:
            raise ValueError("Cannot add operators with different qubit counts.")
        terms: defaultdict[str, complex] = defaultdict(complex)
        terms.update(self.terms)
        for label, coeff in other.terms.items():
            terms[label] += scale * coeff
        return SparsePauliOperator(dict(terms), n_qubits=self.n_qubits)

    def __add__(self, other: "SparsePauliOperator") -> "SparsePauliOperator":
        return self.add(other)

    def __sub__(self, other: "SparsePauliOperator") -> "SparsePauliOperator":
        return self.add(other, scale=-1.0)

    def commutator(self, other: "SparsePauliOperator") -> "SparsePauliOperator":
        if self.n_qubits != other.n_qubits:
            raise ValueError("Cannot commute operators with different qubit counts.")
        terms: defaultdict[str, complex] = defaultdict(complex)
        for left_label, left_coeff in self.terms.items():
            for right_label, right_coeff in other.terms.items():
                item = commutator_pauli_labels(left_label, right_label)
                if item is None:
                    continue
                phase, out_label = item
                terms[out_label] += left_coeff * right_coeff * phase
        return SparsePauliOperator(dict(terms), n_qubits=self.n_qubits)

    def l2_norm_sq(self) -> float:
        return float(sum(abs(coeff) ** 2 for coeff in self.terms.values()))

    def to_vector(
        self,
        basis_labels: Sequence[str],
        *,
        dtype: torch.dtype = torch.complex64,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        basis_labels = [validate_pauli_label(label, self.n_qubits) for label in basis_labels]
        vector = torch.zeros(len(basis_labels), dtype=dtype, device=device)
        for idx, label in enumerate(basis_labels):
            vector[idx] = self.coefficient(label)
        return vector


def interpolate_operator(
    initial: SparsePauliOperator,
    final: SparsePauliOperator,
    lam: float,
) -> SparsePauliOperator:
    """Return ``(1 - lam) initial + lam final`` in sparse Pauli form."""

    return initial.scale(1.0 - lam).add(final, scale=lam)


def exact_agp_residual_operator(
    initial: SparsePauliOperator,
    final: SparsePauliOperator,
    agp: SparsePauliOperator,
    lam: float,
) -> SparsePauliOperator:
    """Euler-Lagrange residual ``[i dH/dlambda - [A,H], H]``.

    This is exact inside the supplied sparse Pauli support. It does not build
    dense Hilbert-space matrices.
    """

    h_ad = interpolate_operator(initial, final, lam)
    d_h = final - initial
    generator = d_h.scale(1.0j) - agp.commutator(h_ad)
    return generator.commutator(h_ad)


def all_local_pauli_labels(
    n_qubits: int,
    max_weight: int,
    *,
    alphabet: Sequence[str] = ("X", "Y", "Z"),
    include_identity: bool = False,
) -> list[str]:
    """Generate all Pauli strings up to a chosen locality/weight."""

    if max_weight < 0:
        raise ValueError("max_weight must be non-negative.")
    labels: set[str] = {"I" * n_qubits} if include_identity else set()
    for weight in range(1, max_weight + 1):
        for sites in combinations(range(n_qubits), weight):
            for local_symbols in product(alphabet, repeat=weight):
                chars = ["I"] * n_qubits
                for site, symbol in zip(sites, local_symbols):
                    chars[site] = symbol
                labels.add("".join(chars))
    return sort_pauli_labels(labels)


def transverse_field_ising_problem(
    n_qubits: int,
    *,
    field: float = 1.0,
    coupling: float = 1.0,
    periodic: bool = False,
) -> tuple[SparsePauliOperator, SparsePauliOperator]:
    """Return a common scalable benchmark ``H0=-sum X_i``, ``H1=-sum Z_i Z_j``."""

    if n_qubits < 2:
        raise ValueError("Use at least two qubits for an Ising-chain problem.")
    h0_terms: dict[str, complex] = {}
    h1_terms: dict[str, complex] = {}
    for site in range(n_qubits):
        label = ["I"] * n_qubits
        label[site] = "X"
        h0_terms["".join(label)] = -float(field)
    edges = [(site, site + 1) for site in range(n_qubits - 1)]
    if periodic and n_qubits > 2:
        edges.append((n_qubits - 1, 0))
    for left, right in edges:
        label = ["I"] * n_qubits
        label[left] = "Z"
        label[right] = "Z"
        h1_terms["".join(label)] = -float(coupling)
    return SparsePauliOperator(h0_terms, n_qubits), SparsePauliOperator(h1_terms, n_qubits)


def build_commutator_closure(
    seed_labels: Iterable[str],
    *,
    max_rounds: int = 2,
    max_weight: int | None = None,
    max_terms: int = 20000,
) -> list[str]:
    """Expand labels under pairwise commutators for a fixed number of rounds."""

    labels = set(sort_pauli_labels(seed_labels))
    n_qubits = infer_n_qubits(labels)
    for _ in range(max_rounds):
        additions: set[str] = set()
        current = list(labels)
        for left in current:
            for right in current:
                item = commutator_pauli_labels(left, right)
                if item is None:
                    continue
                _, out_label = item
                if max_weight is not None and pauli_weight(out_label) > max_weight:
                    continue
                validate_pauli_label(out_label, n_qubits)
                additions.add(out_label)
                if len(labels) + len(additions) > max_terms:
                    raise RuntimeError(
                        "Commutator closure exceeded max_terms. Reduce the ansatz, "
                        "lower max_weight, or raise max_terms deliberately."
                    )
        old_size = len(labels)
        labels.update(additions)
        if len(labels) == old_size:
            break
    return sort_pauli_labels(labels)


class PauliAlgebra:
    """Differentiable Pauli algebra over a fixed sparse operator basis."""

    def __init__(self, basis_labels: Sequence[str]):
        self.basis_labels = sort_pauli_labels(basis_labels)
        self.n_qubits = infer_n_qubits(self.basis_labels)
        self.index = {label: idx for idx, label in enumerate(self.basis_labels)}
        out_idx: list[int] = []
        left_idx: list[int] = []
        right_idx: list[int] = []
        coeffs: list[complex] = []
        for left_label, left_position in self.index.items():
            for right_label, right_position in self.index.items():
                item = commutator_pauli_labels(left_label, right_label)
                if item is None:
                    continue
                phase, out_label = item
                if out_label not in self.index:
                    continue
                out_idx.append(self.index[out_label])
                left_idx.append(left_position)
                right_idx.append(right_position)
                coeffs.append(phase)
        self._out_idx = tuple(out_idx)
        self._left_idx = tuple(left_idx)
        self._right_idx = tuple(right_idx)
        self._coeffs = tuple(coeffs)

    @property
    def size(self) -> int:
        return len(self.basis_labels)

    def vector_from_operator(
        self,
        operator: SparsePauliOperator,
        *,
        dtype: torch.dtype = torch.complex64,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        return operator.to_vector(self.basis_labels, dtype=dtype, device=device)

    def embed_subset(
        self,
        subset_labels: Sequence[str],
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        """Embed coefficients on ``subset_labels`` into the full algebra basis."""

        indices = [self.index[validate_pauli_label(label, self.n_qubits)] for label in subset_labels]
        out = torch.zeros(
            coefficients.shape[:-1] + (self.size,),
            dtype=torch.complex64 if not coefficients.is_complex() else coefficients.dtype,
            device=coefficients.device,
        )
        out[..., torch.tensor(indices, device=coefficients.device)] = coefficients.to(out.dtype)
        return out

    def commutator(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Compute ``[left, right]`` for coefficient tensors on this basis."""

        left, right = torch.broadcast_tensors(left, right)
        dtype = torch.promote_types(left.dtype, right.dtype)
        if not torch.empty((), dtype=dtype).is_complex():
            dtype = torch.complex64
        left = left.to(dtype)
        right = right.to(dtype)
        result = torch.zeros_like(left, dtype=dtype)
        if not self._coeffs:
            return result
        device = left.device
        out_idx = torch.tensor(self._out_idx, dtype=torch.long, device=device)
        left_idx = torch.tensor(self._left_idx, dtype=torch.long, device=device)
        right_idx = torch.tensor(self._right_idx, dtype=torch.long, device=device)
        coeffs = torch.tensor(self._coeffs, dtype=dtype, device=device)
        source = coeffs * left[..., left_idx] * right[..., right_idx]
        result.index_add_(-1, out_idx, source)
        return result

    @staticmethod
    def norm_sq(coefficients: torch.Tensor) -> torch.Tensor:
        """Mean squared sparse-operator norm over the last axis."""

        return torch.mean(torch.sum(torch.abs(coefficients) ** 2, dim=-1).real)

