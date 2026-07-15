"""Temporal factorization and deterministic ordering for MPO AGP evaluation.

This module deliberately depends only on NumPy. TeNPy is imported only by the
later MPO construction and evolution layers so training remains independent of
the optional tensor-network extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from numbers import Integral
from typing import Sequence

import numpy as np


_PAULI_SYMBOLS = frozenset("IXYZ")
_SUPPORTED_ORDER_CANDIDATES = frozenset(("native", "reversed", "spectral"))
_ENDPOINT_TOLERANCE = 1.0e-12
_TERM_PRESERVATION_RELATIVE_ATOL = 1.0e-12
_SPECTRAL_DEGENERACY_RELATIVE_ATOL = 1.0e-12


@dataclass(frozen=True)
class TemporalFactorization:
    """Low-rank temporal representation of the full direct-CD coefficient matrix."""

    tau: np.ndarray
    temporal_factors: np.ndarray
    static_modes: np.ndarray
    singular_values: np.ndarray
    rank: int
    retained_norm_fraction: float
    max_abs_error: float
    endpoint_max_abs_error: float
    rank_for_retained_norm: int
    rank_increased_for_term_preservation: bool
    rank_increase_reason: str | None
    term_preservation_relative_atol: float

    def reconstruct(self) -> np.ndarray:
        """Reconstruct the exact factorized direct-CD coefficient matrix."""
        return self.temporal_factors @ self.static_modes


@dataclass(frozen=True)
class QubitOrderScore:
    """Cut-count metadata for one deterministic chain-order candidate."""

    candidate: str
    order: tuple[int, ...]
    max_cut_terms: int
    mean_cut_terms: float
    cut_terms_by_bond: tuple[int, ...]


@dataclass(frozen=True)
class QubitOrderSelection:
    """Selected ordering and all candidate scores used to select it."""

    order: tuple[int, ...]
    candidate: str
    max_cut_terms: int
    mean_cut_terms: float
    candidate_scores: tuple[QubitOrderScore, ...]


def factor_direct_cd_coefficients(
    tau: np.ndarray,
    direct_cd_coefficients: np.ndarray,
    *,
    retained_norm: float = 0.999999,
    endpoint_tolerance: float = _ENDPOINT_TOLERANCE,
    term_preservation_relative_atol: float = _TERM_PRESERVATION_RELATIVE_ATOL,
) -> TemporalFactorization:
    """Factor every direct-CD coefficient with the smallest valid SVD rank.

    ``retained_norm`` is a squared-singular-value fraction. Every source
    coefficient column with nonzero norm must retain a norm greater than
    ``term_preservation_relative_atol * source_norm``. The rank is increased
    beyond the global norm target when necessary and records that decision.

    The direct-CD samples must use a normalized, strictly increasing time grid
    from zero to one. Endpoint coefficient rows must be zero to
    ``endpoint_tolerance``; reconstruction never mutates them afterwards.
    """
    tau_array = _finite_real_array(tau, name="tau", ndim=1)
    coefficient_array = _finite_real_array(
        direct_cd_coefficients, name="direct_cd_coefficients", ndim=2
    )
    if tau_array.size < 2:
        raise ValueError("tau must contain at least two endpoint samples.")
    if coefficient_array.shape[0] != tau_array.size or coefficient_array.shape[1] == 0:
        raise ValueError(
            "direct_cd_coefficients shape must be (len(tau), positive number of terms)."
        )
    if not np.isfinite(retained_norm) or not 0.0 < float(retained_norm) <= 1.0:
        raise ValueError("retained_norm must be finite and in the interval (0, 1].")
    if not np.isfinite(endpoint_tolerance) or float(endpoint_tolerance) < 0.0:
        raise ValueError("endpoint_tolerance must be finite and nonnegative.")
    if (
        not np.isfinite(term_preservation_relative_atol)
        or not 0.0 < float(term_preservation_relative_atol) < 1.0
    ):
        raise ValueError("term_preservation_relative_atol must be finite and in (0, 1).")
    if abs(float(tau_array[0])) > float(endpoint_tolerance):
        raise ValueError("tau must start at 0 within endpoint_tolerance.")
    if abs(float(tau_array[-1]) - 1.0) > float(endpoint_tolerance):
        raise ValueError("tau must end at 1 within endpoint_tolerance.")
    if np.any(np.diff(tau_array) <= 0.0):
        raise ValueError("tau must be strictly increasing.")
    if np.max(np.abs(coefficient_array[[0, -1]])) > float(endpoint_tolerance):
        raise ValueError("direct-CD coefficient endpoint rows must be zero within endpoint_tolerance.")

    left, singular_values, right = np.linalg.svd(coefficient_array, full_matrices=False)
    squared_singular_values = singular_values * singular_values
    total_norm_sq = float(np.sum(squared_singular_values))
    if total_norm_sq == 0.0:
        rank = 0
        rank_for_retained_norm = 0
        temporal_factors = np.empty((tau_array.size, 0), dtype=np.float64)
        static_modes = np.empty((0, coefficient_array.shape[1]), dtype=np.float64)
        retained_norm_fraction = 1.0
        rank_increase_reason = None
    else:
        cumulative_norm_sq = np.cumsum(squared_singular_values)
        rank_for_retained_norm = int(
            np.searchsorted(cumulative_norm_sq, float(retained_norm) * total_norm_sq) + 1
        )
        rank_for_retained_norm = min(rank_for_retained_norm, singular_values.size)
        while (
            rank_for_retained_norm < singular_values.size
            and cumulative_norm_sq[rank_for_retained_norm - 1] / total_norm_sq < retained_norm
        ):
            rank_for_retained_norm += 1

        source_norms = np.linalg.norm(coefficient_array, axis=0)
        rank = rank_for_retained_norm
        while rank < singular_values.size:
            reconstructed = left[:, :rank] @ (singular_values[:rank, None] * right[:rank, :])
            retained_norms = np.linalg.norm(reconstructed, axis=0)
            if np.all(
                (source_norms == 0.0)
                | (retained_norms > float(term_preservation_relative_atol) * source_norms)
            ):
                break
            rank += 1

        temporal_factors = left[:, :rank]
        static_modes = singular_values[:rank, None] * right[:rank, :]
        retained_norm_fraction = float(cumulative_norm_sq[rank - 1] / total_norm_sq)
        final_reconstruction = temporal_factors @ static_modes
        final_retained_norms = np.linalg.norm(final_reconstruction, axis=0)
        if not np.all(
            (source_norms == 0.0)
            | (final_retained_norms > float(term_preservation_relative_atol) * source_norms)
        ):
            raise ValueError("Unable to preserve every nonzero source coefficient column.")
        rank_increase_reason = (
            None
            if rank == rank_for_retained_norm
            else (
                f"Increased rank from {rank_for_retained_norm} to {rank} to preserve nonzero "
                f"source coefficient columns above relative norm threshold "
                f"{float(term_preservation_relative_atol):.3e}."
            )
        )

    provisional = TemporalFactorization(
        tau=tau_array.copy(),
        temporal_factors=temporal_factors,
        static_modes=static_modes,
        singular_values=singular_values.copy(),
        rank=rank,
        retained_norm_fraction=retained_norm_fraction,
        max_abs_error=0.0,
        endpoint_max_abs_error=0.0,
        rank_for_retained_norm=rank_for_retained_norm,
        rank_increased_for_term_preservation=rank > rank_for_retained_norm,
        rank_increase_reason=rank_increase_reason,
        term_preservation_relative_atol=float(term_preservation_relative_atol),
    )
    reconstructed = provisional.reconstruct()
    return TemporalFactorization(
        tau=provisional.tau,
        temporal_factors=provisional.temporal_factors,
        static_modes=provisional.static_modes,
        singular_values=provisional.singular_values,
        rank=provisional.rank,
        retained_norm_fraction=provisional.retained_norm_fraction,
        max_abs_error=float(np.max(np.abs(reconstructed - coefficient_array))),
        endpoint_max_abs_error=float(
            np.max(np.abs(reconstructed[[0, -1]] - coefficient_array[[0, -1]]))
        ),
        rank_for_retained_norm=provisional.rank_for_retained_norm,
        rank_increased_for_term_preservation=provisional.rank_increased_for_term_preservation,
        rank_increase_reason=provisional.rank_increase_reason,
        term_preservation_relative_atol=provisional.term_preservation_relative_atol,
    )


def permute_pauli_label(label: str, order: Sequence[int]) -> str:
    """Move an original-order Pauli label into the requested chain order."""
    normalized_order = _validate_order(order, n_qubits=len(label))
    _validate_pauli_label(label, n_qubits=len(normalized_order))
    return "".join(label[original_site] for original_site in normalized_order)


def unpermute_pauli_label(label: str, order: Sequence[int]) -> str:
    """Restore an original-order Pauli label from a chain-order label."""
    normalized_order = _validate_order(order, n_qubits=len(label))
    _validate_pauli_label(label, n_qubits=len(normalized_order))
    restored = ["I"] * len(normalized_order)
    for chain_site, original_site in enumerate(normalized_order):
        restored[original_site] = label[chain_site]
    return "".join(restored)


def select_qubit_order(
    terms: Sequence[tuple[str, complex]],
    *,
    n_qubits: int,
    candidates: Sequence[str] = ("native", "reversed", "spectral"),
) -> QubitOrderSelection:
    """Select a deterministic qubit order from native, reversed, and spectral candidates.

    All non-identity term supports enter both the weighted interaction graph and
    cut-count scoring. Candidate ranking uses ``max_cut_terms``, then
    ``mean_cut_terms``, then the candidate name.
    """
    n_qubits = _validate_n_qubits(n_qubits)
    candidate_names = tuple(candidates)
    if not candidate_names:
        raise ValueError("At least one ordering candidate is required.")
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError("Ordering candidate names must be unique.")
    unsupported = set(candidate_names) - _SUPPORTED_ORDER_CANDIDATES
    if unsupported:
        raise ValueError(f"Unsupported ordering candidate(s): {sorted(unsupported)!r}.")

    normalized_terms: list[tuple[str, float]] = []
    for label, coefficient in terms:
        _validate_pauli_label(label, n_qubits=n_qubits)
        weight = _finite_weight(coefficient)
        normalized_terms.append((label, weight))
    normalized_terms.sort(key=lambda item: (item[0], item[1]))

    interaction = _weighted_interaction_graph(normalized_terms, n_qubits=n_qubits)
    candidate_orders = {
        "native": tuple(range(n_qubits)),
        "reversed": tuple(reversed(range(n_qubits))),
        "spectral": _spectral_order(interaction),
    }
    scores = tuple(
        sorted(
            (_score_order(name, candidate_orders[name], normalized_terms) for name in candidate_names),
            key=lambda score: score.candidate,
        )
    )
    selected = min(
        scores,
        key=lambda score: (score.max_cut_terms, score.mean_cut_terms, score.candidate),
    )
    return QubitOrderSelection(
        order=selected.order,
        candidate=selected.candidate,
        max_cut_terms=selected.max_cut_terms,
        mean_cut_terms=selected.mean_cut_terms,
        candidate_scores=scores,
    )


def _finite_real_array(values: np.ndarray, *, name: str, ndim: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued.")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array.") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _validate_pauli_label(label: str, *, n_qubits: int) -> None:
    if not isinstance(label, str) or len(label) != n_qubits or any(symbol not in _PAULI_SYMBOLS for symbol in label):
        raise ValueError(f"Pauli label {label!r} must contain {n_qubits} symbols from IXYZ.")


def _validate_order(order: Sequence[int], *, n_qubits: int) -> tuple[int, ...]:
    if any(isinstance(site, bool) or not isinstance(site, Integral) for site in order):
        raise ValueError("order entries must be integer qubit indices, not bools or fractions.")
    normalized = tuple(int(site) for site in order)
    if len(normalized) != n_qubits or sorted(normalized) != list(range(n_qubits)):
        raise ValueError("order must be a permutation of the qubit indices.")
    return normalized


def _validate_n_qubits(n_qubits: int) -> int:
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, Integral):
        raise ValueError("n_qubits must be a positive integer, not a bool or fraction.")
    normalized = int(n_qubits)
    if normalized < 1:
        raise ValueError("n_qubits must be positive.")
    return normalized


def _finite_weight(coefficient: complex) -> float:
    value = complex(coefficient)
    if not np.isfinite(value.real) or not np.isfinite(value.imag):
        raise ValueError("Term coefficients must be finite.")
    return float(abs(value))


def _weighted_interaction_graph(
    terms: Sequence[tuple[str, float]], *, n_qubits: int
) -> np.ndarray:
    graph = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    for label, weight in terms:
        support = [site for site, symbol in enumerate(label) if symbol != "I"]
        for left, right in combinations(support, 2):
            graph[left, right] += weight
            graph[right, left] += weight
    return graph


def _spectral_order(interaction: np.ndarray) -> tuple[int, ...]:
    n_qubits = interaction.shape[0]
    if n_qubits == 1:
        return (0,)
    laplacian = np.diag(np.sum(interaction, axis=1)) - interaction
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    degeneracy_tolerance = _SPECTRAL_DEGENERACY_RELATIVE_ATOL * scale
    if eigenvalues[1] <= degeneracy_tolerance or (
        n_qubits > 2 and eigenvalues[2] - eigenvalues[1] <= degeneracy_tolerance
    ):
        return tuple(range(n_qubits))
    fiedler = eigenvectors[:, 1].copy()
    nonzero = np.flatnonzero(np.abs(fiedler) > degeneracy_tolerance)
    if nonzero.size and fiedler[nonzero[0]] < 0.0:
        fiedler *= -1.0
    return tuple(int(site) for site in np.lexsort((np.arange(n_qubits), fiedler)))


def _score_order(
    candidate: str,
    order: tuple[int, ...],
    terms: Sequence[tuple[str, float]],
) -> QubitOrderScore:
    chain_position = np.empty(len(order), dtype=np.int64)
    for position, original_site in enumerate(order):
        chain_position[original_site] = position
    cut_terms = np.zeros(max(len(order) - 1, 0), dtype=np.int64)
    for label, _ in terms:
        occupied = [chain_position[site] for site, symbol in enumerate(label) if symbol != "I"]
        if len(occupied) > 1:
            cut_terms[min(occupied) : max(occupied)] += 1
    return QubitOrderScore(
        candidate=candidate,
        order=order,
        max_cut_terms=int(np.max(cut_terms)) if cut_terms.size else 0,
        mean_cut_terms=float(np.mean(cut_terms)) if cut_terms.size else 0.0,
        cut_terms_by_bond=tuple(int(value) for value in cut_terms),
    )
