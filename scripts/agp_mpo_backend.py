"""Temporal factorization and deterministic ordering for MPO AGP evaluation.

This module deliberately depends only on NumPy. TeNPy is imported only by the
later MPO construction and evolution layers so training remains independent of
the optional tensor-network extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from itertools import combinations
import math
from numbers import Integral
import time
import warnings
from typing import Any, Callable, Mapping, Sequence

import numpy as np


_PAULI_SYMBOLS = frozenset("IXYZ")
_SUPPORTED_ORDER_CANDIDATES = frozenset(("native", "reversed", "spectral"))
_ENDPOINT_TOLERANCE = 1.0e-12
_COLUMN_RETAINED_ENERGY_ATOL = 128.0 * np.finfo(np.float64).eps
_SPECTRAL_EIGENVALUE_RELATIVE_ATOL = 128.0 * np.finfo(np.float64).eps
_SPECTRAL_COMPONENT_ATOL_SCALE = 32.0
_DEFAULT_MPO_WORKSPACE_CAP_BYTES = 256 * 1024 * 1024
_MPO_WORKSPACE_SAFETY_MARGIN_BYTES = 1024 * 1024
_COMPLEX_BYTES = np.dtype(np.complex128).itemsize
_INDEX_BYTES = np.dtype(np.uint64).itemsize
_PAULI_INDEX = {symbol: index for index, symbol in enumerate("IXYZ")}
_GAMMA_N_MODEL = "gamma_n = n * eps / (1 - n * eps) for IEEE-754 binary64"
_GAMMA_N_ASSUMPTIONS = (
    "Operation estimate counts floating-point contractions from MPO bond dimensions "
    "and chain length; it bounds first-order binary64 accumulation error."
)


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
    column_retained_energy_fractions: np.ndarray
    minimum_column_retained_energy_fraction: float

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


@dataclass
class PreparedTDVPOperators:
    """Compressed static operators and full temporal-mode accounting for evolution."""

    sites: list[object]
    h0_mpo: object | None
    h1_mpo: object | None
    cd_mode_mpos: list[object]
    temporal_factorization: TemporalFactorization | None
    order: tuple[int, ...]
    diagnostics: dict[str, object]
    h0_exact_mpo: object | None = None
    h1_exact_mpo: object | None = None
    cd_mode_exact_mpos: list[object] = field(default_factory=list)


@dataclass
class PauliCoordinateSource:
    """Lightweight exact Pauli provenance consumed by bounded MPO compression."""

    L: int
    sites: list[object]
    _agp_pauli_terms: tuple[tuple[str, complex], ...]
    bc: str = "finite"

    @property
    def chi(self) -> tuple[int, ...]:
        return (1,) * (self.L + 1)


@dataclass(frozen=True)
class FullSupportIdentity:
    """Mutation-sensitive identity for an ordered full learned support."""

    source_terms: int
    n_qubits: int
    coefficient_shape: tuple[int, ...]
    sha256: str


@dataclass
class TimePauliTensorTrain:
    """One full-support tensor train over a finite time grid and Pauli space."""

    sites: list[object]
    time_core: np.ndarray
    pauli_cores: tuple[np.ndarray, ...]
    coefficient_samples: np.ndarray
    labels: tuple[str, ...]
    order: tuple[int, ...]
    coefficient_scale: float
    full_support_identity: FullSupportIdentity
    diagnostics: dict[str, object]
    time_axis_position: int = 0

    @property
    def sample_count(self) -> int:
        return int(self.time_core.shape[1])

    @property
    def n_qubits(self) -> int:
        return len(self.pauli_cores)


def build_full_support_identity(
    labels: Sequence[str], coefficient_samples: np.ndarray
) -> FullSupportIdentity:
    """Hash every ordered learned label and coefficient sample."""

    normalized_labels = tuple(str(label) for label in labels)
    if not normalized_labels:
        raise ValueError("Full learned support must contain at least one label.")
    if len(set(normalized_labels)) != len(normalized_labels):
        raise ValueError("Full learned support labels must be unique.")
    n_qubits = len(normalized_labels[0])
    for label in normalized_labels:
        _validate_pauli_label(label, n_qubits=n_qubits)
    coefficients = np.asarray(coefficient_samples)
    if coefficients.ndim < 1 or coefficients.shape[-1] != len(normalized_labels):
        raise ValueError(
            "Coefficient samples must end with one column per full-support label."
        )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Full-support coefficient samples must be finite.")
    canonical_coefficients = np.ascontiguousarray(coefficients, dtype=np.complex128)
    digest = hashlib.sha256()
    digest.update(b"full-learned-support-v1\0")
    digest.update(repr((n_qubits, normalized_labels)).encode("ascii"))
    digest.update(repr(canonical_coefficients.shape).encode("ascii"))
    digest.update(canonical_coefficients.tobytes(order="C"))
    return FullSupportIdentity(
        source_terms=len(normalized_labels),
        n_qubits=n_qubits,
        coefficient_shape=tuple(int(size) for size in canonical_coefficients.shape),
        sha256=digest.hexdigest(),
    )


def assert_full_support_identity(
    expected: FullSupportIdentity,
    labels: Sequence[str],
    coefficient_samples: np.ndarray,
) -> FullSupportIdentity:
    """Fail before evaluation if any learned label or coefficient changed."""

    actual = build_full_support_identity(labels, coefficient_samples)
    if actual != expected:
        raise ValueError(
            "Learned full-support identity changed; canonical evaluation must use "
            "the complete ordered checkpoint support and coefficients."
        )
    return actual


def combine_instantaneous_full_support_terms(
    *,
    h0_terms: Sequence[tuple[str, complex]],
    h1_terms: Sequence[tuple[str, complex]],
    learned_labels: Sequence[str],
    direct_cd_coefficients: np.ndarray,
    lam: float,
    full_support_identity: FullSupportIdentity | None = None,
    identity_coefficient_samples: np.ndarray | None = None,
) -> tuple[tuple[tuple[str, complex], ...], dict[str, object]]:
    """Combine one complete instantaneous CD Hamiltonian in Pauli coordinates."""

    labels = tuple(str(label) for label in learned_labels)
    coefficients = np.asarray(direct_cd_coefficients, dtype=np.complex128)
    if coefficients.ndim != 1 or coefficients.size != len(labels):
        raise ValueError("direct_cd_coefficients must contain exactly one value per learned label.")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("direct_cd_coefficients must be finite.")
    if not np.isfinite(lam):
        raise ValueError("lam must be finite.")
    if full_support_identity is not None:
        if identity_coefficient_samples is None:
            raise ValueError(
                "identity_coefficient_samples are required with full_support_identity."
            )
        assert_full_support_identity(
            full_support_identity, labels, identity_coefficient_samples
        )
        support_sha256: str | None = full_support_identity.sha256
    else:
        support_sha256 = None
    if labels:
        n_qubits = len(labels[0])
        if len(set(labels)) != len(labels):
            raise ValueError("Learned full-support labels must be unique.")
        for label in labels:
            _validate_pauli_label(label, n_qubits=n_qubits)
    else:
        raise ValueError("Learned full support must contain at least one label.")

    contributions: dict[str, list[complex]] = {}
    for source_terms, scale in ((h0_terms, 1.0 - float(lam)), (h1_terms, float(lam))):
        for label, coefficient in source_terms:
            _validate_pauli_label(str(label), n_qubits=n_qubits)
            contributions.setdefault(str(label), []).append(
                _finite_complex(complex(coefficient) * scale)
            )
    for label, coefficient in zip(labels, coefficients):
        contributions.setdefault(label, []).append(_finite_complex(complex(coefficient)))

    combined = {
        label: _stable_complex_sum(values)
        for label, values in contributions.items()
    }
    terms = tuple(
        (label, coefficient)
        for label, coefficient in sorted(combined.items())
        if coefficient != 0.0j
    )
    coefficient_magnitudes = np.abs(coefficients)
    nonzero_magnitudes = coefficient_magnitudes[coefficient_magnitudes > 0.0]
    metadata: dict[str, object] = {
        "learned_source_terms": len(labels),
        "learned_terms_accounted": len(labels),
        "learned_nonzero_coefficients": int(np.count_nonzero(coefficients)),
        "combined_nonzero_terms": len(terms),
        "combined_zero_terms": sum(
            coefficient == 0.0j for coefficient in combined.values()
        ),
        "combined_zero_label_sample": sorted(
            label for label, coefficient in combined.items() if coefficient == 0.0j
        )[:16],
        "full_support_sha256": support_sha256,
        "coefficient_max_abs": (
            float(np.max(coefficient_magnitudes)) if coefficient_magnitudes.size else 0.0
        ),
        "coefficient_min_nonzero_abs": (
            float(np.min(nonzero_magnitudes)) if nonzero_magnitudes.size else None
        ),
    }
    return terms, metadata


def factor_direct_cd_coefficients(
    tau: np.ndarray,
    direct_cd_coefficients: np.ndarray,
    *,
    retained_norm: float = 0.999999,
    endpoint_tolerance: float = _ENDPOINT_TOLERANCE,
) -> TemporalFactorization:
    """Factor every direct-CD coefficient with the smallest valid SVD rank.

    ``retained_norm`` is a squared-singular-value fraction. Every source
    coefficient column with nonzero temporal energy must retain at least
    ``retained_norm`` of its own squared temporal norm, within a tight
    floating-point tolerance. The rank is increased beyond the global norm
    target when necessary and records that decision.

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
    if abs(float(tau_array[0])) > float(endpoint_tolerance):
        raise ValueError("tau must start at 0 within endpoint_tolerance.")
    if abs(float(tau_array[-1]) - 1.0) > float(endpoint_tolerance):
        raise ValueError("tau must end at 1 within endpoint_tolerance.")
    if np.any(np.diff(tau_array) <= 0.0):
        raise ValueError("tau must be strictly increasing.")
    if np.max(np.abs(coefficient_array[[0, -1]])) > float(endpoint_tolerance):
        raise ValueError("direct-CD coefficient endpoint rows must be zero within endpoint_tolerance.")

    column_energy_atol = max(
        _COLUMN_RETAINED_ENERGY_ATOL,
        16.0 * np.finfo(np.float64).eps * max(coefficient_array.shape),
    )
    left, singular_values, right = np.linalg.svd(coefficient_array, full_matrices=False)
    relative_squared_singular_values = _relative_squared_values(singular_values)
    total_relative_norm_sq = float(np.sum(relative_squared_singular_values))
    if total_relative_norm_sq == 0.0:
        rank = 0
        rank_for_retained_norm = 0
        temporal_factors = np.empty((tau_array.size, 0), dtype=np.float64)
        static_modes = np.empty((0, coefficient_array.shape[1]), dtype=np.float64)
        retained_norm_fraction = 1.0
        rank_increase_reason = None
        column_retained_energy_fractions = np.ones(coefficient_array.shape[1], dtype=np.float64)
    else:
        cumulative_relative_norm_sq = np.cumsum(relative_squared_singular_values)
        rank_for_retained_norm = int(
            np.searchsorted(
                cumulative_relative_norm_sq, float(retained_norm) * total_relative_norm_sq
            )
            + 1
        )
        rank_for_retained_norm = min(rank_for_retained_norm, singular_values.size)
        while (
            rank_for_retained_norm < singular_values.size
            and cumulative_relative_norm_sq[rank_for_retained_norm - 1] / total_relative_norm_sq
            < retained_norm
        ):
            rank_for_retained_norm += 1

        source_scales = np.max(np.abs(coefficient_array), axis=0)
        nonzero_columns = source_scales > 0.0
        source_relative_energies = _scaled_column_squared_norms(coefficient_array, source_scales)
        rank = rank_for_retained_norm
        while rank < singular_values.size:
            reconstructed = left[:, :rank] @ (singular_values[:rank, None] * right[:rank, :])
            column_retained_energy_fractions = _column_retained_energy_fractions(
                reconstructed, source_scales, source_relative_energies
            )
            if np.all(
                column_retained_energy_fractions[nonzero_columns]
                >= float(retained_norm) - column_energy_atol
            ):
                break
            rank += 1

        temporal_factors = left[:, :rank]
        static_modes = singular_values[:rank, None] * right[:rank, :]
        retained_norm_fraction = float(
            cumulative_relative_norm_sq[rank - 1] / total_relative_norm_sq
        )
        final_reconstruction = temporal_factors @ static_modes
        column_retained_energy_fractions = _column_retained_energy_fractions(
            final_reconstruction, source_scales, source_relative_energies
        )
        if not np.all(
            column_retained_energy_fractions[nonzero_columns]
            >= float(retained_norm) - column_energy_atol
        ):
            raise ValueError("Unable to retain the requested energy for every nonzero source coefficient column.")
        rank_increase_reason = (
            None
            if rank == rank_for_retained_norm
            else (
                f"Increased rank from {rank_for_retained_norm} to {rank} so every nonzero source "
                f"coefficient column retains at least {float(retained_norm):.6g} of its squared "
                f"temporal norm within {column_energy_atol:.3e} floating tolerance."
            )
        )

    minimum_column_retained_energy_fraction = float(np.min(column_retained_energy_fractions))

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
        column_retained_energy_fractions=column_retained_energy_fractions.copy(),
        minimum_column_retained_energy_fraction=minimum_column_retained_energy_fraction,
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
        column_retained_energy_fractions=provisional.column_retained_energy_fractions,
        minimum_column_retained_energy_fraction=provisional.minimum_column_retained_energy_fraction,
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


def _normalized_pauli_coordinate_terms(
    terms: Sequence[tuple[str, complex]],
    *,
    n_qubits: int,
    order: Sequence[int],
    arithmetic_zero_tolerance: float = 0.0,
) -> tuple[tuple[tuple[str, complex], ...], dict[str, object]]:
    """Combine, permute, and account for exact Pauli-coordinate provenance."""

    n_qubits = _validate_n_qubits(n_qubits)
    normalized_order = _validate_order(order, n_qubits=n_qubits)
    if (
        isinstance(arithmetic_zero_tolerance, bool)
        or not np.isfinite(arithmetic_zero_tolerance)
        or float(arithmetic_zero_tolerance) < 0.0
    ):
        raise ValueError("arithmetic_zero_tolerance must be finite and nonnegative.")
    zero_tolerance = float(arithmetic_zero_tolerance)

    contributions: dict[str, list[complex]] = {}
    input_terms = 0
    for label, coefficient in terms:
        _validate_pauli_label(label, n_qubits=n_qubits)
        value = _finite_complex(coefficient)
        contributions.setdefault(label, []).append(value)
        input_terms += 1

    combined_coefficients = {
        label: _stable_complex_sum(values) for label, values in sorted(contributions.items())
    }
    duplicate_labels = sorted(
        label for label, values in contributions.items() if len(values) > 1
    )
    dropped_coefficients = {
        label: value
        for label, value in combined_coefficients.items()
        if abs(value) <= zero_tolerance
    }
    included_coefficients = {
        label: value
        for label, value in combined_coefficients.items()
        if label not in dropped_coefficients
    }

    included_chain_labels: list[str] = []
    included_chain_terms: list[tuple[str, complex]] = []
    for label, coefficient in included_coefficients.items():
        chain_label = permute_pauli_label(label, normalized_order)
        included_chain_labels.append(chain_label)
        included_chain_terms.append((chain_label, coefficient))
    metadata: dict[str, object] = {
        "input_terms": input_terms,
        "unique_labels": len(combined_coefficients),
        "duplicate_labels": duplicate_labels,
        "combined_coefficients": combined_coefficients,
        "included_terms": len(included_coefficients),
        "included_labels": sorted(included_coefficients),
        "included_chain_labels": sorted(included_chain_labels),
        "dropped_terms": len(dropped_coefficients),
        "dropped_labels": sorted(dropped_coefficients),
        "dropped_coefficients": dropped_coefficients,
        "arithmetic_zero_tolerance": zero_tolerance,
        "order": normalized_order,
    }
    return tuple(sorted(included_chain_terms)), metadata


def _pauli_sites(n_qubits: int) -> list[object]:
    SpinHalfSite, _, _, _, _ = _require_tenpy()
    site = SpinHalfSite(conserve=None)
    site.add_op("X", site.get_op("Sigmax"), hc="X")
    site.add_op("Y", site.get_op("Sigmay"), hc="Y")
    site.add_op("Z", site.get_op("Sigmaz"), hc="Z")
    return [site] * int(n_qubits)


def build_pauli_coordinate_source(
    terms: Sequence[tuple[str, complex]],
    *,
    n_qubits: int,
    order: Sequence[int],
    arithmetic_zero_tolerance: float = 0.0,
) -> tuple[PauliCoordinateSource, dict[str, object]]:
    """Build exact Pauli provenance without materializing a generic MPO graph."""

    pauli_terms, metadata = _normalized_pauli_coordinate_terms(
        terms,
        n_qubits=n_qubits,
        order=order,
        arithmetic_zero_tolerance=arithmetic_zero_tolerance,
    )
    source = PauliCoordinateSource(
        L=int(n_qubits),
        sites=_pauli_sites(int(n_qubits)),
        _agp_pauli_terms=pauli_terms,
    )
    metadata = {**metadata, "representation": "pauli_coordinate_source"}
    return source, metadata


def build_exact_pauli_mpo(
    terms: Sequence[tuple[str, complex]],
    *,
    n_qubits: int,
    order: Sequence[int],
    arithmetic_zero_tolerance: float = 0.0,
) -> tuple[Any, dict[str, object]]:
    """Build a finite TeNPy MPO from every nonzero combined Pauli label."""

    _, TermList, MPOGraph, _, _ = _require_tenpy()
    pauli_terms, metadata = _normalized_pauli_coordinate_terms(
        terms,
        n_qubits=n_qubits,
        order=order,
        arithmetic_zero_tolerance=arithmetic_zero_tolerance,
    )
    sites = _pauli_sites(int(n_qubits))
    operator_terms: list[list[tuple[str, int]]] = []
    strengths: list[complex] = []
    for chain_label, coefficient in pauli_terms:
        local_terms = [
            (symbol, chain_site)
            for chain_site, symbol in enumerate(chain_label)
            if symbol != "I"
        ]
        operator_terms.append(local_terms or [("Id", 0)])
        strengths.append(coefficient)
    if not operator_terms:
        operator_terms = [[("Id", 0)]]
        strengths = [0.0]
    graph = MPOGraph.from_term_list(
        TermList(operator_terms, strengths),
        sites,
        bc="finite",
        insert_all_id=True,
        unit_cell_width=int(n_qubits),
    )
    mpo = graph.build_MPO()
    mpo._agp_pauli_terms = pauli_terms
    return mpo, metadata


def _combined_time_pauli_coefficients(
    *,
    h0_terms: Sequence[tuple[str, complex]],
    h1_terms: Sequence[tuple[str, complex]],
    learned_labels: Sequence[str],
    direct_cd_coefficients: np.ndarray,
    lambda_samples: np.ndarray,
    n_qubits: int,
) -> tuple[tuple[str, ...], np.ndarray, dict[str, object]]:
    """Build the complete sampled Hamiltonian in a shared Pauli coordinate map."""

    labels = tuple(str(label) for label in learned_labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Learned full-support labels must be nonempty and unique.")
    for label in labels:
        _validate_pauli_label(label, n_qubits=n_qubits)
    direct = np.asarray(direct_cd_coefficients, dtype=np.complex128)
    lambdas = np.asarray(lambda_samples, dtype=np.float64)
    if direct.ndim != 2 or direct.shape[1] != len(labels):
        raise ValueError(
            "direct_cd_coefficients must have shape (samples, learned labels)."
        )
    if lambdas.ndim != 1 or lambdas.size != direct.shape[0] or lambdas.size < 1:
        raise ValueError("lambda_samples must contain one value per coefficient row.")
    if not np.all(np.isfinite(direct)) or not np.all(np.isfinite(lambdas)):
        raise ValueError("Sampled coefficients and lambda values must be finite.")

    static_maps: list[dict[str, complex]] = []
    for terms in (h0_terms, h1_terms):
        contributions: dict[str, list[complex]] = {}
        for label, coefficient in terms:
            normalized_label = str(label)
            _validate_pauli_label(normalized_label, n_qubits=n_qubits)
            contributions.setdefault(normalized_label, []).append(
                _finite_complex(coefficient)
            )
        static_maps.append(
            {
                label: _stable_complex_sum(values)
                for label, values in contributions.items()
            }
        )
    h0_map, h1_map = static_maps
    union_labels = tuple(sorted(set(labels) | set(h0_map) | set(h1_map)))
    label_indices = {label: index for index, label in enumerate(union_labels)}
    coefficients = np.zeros(
        (direct.shape[0], len(union_labels)), dtype=np.complex128
    )
    for label, coefficient in h0_map.items():
        coefficients[:, label_indices[label]] += (1.0 - lambdas) * coefficient
    for label, coefficient in h1_map.items():
        coefficients[:, label_indices[label]] += lambdas * coefficient
    for source_index, label in enumerate(labels):
        coefficients[:, label_indices[label]] += direct[:, source_index]

    maximum_absolute = float(np.max(np.abs(coefficients))) if coefficients.size else 0.0
    maximum_imaginary = (
        float(np.max(np.abs(coefficients.imag))) if coefficients.size else 0.0
    )
    hermiticity_tolerance = (
        128.0 * np.finfo(np.float64).eps * max(maximum_absolute, 1.0)
    )
    if maximum_imaginary > hermiticity_tolerance:
        raise ValueError(
            "The sampled time-dependent Pauli Hamiltonian is not Hermitian within "
            "floating tolerance."
        )
    metadata: dict[str, object] = {
        "learned_source_terms": len(labels),
        "learned_terms_accounted": len(labels),
        "combined_unique_labels": len(union_labels),
        "samples": int(direct.shape[0]),
        "hermiticity_max_imaginary": maximum_imaginary,
        "hermiticity_tolerance": float(hermiticity_tolerance),
        "hermiticity_status": "pass",
    }
    return union_labels, np.asarray(coefficients.real, dtype=np.float64), metadata


def _time_pauli_retained_rank(
    singular_values: np.ndarray,
    *,
    max_bond: int,
    cutoff: float,
    maximum_rank: int,
) -> tuple[int, float, float]:
    """Select a TT rank and return relative plus absolute discarded norms."""

    available = min(int(singular_values.size), int(maximum_rank))
    if available < 1:
        return 1, 0.0, 0.0
    # An m x n unfolding has at most min(m, n) singular directions. Gram-matrix
    # eigensolvers can return tiny positive values in the algebraic null space;
    # they are roundoff, not discarded operator content.
    squared = np.square(
        np.asarray(singular_values[:available], dtype=np.float64)
    )
    total = float(np.sum(squared))
    if total == 0.0:
        return 1, 0.0, 0.0
    if float(cutoff) == 0.0:
        requested = available
    else:
        requested = available
        for rank in range(1, available + 1):
            if float(np.sum(squared[rank:]) / total) <= float(cutoff):
                requested = rank
                break
    retained = max(1, min(requested, int(max_bond), available))
    discarded_absolute = float(np.sum(squared[retained:]))
    return retained, discarded_absolute / total, discarded_absolute


def _compress_time_pauli_samples(
    labels: Sequence[str],
    coefficient_samples: np.ndarray,
    *,
    n_qubits: int,
    max_bond: int,
    cutoff: float,
    workspace_cap_bytes: int,
    time_axis_position: int = 0,
) -> tuple[np.ndarray | None, tuple[np.ndarray, ...], dict[str, object]]:
    """Workspace-bound TT-SVD of a sampled sparse time x Pauli tensor."""

    if (
        isinstance(time_axis_position, bool)
        or not isinstance(time_axis_position, Integral)
        or not 0 <= int(time_axis_position) <= int(n_qubits)
    ):
        raise ValueError("time_axis_position must be an integer in [0, n_qubits].")
    if int(time_axis_position) != 0:
        return _compress_positioned_time_pauli_samples(
            labels,
            coefficient_samples,
            n_qubits=int(n_qubits),
            max_bond=int(max_bond),
            cutoff=float(cutoff),
            workspace_cap_bytes=int(workspace_cap_bytes),
            time_axis_position=int(time_axis_position),
        )

    samples = np.asarray(coefficient_samples, dtype=np.complex128)
    if samples.ndim != 2 or samples.shape[1] != len(labels):
        raise ValueError("coefficient_samples must have one column per Pauli label.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("coefficient_samples must be finite.")
    if int(max_bond) < 1 or not 0.0 <= float(cutoff) < 1.0:
        raise ValueError("Invalid time-Pauli TT bond or cutoff.")

    sorted_rows = sorted(
        (
            _encode_pauli_label(str(label)),
            str(label),
            index,
        )
        for index, label in enumerate(labels)
    )
    codes = [row[0] for row in sorted_rows]
    column_order = [row[2] for row in sorted_rows]
    values = np.asarray(samples[:, column_order], dtype=np.complex128)
    source_norm_squared = float(np.vdot(values, values).real)
    base_bytes = values.nbytes + len(codes) * 64
    initial_required = (
        7 * values.nbytes
        + min(values.shape) ** 2 * _COMPLEX_BYTES
        + _MPO_WORKSPACE_SAFETY_MARGIN_BYTES
    )
    diagnostics: dict[str, object] = {
        "status": "ok",
        "algorithm": "workspace_bounded_joint_time_pauli_tt_svd",
        "pauli_encoding": "python_arbitrary_precision_int",
        "max_bond": int(max_bond),
        "cutoff": float(cutoff),
        "workspace_cap_bytes": int(workspace_cap_bytes),
        "required_workspace_bytes": int(initial_required),
        "peak_workspace_bytes": int(base_bytes),
        "retained_ranks": [],
        "per_bond_discarded_relative_weights": [],
        "per_bond_discarded_squared_norms": [],
        "time_axis_position": 0,
        "tt_axis_order": ["time", *[f"pauli_{site}" for site in range(int(n_qubits))]],
    }
    if initial_required > int(workspace_cap_bytes):
        diagnostics.update(
            {
                "status": "not_feasible",
                "resource_reason": "initial time unfolding SVD exceeds workspace cap",
            }
        )
        return None, (), diagnostics

    try:
        left, singular_values, right = np.linalg.svd(values, full_matrices=False)
    except np.linalg.LinAlgError as error:
        diagnostics.update(
            {"status": "not_feasible", "resource_reason": f"time SVD failed: {error}"}
        )
        return None, (), diagnostics
    time_rank, relative_discarded, absolute_discarded = _time_pauli_retained_rank(
        singular_values,
        max_bond=int(max_bond),
        cutoff=float(cutoff),
        maximum_rank=min(values.shape),
    )
    time_core = np.asarray(left[:, :time_rank], dtype=np.complex128)
    values = np.asarray(
        singular_values[:time_rank, None] * right[:time_rank],
        dtype=np.complex128,
    )
    retained_ranks: list[int] = [time_rank]
    discarded_relative: list[float] = [relative_discarded]
    discarded_squared_norms: list[float] = [absolute_discarded]
    cores: list[np.ndarray] = []
    peak_workspace = max(
        int(diagnostics["peak_workspace_bytes"]),
        initial_required,
    )
    required_workspace = initial_required

    for site in range(int(n_qubits) - 1):
        previous_rank, entry_count = values.shape
        remaining_sites = int(n_qubits) - site
        shift = 2 * (remaining_sites - 1)
        suffix_mask = (1 << shift) - 1 if shift else 0
        symbols = np.fromiter(
            ((code >> shift) & 3 for code in codes),
            dtype=np.int64,
            count=entry_count,
        )
        suffixes = [code & suffix_mask for code in codes]
        unique_suffixes = sorted(set(suffixes))
        suffix_indices = {code: index for index, code in enumerate(unique_suffixes)}
        inverse = np.fromiter(
            (suffix_indices[code] for code in suffixes),
            dtype=np.int64,
            count=entry_count,
        )
        row_count = previous_rank * 4
        column_count = len(unique_suffixes)
        retained_bound = min(int(max_bond), row_count, column_count)
        matrix_bytes = row_count * column_count * _COMPLEX_BYTES
        gram_bytes = row_count * row_count * _COMPLEX_BYTES
        core_bytes = sum(core.nbytes for core in cores)
        local_required = (
            core_bytes
            + values.nbytes
            + matrix_bytes * 2
            + gram_bytes * 4
            + row_count * retained_bound * _COMPLEX_BYTES * 2
            + len(codes) * 80
            + _MPO_WORKSPACE_SAFETY_MARGIN_BYTES
        )
        required_workspace = max(required_workspace, local_required)
        if local_required > int(workspace_cap_bytes):
            diagnostics.update(
                {
                    "status": "not_feasible",
                    "resource_reason": "local joint Pauli unfolding exceeds workspace cap",
                    "failed_bond": site,
                    "required_workspace_bytes": int(required_workspace),
                    "peak_workspace_bytes": int(peak_workspace),
                }
            )
            return None, (), diagnostics

        matrix = np.zeros((row_count, column_count), dtype=np.complex128)
        for left_index in range(previous_rank):
            np.add.at(
                matrix,
                (left_index * 4 + symbols, inverse),
                values[left_index],
            )
        gram = matrix @ matrix.conj().T
        eigenvalues, left_vectors = np.linalg.eigh(gram)
        singular_values = np.sqrt(np.maximum(eigenvalues[::-1], 0.0))
        retained_rank, relative_weight, absolute_weight = _time_pauli_retained_rank(
            singular_values,
            max_bond=int(max_bond),
            cutoff=float(cutoff),
            maximum_rank=min(row_count, column_count),
        )
        retained_core = np.asarray(
            left_vectors[:, -retained_rank:][:, ::-1].reshape(
                previous_rank, 4, retained_rank
            ),
            dtype=np.complex128,
        )
        retained_vectors = retained_core.reshape(row_count, retained_rank)
        next_values = np.asarray(retained_vectors.conj().T @ matrix)
        cores.append(retained_core)
        retained_ranks.append(retained_rank)
        discarded_relative.append(relative_weight)
        discarded_squared_norms.append(absolute_weight)
        peak_workspace = max(peak_workspace, local_required)
        codes = unique_suffixes
        values = next_values

    final_core = np.zeros((values.shape[0], 4, 1), dtype=np.complex128)
    for entry, code in enumerate(codes):
        final_core[:, int(code) & 3, 0] += values[:, entry]
    cores.append(final_core)
    total_discarded = float(sum(discarded_squared_norms))
    roundoff_relative_bound = float(
        64.0
        * (int(n_qubits) + 1)
        * max(samples.shape)
        * np.finfo(np.float64).eps
    )
    roundoff_absolute_bound = float(
        roundoff_relative_bound * np.sqrt(max(source_norm_squared, 0.0))
    )
    coefficient_error_absolute_bound = float(
        np.sqrt(max(total_discarded, 0.0)) + roundoff_absolute_bound
    )
    sample_norms = np.linalg.norm(samples, axis=1)
    per_sample_bounds = [
        (
            math.inf
            if float(norm) == 0.0
            else float(coefficient_error_absolute_bound / float(norm))
        )
        for norm in sample_norms
    ]
    diagnostics.update(
        {
            "required_workspace_bytes": int(required_workspace),
            "peak_workspace_bytes": int(peak_workspace),
            "retained_ranks": retained_ranks,
            "per_bond_discarded_relative_weights": discarded_relative,
            "per_bond_discarded_squared_norms": discarded_squared_norms,
            "total_discarded_squared_norm": total_discarded,
            "source_coefficient_norm_squared": source_norm_squared,
            "roundoff_relative_bound": roundoff_relative_bound,
            "coefficient_error_absolute_upper_bound": coefficient_error_absolute_bound,
            "per_sample_relative_coefficient_error_upper_bounds": per_sample_bounds,
            "max_relative_coefficient_error_upper_bound": max(per_sample_bounds),
            "post_bonds": [1, *retained_ranks[1:], 1],
            "exact_identity_status": (
                "verified" if total_discarded == 0.0 else "not_established"
            ),
        }
    )
    return time_core[None, :, :], tuple(cores), diagnostics


def _compress_positioned_time_pauli_samples(
    labels: Sequence[str],
    coefficient_samples: np.ndarray,
    *,
    n_qubits: int,
    max_bond: int,
    cutoff: float,
    workspace_cap_bytes: int,
    time_axis_position: int,
) -> tuple[np.ndarray | None, tuple[np.ndarray, ...], dict[str, object]]:
    """Sparse TT-SVD with the finite time axis inserted along the Pauli chain."""

    samples = np.asarray(coefficient_samples, dtype=np.complex128)
    if samples.ndim != 2 or samples.shape[1] != len(labels):
        raise ValueError("coefficient_samples must have one column per Pauli label.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("coefficient_samples must be finite.")
    if int(max_bond) < 1 or not 0.0 <= float(cutoff) < 1.0:
        raise ValueError("Invalid time-Pauli TT bond or cutoff.")

    sample_count = int(samples.shape[0])
    if sample_count < 1:
        raise ValueError("coefficient_samples must contain at least one time sample.")
    sorted_rows = sorted(
        (_encode_pauli_label(str(label)), index)
        for index, label in enumerate(labels)
    )
    pauli_codes = [row[0] for row in sorted_rows]
    column_order = [row[1] for row in sorted_rows]
    sorted_samples = np.asarray(samples[:, column_order], dtype=np.complex128)
    source_norm_squared = float(np.vdot(sorted_samples, sorted_samples).real)

    trailing_sites = int(n_qubits) - int(time_axis_position)
    trailing_radix = 1 << (2 * trailing_sites)
    coordinates: list[int] = []
    entries: list[complex] = []
    for sample in range(sample_count):
        for column, pauli_code in enumerate(pauli_codes):
            value = complex(sorted_samples[sample, column])
            if value == 0.0:
                continue
            prefix = pauli_code >> (2 * trailing_sites)
            suffix = pauli_code & (trailing_radix - 1)
            coordinates.append(
                (prefix * sample_count + sample) * trailing_radix + suffix
            )
            entries.append(value)
    if not coordinates:
        coordinates = [0]
        entries = [0.0j]

    values = np.asarray(entries, dtype=np.complex128)[None, :]
    axis_dimensions = (
        [4] * int(time_axis_position)
        + [sample_count]
        + [4] * trailing_sites
    )
    axis_names = (
        [f"pauli_{site}" for site in range(int(time_axis_position))]
        + ["time"]
        + [
            f"pauli_{site}"
            for site in range(int(time_axis_position), int(n_qubits))
        ]
    )
    suffix_radices: list[int] = []
    suffix_product = 1
    for dimension in reversed(axis_dimensions[1:]):
        suffix_product *= int(dimension)
        suffix_radices.append(suffix_product)
    suffix_radices = list(reversed(suffix_radices))

    base_bytes = values.nbytes + sorted_samples.nbytes + len(coordinates) * 80
    diagnostics: dict[str, object] = {
        "status": "ok",
        "algorithm": "workspace_bounded_positioned_time_pauli_tt_svd",
        "pauli_encoding": "python_arbitrary_precision_mixed_radix_int",
        "max_bond": int(max_bond),
        "cutoff": float(cutoff),
        "workspace_cap_bytes": int(workspace_cap_bytes),
        "required_workspace_bytes": int(
            base_bytes + _MPO_WORKSPACE_SAFETY_MARGIN_BYTES
        ),
        "peak_workspace_bytes": int(base_bytes),
        "retained_ranks": [],
        "per_bond_discarded_relative_weights": [],
        "per_bond_discarded_squared_norms": [],
        "time_axis_position": int(time_axis_position),
        "tt_axis_order": axis_names,
        "sparse_coordinate_count": len(coordinates),
    }
    if int(diagnostics["required_workspace_bytes"]) > int(workspace_cap_bytes):
        diagnostics.update(
            {
                "status": "not_feasible",
                "resource_reason": "positioned sparse coordinates exceed workspace cap",
            }
        )
        return None, (), diagnostics

    cores: list[np.ndarray] = []
    retained_ranks: list[int] = []
    discarded_relative: list[float] = []
    discarded_squared_norms: list[float] = []
    peak_workspace = int(diagnostics["peak_workspace_bytes"])
    required_workspace = int(diagnostics["required_workspace_bytes"])
    codes = coordinates

    for axis, dimension in enumerate(axis_dimensions[:-1]):
        previous_rank, entry_count = values.shape
        suffix_radix = suffix_radices[axis]
        symbols = np.fromiter(
            (code // suffix_radix for code in codes),
            dtype=np.int64,
            count=entry_count,
        )
        suffixes = [code % suffix_radix for code in codes]
        unique_suffixes = sorted(set(suffixes))
        suffix_indices = {code: index for index, code in enumerate(unique_suffixes)}
        inverse = np.fromiter(
            (suffix_indices[code] for code in suffixes),
            dtype=np.int64,
            count=entry_count,
        )
        row_count = previous_rank * int(dimension)
        column_count = len(unique_suffixes)
        gram_dimension = min(row_count, column_count)
        retained_bound = min(int(max_bond), gram_dimension)
        matrix_bytes = row_count * column_count * _COMPLEX_BYTES
        gram_bytes = gram_dimension * gram_dimension * _COMPLEX_BYTES
        core_bytes = sum(core.nbytes for core in cores)
        local_required = (
            base_bytes
            + core_bytes
            + values.nbytes
            + matrix_bytes * 2
            + gram_bytes * 4
            + (row_count + column_count)
            * retained_bound
            * _COMPLEX_BYTES
            * 2
            + _MPO_WORKSPACE_SAFETY_MARGIN_BYTES
        )
        required_workspace = max(required_workspace, local_required)
        if local_required > int(workspace_cap_bytes):
            diagnostics.update(
                {
                    "status": "not_feasible",
                    "resource_reason": "positioned joint unfolding exceeds workspace cap",
                    "failed_axis": axis,
                    "failed_axis_name": axis_names[axis],
                    "required_workspace_bytes": int(required_workspace),
                    "peak_workspace_bytes": int(peak_workspace),
                }
            )
            return None, (), diagnostics

        matrix = np.zeros((row_count, column_count), dtype=np.complex128)
        for left_index in range(previous_rank):
            np.add.at(
                matrix,
                (left_index * int(dimension) + symbols, inverse),
                values[left_index],
            )
        try:
            if row_count <= column_count:
                gram = matrix @ matrix.conj().T
                eigenvalues, eigenvectors = np.linalg.eigh(gram)
                singular_values = np.sqrt(np.maximum(eigenvalues[::-1], 0.0))
                candidate_vectors = eigenvectors[:, ::-1]
                vectors_on_left = True
            else:
                gram = matrix.conj().T @ matrix
                eigenvalues, eigenvectors = np.linalg.eigh(gram)
                singular_values = np.sqrt(np.maximum(eigenvalues[::-1], 0.0))
                candidate_vectors = eigenvectors[:, ::-1]
                vectors_on_left = False
        except np.linalg.LinAlgError as error:
            diagnostics.update(
                {
                    "status": "not_feasible",
                    "resource_reason": f"positioned joint eigensolve failed: {error}",
                    "failed_axis": axis,
                    "failed_axis_name": axis_names[axis],
                }
            )
            return None, (), diagnostics

        if singular_values.size and singular_values[0] > 0.0:
            numerical_tolerance = float(
                64.0
                * max(row_count, column_count)
                * np.finfo(np.float64).eps
                * singular_values[0]
            )
            numerical_rank = int(np.count_nonzero(singular_values > numerical_tolerance))
        else:
            numerical_rank = 1
        numerical_rank = max(1, min(numerical_rank, gram_dimension))
        retained_rank, relative_weight, absolute_weight = _time_pauli_retained_rank(
            singular_values[:numerical_rank],
            max_bond=int(max_bond),
            cutoff=float(cutoff),
            maximum_rank=numerical_rank,
        )
        if vectors_on_left:
            retained_vectors = np.asarray(
                candidate_vectors[:, :retained_rank], dtype=np.complex128
            )
        else:
            right_vectors = np.asarray(
                candidate_vectors[:, :retained_rank], dtype=np.complex128
            )
            retained_vectors = np.asarray(matrix @ right_vectors, dtype=np.complex128)
            retained_vectors /= singular_values[:retained_rank][None, :]
        retained_core = retained_vectors.reshape(
            previous_rank, int(dimension), retained_rank
        )
        values = np.asarray(retained_vectors.conj().T @ matrix)
        cores.append(retained_core)
        retained_ranks.append(retained_rank)
        discarded_relative.append(relative_weight)
        discarded_squared_norms.append(absolute_weight)
        peak_workspace = max(peak_workspace, local_required)
        codes = unique_suffixes

    final_dimension = int(axis_dimensions[-1])
    final_core = np.zeros((values.shape[0], final_dimension, 1), dtype=np.complex128)
    for entry, code in enumerate(codes):
        final_core[:, int(code), 0] += values[:, entry]
    cores.append(final_core)

    total_discarded = float(sum(discarded_squared_norms))
    roundoff_relative_bound = float(
        64.0
        * (int(n_qubits) + 1)
        * max(samples.shape)
        * np.finfo(np.float64).eps
    )
    roundoff_absolute_bound = float(
        roundoff_relative_bound * np.sqrt(max(source_norm_squared, 0.0))
    )
    coefficient_error_absolute_bound = float(
        np.sqrt(max(total_discarded, 0.0)) + roundoff_absolute_bound
    )
    sample_norms = np.linalg.norm(samples, axis=1)
    per_sample_bounds = [
        (
            math.inf
            if float(norm) == 0.0
            else float(coefficient_error_absolute_bound / float(norm))
        )
        for norm in sample_norms
    ]
    tt_bonds = [1, *retained_ranks, 1]
    removed_bond = (
        int(time_axis_position) + 1
        if int(time_axis_position) < int(n_qubits)
        else int(time_axis_position)
    )
    post_bonds = [
        bond for index, bond in enumerate(tt_bonds) if index != removed_bond
    ]
    diagnostics.update(
        {
            "required_workspace_bytes": int(required_workspace),
            "peak_workspace_bytes": int(peak_workspace),
            "retained_ranks": retained_ranks,
            "per_bond_discarded_relative_weights": discarded_relative,
            "per_bond_discarded_squared_norms": discarded_squared_norms,
            "total_discarded_squared_norm": total_discarded,
            "source_coefficient_norm_squared": source_norm_squared,
            "roundoff_relative_bound": roundoff_relative_bound,
            "coefficient_error_absolute_upper_bound": coefficient_error_absolute_bound,
            "per_sample_relative_coefficient_error_upper_bounds": per_sample_bounds,
            "max_relative_coefficient_error_upper_bound": max(per_sample_bounds),
            "tt_bonds": tt_bonds,
            "post_bonds": post_bonds,
            "exact_identity_status": (
                "verified" if total_discarded == 0.0 else "not_established"
            ),
        }
    )
    time_core = cores[int(time_axis_position)]
    pauli_cores = tuple(
        core for index, core in enumerate(cores) if index != int(time_axis_position)
    )
    return time_core, pauli_cores, diagnostics


def build_time_pauli_tensor_train(
    *,
    h0_terms: Sequence[tuple[str, complex]],
    h1_terms: Sequence[tuple[str, complex]],
    learned_labels: Sequence[str],
    direct_cd_coefficients: np.ndarray,
    lambda_samples: np.ndarray,
    n_qubits: int,
    order: Sequence[int],
    max_bond: int,
    cutoff: float,
    workspace_cap_bytes: int = _DEFAULT_MPO_WORKSPACE_CAP_BYTES,
    full_support_identity: FullSupportIdentity,
    identity_coefficient_samples: np.ndarray,
    time_axis_position: int = 0,
) -> tuple[TimePauliTensorTrain | None, dict[str, object]]:
    """Factor all requested full-K midpoint Hamiltonians in one tensor train."""

    started = time.perf_counter()
    normalized_n_qubits = _validate_n_qubits(n_qubits)
    normalized_order = _validate_order(order, n_qubits=normalized_n_qubits)
    labels = tuple(str(label) for label in learned_labels)
    assert_full_support_identity(
        full_support_identity, labels, identity_coefficient_samples
    )
    if full_support_identity.n_qubits != normalized_n_qubits:
        raise ValueError("Full-support identity qubit count does not match n_qubits.")
    union_labels, samples, source = _combined_time_pauli_coefficients(
        h0_terms=h0_terms,
        h1_terms=h1_terms,
        learned_labels=labels,
        direct_cd_coefficients=direct_cd_coefficients,
        lambda_samples=lambda_samples,
        n_qubits=normalized_n_qubits,
    )
    coefficient_scale = float(np.max(np.abs(samples))) if samples.size else 0.0
    normalization_scale = coefficient_scale if coefficient_scale > 0.0 else 1.0
    chain_labels = tuple(
        permute_pauli_label(label, normalized_order) for label in union_labels
    )
    time_core, pauli_cores, compression = _compress_time_pauli_samples(
        chain_labels,
        samples / normalization_scale,
        n_qubits=normalized_n_qubits,
        max_bond=int(max_bond),
        cutoff=float(cutoff),
        workspace_cap_bytes=int(workspace_cap_bytes),
        time_axis_position=int(time_axis_position),
    )
    diagnostics: dict[str, object] = {
        **source,
        "status": compression["status"],
        "representation": "joint_time_full_support",
        "source_completeness_status": (
            "pass"
            if int(source["learned_terms_accounted"])
            == int(source["learned_source_terms"])
            else "fail"
        ),
        "full_support_sha256": full_support_identity.sha256,
        "coefficient_normalization_scale": normalization_scale,
        "pauli_encoding": compression.get("pauli_encoding"),
        "time_axis_position": int(time_axis_position),
        "compression": compression,
        "build_seconds": float(time.perf_counter() - started),
    }
    if time_core is None:
        return None, diagnostics
    train = TimePauliTensorTrain(
        sites=_pauli_sites(normalized_n_qubits),
        time_core=time_core,
        pauli_cores=pauli_cores,
        coefficient_samples=np.asarray(samples, dtype=np.float64),
        labels=union_labels,
        order=normalized_order,
        coefficient_scale=normalization_scale,
        full_support_identity=full_support_identity,
        diagnostics=diagnostics,
        time_axis_position=int(time_axis_position),
    )
    return train, diagnostics


def slice_time_pauli_mpo(train: TimePauliTensorTrain, sample: int) -> Any:
    """Contract one sampled time core into an ordinary finite TeNPy MPO."""

    if isinstance(sample, bool) or not isinstance(sample, Integral):
        raise ValueError("sample must be an integer time index.")
    sample_index = int(sample)
    if not 0 <= sample_index < train.sample_count:
        raise IndexError("time-Pauli tensor-train sample is out of range.")
    _, _, _, MPO, npc = _require_tenpy()
    position = int(train.time_axis_position)
    selected_time = np.asarray(
        train.time_core[:, sample_index, :], dtype=np.complex128
    )
    cores = [np.array(core, copy=True) for core in train.pauli_cores]
    if position < train.n_qubits:
        cores[position] = np.asarray(
            np.tensordot(selected_time, cores[position], axes=(1, 0)),
            dtype=np.complex128,
        )
    else:
        cores[-1] = np.asarray(
            np.tensordot(cores[-1], selected_time, axes=(2, 0)),
            dtype=np.complex128,
        )
    cores[0] *= float(train.coefficient_scale)
    mpo = _mpo_from_pauli_cores(MPO, npc, sites=train.sites, cores=cores)
    row_terms = tuple(
        sorted(
            (
                permute_pauli_label(label, train.order),
                complex(train.coefficient_samples[sample_index, index]),
            )
            for index, label in enumerate(train.labels)
            if train.coefficient_samples[sample_index, index] != 0.0
        )
    )
    mpo._agp_pauli_terms = row_terms
    if (
        train.diagnostics.get("compression", {}).get("exact_identity_status")
        == "verified"
    ):
        _mark_exact_identity(
            mpo,
            row_terms,
            maximum_bond=max(max(core.shape[0], core.shape[2]) for core in cores),
        )
    return mpo


def build_direct_time_mpo(
    *,
    h0_terms: Sequence[tuple[str, complex]],
    h1_terms: Sequence[tuple[str, complex]],
    learned_labels: Sequence[str],
    direct_cd_coefficients: np.ndarray,
    lam: float,
    n_qubits: int,
    order: Sequence[int],
    max_bond: int,
    cutoff: float,
    workspace_cap_bytes: int = _DEFAULT_MPO_WORKSPACE_CAP_BYTES,
    full_support_identity: FullSupportIdentity | None = None,
    identity_coefficient_samples: np.ndarray | None = None,
    action_error_max: float = 1.0e-3,
    action_probe_product_states: int = 0,
    action_probe_seed: int = 0,
    action_probe_exact_work_cap: int = 10_000_000,
) -> tuple[Any | None, dict[str, object]]:
    """Build one complete instantaneous Hamiltonian MPO from every learned term."""

    started = time.monotonic()
    normalized_n_qubits = _validate_n_qubits(n_qubits)
    normalized_order = _validate_order(order, n_qubits=normalized_n_qubits)
    terms, combination = combine_instantaneous_full_support_terms(
        h0_terms=h0_terms,
        h1_terms=h1_terms,
        learned_labels=learned_labels,
        direct_cd_coefficients=direct_cd_coefficients,
        lam=lam,
        full_support_identity=full_support_identity,
        identity_coefficient_samples=identity_coefficient_samples,
    )
    maximum_imaginary = max((abs(complex(value).imag) for _, value in terms), default=0.0)
    maximum_absolute = max((abs(complex(value)) for _, value in terms), default=0.0)
    hermiticity_tolerance = 128.0 * np.finfo(np.float64).eps * max(maximum_absolute, 1.0)
    if maximum_imaginary > hermiticity_tolerance:
        raise ValueError(
            "The instantaneous Pauli Hamiltonian is not Hermitian within floating tolerance."
        )
    real_terms = tuple((label, complex(value).real) for label, value in terms)
    coefficient_scale = max((abs(value) for _, value in real_terms), default=0.0)
    normalization_scale = coefficient_scale if coefficient_scale > 0.0 else 1.0
    normalized_terms = tuple(
        (label, value / normalization_scale) for label, value in real_terms
    )
    source, source_metadata = build_pauli_coordinate_source(
        normalized_terms,
        n_qubits=normalized_n_qubits,
        order=normalized_order,
    )
    compressed, compression = compress_mpo_hilbert_schmidt(
        source,
        max_bond=max_bond,
        cutoff=cutoff,
        workspace_cap_bytes=workspace_cap_bytes,
    )
    source_summary = {
        key: source_metadata[key]
        for key in (
            "input_terms",
            "unique_labels",
            "included_terms",
            "dropped_terms",
            "arithmetic_zero_tolerance",
            "order",
            "representation",
        )
    }
    diagnostics: dict[str, object] = {
        **combination,
        "status": "not_feasible" if compressed is None else "ok",
        "representation": "direct_time_full_support",
        "n_qubits": normalized_n_qubits,
        "order": normalized_order,
        "coefficient_normalization_scale": float(normalization_scale),
        "hermiticity_max_imaginary": float(maximum_imaginary),
        "hermiticity_tolerance": float(hermiticity_tolerance),
        "hermiticity_status": "pass",
        "source_completeness_status": (
            "pass"
            if int(combination["learned_terms_accounted"])
            == int(combination["learned_source_terms"])
            else "fail"
        ),
        "source": source_summary,
        "compression": compression,
    }
    if compressed is None:
        diagnostics["exact_identity_status"] = "not_established"
        diagnostics["build_seconds"] = float(time.monotonic() - started)
        return None, diagnostics

    _, _, _, MPO, npc = _require_tenpy()
    scaled_tensors = _effective_finite_mpo_tensors(compressed)
    scaled_tensors[0] *= normalization_scale
    scaled = _mpo_from_effective_tensors(
        MPO,
        npc,
        sites=compressed.sites,
        tensors=scaled_tensors,
    )
    original_chain_terms = tuple(
        sorted(
            (permute_pauli_label(label, normalized_order), complex(value))
            for label, value in real_terms
        )
    )
    scaled._agp_pauli_terms = original_chain_terms
    if float(compression.get("discarded_weight", math.inf)) == 0.0:
        _mark_exact_identity(scaled, original_chain_terms, maximum_bond=int(max_bond))
    diagnostics["exact_identity_status"] = _exact_identity_certificate_status(
        scaled, original_chain_terms
    )
    if diagnostics["exact_identity_status"] == "verified":
        action_certificate: dict[str, object] = {
            "status": "pass",
            "method": "exact_identity",
            "max_relative_action_error_upper_bound": 0.0,
            "tested_probes": 0,
        }
    elif int(action_probe_product_states) > 0:
        original_source = PauliCoordinateSource(
            L=normalized_n_qubits,
            sites=scaled.sites,
            _agp_pauli_terms=original_chain_terms,
        )
        generator = np.random.default_rng(int(action_probe_seed))
        product_states = tuple(
            tuple(
                "up" if bit == 0 else "down"
                for bit in generator.integers(0, 2, size=normalized_n_qubits)
            )
            for _ in range(int(action_probe_product_states))
        )
        probes = probe_mpo_compression(
            original_source,
            scaled,
            product_states=product_states,
            random_state_count=0,
            seed=int(action_probe_seed),
            exact_work_cap=int(action_probe_exact_work_cap),
            workspace_cap_bytes=int(workspace_cap_bytes),
        )
        upper_bounds = []
        for row in probes["probes"]:
            upper = row.get(
                "relative_action_error_upper_bound",
                row.get("relative_action_error"),
            )
            if upper is not None and np.isfinite(float(upper)):
                upper_bounds.append(float(upper))
        complete = len(upper_bounds) == int(action_probe_product_states)
        maximum_error = max(upper_bounds) if upper_bounds else None
        action_certificate = {
            **probes,
            "status": (
                "pass"
                if complete
                and maximum_error is not None
                and maximum_error <= float(action_error_max)
                else "fail"
            ),
            "method": "exact_full_support_product_action",
            "max_relative_action_error_upper_bound": maximum_error,
            "action_error_max": float(action_error_max),
        }
    else:
        action_certificate = {
            "status": "not_tested",
            "method": "not_configured",
            "max_relative_action_error_upper_bound": None,
            "tested_probes": 0,
            "action_error_max": float(action_error_max),
        }
    diagnostics["action_certificate"] = action_certificate
    diagnostics["operator_gate_status"] = action_certificate["status"]
    diagnostics["post_bonds"] = list(scaled.chi)
    diagnostics["build_seconds"] = float(time.monotonic() - started)
    return scaled, diagnostics


def compress_mpo_hilbert_schmidt(
    mpo: Any,
    *,
    max_bond: int,
    cutoff: float,
    workspace_cap_bytes: int = _DEFAULT_MPO_WORKSPACE_CAP_BYTES,
) -> tuple[Any | None, dict[str, object]]:
    """Compress a full Pauli sum with a workspace-bounded TT-SVD sweep.

    ``cutoff`` is the maximum relative cumulative squared singular weight to
    discard at each bond. ``max_bond`` is a hard cap and can force a larger
    discarded weight. The exact TeNPy MPO tensors are never densified: the
    sweep consumes the combined Pauli-coordinate provenance retained by
    :func:`build_exact_pauli_mpo`.
    """
    _, _, _, MPO, npc = _require_tenpy()
    if isinstance(max_bond, bool) or not isinstance(max_bond, Integral) or int(max_bond) < 1:
        raise ValueError("max_bond must be a positive integer.")
    if (
        isinstance(cutoff, bool)
        or not np.isfinite(cutoff)
        or not 0.0 <= float(cutoff) < 1.0
    ):
        raise ValueError("cutoff must be finite and in the interval [0, 1).")
    if getattr(mpo, "bc", None) != "finite":
        raise ValueError("Hilbert-Schmidt MPO compression requires finite boundary conditions.")
    if getattr(mpo, "L", 0) < 1:
        raise ValueError("MPO must contain at least one site.")
    if (
        isinstance(workspace_cap_bytes, bool)
        or not isinstance(workspace_cap_bytes, Integral)
        or int(workspace_cap_bytes) < 1
    ):
        raise ValueError("workspace_cap_bytes must be a positive integer.")

    hard_max_bond = int(max_bond)
    relative_cutoff = float(cutoff)
    workspace_cap = int(workspace_cap_bytes)
    pauli_terms = getattr(mpo, "_agp_pauli_terms", None)
    diagnostics: dict[str, object] = {
        "status": "ok",
        "algorithm": "workspace_bounded_pauli_tt_svd",
        "max_bond": hard_max_bond,
        "cutoff": relative_cutoff,
        "cutoff_semantics": (
            "maximum relative cumulative discarded squared singular weight per Pauli unfolding"
        ),
        "workspace_cap_bytes": workspace_cap,
        "workspace_safety_margin_bytes": _MPO_WORKSPACE_SAFETY_MARGIN_BYTES,
        "workspace_semantics": (
            "hard cap on compression-created array and MPO-construction workspace; "
            "caller-owned MPO and Pauli provenance are excluded"
        ),
        "peak_workspace_bytes": 0,
        "peak_explicit_workspace_bytes": 0,
        "required_workspace_bytes": 0,
        "source_bonds": list(mpo.chi),
        "input_terms": 0 if pauli_terms is None else len(pauli_terms),
    }
    if pauli_terms is None:
        return _compression_not_feasible(
            diagnostics,
            required_workspace_bytes=0,
            peak_workspace_bytes=0,
            failed_bond=None,
            reason="exact Pauli-coordinate provenance is unavailable",
        )
    arbitrary_precision_encoding = mpo.L > 32
    diagnostics["pauli_encoding"] = (
        "python_arbitrary_precision_int"
        if arbitrary_precision_encoding
        else "numpy_uint64"
    )

    source_coefficient_norm_squared = float(
        math.fsum(abs(coefficient) ** 2 for _, coefficient in pauli_terms)
    )
    if not pauli_terms:
        zero_core_bytes = mpo.L * 4 * _COMPLEX_BYTES
        zero_output_required = (
            5 * zero_core_bytes + _MPO_WORKSPACE_SAFETY_MARGIN_BYTES
        )
        if zero_output_required > workspace_cap:
            return _compression_not_feasible(
                diagnostics,
                required_workspace_bytes=zero_output_required,
                peak_workspace_bytes=0,
                failed_bond=None,
                reason="zero compressed MPO output exceeds workspace cap",
            )
        cores = [np.zeros((1, 4, 1), dtype=np.complex128)]
        for _ in range(1, mpo.L):
            core = np.zeros((1, 4, 1), dtype=np.complex128)
            core[0, 0, 0] = 1.0
            cores.append(core)
        compressed = _mpo_from_pauli_cores(MPO, npc, sites=mpo.sites, cores=cores)
        _mark_exact_identity(compressed, pauli_terms, maximum_bond=int(max_bond))
        diagnostics.update(
            {
                "peak_workspace_bytes": zero_output_required,
                "peak_explicit_workspace_bytes": sum(core.nbytes for core in cores),
                "required_workspace_bytes": zero_output_required,
                "post_bonds": list(compressed.chi),
                "retained_ranks": [1] * max(mpo.L - 1, 0),
                "per_bond_cutoff_weights": [0.0] * max(mpo.L - 1, 0),
                "per_bond_discarded_weights": [0.0] * max(mpo.L - 1, 0),
                "discarded_weight": 0.0,
                "per_bond_discarded_squared_norms": [0.0] * max(mpo.L - 1, 0),
                "total_discarded_squared_norm": 0.0,
                "source_hilbert_schmidt_norm_squared": 0.0,
                "relative_hilbert_schmidt_error_squared": 0.0,
                "cutoff_satisfied_by_bond": [True] * max(mpo.L - 1, 0),
                "exact_identity_status": "verified",
            }
        )
        return compressed, diagnostics

    code_bytes_per_entry = (
        max(80, 32 + (2 * int(mpo.L) + 7) // 8)
        if arbitrary_precision_encoding
        else 4 * _INDEX_BYTES
    )
    initial_required = (
        len(pauli_terms) * (code_bytes_per_entry + 3 * _COMPLEX_BYTES)
        + _MPO_WORKSPACE_SAFETY_MARGIN_BYTES
    )
    diagnostics["required_workspace_bytes"] = initial_required
    if initial_required > workspace_cap:
        return _compression_not_feasible(
            diagnostics,
            required_workspace_bytes=initial_required,
            peak_workspace_bytes=0,
            failed_bond=0,
            reason="Pauli coefficient workspace exceeds cap",
        )

    if arbitrary_precision_encoding:
        encoded_rows = sorted(
            (_encode_pauli_label(label), coefficient)
            for label, coefficient in pauli_terms
        )
        codes: list[int] | np.ndarray = [row[0] for row in encoded_rows]
        values = np.asarray(
            [[row[1] for row in encoded_rows]], dtype=np.complex128
        )
    else:
        codes = np.fromiter(
            (_encode_pauli_label(label) for label, _ in pauli_terms),
            dtype=np.uint64,
            count=len(pauli_terms),
        )
        values = np.asarray(
            [[coefficient for _, coefficient in pauli_terms]], dtype=np.complex128
        )
        sort_order = np.argsort(codes, kind="stable")
        codes = codes[sort_order]
        values = values[:, sort_order]

    def encoded_workspace_bytes(encoded: list[int] | np.ndarray) -> int:
        if isinstance(encoded, np.ndarray):
            return int(encoded.nbytes)
        return len(encoded) * code_bytes_per_entry

    cores: list[np.ndarray] = []
    per_bond_cutoff_weights: list[float] = []
    per_bond_discarded_coefficient_norms: list[float] = []
    cutoff_satisfied_by_bond: list[bool] = []
    retained_ranks: list[int] = []
    peak_workspace = initial_required
    peak_explicit_workspace = int(encoded_workspace_bytes(codes) + values.nbytes)
    required_workspace = initial_required

    for bond in range(mpo.L - 1):
        previous_rank, entry_count = values.shape
        remaining_sites = mpo.L - bond
        shift = 2 * (remaining_sites - 1)
        code_workspace = encoded_workspace_bytes(codes)
        index_workspace = entry_count * (
            10 * _INDEX_BYTES if arbitrary_precision_encoding else 7 * _INDEX_BYTES
        )
        pre_index_required = (
            sum(core.nbytes for core in cores)
            + code_workspace
            + values.nbytes
            + index_workspace
            + _MPO_WORKSPACE_SAFETY_MARGIN_BYTES
        )
        required_workspace = max(required_workspace, pre_index_required)
        if pre_index_required > workspace_cap:
            return _compression_not_feasible(
                diagnostics,
                required_workspace_bytes=pre_index_required,
                peak_workspace_bytes=peak_workspace,
                failed_bond=bond,
                reason="Pauli suffix indexing exceeds workspace cap",
            )

        if arbitrary_precision_encoding:
            suffix_mask = (1 << shift) - 1 if shift else 0
            symbols = np.fromiter(
                ((int(code) >> shift) & 3 for code in codes),
                dtype=np.int64,
                count=entry_count,
            )
            suffixes: list[int] | np.ndarray = [
                int(code) & suffix_mask for code in codes
            ]
            unique_suffixes: list[int] | np.ndarray = sorted(set(suffixes))
            suffix_indices = {
                code: index for index, code in enumerate(unique_suffixes)
            }
            inverse = np.fromiter(
                (suffix_indices[int(code)] for code in suffixes),
                dtype=np.int64,
                count=entry_count,
            )
        else:
            assert isinstance(codes, np.ndarray)
            symbols = np.asarray(codes >> np.uint64(shift), dtype=np.int64)
            if shift:
                suffix_mask = np.uint64((1 << shift) - 1)
                suffixes = codes & suffix_mask
            else:
                suffixes = np.zeros_like(codes)
            unique_suffixes, inverse = np.unique(suffixes, return_inverse=True)
        column_count = len(unique_suffixes)
        suffix_workspace = encoded_workspace_bytes(suffixes)
        unique_suffix_workspace = encoded_workspace_bytes(unique_suffixes)
        row_count = previous_rank * 4
        retained_bound = min(hard_max_bond, row_count, column_count)
        matrix_bytes = row_count * column_count * _COMPLEX_BYTES
        gram_bytes = row_count * row_count * _COMPLEX_BYTES
        retained_core_bytes = previous_rank * 4 * retained_bound * _COMPLEX_BYTES
        next_values_bytes = retained_bound * column_count * _COMPLEX_BYTES
        core_bytes = sum(core.nbytes for core in cores)
        conservative_required = (
            core_bytes
            + code_workspace
            + values.nbytes
            + symbols.nbytes
            + suffix_workspace
            + unique_suffix_workspace
            + inverse.nbytes
            + 2 * matrix_bytes
            + 5 * gram_bytes
            + retained_core_bytes
            + 2 * next_values_bytes
            + row_count * np.dtype(np.float64).itemsize
            + _MPO_WORKSPACE_SAFETY_MARGIN_BYTES
        )
        required_workspace = max(required_workspace, conservative_required)
        if conservative_required > workspace_cap:
            return _compression_not_feasible(
                diagnostics,
                required_workspace_bytes=conservative_required,
                peak_workspace_bytes=peak_workspace,
                failed_bond=bond,
                reason="local Pauli unfolding SVD exceeds workspace cap",
            )

        matrix = np.zeros((row_count, column_count), dtype=np.complex128)
        for left_index in range(previous_rank):
            np.add.at(
                matrix,
                (left_index * 4 + symbols, inverse),
                values[left_index],
            )
        gram = matrix @ matrix.conj().T
        eigenvalues, left_vectors = np.linalg.eigh(gram)
        descending_eigenvalues = np.maximum(eigenvalues[::-1], 0.0)
        singular_values = np.sqrt(descending_eigenvalues)
        retained_rank, discarded_weight = _svd_retained_rank(
            singular_values,
            max_bond=hard_max_bond,
            cutoff=relative_cutoff,
            unfolding_dimension=max(row_count, column_count),
        )
        retained_ranks.append(retained_rank)
        discarded_values = singular_values[retained_rank:]
        if relative_cutoff == 0.0 and discarded_values.size:
            numerical_squared_floor = (
                8.0
                * max(row_count, column_count)
                * np.finfo(np.float64).eps
            )
            singular_scale = float(np.max(singular_values))
            if singular_scale > 0.0:
                discarded_values = discarded_values[
                    np.square(discarded_values / singular_scale)
                    > numerical_squared_floor
                ]
        singular_norm_squared = float(np.vdot(singular_values, singular_values).real)
        effective_discarded_weight = (
            0.0
            if singular_norm_squared == 0.0
            else float(
                np.vdot(discarded_values, discarded_values).real
                / singular_norm_squared
            )
        )
        per_bond_cutoff_weights.append(effective_discarded_weight)
        per_bond_discarded_coefficient_norms.append(
            float(np.vdot(discarded_values, discarded_values).real)
        )
        cutoff_satisfied_by_bond.append(
            effective_discarded_weight
            <= relative_cutoff + 64.0 * np.finfo(np.float64).eps
        )

        retained_core = np.array(
            left_vectors[:, -retained_rank:][:, ::-1].reshape(
                previous_rank, 4, retained_rank
            ),
            dtype=np.complex128,
            order="C",
            copy=True,
        )
        cores.append(retained_core)
        retained_vectors = retained_core.reshape(row_count, retained_rank)
        next_values = retained_vectors.conj().T @ matrix
        explicit_workspace = (
            core_bytes
            + code_workspace
            + values.nbytes
            + symbols.nbytes
            + suffix_workspace
            + unique_suffix_workspace
            + inverse.nbytes
            + matrix.nbytes
            + gram.nbytes
            + eigenvalues.nbytes
            + left_vectors.nbytes
            + descending_eigenvalues.nbytes
            + singular_values.nbytes
            + retained_core.nbytes
            + next_values.nbytes
        )
        peak_explicit_workspace = max(peak_explicit_workspace, explicit_workspace)
        peak_workspace = max(peak_workspace, conservative_required)
        codes = unique_suffixes
        values = next_values
        del (
            symbols,
            suffixes,
            inverse,
            matrix,
            gram,
            eigenvalues,
            left_vectors,
            descending_eigenvalues,
            singular_values,
            discarded_values,
            retained_vectors,
        )

    final_core_bytes = values.shape[0] * 4 * _COMPLEX_BYTES
    prospective_core_bytes = sum(core.nbytes for core in cores) + final_core_bytes
    final_required = max(
        required_workspace,
        encoded_workspace_bytes(codes)
        + values.nbytes
        + 5 * prospective_core_bytes
        + _MPO_WORKSPACE_SAFETY_MARGIN_BYTES,
    )
    if final_required > workspace_cap:
        return _compression_not_feasible(
            diagnostics,
            required_workspace_bytes=final_required,
            peak_workspace_bytes=peak_workspace,
            failed_bond=mpo.L - 1,
            reason="compressed MPO output exceeds workspace cap",
        )

    final_core = np.zeros((values.shape[0], 4, 1), dtype=np.complex128)
    for entry, code in enumerate(codes):
        final_core[:, int(code), 0] += values[:, entry]
    cores.append(final_core)
    compressed_output_bytes = sum(core.nbytes for core in cores) * 4

    compressed = _mpo_from_pauli_cores(MPO, npc, sites=mpo.sites, cores=cores)
    hilbert_schmidt_scale = float(2**mpo.L)
    per_bond_discarded_squared_norms = [
        weight * hilbert_schmidt_scale for weight in per_bond_discarded_coefficient_norms
    ]
    total_discarded_squared_norm = float(sum(per_bond_discarded_squared_norms))
    source_hilbert_schmidt_norm_squared = (
        source_coefficient_norm_squared * hilbert_schmidt_scale
    )
    if source_coefficient_norm_squared > 0.0:
        per_bond_discarded_weights = [
            weight / source_coefficient_norm_squared
            for weight in per_bond_discarded_coefficient_norms
        ]
    else:
        per_bond_discarded_weights = [0.0] * len(per_bond_discarded_squared_norms)
    discarded_weight = float(sum(per_bond_discarded_weights))
    if not any(weight != 0.0 for weight in per_bond_discarded_coefficient_norms):
        _mark_exact_identity(compressed, pauli_terms, maximum_bond=int(max_bond))
    diagnostics.update(
        {
            "peak_workspace_bytes": max(peak_workspace, final_required),
            "peak_explicit_workspace_bytes": peak_explicit_workspace,
            "required_workspace_bytes": final_required,
            "post_bonds": list(compressed.chi),
            "retained_ranks": retained_ranks,
            "per_bond_cutoff_weights": per_bond_cutoff_weights,
            "per_bond_discarded_weights": per_bond_discarded_weights,
            "discarded_weight": discarded_weight,
            "per_bond_discarded_squared_norms": per_bond_discarded_squared_norms,
            "total_discarded_squared_norm": total_discarded_squared_norm,
            "source_hilbert_schmidt_norm_squared": (
                source_hilbert_schmidt_norm_squared
            ),
            "relative_hilbert_schmidt_error_squared": discarded_weight,
            "cutoff_satisfied_by_bond": cutoff_satisfied_by_bond,
            "exact_identity_status": _exact_identity_certificate_status(
                compressed, pauli_terms
            ),
        }
    )
    return compressed, diagnostics


def probe_mpo_compression(
    exact_mpo: Any,
    compressed_mpo: Any,
    *,
    product_states: Sequence[Sequence[str]] | None = None,
    random_state_count: int = 2,
    random_bond: int = 4,
    seed: int = 0,
    exact_work_cap: int = 10_000_000,
    workspace_cap_bytes: int = _DEFAULT_MPO_WORKSPACE_CAP_BYTES,
) -> dict[str, object]:
    """Measure exact-versus-compressed action errors without forming exact MPO actions."""
    _, _, _, _, npc = _require_tenpy()
    from tenpy.networks.mps import MPS

    if compressed_mpo is None:
        raise ValueError("compressed_mpo must be feasible before action probing.")
    if exact_mpo.bc != "finite" or compressed_mpo.bc != "finite":
        raise ValueError("Action probes require finite MPOs.")
    if exact_mpo.L != compressed_mpo.L:
        raise ValueError("Exact and compressed MPO lengths must match.")
    if isinstance(random_state_count, bool) or not isinstance(random_state_count, Integral):
        raise ValueError("random_state_count must be a nonnegative integer.")
    if int(random_state_count) < 0:
        raise ValueError("random_state_count must be a nonnegative integer.")
    if (
        isinstance(random_bond, bool)
        or not isinstance(random_bond, Integral)
        or int(random_bond) < 1
    ):
        raise ValueError("random_bond must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if (
        isinstance(exact_work_cap, bool)
        or not isinstance(exact_work_cap, Integral)
        or int(exact_work_cap) < 1
    ):
        raise ValueError("exact_work_cap must be a positive integer.")
    if (
        isinstance(workspace_cap_bytes, bool)
        or not isinstance(workspace_cap_bytes, Integral)
        or int(workspace_cap_bytes) < 1
    ):
        raise ValueError("workspace_cap_bytes must be a positive integer.")
    pauli_terms = getattr(exact_mpo, "_agp_pauli_terms", None)
    if pauli_terms is None:
        raise ValueError("Exact MPO is missing Pauli-coordinate provenance.")
    if compressed_mpo is exact_mpo:
        exact_identity_established = True
        exact_identity_certificate_status = "same_object"
    else:
        exact_identity_certificate_status = _exact_identity_certificate_status(
            compressed_mpo, pauli_terms
        )
        exact_identity_established = exact_identity_certificate_status == "verified"

    if product_states is None:
        alternating = tuple("up" if site % 2 == 0 else "down" for site in range(exact_mpo.L))
        normalized_product_states = (
            tuple("up" for _ in range(exact_mpo.L)),
            tuple("down" for _ in range(exact_mpo.L)),
            alternating,
        )
    else:
        normalized_product_states = tuple(tuple(state) for state in product_states)
    for state in normalized_product_states:
        if len(state) != exact_mpo.L:
            raise ValueError("Each product state must specify one local state per MPO site.")
        if any(local_state not in ("up", "down") for local_state in state):
            raise ValueError("Product probes support only computational-basis states up/down.")

    probes: list[dict[str, object]] = []
    exact_cap = int(exact_work_cap)
    workspace_cap = int(workspace_cap_bytes)
    for index, product_state in enumerate(normalized_product_states):
        estimated_exact_work = len(pauli_terms) * exact_mpo.L
        estimated_sparse_bytes = max(len(pauli_terms), 1) * 192
        if estimated_exact_work > exact_cap or estimated_sparse_bytes > workspace_cap:
            probes.append(
                _not_feasible_probe(
                    f"product_{index}",
                    estimated_exact_work=estimated_exact_work,
                    required_workspace_bytes=estimated_sparse_bytes,
                    reason="sparse exact product action exceeds configured cap",
                )
            )
            continue
        input_bits = tuple(0 if state == "up" else 1 for state in product_state)
        exact_action, aggregation = _sparse_pauli_product_action(pauli_terms, input_bits)
        exact_norm_squared = float(math.fsum(abs(value) ** 2 for value in exact_action.values()))
        exact_action_norm = float(np.sqrt(exact_norm_squared))
        exact_action_norm_uncertainty = float(
            aggregation["absolute_uncertainty"] + _fsum_roundoff_bound(exact_action_norm)
        )
        if exact_norm_squared == 0.0:
            probes.append(
                _zero_action_denominator_probe(
                    f"product_{index}",
                    kind="product",
                    estimated_exact_work=estimated_exact_work,
                    peak_workspace_bytes=estimated_sparse_bytes,
                )
            )
            continue
        compressed_metrics = _compressed_product_action_metrics(
            compressed_mpo,
            input_bits=input_bits,
            exact_action=exact_action,
            workspace_cap_bytes=workspace_cap,
        )
        if compressed_metrics["status"] == "not_feasible":
            probes.append(
                _not_feasible_probe(
                    f"product_{index}",
                    estimated_exact_work=estimated_exact_work,
                    required_workspace_bytes=int(
                        compressed_metrics["required_workspace_bytes"]
                    ),
                    reason=str(compressed_metrics["resource_reason"]),
                )
            )
            continue
        difference_norm_squared = float(
            compressed_metrics["direct_difference_norm_squared"]
        ) + float(compressed_metrics["off_support_norm_squared"])
        base_difference_uncertainty = float(
            compressed_metrics["squared_difference_arithmetic_uncertainty"]
        )
        aggregation_difference_uncertainty = (
            _exact_action_aggregation_squared_difference_uncertainty(
                difference_norm_squared,
                base_difference_uncertainty,
                exact_action_norm_uncertainty,
            )
        )
        probes.append(
            _action_error_probe(
                name=f"product_{index}",
                kind="product",
                exact_norm_squared=exact_norm_squared,
                compressed_norm_squared=float(compressed_metrics["compressed_norm_squared"]),
                cross_overlap=complex(compressed_metrics["cross_overlap"]),
                difference_norm_squared=difference_norm_squared,
                squared_difference_arithmetic_uncertainty=(
                    base_difference_uncertainty + aggregation_difference_uncertainty
                ),
                roundoff_operation_estimate=int(
                    compressed_metrics["roundoff_operation_estimate"]
                ) + int(aggregation["operation_estimate"]),
                numerically_unresolved=bool(
                    compressed_metrics["off_support_numerically_unresolved"]
                ),
                exact_action_aggregation_condition_number=float(
                    aggregation["condition_number"]
                ),
                exact_action_aggregation_method=str(aggregation["method"]),
                exact_action_aggregation_absolute_uncertainty=exact_action_norm_uncertainty,
                exact_action_aggregation_operation_estimate=int(
                    aggregation["operation_estimate"]
                ),
                exact_action_aggregation_numerically_unresolved=bool(
                    aggregation["numerically_unresolved"]
                ),
                exact_action_aggregation_squared_difference_uncertainty=(
                    aggregation_difference_uncertainty
                ),
                exact_identity_established=exact_identity_established,
                estimated_exact_work=estimated_exact_work,
                peak_workspace_bytes=max(
                    estimated_sparse_bytes,
                    int(compressed_metrics["peak_workspace_bytes"]),
                ),
            )
        )

    random_generator_state = np.random.get_state()
    maximum_finite_bond = 2 ** (exact_mpo.L // 2)
    effective_random_bond = min(int(random_bond), maximum_finite_bond)
    estimated_random_work = (
        len(pauli_terms)
        * len(pauli_terms)
        * exact_mpo.L
        * max(effective_random_bond**3, 1)
    )
    estimated_random_workspace = _estimate_random_mps_workspace_bytes(
        compressed_mpo, random_bond=effective_random_bond
    )
    try:
        np.random.seed(int(seed))
        for index in range(int(random_state_count)):
            if estimated_random_work > exact_cap:
                probes.append(
                    _not_feasible_probe(
                        f"random_{index}",
                        estimated_exact_work=estimated_random_work,
                        required_workspace_bytes=0,
                        reason="Pauli-pair random-MPS contractions exceed exact_work_cap",
                    )
                )
                continue
            if estimated_random_workspace > workspace_cap:
                probes.append(
                    _not_feasible_probe(
                        f"random_{index}",
                        estimated_exact_work=estimated_random_work,
                        required_workspace_bytes=estimated_random_workspace,
                        reason=(
                            "random-MPS construction and compressed action exceed "
                            "workspace_cap_bytes"
                        ),
                    )
                )
                continue
            random_state = _seeded_random_mps(
                MPS,
                exact_mpo.sites,
                n_qubits=exact_mpo.L,
                random_bond=effective_random_bond,
            )
            random_metrics = _random_mps_action_metrics(
                pauli_terms,
                compressed_mpo,
                random_state,
                npc=npc,
                workspace_cap_bytes=workspace_cap,
            )
            if random_metrics["status"] == "not_feasible":
                probes.append(
                    _not_feasible_probe(
                        f"random_{index}",
                        estimated_exact_work=estimated_random_work,
                        required_workspace_bytes=int(
                            random_metrics["required_workspace_bytes"]
                        ),
                        reason=str(random_metrics["resource_reason"]),
                    )
                )
                continue
            exact_norm_squared = float(random_metrics["exact_norm_squared"])
            if exact_norm_squared == 0.0:
                probes.append(
                    _zero_action_denominator_probe(
                        f"random_{index}",
                        kind="random_mps",
                        estimated_exact_work=estimated_random_work,
                        peak_workspace_bytes=int(random_metrics["peak_workspace_bytes"]),
                    )
                )
                continue
            probes.append(
                _action_error_probe(
                    name=f"random_{index}",
                    kind="random_mps",
                    exact_norm_squared=exact_norm_squared,
                    compressed_norm_squared=float(
                        random_metrics["compressed_norm_squared"]
                    ),
                    cross_overlap=complex(random_metrics["cross_overlap"]),
                    roundoff_operation_estimate=int(
                        random_metrics["roundoff_operation_estimate"]
                    ),
                    exact_identity_established=exact_identity_established,
                    estimated_exact_work=estimated_random_work,
                    peak_workspace_bytes=max(
                        estimated_random_workspace,
                        int(random_metrics["peak_workspace_bytes"]),
                    ),
                )
            )
    finally:
        np.random.set_state(random_generator_state)

    tested_errors = [
        float(probe["relative_action_error"])
        for probe in probes
        if probe["status"] == "tested"
    ]
    return {
        "seed": int(seed),
        "product_states": [list(state) for state in normalized_product_states],
        "random_state_count": int(random_state_count),
        "random_bond": int(random_bond),
        "effective_random_bond": effective_random_bond,
        "exact_work_cap": exact_cap,
        "workspace_cap_bytes": workspace_cap,
        "exact_identity_certificate_status": exact_identity_certificate_status,
        "roundoff_model": _GAMMA_N_MODEL,
        "roundoff_assumptions": _GAMMA_N_ASSUMPTIONS,
        "peak_workspace_bytes": max(
            (int(probe.get("peak_workspace_bytes", 0)) for probe in probes),
            default=0,
        ),
        "tested_probes": len(tested_errors),
        "not_tested_probes": sum(probe["status"] == "not_tested" for probe in probes),
        "not_feasible_probes": sum(
            probe["status"] == "not_feasible" for probe in probes
        ),
        "numerically_unresolved_probes": sum(
            probe["status"] == "numerically_unresolved" for probe in probes
        ),
        "max_relative_action_error": max(tested_errors) if tested_errors else None,
        "mean_relative_action_error": (
            float(np.mean(tested_errors)) if tested_errors else None
        ),
        "probes": probes,
    }


def _sparse_pauli_product_action(
    pauli_terms: Sequence[tuple[str, complex]],
    input_bits: Sequence[int],
) -> tuple[dict[int, complex], dict[str, object]]:
    """Apply Pauli terms to a product state with stable per-bitstring sums."""
    buckets: dict[int, list[complex]] = {}
    for label, coefficient in pauli_terms:
        output = 0
        phase = 1.0 + 0.0j
        for symbol, bit in zip(label, input_bits):
            output_bit = bit
            if symbol == "X":
                output_bit = 1 - bit
            elif symbol == "Y":
                output_bit = 1 - bit
                phase *= 1.0j if bit == 0 else -1.0j
            elif symbol == "Z" and bit == 1:
                phase *= -1.0
            output = (output << 1) | output_bit
        buckets.setdefault(output, []).append(coefficient * phase)

    amplitudes: dict[int, complex] = {}
    condition_number = 1.0
    squared_absolute_uncertainty = 0.0
    operation_estimate = 0
    numerically_unresolved = False
    for state, contributions in buckets.items():
        real_values = [value.real for value in contributions]
        imaginary_values = [value.imag for value in contributions]
        real = math.fsum(real_values)
        imaginary = math.fsum(imaginary_values)
        amplitude = complex(real, imaginary)
        contribution_scale = math.fsum(abs(value) for value in contributions)
        if contribution_scale > 0.0:
            if amplitude == 0.0j:
                condition_number = math.inf
            else:
                condition_number = max(condition_number, contribution_scale / abs(amplitude))
        component_uncertainty = math.hypot(
            _fsum_roundoff_bound(real), _fsum_roundoff_bound(imaginary)
        )
        squared_absolute_uncertainty += component_uncertainty**2
        operation_estimate += 2 * max(len(contributions), 1)
        if amplitude != 0.0j:
            amplitudes[state] = amplitude

    absolute_uncertainty = float(np.sqrt(squared_absolute_uncertainty))
    action_norm = float(np.sqrt(math.fsum(abs(value) ** 2 for value in amplitudes.values())))
    if not np.isfinite(condition_number) and action_norm <= absolute_uncertainty:
        numerically_unresolved = True
    return amplitudes, {
        "method": "math.fsum_components",
        "condition_number": condition_number,
        "absolute_uncertainty": absolute_uncertainty,
        "operation_estimate": operation_estimate,
        "numerically_unresolved": numerically_unresolved,
    }


def _compressed_product_action_metrics(
    mpo: Any,
    *,
    input_bits: Sequence[int],
    exact_action: dict[int, complex],
    workspace_cap_bytes: int,
) -> dict[str, object]:
    output_count = len(exact_action)
    maximum_bond = max(mpo.chi)
    maximum_local_bytes = max(
        mpo.chi[site] * mpo.chi[site + 1] * 4 * _COMPLEX_BYTES
        for site in range(mpo.L)
    )
    query_bytes = max(output_count, 1) * maximum_bond * _COMPLEX_BYTES
    amplitude_bytes = max(output_count, 1) * _COMPLEX_BYTES
    arbitrary_precision_outputs = mpo.L > 63
    output_index_item_bytes = (
        max(80, 32 + (int(mpo.L) + 7) // 8)
        if arbitrary_precision_outputs
        else _INDEX_BYTES
    )
    output_index_bytes = max(output_count, 1) * output_index_item_bytes
    transfer_bytes = maximum_bond * maximum_bond * _COMPLEX_BYTES
    required_workspace = (
        maximum_local_bytes
        + 2 * query_bytes
        + 4 * transfer_bytes
        + 3 * amplitude_bytes
        + output_index_bytes
    )
    if required_workspace > workspace_cap_bytes:
        return {
            "status": "not_feasible",
            "required_workspace_bytes": required_workspace,
            "resource_reason": "compressed product-state transfer exceeds workspace cap",
        }

    sorted_output_states = tuple(sorted(exact_action))
    output_states: tuple[int, ...] | np.ndarray
    if arbitrary_precision_outputs:
        output_states = sorted_output_states
    else:
        output_states = np.asarray(sorted_output_states, dtype=np.uint64)
    output_state_count = len(output_states)
    exact_amplitudes = np.asarray(
        [exact_action[int(state)] for state in output_states], dtype=np.complex128
    )
    left_boundary = mpo.get_IdL(0)
    right_boundary = mpo.get_IdR(mpo.L - 1)
    if left_boundary is None or right_boundary is None:
        raise ValueError("Compressed finite MPO must expose both boundary indices.")
    query = np.zeros((output_state_count, mpo.chi[0]), dtype=np.complex128)
    query[:, int(left_boundary)] = 1.0
    density = np.zeros((mpo.chi[0], mpo.chi[0]), dtype=np.complex128)
    density[int(left_boundary), int(left_boundary)] = 1.0
    peak_workspace = query.nbytes + density.nbytes
    transfer_operation_estimate = 0

    for site, input_bit in enumerate(input_bits):
        tensor = _mpo_tensor_ndarray(mpo, site)
        next_query = np.zeros((output_state_count, tensor.shape[1]), dtype=np.complex128)
        next_density = np.zeros((tensor.shape[1], tensor.shape[1]), dtype=np.complex128)
        shift = mpo.L - site - 1
        if arbitrary_precision_outputs:
            output_bits = np.fromiter(
                ((int(state) >> shift) & 1 for state in output_states),
                dtype=np.int8,
                count=output_state_count,
            )
        else:
            output_bits = (output_states >> np.uint64(shift)) & np.uint64(1)
        for output_bit in (0, 1):
            local = tensor[:, :, output_bit, input_bit]
            selected = output_bits == output_bit
            if np.any(selected):
                next_query[selected] = query[selected] @ local
            next_density += local.T @ density @ local.conj()
        left_bond, right_bond = tensor.shape[:2]
        transfer_operation_estimate += (
            4 * left_bond * right_bond * (left_bond + right_bond)
            + 4 * max(output_state_count, 1) * left_bond * right_bond
        )
        peak_workspace = max(
            peak_workspace,
            tensor.nbytes
            + query.nbytes
            + next_query.nbytes
            + density.nbytes
            + next_density.nbytes,
        )
        query = next_query
        density = next_density

    compressed_amplitudes = query[:, int(right_boundary)]
    compressed_norm_squared = float(np.real(density[int(right_boundary), int(right_boundary)]))
    compressed_norm_squared = _nonnegative_roundoff(
        compressed_norm_squared, scale=abs(compressed_norm_squared)
    )
    queried_compressed_norm_squared = float(
        np.vdot(compressed_amplitudes, compressed_amplitudes).real
    )
    (
        unqueried_compressed_norm_squared,
        off_support_roundoff_bound,
        off_support_numerically_unresolved,
    ) = (
        _cancellation_safe_nonnegative_difference(
            compressed_norm_squared,
            queried_compressed_norm_squared,
            operation_estimate=transfer_operation_estimate + 2 * output_state_count,
        )
    )
    amplitude_differences = compressed_amplitudes - exact_amplitudes
    direct_difference_norm_squared = float(
        np.vdot(amplitude_differences, amplitude_differences).real
    )
    direct_operation_estimate = max(1, 16 * output_state_count)
    direct_scale = (
        float(np.vdot(exact_amplitudes, exact_amplitudes).real)
        + queried_compressed_norm_squared
    )
    direct_roundoff_bound = _gamma_n_bound(direct_scale, direct_operation_estimate)
    return {
        "status": "ok",
        "compressed_norm_squared": compressed_norm_squared,
        "cross_overlap": np.vdot(exact_amplitudes, compressed_amplitudes),
        "direct_difference_norm_squared": direct_difference_norm_squared,
        "off_support_numerically_unresolved": off_support_numerically_unresolved,
        "off_support_norm_squared": unqueried_compressed_norm_squared,
        "squared_difference_arithmetic_uncertainty": (
            direct_roundoff_bound + off_support_roundoff_bound
        ),
        "roundoff_operation_estimate": (
            direct_operation_estimate + transfer_operation_estimate + 2 * output_state_count
        ),
        "peak_workspace_bytes": max(peak_workspace, required_workspace),
    }


def _estimate_random_mps_workspace_bytes(mpo: Any, *, random_bond: int) -> int:
    state_bonds = [1]
    state_bonds.extend(
        min(random_bond, 2 ** min(cut, mpo.L - cut))
        for cut in range(1, mpo.L)
    )
    state_bonds.append(1)
    action_bonds = [
        state_bond * mpo_bond
        for state_bond, mpo_bond in zip(state_bonds, mpo.chi)
    ]
    state_tensor_bytes = sum(
        state_bonds[site] * 2 * state_bonds[site + 1] * _COMPLEX_BYTES
        for site in range(mpo.L)
    )
    compressed_action_bytes = sum(
        action_bonds[site] * 2 * action_bonds[site + 1] * _COMPLEX_BYTES
        for site in range(mpo.L)
    )
    return 3 * compressed_action_bytes + 3 * state_tensor_bytes


def _seeded_random_mps(
    MPS: Any,
    sites: Sequence[Any],
    *,
    n_qubits: int,
    random_bond: int,
) -> Any:
    if n_qubits == 1:
        local_state = np.random.normal(size=2) + 1.0j * np.random.normal(size=2)
        local_state /= np.linalg.norm(local_state)
        return MPS.from_product_state(
            list(sites),
            [local_state],
            bc="finite",
            dtype=np.complex128,
            unit_cell_width=n_qubits,
        )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="unit_cell_width is a new argument.*",
            category=UserWarning,
            module=r"tenpy\.networks\.mps",
        )
        return MPS.from_random_unitary_evolution(
            sites,
            chi=random_bond,
            p_state=["up"] * n_qubits,
            bc="finite",
            dtype=np.complex128,
        )


def _random_mps_action_metrics(
    pauli_terms: Sequence[tuple[str, complex]],
    compressed_mpo: Any,
    state: Any,
    *,
    npc: Any,
    workspace_cap_bytes: int,
) -> dict[str, object]:
    state_bonds = [1] + list(state.chi) + [1]
    action_bonds = [
        state_bond * mpo_bond
        for state_bond, mpo_bond in zip(state_bonds, compressed_mpo.chi)
    ]
    compressed_action_bytes = sum(
        action_bonds[site] * 2 * action_bonds[site + 1] * _COMPLEX_BYTES
        for site in range(compressed_mpo.L)
    )
    pauli_state_bytes = sum(
        state_bonds[site] * 2 * state_bonds[site + 1] * _COMPLEX_BYTES
        for site in range(compressed_mpo.L)
    )
    required_workspace = 3 * compressed_action_bytes + 2 * pauli_state_bytes
    if required_workspace > workspace_cap_bytes:
        return {
            "status": "not_feasible",
            "required_workspace_bytes": required_workspace,
            "resource_reason": "compressed random-MPS action exceeds workspace cap",
        }

    state_bond = max(state_bonds)
    action_bond = max(action_bonds)
    roundoff_operation_estimate = (
        32 * len(pauli_terms) * len(pauli_terms) * compressed_mpo.L * state_bond**3
        + 32 * compressed_mpo.L * action_bond**3
        + 32 * len(pauli_terms) * compressed_mpo.L * max(state_bond, action_bond) ** 3
    )

    exact_norm_squared = 0.0j
    for left_label, left_coefficient in pauli_terms:
        for right_label, right_coefficient in pauli_terms:
            phase, product_label = _multiply_pauli_labels(left_label, right_label)
            expectation = _mps_pauli_expectation(state, product_label)
            exact_norm_squared += (
                left_coefficient.conjugate()
                * right_coefficient
                * phase
                * expectation
            )
    exact_norm_real = _nonnegative_roundoff(
        float(np.real(exact_norm_squared)),
        scale=float(abs(exact_norm_squared)),
    )
    if exact_norm_real == 0.0:
        return {
            "status": "ok",
            "exact_norm_squared": 0.0,
            "compressed_norm_squared": 0.0,
            "cross_overlap": 0.0j,
            "roundoff_operation_estimate": roundoff_operation_estimate,
            "peak_workspace_bytes": required_workspace,
        }

    compressed_action = _apply_mpo_action(compressed_mpo, state, npc)
    compressed_norm_squared = _real_overlap(compressed_action, compressed_action)
    cross_overlap = 0.0j
    for label, coefficient in pauli_terms:
        pauli_state = _apply_pauli_string(state, label)
        cross_overlap += coefficient.conjugate() * pauli_state.overlap(
            compressed_action, ignore_form=True
        )
    return {
        "status": "ok",
        "exact_norm_squared": exact_norm_real,
        "compressed_norm_squared": compressed_norm_squared,
        "cross_overlap": cross_overlap,
        "roundoff_operation_estimate": roundoff_operation_estimate,
        "peak_workspace_bytes": required_workspace,
    }


def _action_error_probe(
    *,
    name: str,
    kind: str,
    exact_norm_squared: float,
    compressed_norm_squared: float,
    cross_overlap: complex,
    estimated_exact_work: int,
    peak_workspace_bytes: int,
    difference_norm_squared: float | None = None,
    squared_difference_arithmetic_uncertainty: float = 0.0,
    roundoff_operation_estimate: int = 0,
    numerically_unresolved: bool = False,
    exact_action_aggregation_condition_number: float = 1.0,
    exact_action_aggregation_method: str = "not_applicable",
    exact_action_aggregation_absolute_uncertainty: float = 0.0,
    exact_action_aggregation_operation_estimate: int = 0,
    exact_action_aggregation_numerically_unresolved: bool = False,
    exact_action_aggregation_squared_difference_uncertainty: float = 0.0,
    exact_identity_established: bool = False,
) -> dict[str, object]:
    if exact_identity_established:
        difference_norm_squared = 0.0
        squared_difference_arithmetic_uncertainty = 0.0
        roundoff_operation_estimate = 0
        numerically_unresolved = False
        exact_action_aggregation_absolute_uncertainty = 0.0
        exact_action_aggregation_numerically_unresolved = False
        exact_action_aggregation_squared_difference_uncertainty = 0.0
        error_method = "provenance_established_identity"
    elif difference_norm_squared is None:
        (
            difference_norm_squared,
            squared_difference_arithmetic_uncertainty,
            numerically_unresolved,
        ) = _overlap_squared_difference(
            exact_norm_squared,
            compressed_norm_squared,
            cross_overlap,
            operation_estimate=roundoff_operation_estimate,
        )
        error_method = "overlap_with_roundoff_bound"
    else:
        error_method = "direct_product_amplitudes"
    lower_squared, upper_squared = _squared_difference_interval(
        difference_norm_squared, squared_difference_arithmetic_uncertainty
    )
    action_norm = float(np.sqrt(exact_norm_squared))
    action_norm_lower_bound = max(
        0.0, action_norm - exact_action_aggregation_absolute_uncertainty
    )
    action_norm_upper_bound = (
        action_norm + exact_action_aggregation_absolute_uncertainty
    )
    diagnostics = {
        "squared_difference_estimate": float(difference_norm_squared),
        "squared_difference_arithmetic_uncertainty": float(
            squared_difference_arithmetic_uncertainty
        ),
        "squared_difference_lower_bound": lower_squared,
        "squared_difference_upper_bound": upper_squared,
        "roundoff_operation_estimate": int(roundoff_operation_estimate),
        "roundoff_model": _GAMMA_N_MODEL,
        "roundoff_assumptions": _GAMMA_N_ASSUMPTIONS,
        "exact_action_aggregation_condition_number": float(
            exact_action_aggregation_condition_number
        ),
        "exact_action_aggregation_method": exact_action_aggregation_method,
        "exact_action_aggregation_absolute_uncertainty": float(
            exact_action_aggregation_absolute_uncertainty
        ),
        "exact_action_aggregation_operation_estimate": int(
            exact_action_aggregation_operation_estimate
        ),
        "exact_action_aggregation_numerically_unresolved": bool(
            exact_action_aggregation_numerically_unresolved
        ),
        "exact_action_aggregation_squared_difference_uncertainty": float(
            exact_action_aggregation_squared_difference_uncertainty
        ),
        "exact_action_norm_lower_bound": action_norm_lower_bound,
        "exact_action_norm_upper_bound": action_norm_upper_bound,
    }
    if (
        numerically_unresolved
        or exact_action_aggregation_numerically_unresolved
        or action_norm_lower_bound == 0.0
    ):
        relative_upper_bound = (
            math.inf
            if action_norm_lower_bound == 0.0
            else float(np.sqrt(upper_squared) / action_norm_lower_bound)
        )
        return {
            "name": name,
            "kind": kind,
            "status": "numerically_unresolved",
            "action_norm": action_norm,
            "relative_action_error": None,
            "relative_action_error_lower_bound": float(
                np.sqrt(lower_squared) / action_norm_upper_bound
            ),
            "relative_action_error_upper_bound": relative_upper_bound,
            "relative_action_error_numerical_floor": float(
                math.inf
                if action_norm_lower_bound == 0.0
                else np.sqrt(squared_difference_arithmetic_uncertainty)
                / action_norm_lower_bound
            ),
            "action_error_method": (
                "condition_limited_exact_action"
                if exact_action_aggregation_numerically_unresolved
                or action_norm_lower_bound == 0.0
                else "cancellation_limited_norm_difference"
            ),
            "estimated_exact_work": int(estimated_exact_work),
            "peak_workspace_bytes": int(peak_workspace_bytes),
            **diagnostics,
        }
    difference_norm_squared = _nonnegative_roundoff(
        difference_norm_squared,
        scale=max(exact_norm_squared, compressed_norm_squared),
    )
    return {
        "name": name,
        "kind": kind,
        "status": "tested",
        "action_norm": float(np.sqrt(exact_norm_squared)),
        "relative_action_error": float(
            np.sqrt(difference_norm_squared / exact_norm_squared)
        ),
        "action_error_method": error_method,
        "difference_roundoff_bound": float(squared_difference_arithmetic_uncertainty),
        "estimated_exact_work": int(estimated_exact_work),
        "peak_workspace_bytes": int(peak_workspace_bytes),
        **diagnostics,
    }


def _not_feasible_probe(
    name: str,
    *,
    estimated_exact_work: int,
    required_workspace_bytes: int,
    reason: str,
) -> dict[str, object]:
    return {
        "name": name,
        "status": "not_feasible",
        "relative_action_error": None,
        "estimated_exact_work": int(estimated_exact_work),
        "required_workspace_bytes": int(required_workspace_bytes),
        "peak_workspace_bytes": 0,
        "resource_reason": reason,
    }


def _zero_action_denominator_probe(
    name: str,
    *,
    kind: str,
    estimated_exact_work: int,
    peak_workspace_bytes: int,
) -> dict[str, object]:
    """Return the explicit non-comparable status before relative-error division."""
    return {
        "name": name,
        "kind": kind,
        "status": "not_tested",
        "reason": "zero_action_denominator",
        "action_norm": 0.0,
        "relative_action_error": None,
        "relative_action_error_lower_bound": None,
        "relative_action_error_upper_bound": None,
        "relative_action_error_numerical_floor": None,
        "estimated_exact_work": int(estimated_exact_work),
        "peak_workspace_bytes": int(peak_workspace_bytes),
    }


def _mpo_tensor_ndarray(mpo: Any, site: int) -> np.ndarray:
    tensor = mpo.get_W(site)
    labels = tensor.get_leg_labels()
    axes = [labels.index(label) for label in ("wL", "wR", "p", "p*")]
    return np.transpose(tensor.to_ndarray(), axes)


def _multiply_pauli_labels(left: str, right: str) -> tuple[complex, str]:
    multiplication = {
        ("I", "I"): (1.0, "I"),
        ("I", "X"): (1.0, "X"),
        ("I", "Y"): (1.0, "Y"),
        ("I", "Z"): (1.0, "Z"),
        ("X", "I"): (1.0, "X"),
        ("Y", "I"): (1.0, "Y"),
        ("Z", "I"): (1.0, "Z"),
        ("X", "X"): (1.0, "I"),
        ("Y", "Y"): (1.0, "I"),
        ("Z", "Z"): (1.0, "I"),
        ("X", "Y"): (1.0j, "Z"),
        ("Y", "X"): (-1.0j, "Z"),
        ("Y", "Z"): (1.0j, "X"),
        ("Z", "Y"): (-1.0j, "X"),
        ("Z", "X"): (1.0j, "Y"),
        ("X", "Z"): (-1.0j, "Y"),
    }
    phase = 1.0 + 0.0j
    symbols = []
    for left_symbol, right_symbol in zip(left, right):
        local_phase, symbol = multiplication[(left_symbol, right_symbol)]
        phase *= local_phase
        symbols.append(symbol)
    return phase, "".join(symbols)


def _mps_pauli_expectation(state: Any, label: str) -> complex:
    term = [(symbol, site) for site, symbol in enumerate(label) if symbol != "I"]
    if not term:
        return complex(abs(state.norm) ** 2)
    return complex(state.expectation_value_term(term)) * abs(state.norm) ** 2


def _apply_pauli_string(state: Any, label: str) -> Any:
    result = state.copy()
    for site, symbol in enumerate(label):
        if symbol != "I":
            result.apply_local_op(site, symbol, unitary=True)
    return result


def _nonnegative_roundoff(
    value: float, *, scale: float, operation_estimate: int = 1
) -> float:
    if value < 0.0 and abs(value) <= _gamma_n_bound(scale, operation_estimate):
        return 0.0
    if value < 0.0:
        raise ValueError("Action contraction produced a negative squared norm.")
    return value


def _cancellation_safe_nonnegative_difference(
    total: float, part: float, *, operation_estimate: int
) -> tuple[float, float, bool]:
    """Return an off-support norm or an explicit bound when subtraction is ambiguous."""
    difference = total - part
    scale = max(abs(total), abs(part))
    roundoff_bound = _gamma_n_bound(scale, operation_estimate)
    return difference, roundoff_bound, abs(difference) <= roundoff_bound


def _gamma_n_bound(scale: float, operation_estimate: int) -> float:
    """Bound absolute binary64 first-order accumulation error at the given scale."""
    if scale == 0.0:
        return 0.0
    operations = max(1, int(operation_estimate))
    epsilon = np.finfo(np.float64).eps
    product = operations * epsilon
    if product >= 1.0:
        return math.inf
    return float(abs(scale) * product / (1.0 - product))


def _fsum_roundoff_bound(value: float) -> float:
    """Bound final binary64 rounding after ``math.fsum`` component accumulation."""
    if not np.isfinite(value):
        return math.inf
    return float(2.0 * math.ulp(float(value)))


def _exact_action_aggregation_squared_difference_uncertainty(
    difference_norm_squared: float,
    base_uncertainty: float,
    action_norm_uncertainty: float,
) -> float:
    """Bound squared-error drift from a stable exact-action amplitude uncertainty."""
    base_upper_bound = max(0.0, float(difference_norm_squared)) + max(
        0.0, float(base_uncertainty)
    )
    amplitude_uncertainty = max(0.0, float(action_norm_uncertainty))
    return float(
        2.0 * np.sqrt(base_upper_bound) * amplitude_uncertainty
        + amplitude_uncertainty**2
    )


def _squared_difference_interval(
    difference_norm_squared: float, arithmetic_uncertainty: float
) -> tuple[float, float]:
    """Return the conservative interval for a nonnegative squared difference."""
    uncertainty = max(0.0, float(arithmetic_uncertainty))
    difference = float(difference_norm_squared)
    return (
        max(0.0, difference - uncertainty),
        max(0.0, difference) + uncertainty,
    )


def _overlap_squared_difference(
    exact_norm_squared: float,
    compressed_norm_squared: float,
    cross_overlap: complex,
    *,
    operation_estimate: int,
) -> tuple[float, float, bool]:
    """Return an overlap-derived squared error with an explicit cancellation bound."""
    scale = max(
        abs(exact_norm_squared),
        abs(compressed_norm_squared),
        abs(cross_overlap),
    )
    roundoff_bound = _gamma_n_bound(scale, operation_estimate)
    difference = (
        exact_norm_squared
        + compressed_norm_squared
        - 2.0 * float(np.real(cross_overlap))
    )
    return difference, roundoff_bound, abs(difference) <= roundoff_bound


def dense_pauli_sum(terms: Sequence[tuple[str, complex]]) -> np.ndarray:
    """Return a tiny dense Pauli sum for tests, rejecting systems above four qubits."""
    normalized_terms = list(terms)
    if not normalized_terms:
        raise ValueError("dense_pauli_sum requires at least one Pauli term.")
    first_label = normalized_terms[0][0]
    if not isinstance(first_label, str):
        raise ValueError("Pauli labels must be strings.")
    n_qubits = len(first_label)
    _validate_dense_helper_size(n_qubits)
    matrices = {
        "I": np.eye(2, dtype=np.complex128),
        "X": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        "Y": np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
        "Z": np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
    }
    result = np.zeros((2**n_qubits, 2**n_qubits), dtype=np.complex128)
    for label, coefficient in normalized_terms:
        _validate_pauli_label(label, n_qubits=n_qubits)
        local_operator = np.asarray([[1.0]], dtype=np.complex128)
        for symbol in label:
            local_operator = np.kron(local_operator, matrices[symbol])
        result += _finite_complex(coefficient) * local_operator
    return result


def mpo_to_dense(mpo: Any) -> np.ndarray:
    """Contract a finite MPO to a dense matrix only for test systems with q <= 4."""
    n_qubits = int(mpo.L)
    _validate_dense_helper_size(n_qubits)
    if mpo.bc != "finite":
        raise ValueError("mpo_to_dense requires finite boundary conditions.")
    tensors = _effective_finite_mpo_tensors(mpo)
    contraction = tensors[0][0]
    for tensor in tensors[1:]:
        contraction = np.tensordot(contraction, tensor, axes=(0, 0))
        contraction = np.moveaxis(contraction, -3, 0)
    contraction = contraction[0]
    output_axes = tuple(range(0, 2 * n_qubits, 2))
    input_axes = tuple(range(1, 2 * n_qubits, 2))
    return contraction.transpose(output_axes + input_axes).reshape(
        2**n_qubits, 2**n_qubits
    )


def _configured_action_probe(
    exact_mpo: Any | None,
    compressed_mpo: Any | None,
    *,
    product_state_count: int,
    random_state_count: int,
    seed: int,
    exact_work_cap: int,
    workspace_cap_bytes: int,
) -> dict[str, object]:
    """Run bounded deterministic action probes, preserving inconclusive outcomes."""

    if exact_mpo is None or compressed_mpo is None:
        return {"status": "not_feasible", "reason": "static MPO compression was not feasible"}
    if int(product_state_count) == 0 and int(random_state_count) == 0:
        return {"status": "not_tested", "reason": "action probes were not configured"}
    generator = np.random.default_rng(int(seed))
    product_states = tuple(
        tuple("up" if bit == 0 else "down" for bit in generator.integers(0, 2, size=int(exact_mpo.L)))
        for _ in range(int(product_state_count))
    )
    try:
        result = probe_mpo_compression(
            exact_mpo,
            compressed_mpo,
            product_states=product_states,
            random_state_count=int(random_state_count),
            seed=int(seed),
            exact_work_cap=int(exact_work_cap),
            workspace_cap_bytes=int(workspace_cap_bytes),
        )
    except (MemoryError, RuntimeError, ValueError) as error:
        return {
            "status": "unresolved_error",
            "reason": str(error),
            "seed": int(seed),
            "product_state_count": int(product_state_count),
            "random_state_count": int(random_state_count),
            "exact_work_cap": int(exact_work_cap),
            "workspace_cap_bytes": int(workspace_cap_bytes),
            "finite_error_intervals": [],
        }
    upper_bounds: list[float] = []
    lower_bounds: list[float] = []
    finite_error_intervals: list[dict[str, object]] = []
    statuses: list[str] = []
    for probe in result["probes"]:
        statuses.append(str(probe["status"]))
        lower = probe.get("relative_action_error_lower_bound", probe.get("relative_action_error"))
        upper = probe.get("relative_action_error_upper_bound", probe.get("relative_action_error"))
        if (
            lower is not None
            and upper is not None
            and np.isfinite(float(lower))
            and np.isfinite(float(upper))
        ):
            lower_value = float(lower)
            upper_value = float(upper)
            lower_bounds.append(lower_value)
            upper_bounds.append(upper_value)
            finite_error_intervals.append(
                {
                    "name": str(probe.get("name", "unnamed")),
                    "status": str(probe["status"]),
                    "lower_bound": lower_value,
                    "upper_bound": upper_value,
                }
            )
    if "not_feasible" in statuses:
        status = "not_feasible"
    elif any(item in {"not_tested", "numerically_unresolved"} for item in statuses):
        status = "numerically_unresolved" if "numerically_unresolved" in statuses else "not_tested"
    elif upper_bounds:
        status = "measured"
    else:
        status = "not_tested"
    return {
        "status": status,
        "probe_statuses": statuses,
        "finite_error_intervals": finite_error_intervals,
        "max_relative_action_error_lower_bound": max(lower_bounds) if lower_bounds else None,
        "max_relative_action_error_upper_bound": max(upper_bounds) if upper_bounds else None,
        **result,
    }


def prepare_tdvp_operators(
    *,
    labels: Sequence[str],
    static_modes: np.ndarray,
    temporal_factors: np.ndarray,
    n_qubits: int,
    order: Sequence[int],
    mpo_max_bond: int,
    mpo_cutoff: float,
    h0_terms: Sequence[tuple[str, complex]] | None = None,
    h1_terms: Sequence[tuple[str, complex]] | None = None,
    temporal_factorization: TemporalFactorization | None = None,
    mpo_workspace_cap_bytes: int = _DEFAULT_MPO_WORKSPACE_CAP_BYTES,
    action_probe_product_states: int = 0,
    action_probe_random_mps: int = 0,
    action_probe_seed: int = 0,
    action_probe_exact_work_cap: int = 10_000_000,
) -> PreparedTDVPOperators:
    """Build compressed static MPOs without dropping any declared CD mode or term."""
    n_qubits = _validate_n_qubits(n_qubits)
    normalized_order = _validate_order(order, n_qubits=n_qubits)
    mode_array = _finite_real_array(static_modes, name="static_modes", ndim=2)
    factor_array = _finite_real_array(temporal_factors, name="temporal_factors", ndim=2)
    normalized_labels = tuple(labels)
    if len(normalized_labels) != mode_array.shape[1]:
        raise ValueError("labels and static_modes must contain the same number of terms.")
    if factor_array.shape[1] != mode_array.shape[0]:
        raise ValueError("temporal_factors and static_modes must have the same temporal rank.")
    if not factor_array.shape[0]:
        raise ValueError("temporal_factors must contain at least one time sample.")
    for label in normalized_labels:
        _validate_pauli_label(label, n_qubits=n_qubits)
    if normalized_labels and np.any(np.all(mode_array == 0.0, axis=0)):
        raise ValueError("Every declared CD label must contribute to at least one temporal mode.")
    if mode_array.shape[0] and np.any(np.all(mode_array == 0.0, axis=1)):
        raise ValueError("Every declared temporal mode must contain at least one CD term.")

    static_compression: dict[str, object] = {"cd_modes": []}
    sites: list[object] = []
    h0_mpo = h1_mpo = h0_exact_mpo = h1_exact_mpo = None
    cd_mode_exact_mpos: list[object] = []
    status = "ok"

    def build_static(
        name: str, terms: Sequence[tuple[str, complex]]
    ) -> tuple[Any | None, Any | None]:
        nonlocal sites, status
        exact, metadata = build_pauli_coordinate_source(
            terms, n_qubits=n_qubits, order=normalized_order
        )
        if not sites:
            sites = list(exact.sites)
        compressed, compression = compress_mpo_hilbert_schmidt(
            exact,
            max_bond=mpo_max_bond,
            cutoff=mpo_cutoff,
            workspace_cap_bytes=mpo_workspace_cap_bytes,
        )
        static_compression[name] = {"build": metadata, "compression": compression}
        if compressed is None:
            status = "not_feasible"
        else:
            compressed._agp_pauli_terms = exact._agp_pauli_terms
        return exact, compressed

    if h0_terms is not None:
        h0_exact_mpo, h0_mpo = build_static("h0", h0_terms)
    if h1_terms is not None:
        h1_exact_mpo, h1_mpo = build_static("h1", h1_terms)

    cd_mode_mpos: list[object] = []
    for mode_index, row in enumerate(mode_array):
        exact, metadata = build_pauli_coordinate_source(
            list(zip(normalized_labels, row, strict=True)),
            n_qubits=n_qubits,
            order=normalized_order,
        )
        if not sites:
            sites = list(exact.sites)
        compressed, compression = compress_mpo_hilbert_schmidt(
            exact,
            max_bond=mpo_max_bond,
            cutoff=mpo_cutoff,
            workspace_cap_bytes=mpo_workspace_cap_bytes,
        )
        static_compression["cd_modes"].append(
            {"mode": mode_index, "build": metadata, "compression": compression,
             "status": "ok" if compressed is not None else "not_feasible"}
        )
        if compressed is None:
            status = "not_feasible"
        else:
            compressed._agp_pauli_terms = exact._agp_pauli_terms
            cd_mode_mpos.append(compressed)
        cd_mode_exact_mpos.append(exact)

    if not sites:
        identity, _ = build_pauli_coordinate_source(
            [("I" * n_qubits, 0.0)], n_qubits=n_qubits, order=normalized_order
        )
        sites = list(identity.sites)

    static_action_probes: dict[str, object] = {}
    for index, (name, exact, compressed) in enumerate(
        [
            ("h0", h0_exact_mpo, h0_mpo),
            ("h1", h1_exact_mpo, h1_mpo),
            *[
                (f"cd_mode_{mode}", exact, compressed)
                for mode, (exact, compressed) in enumerate(zip(cd_mode_exact_mpos, cd_mode_mpos))
            ],
        ]
    ):
        static_action_probes[name] = _configured_action_probe(
            exact,
            compressed,
            product_state_count=action_probe_product_states,
            random_state_count=action_probe_random_mps,
            seed=int(action_probe_seed) + index,
            exact_work_cap=action_probe_exact_work_cap,
            workspace_cap_bytes=mpo_workspace_cap_bytes,
        )

    diagnostics: dict[str, object] = {
        "status": status,
        "learned_input_terms": len(normalized_labels),
        "temporal_rank": int(mode_array.shape[0]),
        "support_fraction": 1.0 if normalized_labels else 0.0,
        "full_support_status": "pass" if status == "ok" else "not_feasible",
        "static_mpo_compression": static_compression,
        "static_mpo_action_probes": static_action_probes,
        "static_mpo_bonds": {
            name: _mpo_bonds(mpo)
            for name, mpo in (("h0", h0_mpo), ("h1", h1_mpo))
            if mpo is not None
        },
    }
    return PreparedTDVPOperators(
        sites=sites,
        h0_mpo=h0_mpo,
        h1_mpo=h1_mpo,
        cd_mode_mpos=cd_mode_mpos,
        temporal_factorization=temporal_factorization,
        order=normalized_order,
        diagnostics=diagnostics,
        h0_exact_mpo=h0_exact_mpo,
        h1_exact_mpo=h1_exact_mpo,
        cd_mode_exact_mpos=cd_mode_exact_mpos,
    )


def evolve_protocol_tdvp(
    *,
    h0_terms: Sequence[tuple[str, complex]],
    h1_terms: Sequence[tuple[str, complex]],
    cd_factorization: TemporalFactorization | None,
    total_time: float = 1.0,
    steps: int = 128,
    cd_labels: Sequence[str] = (),
    protocol: str | None = None,
    schedule: Any | None = None,
    order: Sequence[int] | None = None,
    initial_state: Sequence[object] | None = None,
    ground_bitstring: str | None = None,
    mps_max_bond: int = 128,
    mps_cutoff: float = 1.0e-12,
    mpo_max_bond: int = 256,
    mpo_cutoff: float = 1.0e-12,
    lanczos_max: int = 20,
    mpo_workspace_cap_bytes: int = _DEFAULT_MPO_WORKSPACE_CAP_BYTES,
    action_probe_product_states: int = 0,
    action_probe_random_mps: int = 0,
    action_probe_seed: int = 0,
    action_probe_exact_work_cap: int = 10_000_000,
    action_probe_dynamic_samples: int = 0,
) -> tuple[Any, dict[str, object]]:
    """Evolve the full declared MPO support with midpoint TDVP (two-site except q=1)."""
    return _evolve_protocol_mpo(
        integrator="tdvp",
        h0_terms=h0_terms,
        h1_terms=h1_terms,
        cd_factorization=cd_factorization,
        total_time=total_time,
        steps=steps,
        cd_labels=cd_labels,
        protocol=protocol,
        schedule=schedule,
        order=order,
        initial_state=initial_state,
        ground_bitstring=ground_bitstring,
        mps_max_bond=mps_max_bond,
        mps_cutoff=mps_cutoff,
        mpo_max_bond=mpo_max_bond,
        mpo_cutoff=mpo_cutoff,
        lanczos_max=lanczos_max,
        mpo_workspace_cap_bytes=mpo_workspace_cap_bytes,
        action_probe_product_states=action_probe_product_states,
        action_probe_random_mps=action_probe_random_mps,
        action_probe_seed=action_probe_seed,
        action_probe_exact_work_cap=action_probe_exact_work_cap,
        action_probe_dynamic_samples=action_probe_dynamic_samples,
    )


def evolve_protocol_direct_tdvp(
    *,
    h0_terms: Sequence[tuple[str, complex]],
    h1_terms: Sequence[tuple[str, complex]],
    learned_tau: np.ndarray,
    learned_direct_cd_coefficients: np.ndarray,
    learned_labels: Sequence[str],
    full_support_identity: FullSupportIdentity,
    total_time: float = 1.0,
    steps: int = 128,
    schedule: Any | None = None,
    order: Sequence[int] | None = None,
    initial_state: Sequence[object] | None = None,
    ground_bitstring: str | None = None,
    mps_max_bond: int = 128,
    mps_cutoff: float = 1.0e-12,
    mpo_max_bond: int = 256,
    mpo_cutoff: float = 0.0,
    lanczos_max: int = 20,
    mpo_workspace_cap_bytes: int = _DEFAULT_MPO_WORKSPACE_CAP_BYTES,
    action_error_max: float = 1.0e-3,
    action_probe_product_states: int = 4,
    action_probe_seed: int = 0,
    action_probe_exact_work_cap: int = 10_000_000,
) -> tuple[Any, dict[str, object]]:
    """Evolve one full-K learned AGP by direct instantaneous MPO construction."""

    n_qubits = _protocol_n_qubits(h0_terms, h1_terms)
    dt = _validate_evolution_parameters(total_time, steps)
    normalized_order = (
        tuple(range(n_qubits))
        if order is None
        else _validate_order(order, n_qubits=n_qubits)
    )
    labels = tuple(str(label) for label in learned_labels)
    tau_grid = _finite_real_array(learned_tau, name="learned_tau", ndim=1)
    coefficient_samples = _finite_real_array(
        learned_direct_cd_coefficients,
        name="learned_direct_cd_coefficients",
        ndim=2,
    )
    if coefficient_samples.shape != (tau_grid.size, len(labels)):
        raise ValueError(
            "learned_direct_cd_coefficients must have shape "
            "(len(learned_tau), len(learned_labels))."
        )
    if tau_grid.size < 2 or np.any(np.diff(tau_grid) <= 0.0):
        raise ValueError("learned_tau must be strictly increasing with at least two samples.")
    assert_full_support_identity(full_support_identity, labels, coefficient_samples)
    if full_support_identity.n_qubits != n_qubits:
        raise ValueError("Full-support identity qubit count does not match the Hamiltonians.")
    if not np.isfinite(action_error_max) or float(action_error_max) < 0.0:
        raise ValueError("action_error_max must be finite and nonnegative.")

    sites = _pauli_sites(n_qubits)
    chain_initial, original_initial = _chain_initial_state(initial_state, normalized_order)
    state = _make_product_mps(sites, chain_initial)
    state._agp_qubit_order = normalized_order
    chain_ground, original_ground = _chain_ground_bitstring(
        ground_bitstring, normalized_order
    )
    operator_certificates: list[dict[str, object]] = []
    midpoint_lambdas: list[float] = []
    truncation_error_by_step: list[float] = []
    peak_mps_bond = _mps_peak_bond(state)
    dynamic_mpo_peak_bond = 1
    operator_build_seconds = 0.0
    evolution_seconds = 0.0
    started = time.perf_counter()

    def coefficients_at(tau: float) -> np.ndarray:
        clipped = float(np.clip(tau, tau_grid[0], tau_grid[-1]))
        if clipped <= tau_grid[0]:
            return coefficient_samples[0].copy()
        if clipped >= tau_grid[-1]:
            return coefficient_samples[-1].copy()
        upper = int(np.searchsorted(tau_grid, clipped, side="right"))
        lower = upper - 1
        weight = (clipped - tau_grid[lower]) / (tau_grid[upper] - tau_grid[lower])
        return (1.0 - weight) * coefficient_samples[lower] + weight * coefficient_samples[upper]

    def base_diagnostics(status: str, completed_steps: int) -> dict[str, object]:
        return {
            "status": status,
            "integrator": "tdvp",
            "protocol": "learned",
            "representation": "direct_time_full_support",
            "steps": int(steps),
            "completed_steps": int(completed_steps),
            "total_time": float(total_time),
            "dt": float(dt),
            "evaluated_cd_terms": len(labels),
            "full_support_sha256": full_support_identity.sha256,
            "source_completeness_status": "pass",
            "operator_gate_status": (
                "pass"
                if operator_certificates
                and all(row["operator_gate_status"] == "pass" for row in operator_certificates)
                else "not_tested"
            ),
            "operator_certificates": operator_certificates,
            "midpoint_lambdas": midpoint_lambdas,
            "operator_build_seconds": float(operator_build_seconds),
            "evolution_seconds": float(evolution_seconds),
            "runtime_seconds": float(time.perf_counter() - started),
            "truncation_error": float(sum(truncation_error_by_step)),
            "truncation_error_by_step": truncation_error_by_step,
            "norm_drift": abs(_physical_mps_norm_squared(state) - 1.0),
            "peak_mps_bond": int(peak_mps_bond),
            "final_mps_bond": int(_mps_peak_bond(state)),
            "dynamic_mpo_peak_bond": int(dynamic_mpo_peak_bond),
            "resource_statuses": {
                "static_mpo_compression": "not_applicable",
                "dynamic_mpo_assembly": "ok" if status == "ok" else "not_feasible",
            },
            "initial_state_original": original_initial,
            "initial_state_chain": [original_initial[index] for index in normalized_order],
            "ground_bitstring_original": original_ground,
            "ground_bitstring_chain": chain_ground,
            "final_energy": None,
            "final_energy_status": "not_tested",
            "ground_fidelity": None,
            "ground_fidelity_status": "not_tested",
        }

    for step in range(int(steps)):
        tau = (step + 0.5) / int(steps)
        lam, _ = _schedule_values(schedule, tau, total_time)
        midpoint_lambdas.append(float(lam))
        build_started = time.perf_counter()
        instantaneous_mpo, build = build_direct_time_mpo(
            h0_terms=h0_terms,
            h1_terms=h1_terms,
            learned_labels=labels,
            direct_cd_coefficients=coefficients_at(tau),
            lam=lam,
            n_qubits=n_qubits,
            order=normalized_order,
            max_bond=int(mpo_max_bond),
            cutoff=float(mpo_cutoff),
            workspace_cap_bytes=int(mpo_workspace_cap_bytes),
            full_support_identity=full_support_identity,
            identity_coefficient_samples=coefficient_samples,
            action_error_max=float(action_error_max),
            action_probe_product_states=int(action_probe_product_states),
            action_probe_seed=int(action_probe_seed) + step,
            action_probe_exact_work_cap=int(action_probe_exact_work_cap),
        )
        operator_build_seconds += time.perf_counter() - build_started
        compression = build.get("compression", {})
        certificate = {
            "step": step,
            "tau": float(tau),
            "lambda": float(lam),
            "learned_source_terms": int(build["learned_source_terms"]),
            "learned_terms_accounted": int(build["learned_terms_accounted"]),
            "full_support_sha256": build.get("full_support_sha256"),
            "source_completeness_status": build.get("source_completeness_status"),
            "exact_identity_status": build.get("exact_identity_status"),
            "operator_gate_status": build.get("operator_gate_status", "not_tested"),
            "max_relative_action_error_upper_bound": (
                build.get("action_certificate", {}).get(
                    "max_relative_action_error_upper_bound"
                )
                if isinstance(build.get("action_certificate"), dict)
                else None
            ),
            "mpo_bonds": build.get("post_bonds", []),
            "discarded_weight": (
                compression.get("discarded_weight")
                if isinstance(compression, dict)
                else None
            ),
            "build_seconds": build.get("build_seconds"),
        }
        operator_certificates.append(certificate)
        if instantaneous_mpo is None or certificate["operator_gate_status"] != "pass":
            diagnostics = base_diagnostics("not_feasible", step)
            diagnostics["operator_gate_status"] = "fail"
            diagnostics["resource_reason"] = (
                "The complete instantaneous learned operator failed its action or resource gate."
            )
            return state, diagnostics

        dynamic_mpo_peak_bond = max(
            dynamic_mpo_peak_bond, max(_mpo_bonds(instantaneous_mpo), default=1)
        )
        evolve_started = time.perf_counter()
        engine, engine_name = _make_tdvp_engine(
            state,
            instantaneous_mpo,
            dt=dt,
            mps_max_bond=int(mps_max_bond),
            mps_cutoff=float(mps_cutoff),
            lanczos_max=int(lanczos_max),
        )
        step_error = _evolve_one_mpo_step(engine, integrator="tdvp", dt=dt)
        evolution_seconds += time.perf_counter() - evolve_started
        state = engine.psi
        truncation_error_by_step.append(float(step_error))
        peak_mps_bond = max(peak_mps_bond, _mps_peak_bond(state))

    diagnostics = base_diagnostics("ok", int(steps))
    diagnostics["tdvp_engine"] = engine_name
    chain_h1_terms = tuple(
        (permute_pauli_label(label, normalized_order), complex(coefficient))
        for label, coefficient in h1_terms
    )
    energy = _stable_complex_sum(
        coefficient * _mps_pauli_expectation(state, label)
        for label, coefficient in chain_h1_terms
    )
    energy_tolerance = 128.0 * np.finfo(np.float64).eps * max(abs(energy), 1.0)
    if abs(energy.imag) > energy_tolerance:
        diagnostics["status"] = "unresolved_error"
        diagnostics["operator_gate_status"] = "fail"
        diagnostics["resource_reason"] = "Final energy contraction is not real within tolerance."
        return state, diagnostics
    diagnostics["final_energy"] = float(energy.real)
    diagnostics["final_energy_status"] = "ok"
    if chain_ground is not None:
        ground_state = _make_product_mps(
            sites,
            tuple("up" if bit == "0" else "down" for bit in chain_ground),
        )
        overlap = _canonical_mps_overlap(ground_state, state)
        norm_squared = _physical_mps_norm_squared(state)
        diagnostics["ground_fidelity"] = float(abs(overlap) ** 2 / norm_squared)
        diagnostics["ground_fidelity_status"] = "ok"
    return state, diagnostics


def _interpolate_coefficient_rows(
    tau_grid: np.ndarray,
    coefficient_samples: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate complete coefficient rows without column loops."""

    output = np.empty((targets.size, coefficient_samples.shape[1]), dtype=np.float64)
    for index, target in enumerate(targets):
        clipped = float(np.clip(target, tau_grid[0], tau_grid[-1]))
        if clipped <= tau_grid[0]:
            output[index] = coefficient_samples[0]
            continue
        if clipped >= tau_grid[-1]:
            output[index] = coefficient_samples[-1]
            continue
        upper = int(np.searchsorted(tau_grid, clipped, side="right"))
        lower = upper - 1
        weight = float(
            (clipped - tau_grid[lower]) / (tau_grid[upper] - tau_grid[lower])
        )
        output[index] = (
            (1.0 - weight) * coefficient_samples[lower]
            + weight * coefficient_samples[upper]
        )
    return output


def _certify_time_pauli_tensor_train(
    train: TimePauliTensorTrain,
    *,
    coefficient_error_max: float,
    action_error_max: float,
    action_probe_product_states: int,
    action_probe_time_samples: int,
    action_probe_seed: int,
    action_probe_exact_work_cap: int,
    workspace_cap_bytes: int,
) -> dict[str, object]:
    """Apply coefficient and exact sparse-action gates to sampled MPO slices."""

    compression = train.diagnostics["compression"]
    coefficient_bound = float(
        compression["max_relative_coefficient_error_upper_bound"]
    )
    coefficient_status = (
        "pass" if coefficient_bound <= float(coefficient_error_max) else "fail"
    )
    if coefficient_status == "fail":
        return {
            "status": "fail",
            "coefficient_status": "fail",
            "coefficient_error_max": float(coefficient_error_max),
            "max_relative_coefficient_error_upper_bound": coefficient_bound,
            "action_status": "not_tested_due_to_coefficient_failure",
            "action_error_max": float(action_error_max),
            "max_relative_action_error_upper_bound": None,
            "action_probe_samples": [],
            "action_probes": [],
        }
    count = min(max(int(action_probe_time_samples), 0), train.sample_count)
    if count:
        sample_indices = sorted(
            {
                int(round(value))
                for value in np.linspace(0, train.sample_count - 1, count)
            }
        )
    else:
        sample_indices = []
    probes: list[dict[str, object]] = []
    exact_identity = compression.get("exact_identity_status") == "verified"
    if exact_identity:
        action_status = "pass"
        maximum_action_error: float | None = 0.0
        probes = [
            {
                "sample": sample,
                "status": "pass",
                "method": "complete_tt_without_discarded_singular_directions",
                "max_relative_action_error_upper_bound": 0.0,
            }
            for sample in sample_indices
        ]
    elif int(action_probe_product_states) < 1 or not sample_indices:
        action_status = "not_tested"
        maximum_action_error = None
    else:
        upper_bounds: list[float] = []
        for sample in sample_indices:
            exact_terms = tuple(
                (
                    permute_pauli_label(label, train.order),
                    complex(train.coefficient_samples[sample, column]),
                )
                for column, label in enumerate(train.labels)
                if train.coefficient_samples[sample, column] != 0.0
            )
            exact_source = PauliCoordinateSource(
                L=train.n_qubits,
                sites=train.sites,
                _agp_pauli_terms=tuple(sorted(exact_terms)),
            )
            generator = np.random.default_rng(int(action_probe_seed) + sample)
            product_states = tuple(
                tuple(
                    "up" if bit == 0 else "down"
                    for bit in generator.integers(0, 2, size=train.n_qubits)
                )
                for _ in range(int(action_probe_product_states))
            )
            result = probe_mpo_compression(
                exact_source,
                slice_time_pauli_mpo(train, sample),
                product_states=product_states,
                random_state_count=0,
                seed=int(action_probe_seed) + sample,
                exact_work_cap=int(action_probe_exact_work_cap),
                workspace_cap_bytes=int(workspace_cap_bytes),
            )
            sample_bounds: list[float] = []
            for row in result.get("probes", []):
                upper = row.get(
                    "relative_action_error_upper_bound",
                    row.get("relative_action_error"),
                )
                if upper is not None and np.isfinite(float(upper)):
                    sample_bounds.append(float(upper))
            complete = len(sample_bounds) == int(action_probe_product_states)
            sample_maximum = max(sample_bounds) if sample_bounds else None
            sample_status = (
                "pass"
                if complete
                and sample_maximum is not None
                and sample_maximum <= float(action_error_max)
                else "fail"
            )
            if sample_maximum is not None:
                upper_bounds.append(sample_maximum)
            probes.append(
                {
                    "sample": sample,
                    "status": sample_status,
                    "method": "exact_full_support_product_action",
                    "max_relative_action_error_upper_bound": sample_maximum,
                    "tested_probes": len(sample_bounds),
                    "details": result,
                }
            )
        maximum_action_error = max(upper_bounds) if upper_bounds else None
        action_status = (
            "pass"
            if len(probes) == len(sample_indices)
            and all(row["status"] == "pass" for row in probes)
            else "fail"
        )
    overall = (
        "pass"
        if coefficient_status == "pass" and action_status == "pass"
        else "fail"
    )
    return {
        "status": overall,
        "coefficient_status": coefficient_status,
        "coefficient_error_max": float(coefficient_error_max),
        "max_relative_coefficient_error_upper_bound": coefficient_bound,
        "action_status": action_status,
        "action_error_max": float(action_error_max),
        "max_relative_action_error_upper_bound": maximum_action_error,
        "action_probe_samples": sample_indices,
        "action_probes": probes,
    }


def evolve_protocol_time_tensor_tdvp(
    *,
    h0_terms: Sequence[tuple[str, complex]],
    h1_terms: Sequence[tuple[str, complex]],
    learned_tau: np.ndarray,
    learned_direct_cd_coefficients: np.ndarray,
    learned_labels: Sequence[str],
    full_support_identity: FullSupportIdentity,
    total_time: float = 1.0,
    steps: int = 128,
    schedule: Any | None = None,
    order: Sequence[int] | None = None,
    initial_state: Sequence[object] | None = None,
    ground_bitstring: str | None = None,
    mps_max_bond: int = 128,
    mps_cutoff: float = 1.0e-12,
    mpo_max_bond: int = 256,
    mpo_cutoff: float = 1.0e-10,
    lanczos_max: int = 20,
    mpo_workspace_cap_bytes: int = _DEFAULT_MPO_WORKSPACE_CAP_BYTES,
    coefficient_error_max: float = 1.0e-3,
    action_error_max: float = 1.0e-3,
    action_probe_product_states: int = 4,
    action_probe_time_samples: int = 3,
    action_probe_seed: int = 0,
    action_probe_exact_work_cap: int = 10_000_000,
    time_window_size: int | None = None,
    adaptive_time_windows: bool = True,
    time_axis_position: int = 0,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> tuple[Any, dict[str, object]]:
    """Evolve a full-K AGP through independently certified contiguous windows."""

    n_qubits = _protocol_n_qubits(h0_terms, h1_terms)
    dt = _validate_evolution_parameters(total_time, steps)
    normalized_order = (
        tuple(range(n_qubits))
        if order is None
        else _validate_order(order, n_qubits=n_qubits)
    )
    labels = tuple(str(label) for label in learned_labels)
    if (
        isinstance(time_axis_position, bool)
        or not isinstance(time_axis_position, Integral)
        or not 0 <= int(time_axis_position) <= int(n_qubits)
    ):
        raise ValueError("time_axis_position must be an integer in [0, n_qubits].")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None.")
    tau_grid = _finite_real_array(learned_tau, name="learned_tau", ndim=1)
    coefficient_samples = _finite_real_array(
        learned_direct_cd_coefficients,
        name="learned_direct_cd_coefficients",
        ndim=2,
    )
    if coefficient_samples.shape != (tau_grid.size, len(labels)):
        raise ValueError(
            "learned_direct_cd_coefficients must have shape "
            "(len(learned_tau), len(learned_labels))."
        )
    if tau_grid.size < 2 or np.any(np.diff(tau_grid) <= 0.0):
        raise ValueError("learned_tau must be strictly increasing with at least two samples.")
    assert_full_support_identity(full_support_identity, labels, coefficient_samples)
    for value, name in (
        (coefficient_error_max, "coefficient_error_max"),
        (action_error_max, "action_error_max"),
    ):
        if not np.isfinite(value) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if time_window_size is None:
        requested_window_size = int(steps)
    elif (
        isinstance(time_window_size, bool)
        or not isinstance(time_window_size, Integral)
        or int(time_window_size) < 1
    ):
        raise ValueError("time_window_size must be a positive integer or None.")
    else:
        requested_window_size = min(int(time_window_size), int(steps))

    midpoint_tau = (np.arange(int(steps), dtype=np.float64) + 0.5) / int(steps)
    midpoint_coefficients = _interpolate_coefficient_rows(
        tau_grid, coefficient_samples, midpoint_tau
    )
    midpoint_lambdas = np.asarray(
        [
            _schedule_values(schedule, float(tau), total_time)[0]
            for tau in midpoint_tau
        ],
        dtype=np.float64,
    )
    chain_initial, original_initial = _chain_initial_state(
        initial_state, normalized_order
    )
    sites = _pauli_sites(n_qubits)
    state = _make_product_mps(sites, chain_initial)
    state._agp_qubit_order = normalized_order
    chain_ground, original_ground = _chain_ground_bitstring(
        ground_bitstring, normalized_order
    )
    started = time.perf_counter()
    truncation_error_by_step: list[float] = []
    peak_mps_bond = _mps_peak_bond(state)
    dynamic_mpo_peak_bond = 1
    operator_build_seconds = 0.0
    evolution_seconds = 0.0
    completed_steps = 0
    engine_name: str | None = None
    window_records: list[dict[str, object]] = []
    accepted_compressions: list[dict[str, object]] = []
    split_attempts = 0
    terminal_reason: str | None = None

    def report_progress(event: str, **values: object) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "event": event,
                "learned_terms": len(labels),
                "total_steps": int(steps),
                **values,
            }
        )

    def compact_gate(gate: Mapping[str, object]) -> dict[str, object]:
        probes = gate.get("action_probes", [])
        probes = (
            probes
            if isinstance(probes, Sequence) and not isinstance(probes, (str, bytes))
            else []
        )
        return {
            "status": gate.get("status", "fail"),
            "coefficient_status": gate.get("coefficient_status", "not_tested"),
            "max_relative_coefficient_error_upper_bound": gate.get(
                "max_relative_coefficient_error_upper_bound"
            ),
            "action_status": gate.get("action_status", "not_tested"),
            "max_relative_action_error_upper_bound": gate.get(
                "max_relative_action_error_upper_bound"
            ),
            "action_probes": [
                {
                    "sample": row.get("sample"),
                    "status": row.get("status"),
                    "method": row.get("method"),
                    "max_relative_action_error_upper_bound": row.get(
                        "max_relative_action_error_upper_bound"
                    ),
                    "tested_probes": row.get("tested_probes", 0),
                }
                for row in probes
                if isinstance(row, Mapping)
            ],
        }

    def compact_compression(compression: Mapping[str, object]) -> dict[str, object]:
        ranks = compression.get("retained_ranks", [])
        ranks = (
            [int(value) for value in ranks]
            if isinstance(ranks, Sequence) and not isinstance(ranks, (str, bytes))
            else []
        )
        post_bonds = compression.get("post_bonds", [])
        post_bonds = (
            [int(value) for value in post_bonds]
            if isinstance(post_bonds, Sequence)
            and not isinstance(post_bonds, (str, bytes))
            else []
        )
        return {
            "status": compression.get("status", "not_feasible"),
            "retained_ranks": ranks,
            "post_bonds": post_bonds,
            "peak_bond": max(post_bonds or ranks, default=1),
            "total_discarded_squared_norm": compression.get(
                "total_discarded_squared_norm"
            ),
            "max_relative_coefficient_error_upper_bound": compression.get(
                "max_relative_coefficient_error_upper_bound"
            ),
            "peak_workspace_bytes": compression.get("peak_workspace_bytes"),
            "required_workspace_bytes": compression.get("required_workspace_bytes"),
            "exact_identity_status": compression.get("exact_identity_status"),
            "resource_reason": compression.get("resource_reason"),
            "failed_axis": compression.get("failed_axis"),
            "failed_axis_name": compression.get("failed_axis_name"),
            "time_axis_position": compression.get("time_axis_position"),
        }

    def operator_certificate(status: str) -> dict[str, object]:
        accepted = [row for row in window_records if row["status"] == "pass"]
        coefficient_bounds = [
            float(row["max_relative_coefficient_error_upper_bound"])
            for row in accepted
            if row.get("max_relative_coefficient_error_upper_bound") is not None
        ]
        action_bounds = [
            float(row["max_relative_action_error_upper_bound"])
            for row in accepted
            if row.get("max_relative_action_error_upper_bound") is not None
        ]
        return {
            "status": status,
            "coefficient_status": "pass" if status == "pass" else "fail",
            "action_status": "pass" if status == "pass" else "fail",
            "coefficient_error_max": float(coefficient_error_max),
            "action_error_max": float(action_error_max),
            "requested_window_size": int(requested_window_size),
            "adaptive_time_windows": bool(adaptive_time_windows),
            "accepted_windows": len(accepted),
            "split_attempts": int(split_attempts),
            "max_relative_coefficient_error_upper_bound": (
                max(coefficient_bounds) if coefficient_bounds else None
            ),
            "max_relative_action_error_upper_bound": (
                max(action_bounds) if action_bounds else None
            ),
            "windows": window_records,
        }

    def compression_summary(status: str) -> dict[str, object]:
        return {
            "status": status,
            "algorithm": "adaptive_windowed_joint_time_pauli_tt_svd",
            "window_count": len(accepted_compressions),
            "peak_bond": max(
                (int(row.get("peak_bond", 1)) for row in accepted_compressions),
                default=1,
            ),
            "peak_workspace_bytes": max(
                (
                    int(row.get("peak_workspace_bytes") or 0)
                    for row in accepted_compressions
                ),
                default=0,
            ),
            "windows": accepted_compressions,
        }

    def diagnostics(status: str, completed_steps: int) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": status,
            "integrator": "tdvp",
            "protocol": "learned",
            "representation": "joint_time_full_support",
            "steps": int(steps),
            "completed_steps": int(completed_steps),
            "total_time": float(total_time),
            "dt": float(dt),
            "evaluated_cd_terms": len(labels),
            "time_axis_position": int(time_axis_position),
            "full_support_sha256": full_support_identity.sha256,
            "source_completeness_status": (
                "pass"
                if window_records
                and all(
                    row.get("source_completeness_status") == "pass"
                    for row in window_records
                )
                else "fail"
            ),
            "operator_gate_status": "pass" if status == "ok" else "fail",
            "operator_certificate": operator_certificate(
                "pass" if status == "ok" else "fail"
            ),
            "joint_tt_compression": compression_summary(
                "ok" if status == "ok" else "not_feasible"
            ),
            "midpoint_lambdas": midpoint_lambdas.tolist(),
            "operator_build_seconds": operator_build_seconds,
            "evolution_seconds": float(evolution_seconds),
            "runtime_seconds": float(time.perf_counter() - started),
            "truncation_error": float(sum(truncation_error_by_step)),
            "truncation_error_by_step": truncation_error_by_step,
            "norm_drift": abs(_physical_mps_norm_squared(state) - 1.0),
            "peak_mps_bond": int(peak_mps_bond),
            "final_mps_bond": int(_mps_peak_bond(state)),
            "dynamic_mpo_peak_bond": int(dynamic_mpo_peak_bond),
            "resource_statuses": {
                "joint_time_pauli_compression": "ok" if status == "ok" else "not_feasible",
                "dynamic_mpo_slicing": "ok" if status == "ok" else "not_feasible",
            },
            "initial_state_original": original_initial,
            "initial_state_chain": [
                original_initial[index] for index in normalized_order
            ],
            "ground_bitstring_original": original_ground,
            "ground_bitstring_chain": chain_ground,
            "final_energy": None,
            "final_energy_status": "not_tested",
            "ground_fidelity": None,
            "ground_fidelity_status": "not_tested",
        }
        return payload

    def process_window(start: int, stop: int) -> bool:
        nonlocal state
        nonlocal operator_build_seconds, evolution_seconds, completed_steps
        nonlocal peak_mps_bond, dynamic_mpo_peak_bond, engine_name
        nonlocal split_attempts, terminal_reason

        build_started = time.perf_counter()
        train, build = build_time_pauli_tensor_train(
            h0_terms=h0_terms,
            h1_terms=h1_terms,
            learned_labels=labels,
            direct_cd_coefficients=midpoint_coefficients[start:stop],
            lambda_samples=midpoint_lambdas[start:stop],
            n_qubits=n_qubits,
            order=normalized_order,
            max_bond=int(mpo_max_bond),
            cutoff=float(mpo_cutoff),
            workspace_cap_bytes=int(mpo_workspace_cap_bytes),
            full_support_identity=full_support_identity,
            identity_coefficient_samples=coefficient_samples,
            time_axis_position=int(time_axis_position),
        )
        gate: dict[str, object]
        if train is None:
            gate = {
                "status": "fail",
                "coefficient_status": "not_tested",
                "action_status": "not_tested",
                "max_relative_coefficient_error_upper_bound": None,
                "max_relative_action_error_upper_bound": None,
            }
        else:
            gate = _certify_time_pauli_tensor_train(
                train,
                coefficient_error_max=float(coefficient_error_max),
                action_error_max=float(action_error_max),
                action_probe_product_states=int(action_probe_product_states),
                action_probe_time_samples=min(
                    int(action_probe_time_samples), stop - start
                ),
                action_probe_seed=int(action_probe_seed) + start,
                action_probe_exact_work_cap=int(action_probe_exact_work_cap),
                workspace_cap_bytes=int(mpo_workspace_cap_bytes),
            )
        window_build_seconds = time.perf_counter() - build_started
        operator_build_seconds += window_build_seconds
        compression = build.get("compression", {})
        compression = compression if isinstance(compression, Mapping) else {}
        compact = compact_gate(gate)
        record: dict[str, object] = {
            "start_step": int(start),
            "stop_step": int(stop),
            "samples": int(stop - start),
            "status": compact["status"],
            "learned_source_terms": int(build.get("learned_source_terms", len(labels))),
            "learned_terms_accounted": int(
                build.get("learned_terms_accounted", 0)
            ),
            "full_support_sha256": build.get("full_support_sha256"),
            "source_completeness_status": build.get(
                "source_completeness_status", "fail"
            ),
            **compact,
            "compression": compact_compression(compression),
        }
        if gate.get("status") != "pass":
            if bool(adaptive_time_windows) and stop - start > 1:
                record["status"] = "split"
                window_records.append(record)
                report_progress(
                    "operator_window",
                    start_step=int(start),
                    stop_step=int(stop),
                    status="split",
                    build_seconds=float(window_build_seconds),
                )
                split_attempts += 1
                midpoint = start + (stop - start) // 2
                train = None
                return process_window(start, midpoint) and process_window(midpoint, stop)
            window_records.append(record)
            report_progress(
                "operator_window",
                start_step=int(start),
                stop_step=int(stop),
                status="fail",
                build_seconds=float(window_build_seconds),
            )
            terminal_reason = (
                "A minimum-size full-support time window failed its resource, "
                "coefficient, or action gate."
            )
            return False

        record["status"] = "pass"
        window_records.append(record)
        accepted_compressions.append(compact_compression(compression))
        report_progress(
            "operator_window",
            start_step=int(start),
            stop_step=int(stop),
            status="pass",
            build_seconds=float(window_build_seconds),
            coefficient_error_upper_bound=gate.get(
                "max_relative_coefficient_error_upper_bound"
            ),
            action_error_upper_bound=gate.get(
                "max_relative_action_error_upper_bound"
            ),
        )
        assert train is not None
        for local_step, global_step in enumerate(range(start, stop)):
            instantaneous_mpo = slice_time_pauli_mpo(train, local_step)
            dynamic_mpo_peak_bond = max(
                dynamic_mpo_peak_bond,
                max(_mpo_bonds(instantaneous_mpo), default=1),
            )
            evolve_started = time.perf_counter()
            engine, engine_name = _make_tdvp_engine(
                state,
                instantaneous_mpo,
                dt=dt,
                mps_max_bond=int(mps_max_bond),
                mps_cutoff=float(mps_cutoff),
                lanczos_max=int(lanczos_max),
            )
            step_error = _evolve_one_mpo_step(engine, integrator="tdvp", dt=dt)
            evolution_seconds += time.perf_counter() - evolve_started
            state = engine.psi
            truncation_error_by_step.append(float(step_error))
            peak_mps_bond = max(peak_mps_bond, _mps_peak_bond(state))
            completed_steps = global_step + 1
            report_progress(
                "tdvp_step",
                completed_steps=int(completed_steps),
                peak_mps_bond=int(peak_mps_bond),
                dynamic_mpo_peak_bond=int(dynamic_mpo_peak_bond),
                step_seconds=float(time.perf_counter() - evolve_started),
            )
        return True

    successful = True
    for start in range(0, int(steps), requested_window_size):
        if not process_window(start, min(start + requested_window_size, int(steps))):
            successful = False
            break
    if not successful:
        payload = diagnostics("not_feasible", completed_steps)
        payload["resource_reason"] = terminal_reason
        return state, payload

    payload = diagnostics("ok", int(steps))
    payload["tdvp_engine"] = engine_name
    chain_h1_terms = tuple(
        (permute_pauli_label(label, normalized_order), complex(coefficient))
        for label, coefficient in h1_terms
    )
    energy = _stable_complex_sum(
        coefficient * _mps_pauli_expectation(state, label)
        for label, coefficient in chain_h1_terms
    )
    energy_tolerance = 128.0 * np.finfo(np.float64).eps * max(abs(energy), 1.0)
    if abs(energy.imag) > energy_tolerance:
        payload["status"] = "unresolved_error"
        payload["operator_gate_status"] = "fail"
        payload["resource_reason"] = "Final energy contraction is not real within tolerance."
        return state, payload
    payload["final_energy"] = float(energy.real)
    payload["final_energy_status"] = "ok"
    if chain_ground is not None:
        ground_state = _make_product_mps(
            sites,
            tuple("up" if bit == "0" else "down" for bit in chain_ground),
        )
        overlap = _canonical_mps_overlap(ground_state, state)
        norm_squared = _physical_mps_norm_squared(state)
        payload["ground_fidelity"] = float(abs(overlap) ** 2 / norm_squared)
        payload["ground_fidelity_status"] = "ok"
    return state, payload


def evolve_protocol_expm_mpo(**kwargs: Any) -> tuple[Any, dict[str, object]]:
    """Run the same midpoint MPO protocol with TeNPy's ExpMPO comparison engine."""
    kwargs.setdefault("cd_labels", ())
    kwargs.setdefault("protocol", None)
    kwargs.setdefault("schedule", None)
    kwargs.setdefault("order", None)
    kwargs.setdefault("initial_state", None)
    kwargs.setdefault("ground_bitstring", None)
    kwargs.setdefault("lanczos_max", 20)
    kwargs.setdefault("mpo_workspace_cap_bytes", _DEFAULT_MPO_WORKSPACE_CAP_BYTES)
    kwargs.setdefault("action_probe_product_states", 0)
    kwargs.setdefault("action_probe_random_mps", 0)
    kwargs.setdefault("action_probe_seed", 0)
    kwargs.setdefault("action_probe_exact_work_cap", 10_000_000)
    kwargs.setdefault("action_probe_dynamic_samples", 0)
    return _evolve_protocol_mpo(integrator="expm_mpo", **kwargs)


def dense_midpoint_evolution(
    h0_terms: Sequence[tuple[str, complex]],
    h1_terms: Sequence[tuple[str, complex]],
    *,
    total_time: float = 1.0,
    steps: int = 128,
    cd_labels: Sequence[str] = (),
    cd_factorization: TemporalFactorization | None = None,
    protocol: str | None = None,
    schedule: Any | None = None,
    order: Sequence[int] | None = None,
    initial_state: Sequence[object] | None = None,
) -> np.ndarray:
    """Test-only dense midpoint reference for systems with at most four qubits."""
    n_qubits = _protocol_n_qubits(h0_terms, h1_terms)
    _validate_dense_helper_size(n_qubits)
    normalized_order = tuple(range(n_qubits)) if order is None else _validate_order(order, n_qubits=n_qubits)
    resolved_protocol, factorization, resolved_cd_labels = _resolve_protocol_factorization(
        protocol=protocol,
        cd_factorization=cd_factorization,
        cd_labels=cd_labels,
        h0_terms=h0_terms,
        h1_terms=h1_terms,
        total_time=total_time,
        steps=steps,
        schedule=schedule,
    )
    chain_initial, _ = _chain_initial_state(initial_state, normalized_order)
    state = _dense_product_state(chain_initial)
    dt = _validate_evolution_parameters(total_time, steps)
    for step in range(int(steps)):
        tau = (step + 0.5) / int(steps)
        lam, _ = _schedule_values(schedule, tau, total_time)
        coefficients = _temporal_mode_values(factorization, tau)
        terms = _protocol_terms_at_time(
            h0_terms, h1_terms, lam, resolved_cd_labels, factorization, coefficients, resolved_protocol
        )
        chain_terms = [(permute_pauli_label(label, normalized_order), value) for label, value in terms]
        hamiltonian = dense_pauli_sum(chain_terms)
        if not np.allclose(hamiltonian, hamiltonian.conj().T, atol=1.0e-10):
            raise ValueError("Midpoint Hamiltonian is not Hermitian.")
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        state = (eigenvectors * np.exp(-1.0j * dt * eigenvalues)) @ (eigenvectors.conj().T @ state)
    return state


def state_overlap_dense(state: Any, dense_state: np.ndarray) -> complex:
    """Return an MPS/dense-state overlap only for q <= 4 test references."""
    _validate_dense_helper_size(int(state.L))
    vector = _mps_to_dense_statevector(state)
    reference = np.asarray(dense_state, dtype=np.complex128).reshape(-1)
    if vector.shape != reference.shape:
        raise ValueError("Dense reference state has incompatible dimension.")
    return np.vdot(vector, reference)


class _DynamicMPOResourceError(MemoryError):
    """Raised before a dynamic MPO allocation whose bounded workspace exceeds the cap."""

    def __init__(self, plan: dict[str, object]):
        super().__init__(str(plan["reason"]))
        self.plan = plan


def _static_mpo_compression_certificate(prepared: PreparedTDVPOperators) -> dict[str, object]:
    """Return the serializable compression evidence needed by certification."""

    source = prepared.diagnostics.get("static_mpo_compression", {})
    if not isinstance(source, dict):
        return {}

    def json_value(value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_value(item) for item in value]
        return value

    def compact(item: object) -> dict[str, object]:
        item = item if isinstance(item, dict) else {}
        build = item.get("build", {})
        build = build if isinstance(build, dict) else {}
        return {
            "input_terms": build.get("input_terms"),
            "included_terms": build.get("included_terms"),
            "dropped_terms": build.get("dropped_terms"),
            "compression": json_value(item.get("compression", {})),
        }

    certificate = {
        name: compact(source[name])
        for name in ("h0", "h1")
        if name in source
    }
    cd_modes = source.get("cd_modes", [])
    if isinstance(cd_modes, list):
        certificate["cd_modes"] = [
            {
                "mode": item.get("mode"),
                "status": item.get("status"),
                **compact(item),
            }
            for item in cd_modes
            if isinstance(item, dict)
        ]
    return certificate


def _static_mpo_evolution_diagnostics(prepared: PreparedTDVPOperators) -> dict[str, object]:
    compression = prepared.diagnostics.get("static_mpo_compression", {})
    components: list[tuple[str, Any, dict[str, object] | None]] = []
    for name, mpo in (("h0", prepared.h0_mpo), ("h1", prepared.h1_mpo)):
        item = compression.get(name, {}) if isinstance(compression, dict) else {}
        components.append((name, mpo, item.get("compression") if isinstance(item, dict) else None))
    mode_items = compression.get("cd_modes", []) if isinstance(compression, dict) else []
    for index, mpo in enumerate(prepared.cd_mode_mpos):
        item = mode_items[index] if index < len(mode_items) else {}
        components.append((f"cd_mode_{index}", mpo, item.get("compression") if isinstance(item, dict) else None))

    discarded_weight = 0.0
    action_statuses: dict[str, str] = {}
    error_statuses: dict[str, str] = {}
    equivalence_components: dict[str, dict[str, object]] = {}
    for name, mpo, item in components:
        if mpo is None or not isinstance(item, dict):
            action_statuses[name] = "not_tested"
            error_statuses[name] = "not_feasible"
            equivalence_components[name] = {"status": "not_tested"}
            continue
        weight = item.get("discarded_weight")
        if weight is None:
            action_statuses[name] = "not_tested"
            error_statuses[name] = "not_tested"
            equivalence_components[name] = {"status": "not_tested"}
            continue
        discarded_weight += float(weight)
        certificate = _exact_identity_certificate_status(
            mpo, getattr(mpo, "_agp_pauli_terms", ())
        )
        action_statuses[name] = "not_tested"
        if certificate == "verified" and float(weight) == 0.0:
            error_statuses[name] = "exact_identity_verified"
            equivalence_components[name] = {
                "status": "pass",
                "discarded_weight": float(weight),
                "exact_identity_certificate_status": certificate,
            }
        else:
            error_statuses[name] = "lossy_or_unverified"
            equivalence_components[name] = {
                "status": "not_comparable",
                "discarded_weight": float(weight),
                "exact_identity_certificate_status": certificate,
            }

    status = (
        "pass"
        if equivalence_components
        and all(item["status"] == "pass" for item in equivalence_components.values())
        else "not_comparable"
    )
    return {
        "static_mpo_discarded_weight": float(discarded_weight),
        "static_mpo_action_statuses": action_statuses,
        "static_mpo_error_statuses": error_statuses,
        "operator_equivalence_status": status,
        "operator_equivalence": {
            "status": status,
            "basis": "exact ExpMPO Pauli graph versus compressed TDVP block-sum components",
            "components": equivalence_components,
        },
    }


def _canonical_mps_overlap(left: Any, right: Any) -> complex:
    """Contract finite MPS copies after restoring TeNPy's canonical form metadata."""
    if int(left.L) == 1 and int(right.L) == 1:
        return complex(left.overlap(right))
    left_canonical = left.copy()
    right_canonical = right.copy()
    left_canonical.canonical_form()
    right_canonical.canonical_form()
    return complex(left_canonical.overlap(right_canonical))


def _physical_mps_norm_squared(state: Any) -> float:
    overlap = _canonical_mps_overlap(state, state)
    if abs(overlap.imag) > 1.0e-9 or overlap.real < -1.0e-12:
        raise ValueError("MPS self-overlap is not a nonnegative real physical norm.")
    return float(max(overlap.real, 0.0))


def _require_tenpy() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from tenpy.linalg import np_conserved as npc
        from tenpy.networks.mpo import MPO, MPOGraph
        from tenpy.networks.site import SpinHalfSite
        from tenpy.networks.terms import TermList
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "TeNPy is required for MPO operations; install the tensor-network optional dependency."
        ) from error
    return SpinHalfSite, TermList, MPOGraph, MPO, npc


def _evolve_protocol_mpo(
    *,
    integrator: str,
    h0_terms: Sequence[tuple[str, complex]],
    h1_terms: Sequence[tuple[str, complex]],
    cd_factorization: TemporalFactorization | None,
    total_time: float,
    steps: int,
    cd_labels: Sequence[str],
    protocol: str | None,
    schedule: Any | None,
    order: Sequence[int] | None,
    initial_state: Sequence[object] | None,
    ground_bitstring: str | None,
    mps_max_bond: int,
    mps_cutoff: float,
    mpo_max_bond: int,
    mpo_cutoff: float,
    lanczos_max: int,
    mpo_workspace_cap_bytes: int,
    action_probe_product_states: int,
    action_probe_random_mps: int,
    action_probe_seed: int,
    action_probe_exact_work_cap: int,
    action_probe_dynamic_samples: int,
) -> tuple[Any, dict[str, object]]:
    n_qubits = _protocol_n_qubits(h0_terms, h1_terms)
    dt = _validate_evolution_parameters(total_time, steps)
    if integrator not in {"tdvp", "expm_mpo"}:
        raise ValueError("integrator must be 'tdvp' or 'expm_mpo'.")
    if int(mps_max_bond) < 1 or int(mpo_max_bond) < 1 or int(lanczos_max) < 1:
        raise ValueError("Bond and Lanczos limits must be positive.")
    if not np.isfinite(mps_cutoff) or float(mps_cutoff) < 0.0:
        raise ValueError("mps_cutoff must be finite and nonnegative.")
    if not np.isfinite(mpo_cutoff) or float(mpo_cutoff) < 0.0:
        raise ValueError("mpo_cutoff must be finite and nonnegative.")
    normalized_order = tuple(range(n_qubits)) if order is None else _validate_order(order, n_qubits=n_qubits)
    resolved_protocol, factorization, resolved_cd_labels = _resolve_protocol_factorization(
        protocol=protocol,
        cd_factorization=cd_factorization,
        cd_labels=cd_labels,
        h0_terms=h0_terms,
        h1_terms=h1_terms,
        total_time=total_time,
        steps=steps,
        schedule=schedule,
    )
    factor_values = (
        factorization.temporal_factors
        if factorization is not None
        else np.empty((int(steps) + 1, 0), dtype=np.float64)
    )
    mode_values = (
        factorization.static_modes
        if factorization is not None
        else np.empty((0, 0), dtype=np.float64)
    )
    prepared = prepare_tdvp_operators(
        labels=resolved_cd_labels,
        static_modes=mode_values,
        temporal_factors=factor_values,
        n_qubits=n_qubits,
        order=normalized_order,
        mpo_max_bond=int(mpo_max_bond),
        mpo_cutoff=float(mpo_cutoff),
        h0_terms=h0_terms,
        h1_terms=h1_terms,
        temporal_factorization=factorization,
        mpo_workspace_cap_bytes=mpo_workspace_cap_bytes,
        action_probe_product_states=action_probe_product_states,
        action_probe_random_mps=action_probe_random_mps,
        action_probe_seed=action_probe_seed,
        action_probe_exact_work_cap=action_probe_exact_work_cap,
    )
    chain_initial, original_initial = _chain_initial_state(initial_state, normalized_order)
    state = _make_product_mps(prepared.sites, chain_initial)
    state._agp_qubit_order = normalized_order
    chain_ground, original_ground = _chain_ground_bitstring(ground_bitstring, normalized_order)
    static_evolution = _static_mpo_evolution_diagnostics(prepared)
    base_diagnostics: dict[str, object] = {
        "status": prepared.diagnostics["status"],
        "integrator": integrator,
        "protocol": resolved_protocol,
        "steps": int(steps),
        "completed_steps": 0,
        "total_time": float(total_time),
        "dt": dt,
        "temporal_rank": 0 if factorization is None else factorization.rank,
        "evaluated_cd_terms": len(resolved_cd_labels),
        "static_mpo_bonds": prepared.diagnostics["static_mpo_bonds"],
        "static_mpo_compression": _static_mpo_compression_certificate(prepared),
        "dynamic_mpo_peak_bond": 1,
        "dynamic_mpo_discarded_weight": 0.0,
        "dynamic_mpo_action_status": "not_tested",
        "dynamic_mpo_error_status": "not_tested",
        "dynamic_mpo_workspace": {"status": "not_tested"},
        "dynamic_mpo_action_probes": [],
        "static_mpo_action_probes": prepared.diagnostics["static_mpo_action_probes"],
        "operator_build_seconds": 0.0,
        "evolution_seconds": 0.0,
        "truncation_error": 0.0,
        "truncation_error_by_step": [],
        "norm_drift": 0.0,
        "peak_mps_bond": _mps_peak_bond(state),
        "final_mps_bond": _mps_peak_bond(state),
        "resource_statuses": {
            "static_mpo_compression": prepared.diagnostics["status"],
            "dynamic_mpo_assembly": "not_tested",
        },
        "initial_state_original": original_initial,
        "initial_state_chain": [original_initial[site] for site in normalized_order],
        "ground_bitstring_original": original_ground,
        "ground_bitstring_chain": chain_ground,
        "midpoint_lambdas": [],
        "ground_fidelity_status": "not tested" if chain_ground is None else "pending",
        **static_evolution,
    }
    if prepared.diagnostics["status"] != "ok":
        base_diagnostics["status"] = "not_feasible"
        base_diagnostics["ground_fidelity_status"] = "not tested"
        return state, base_diagnostics

    evolution_start = time.perf_counter()
    operator_build_seconds = 0.0
    truncation_error = 0.0
    truncation_error_by_step: list[float] = []
    peak_bond = _mps_peak_bond(state)
    dynamic_peak_bond = 1
    midpoint_lambdas: list[float] = []
    dynamic_details: dict[str, object] = {"status": "not_tested"}
    dynamic_action_probes: list[dict[str, object]] = []
    dynamic_probe_steps = set()
    if int(action_probe_dynamic_samples) > 0:
        dynamic_probe_steps = {
            int(round(index * (int(steps) - 1) / max(int(action_probe_dynamic_samples) - 1, 1)))
            for index in range(min(int(action_probe_dynamic_samples), int(steps)))
        }
    engine_name: str | None = None
    for step in range(int(steps)):
        tau = (step + 0.5) / int(steps)
        lam, _ = _schedule_values(schedule, tau, total_time)
        midpoint_lambdas.append(lam)
        temporal_values = _temporal_mode_values(factorization, tau)
        build_start = time.perf_counter()
        try:
            dynamic_mpo, dynamic_details = _assemble_dynamic_mpo(
                prepared,
                h0_coefficient=1.0 - lam,
                h1_coefficient=lam,
                cd_mode_coefficients=temporal_values,
                workspace_cap_bytes=mpo_workspace_cap_bytes,
                expm_compatible=integrator == "expm_mpo",
            )
        except _DynamicMPOResourceError as error:
            operator_build_seconds += time.perf_counter() - build_start
            base_diagnostics.update(
                {
                    "status": "not_feasible",
                    "completed_steps": step,
                    "midpoint_lambdas": midpoint_lambdas,
                    "operator_build_seconds": operator_build_seconds,
                    "evolution_seconds": time.perf_counter() - evolution_start,
                    "truncation_error": truncation_error,
                    "truncation_error_by_step": truncation_error_by_step,
                    "norm_drift": abs(_physical_mps_norm_squared(state) - 1.0),
                    "peak_mps_bond": peak_bond,
                    "final_mps_bond": _mps_peak_bond(state),
                    "dynamic_mpo_peak_bond": dynamic_peak_bond,
                    "dynamic_mpo_workspace": error.plan,
                    "resource_statuses": {
                        "static_mpo_compression": "ok",
                        "dynamic_mpo_assembly": "not_feasible",
                    },
                }
            )
            base_diagnostics.update(_final_mpo_metrics(state, prepared.h1_exact_mpo, chain_ground))
            return state, base_diagnostics
        except MemoryError as error:
            operator_build_seconds += time.perf_counter() - build_start
            dynamic_details = {
                "status": "not_feasible",
                "reason": f"allocation failed after dynamic preflight: {error}",
            }
            base_diagnostics.update(
                {
                    "status": "not_feasible",
                    "completed_steps": step,
                    "midpoint_lambdas": midpoint_lambdas,
                    "operator_build_seconds": operator_build_seconds,
                    "evolution_seconds": time.perf_counter() - evolution_start,
                    "truncation_error": truncation_error,
                    "truncation_error_by_step": truncation_error_by_step,
                    "norm_drift": abs(_physical_mps_norm_squared(state) - 1.0),
                    "peak_mps_bond": peak_bond,
                    "final_mps_bond": _mps_peak_bond(state),
                    "dynamic_mpo_peak_bond": dynamic_peak_bond,
                    "dynamic_mpo_workspace": dynamic_details,
                    "resource_statuses": {
                        "static_mpo_compression": "ok",
                        "dynamic_mpo_assembly": "not_feasible",
                    },
                }
            )
            base_diagnostics.update(_final_mpo_metrics(state, prepared.h1_exact_mpo, chain_ground))
            return state, base_diagnostics
        operator_build_seconds += time.perf_counter() - build_start
        dynamic_peak_bond = max(dynamic_peak_bond, max(_mpo_bonds(dynamic_mpo), default=1))
        if step in dynamic_probe_steps:
            try:
                exact_dynamic, _ = _assemble_dynamic_mpo(
                    prepared,
                    h0_coefficient=1.0 - lam,
                    h1_coefficient=lam,
                    cd_mode_coefficients=temporal_values,
                    workspace_cap_bytes=mpo_workspace_cap_bytes,
                    exact_components=True,
                )
                dynamic_action_probes.append(
                    _configured_action_probe(
                        exact_dynamic,
                        dynamic_mpo,
                        product_state_count=action_probe_product_states,
                        random_state_count=action_probe_random_mps,
                        seed=int(action_probe_seed) + step,
                        exact_work_cap=action_probe_exact_work_cap,
                        workspace_cap_bytes=mpo_workspace_cap_bytes,
                    )
                )
            except (_DynamicMPOResourceError, MemoryError, ValueError) as error:
                dynamic_action_probes.append({"status": "not_feasible", "reason": str(error)})
        if integrator == "tdvp":
            engine, engine_name = _make_tdvp_engine(
                state,
                dynamic_mpo,
                dt=dt,
                mps_max_bond=int(mps_max_bond),
                mps_cutoff=float(mps_cutoff),
                lanczos_max=int(lanczos_max),
            )
        else:
            engine, engine_name = _make_expm_engine(
                state,
                dynamic_mpo,
                dt=dt,
                mps_max_bond=int(mps_max_bond),
                mps_cutoff=float(mps_cutoff),
            )
        # TeNPy 1.1.0's run() logging assumes at least one MPS bond; evolve directly
        # so the explicit q=1 single-site path remains supported.
        step_truncation_error = _evolve_one_mpo_step(engine, integrator=integrator, dt=dt)
        state = engine.psi
        truncation_error += step_truncation_error
        truncation_error_by_step.append(step_truncation_error)
        peak_bond = max(peak_bond, _mps_peak_bond(state))

    evolution_seconds = time.perf_counter() - evolution_start
    base_diagnostics.update(
        {
            "status": "ok",
            "completed_steps": int(steps),
            "tdvp_engine": engine_name if integrator == "tdvp" else None,
            "midpoint_lambdas": midpoint_lambdas,
            "operator_build_seconds": operator_build_seconds,
            "evolution_seconds": evolution_seconds,
            "truncation_error": truncation_error,
            "truncation_error_by_step": truncation_error_by_step,
            "norm_drift": abs(_physical_mps_norm_squared(state) - 1.0),
            "peak_mps_bond": peak_bond,
            "final_mps_bond": _mps_peak_bond(state),
            "dynamic_mpo_peak_bond": dynamic_peak_bond,
            "dynamic_mpo_discarded_weight": 0.0,
            "dynamic_mpo_action_status": str(dynamic_details["action_status"]),
            "dynamic_mpo_error_status": str(dynamic_details["error_status"]),
            "dynamic_mpo_workspace": dynamic_details,
            "dynamic_mpo_action_probes": dynamic_action_probes,
            "resource_statuses": {
                "static_mpo_compression": "ok",
                "dynamic_mpo_assembly": "ok",
            },
        }
    )
    base_diagnostics.update(_final_mpo_metrics(state, prepared.h1_exact_mpo, chain_ground))
    return state, base_diagnostics


def _evolve_one_mpo_step(engine: Any, *, integrator: str, dt: float) -> float:
    """Return TeNPy's actual one-step discarded weight for each engine path."""
    if integrator == "expm_mpo":
        # TimeDependentExpMPOEvolution.run_evolution() discards the TruncationError
        # returned by evolve(); reproduce its update sequence while retaining it.
        engine.prepare_evolve(dt)
        truncation = engine.evolve(1, dt)
        engine.reinit_model()
        return max(float(truncation.eps), 0.0)
    previous = float(engine.trunc_err.eps)
    engine.run_evolution(1, dt)
    return max(float(engine.trunc_err.eps) - previous, 0.0)


def _resolve_protocol_factorization(
    *,
    protocol: str | None,
    cd_factorization: TemporalFactorization | None,
    cd_labels: Sequence[str],
    h0_terms: Sequence[tuple[str, complex]],
    h1_terms: Sequence[tuple[str, complex]],
    total_time: float,
    steps: int,
    schedule: Any | None,
) -> tuple[str, TemporalFactorization | None, tuple[str, ...]]:
    resolved = protocol or ("learned" if cd_factorization is not None else "no_cd")
    if resolved not in {"no_cd", "learned", "nested_l1"}:
        raise ValueError("protocol must be 'no_cd', 'learned', or 'nested_l1'.")
    if resolved == "no_cd":
        if cd_factorization is not None or cd_labels:
            raise ValueError("no_cd does not accept learned CD modes or labels.")
        return resolved, None, ()
    if resolved == "learned":
        if cd_factorization is None or not cd_labels:
            raise ValueError("learned evolution requires cd_factorization and cd_labels.")
        if cd_factorization.static_modes.shape[1] != len(cd_labels):
            raise ValueError("cd_labels must match the learned factorization term count.")
        return resolved, cd_factorization, tuple(cd_labels)
    if cd_factorization is not None or cd_labels:
        raise ValueError("nested_l1 constructs its own full direct-CD factorization.")
    factorization, labels = _factor_nested_l1_direct_cd(
        h0_terms, h1_terms, total_time=total_time, steps=steps, schedule=schedule
    )
    return resolved, factorization, labels


def _factor_nested_l1_direct_cd(
    h0_terms: Sequence[tuple[str, complex]],
    h1_terms: Sequence[tuple[str, complex]],
    *,
    total_time: float,
    steps: int,
    schedule: Any | None,
) -> tuple[TemporalFactorization | None, tuple[str, ...]]:
    # A one-step diagnostic still evaluates the Hamiltonian at tau=1/2. Keep
    # that midpoint in the factorization instead of sampling only endpoints,
    # where constrained schedules have d(lambda)/dt=0.
    tau = np.linspace(0.0, 1.0, max(int(steps) + 1, 3))
    samples: list[dict[str, complex]] = []
    all_labels: set[str] = set()
    for value in tau:
        lam, dlam_dt = _schedule_values(schedule, float(value), total_time)
        terms = _nested_l1_direct_cd_terms(h0_terms, h1_terms, lam, dlam_dt)
        samples.append(terms)
        all_labels.update(terms)
    if not all_labels:
        return None, ()
    labels = tuple(sorted(all_labels))
    coefficients = np.empty((tau.size, len(labels)), dtype=np.float64)
    for row, terms in enumerate(samples):
        for column, label in enumerate(labels):
            value = terms.get(label, 0.0)
            if abs(value.imag) > 1.0e-10:
                raise ValueError("Nested l=1 construction produced a non-Hermitian coefficient.")
            coefficients[row, column] = float(value.real)
    return factor_direct_cd_coefficients(tau, coefficients, retained_norm=1.0), labels


def _nested_l1_direct_cd_terms(
    h0_terms: Sequence[tuple[str, complex]],
    h1_terms: Sequence[tuple[str, complex]],
    lam: float,
    dlam_dt: float,
) -> dict[str, complex]:
    h_ad = _combine_pauli_terms(((1.0 - lam, h0_terms), (lam, h1_terms)))
    d_h = _combine_pauli_terms(((1.0, h1_terms), (-1.0, h0_terms)))
    candidate = {label: 1.0j * value for label, value in _commutator_term_maps(h_ad, d_h).items()}
    if not candidate:
        return {}
    direction = _commutator_term_maps(candidate, h_ad)
    denominator = math.fsum(abs(value) ** 2 for value in direction.values())
    if denominator <= 1.0e-24:
        return {}
    numerator = sum(np.conjugate(value) * (1.0j * d_h.get(label, 0.0)) for label, value in direction.items())
    alpha = float(np.real(numerator) / denominator)
    return {label: dlam_dt * alpha * value for label, value in candidate.items()}


def _combine_pauli_terms(
    weighted_terms: Sequence[tuple[float, Sequence[tuple[str, complex]]]]
) -> dict[str, complex]:
    contributions: dict[str, list[complex]] = {}
    for scale, terms in weighted_terms:
        for label, coefficient in terms:
            value = _finite_complex(coefficient) * float(scale)
            contributions.setdefault(label, []).append(value)
    return {
        label: value
        for label, values in contributions.items()
        if abs(value := _stable_complex_sum(values)) > 0.0
    }


def _commutator_term_maps(
    left: dict[str, complex], right: dict[str, complex]
) -> dict[str, complex]:
    contributions: dict[str, list[complex]] = {}
    for left_label, left_value in left.items():
        for right_label, right_value in right.items():
            phase, label = _multiply_pauli_labels(left_label, right_label)
            reverse_phase, reverse_label = _multiply_pauli_labels(right_label, left_label)
            if label != reverse_label:
                raise RuntimeError("Pauli multiplication returned inconsistent labels.")
            coefficient = left_value * right_value * (phase - reverse_phase)
            if coefficient != 0.0:
                contributions.setdefault(label, []).append(coefficient)
    return {
        label: value
        for label, values in contributions.items()
        if abs(value := _stable_complex_sum(values)) > 0.0
    }


def _protocol_terms_at_time(
    h0_terms: Sequence[tuple[str, complex]],
    h1_terms: Sequence[tuple[str, complex]],
    lam: float,
    cd_labels: Sequence[str],
    factorization: TemporalFactorization | None,
    mode_coefficients: np.ndarray,
    protocol: str,
) -> list[tuple[str, complex]]:
    terms = _combine_pauli_terms(((1.0 - lam, h0_terms), (lam, h1_terms)))
    if protocol != "no_cd":
        if factorization is None:
            raise RuntimeError("Counterdiabatic protocol is missing its factorization.")
        if mode_coefficients.shape != (factorization.rank,):
            raise ValueError("Temporal-mode interpolation did not return every declared mode.")
        for column, label in enumerate(cd_labels):
            value = float(np.dot(mode_coefficients, factorization.static_modes[:, column]))
            terms[label] = terms.get(label, 0.0) + value
    return [(label, coefficient) for label, coefficient in terms.items() if coefficient != 0.0]


def _temporal_mode_values(
    factorization: TemporalFactorization | None, tau: float
) -> np.ndarray:
    if factorization is None:
        return np.empty(0, dtype=np.float64)
    if not 0.0 <= float(tau) <= 1.0:
        raise ValueError("Midpoint tau must be in [0, 1].")
    return np.asarray(
        [np.interp(float(tau), factorization.tau, factorization.temporal_factors[:, mode])
         for mode in range(factorization.rank)],
        dtype=np.float64,
    )


def _schedule_values(schedule: Any | None, tau: float, total_time: float) -> tuple[float, float]:
    if schedule is None:
        lam = float(np.sin(0.5 * np.pi * tau) ** 2)
        derivative = float(0.5 * np.pi / total_time * np.sin(np.pi * tau))
    else:
        result = schedule(float(tau), float(total_time))
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("schedule must return (lambda, d_lambda_dt).")
        lam, derivative = (float(result[0]), float(result[1]))
    if not np.isfinite(lam) or not np.isfinite(derivative):
        raise ValueError("schedule returned non-finite values.")
    return lam, derivative


def _assemble_dynamic_mpo(
    prepared: PreparedTDVPOperators,
    *,
    h0_coefficient: float,
    h1_coefficient: float,
    cd_mode_coefficients: np.ndarray,
    workspace_cap_bytes: int,
    expm_compatible: bool = False,
    exact_components: bool = False,
) -> tuple[Any, dict[str, object]]:
    if exact_components:
        if prepared.h0_exact_mpo is None or prepared.h1_exact_mpo is None:
            raise RuntimeError("Exact dynamic action probing requires exact H_initial and H_final MPOs.")
        components = [prepared.h0_exact_mpo, prepared.h1_exact_mpo, *prepared.cd_mode_exact_mpos]
    else:
        components = [prepared.h0_mpo, prepared.h1_mpo, *prepared.cd_mode_mpos]
    if any(component is None for component in components):
        raise RuntimeError("Time-dependent evolution requires compressed H_initial and H_final MPOs.")
    if cd_mode_coefficients.shape != (len(prepared.cd_mode_mpos),):
        raise ValueError("Dynamic assembly requires one coefficient for every declared temporal mode.")
    if (
        isinstance(workspace_cap_bytes, bool)
        or not isinstance(workspace_cap_bytes, Integral)
        or int(workspace_cap_bytes) < 1
    ):
        raise ValueError("workspace_cap_bytes must be a positive integer.")
    coefficients = [h0_coefficient, h1_coefficient, *cd_mode_coefficients.tolist()]
    if exact_components:
        input_term_count = sum(len(source._agp_pauli_terms) for source in components)
        n_qubits = len(prepared.sites)
        # This branch creates several Python objects per input coordinate
        # (dictionary entry, coefficient list entry, normalized tuple, and
        # metadata). Use a deliberately conservative bound rather than NumPy
        # payload size; measured q24 peaks stay below one quarter of it.
        estimated_entry_bytes = 4096 + 8 * n_qubits
        required_workspace_bytes = (
            _MPO_WORKSPACE_SAFETY_MARGIN_BYTES
            + input_term_count * estimated_entry_bytes
        )
        if required_workspace_bytes > int(workspace_cap_bytes):
            raise _DynamicMPOResourceError(
                {
                    "status": "not_feasible",
                    "reason": "exact Pauli-coordinate action source exceeds workspace cap",
                    "representation": "pauli_coordinate_source",
                    "input_terms": input_term_count,
                    "estimated_entry_bytes": estimated_entry_bytes,
                    "required_workspace_bytes": required_workspace_bytes,
                    "workspace_cap_bytes": int(workspace_cap_bytes),
                }
            )
        contributions: dict[str, list[complex]] = {}
        for scale, source in zip(coefficients, components, strict=True):
            for label, coefficient in source._agp_pauli_terms:
                contributions.setdefault(label, []).append(float(scale) * coefficient)
        exact_terms = [
            (label, _stable_complex_sum(values))
            for label, values in contributions.items()
        ]
        dynamic, metadata = build_pauli_coordinate_source(
            exact_terms,
            n_qubits=len(prepared.sites),
            order=tuple(range(len(prepared.sites))),
        )
        return dynamic, {
            "status": "ok",
            "representation": "pauli_coordinate_source",
            "input_terms": metadata["input_terms"],
            "included_terms": metadata["included_terms"],
            "discarded_weight": 0.0,
            "action_status": "not_tested",
            "error_status": "exact_pauli_coordinates",
        }
    if expm_compatible and not exact_components:
        provenance_terms: list[Sequence[tuple[str, complex]]] = []
        for mpo in components:
            try:
                provenance_terms.append(mpo._agp_pauli_terms)
            except AttributeError as error:
                raise RuntimeError("ExpMPO requires Pauli provenance for every full-support component.") from error
        # The midpoint graph can retain every provenance contribution before duplicate-label
        # cancellation; use that allocation bound before building aggregation dictionaries.
        exact_plan = _exact_pauli_graph_workspace_plan(
            term_count=sum(len(terms) for terms in provenance_terms),
            n_qubits=len(prepared.sites),
        )
        if int(exact_plan["required_workspace_bytes"]) > int(workspace_cap_bytes):
            exact_plan.update({"status": "not_feasible", "reason": "ExpMPO midpoint graph exceeds mpo_workspace_cap_bytes"})
            raise _DynamicMPOResourceError(exact_plan)
        contributions: dict[str, list[complex]] = {}
        for scale, pauli_terms in zip(
            coefficients,
            provenance_terms,
            strict=True,
        ):
            for label, coefficient in pauli_terms:
                contributions.setdefault(label, []).append(float(scale) * coefficient)
        exact_terms = [
            (label, _stable_complex_sum(values))
            for label, values in contributions.items()
        ]
        dynamic, _ = build_exact_pauli_mpo(
            exact_terms,
            n_qubits=len(prepared.sites),
            order=tuple(range(len(prepared.sites))),
        )
        return dynamic, {
            **exact_plan,
            "status": "ok",
            "representation": "exact_pauli_graph",
            "discarded_weight": 0.0,
            "action_status": "not_tested",
            "error_status": "exact_pauli_graph",
        }
    block_plan = _dynamic_block_sum_workspace_plan(components)
    if int(block_plan["required_workspace_bytes"]) > int(workspace_cap_bytes):
        block_plan.update({"status": "not_feasible", "reason": "dynamic block-sum MPO exceeds mpo_workspace_cap_bytes"})
        raise _DynamicMPOResourceError(block_plan)
    component_tensors = [_effective_finite_mpo_tensors(mpo) for mpo in components]
    _, _, _, MPO, npc = _require_tenpy()
    tensors: list[np.ndarray] = []
    length = len(prepared.sites)
    for site in range(length):
        local = [items[site] for items in component_tensors]
        if length == 1:
            tensors.append(sum(float(scale) * item for scale, item in zip(coefficients, local, strict=True)))
        elif site == 0:
            tensors.append(np.concatenate(
                [float(scale) * item for scale, item in zip(coefficients, local, strict=True)], axis=1
            ))
        elif site == length - 1:
            tensors.append(np.concatenate(local, axis=0))
        else:
            left_dim = sum(item.shape[0] for item in local)
            right_dim = sum(item.shape[1] for item in local)
            block = np.zeros((left_dim, right_dim, 2, 2), dtype=np.complex128)
            left = right = 0
            for item in local:
                next_left = left + item.shape[0]
                next_right = right + item.shape[1]
                block[left:next_left, right:next_right] = item
                left, right = next_left, next_right
            tensors.append(block)
    dynamic_mpo = _mpo_from_effective_tensors(MPO, npc, sites=prepared.sites, tensors=tensors)
    if exact_components:
        contributions: dict[str, complex] = {}
        for coefficient, component in zip(coefficients, components, strict=True):
            for label, value in component._agp_pauli_terms:
                contributions[label] = contributions.get(label, 0.0j) + float(coefficient) * value
        dynamic_mpo._agp_pauli_terms = tuple(
            (label, value) for label, value in contributions.items() if value != 0.0
        )
    return dynamic_mpo, {
        **block_plan,
        "status": "ok",
        "representation": (
            "exact_block_sum_of_exact_components"
            if exact_components
            else "exact_block_sum_of_compressed_components"
        ),
        "discarded_weight": 0.0,
        "action_status": "not_tested",
        "error_status": "exact_block_sum",
    }


def _effective_finite_mpo_tensor_shapes(mpo: Any) -> list[tuple[int, int, int, int]]:
    shapes: list[tuple[int, int, int, int]] = []
    for site in range(int(mpo.L)):
        tensor = mpo.get_W(site)
        labels = tensor.get_leg_labels()
        shape = tensor.shape
        axes = [labels.index(label) for label in ("wL", "wR", "p", "p*")]
        shapes.append(tuple(int(shape[axis]) for axis in axes))
    left_boundary = mpo.get_IdL(0)
    right_boundary = mpo.get_IdR(mpo.L - 1)
    if left_boundary is None and shapes[0][0] != 1:
        raise ValueError("Finite MPO has no unambiguous left boundary index.")
    if right_boundary is None and shapes[-1][1] != 1:
        raise ValueError("Finite MPO has no unambiguous right boundary index.")
    first = shapes[0]
    last = shapes[-1]
    shapes[0] = (1, first[1], first[2], first[3])
    shapes[-1] = (last[0], 1, last[2], last[3])
    for site, shape in enumerate(shapes):
        if shape[2:] != (2, 2):
            raise ValueError("MPO tensors must have two-dimensional input and output physical legs.")
        if site and shapes[site - 1][1] != shape[0]:
            raise ValueError("Adjacent MPO bond dimensions are inconsistent.")
    return shapes


def _dynamic_block_sum_workspace_plan(components: Sequence[Any]) -> dict[str, object]:
    if not components:
        raise ValueError("Dynamic MPO assembly requires at least one component.")
    component_shapes = [_effective_finite_mpo_tensor_shapes(mpo) for mpo in components]
    length = len(component_shapes[0])
    if length < 1 or any(len(shapes) != length for shapes in component_shapes):
        raise ValueError("Dynamic MPO components must have a common positive chain length.")
    source_bytes = int(
        sum(np.prod(shape, dtype=np.int64) * _COMPLEX_BYTES for shapes in component_shapes for shape in shapes)
    )
    output_shapes: list[tuple[int, int, int, int]] = []
    scale_temporary_bytes = 0
    for site in range(length):
        local = [shapes[site] for shapes in component_shapes]
        if length == 1:
            output_shape = (1, 1, 2, 2)
        elif site == 0:
            output_shape = (1, sum(shape[1] for shape in local), 2, 2)
        elif site == length - 1:
            output_shape = (sum(shape[0] for shape in local), 1, 2, 2)
        else:
            output_shape = (
                sum(shape[0] for shape in local),
                sum(shape[1] for shape in local),
                2,
                2,
            )
        output_shapes.append(output_shape)
        scale_temporary_bytes = max(
            scale_temporary_bytes,
            int(sum(np.prod(shape, dtype=np.int64) * _COMPLEX_BYTES for shape in local)),
        )
    output_bytes = int(sum(np.prod(shape, dtype=np.int64) * _COMPLEX_BYTES for shape in output_shapes))
    max_output_bytes = int(max(np.prod(shape, dtype=np.int64) * _COMPLEX_BYTES for shape in output_shapes))
    required = int(
        2 * source_bytes
        + 3 * output_bytes
        + 2 * max_output_bytes
        + scale_temporary_bytes
        + _MPO_WORKSPACE_SAFETY_MARGIN_BYTES
    )
    return {
        "status": "planned",
        "representation": "block_sum",
        "component_count": len(component_shapes),
        "component_tensor_shapes": component_shapes,
        "output_tensor_shapes": output_shapes,
        "component_tensor_bytes": source_bytes,
        "output_tensor_bytes": output_bytes,
        "largest_output_tensor_bytes": max_output_bytes,
        "scale_temporary_bytes": scale_temporary_bytes,
        "copy_and_conversion_bytes": int(2 * source_bytes + 2 * output_bytes + max_output_bytes),
        "required_workspace_bytes": required,
    }


def _exact_pauli_graph_workspace_plan(*, term_count: int, n_qubits: int) -> dict[str, object]:
    virtual_dimension = max(int(term_count) + 1, 1)
    tensor_bytes = int(n_qubits * virtual_dimension * virtual_dimension * 4 * _COMPLEX_BYTES)
    required = int(4 * tensor_bytes + _MPO_WORKSPACE_SAFETY_MARGIN_BYTES)
    return {
        "status": "planned",
        "representation": "exact_pauli_graph",
        "provenance_term_upper_bound": int(term_count),
        "worst_case_virtual_dimension": virtual_dimension,
        "tensor_bytes": tensor_bytes,
        "copy_and_conversion_bytes": int(3 * tensor_bytes),
        "required_workspace_bytes": required,
    }


def _make_tdvp_engine(
    state: Any,
    mpo: Any,
    *,
    dt: float,
    mps_max_bond: int,
    mps_cutoff: float,
    lanczos_max: int,
) -> tuple[Any, str]:
    from tenpy.algorithms.tdvp import SingleSiteTDVPEngine, TwoSiteTDVPEngine

    engine_class = SingleSiteTDVPEngine if int(state.L) == 1 else TwoSiteTDVPEngine
    name = "single_site_tdvp_l1" if int(state.L) == 1 else "two_site_tdvp"
    options = {
        "dt": dt,
        "max_dt": max(abs(float(dt)), 1.0),
        "N_steps": 1,
        "max_trunc_err": float("inf"),
        "trunc_params": {"chi_max": mps_max_bond, "svd_min": mps_cutoff},
        "lanczos_params": {"N_max": lanczos_max},
    }
    engine = engine_class(state, _static_mpo_model(mpo), options)
    _mark_engine_options_consumed(engine)
    return engine, name


def _make_expm_engine(
    state: Any,
    mpo: Any,
    *,
    dt: float,
    mps_max_bond: int,
    mps_cutoff: float,
) -> tuple[Any, str]:
    from tenpy.algorithms.mpo_evolution import TimeDependentExpMPOEvolution

    options = {
        "dt": dt,
        "max_dt": max(abs(float(dt)), 1.0),
        "N_steps": 1,
        "max_trunc_err": float("inf"),
        "trunc_params": {"chi_max": mps_max_bond, "svd_min": mps_cutoff},
        "compression_method": "zip_up",
        "approximation": "I",
        "order": 2,
    }
    engine = TimeDependentExpMPOEvolution(state, _frozen_time_mpo_model(mpo), options)
    _mark_engine_options_consumed(engine)
    return engine, "time_dependent_expm_mpo"


def _mark_engine_options_consumed(engine: Any) -> None:
    """Avoid TeNPy 1.1.0 false unused-option warnings when stepping manually."""
    engine.options.get("dt", None)
    engine.options.get("max_dt", None)
    engine.options.get("N_steps", None)
    engine.options.get("max_trunc_err", None)
    truncation = engine.options.subconfig("trunc_params")
    truncation.get("chi_max", None)
    truncation.get("svd_min", None)


def _static_mpo_model(mpo: Any) -> Any:
    from tenpy.models.lattice import Chain
    from tenpy.models.model import MPOModel

    lattice = Chain(len(mpo.sites), mpo.sites[0], bc="open", bc_MPS="finite")
    return MPOModel(lattice, mpo)


def _frozen_time_mpo_model(mpo: Any) -> Any:
    from tenpy.models.lattice import Chain
    from tenpy.models.model import MPOModel
    from tenpy.tools.params import asConfig

    class FrozenTimeMPOModel(MPOModel):
        def __init__(self, hamiltonian: Any):
            lattice = Chain(
                len(hamiltonian.sites),
                hamiltonian.sites[0],
                bc="open",
                bc_MPS="finite",
            )
            super().__init__(lattice, hamiltonian)
            self.options = asConfig({"time": 0.0}, "FrozenTimeMPOModel")

        def update_time_parameter(self, new_time: float) -> "FrozenTimeMPOModel":
            self.options["time"] = new_time
            return self

    return FrozenTimeMPOModel(mpo)


def _protocol_n_qubits(
    h0_terms: Sequence[tuple[str, complex]], h1_terms: Sequence[tuple[str, complex]]
) -> int:
    candidates = [label for terms in (h0_terms, h1_terms) for label, _ in terms]
    if not candidates:
        raise ValueError("At least one H_initial or H_final Pauli term is required.")
    n_qubits = len(candidates[0])
    _validate_n_qubits(n_qubits)
    for label in candidates:
        _validate_pauli_label(label, n_qubits=n_qubits)
    return n_qubits


def _validate_evolution_parameters(total_time: float, steps: int) -> float:
    if not np.isfinite(total_time) or float(total_time) <= 0.0:
        raise ValueError("total_time must be finite and positive.")
    if isinstance(steps, bool) or not isinstance(steps, Integral) or int(steps) < 1:
        raise ValueError("steps must be a positive integer.")
    return float(total_time) / int(steps)


def _chain_initial_state(
    initial_state: Sequence[object] | None, order: Sequence[int]
) -> tuple[list[object], list[object]]:
    n_qubits = len(order)
    original = list(initial_state) if initial_state is not None else ["+"] * n_qubits
    if len(original) != n_qubits:
        raise ValueError("initial_state must specify one local state per original qubit.")
    chain = [original[site] for site in order]
    return [_normalize_local_state(value) for value in chain], original


def _normalize_local_state(value: object) -> object:
    if value == "0":
        return "up"
    if value == "1":
        return "down"
    if value == "+":
        return np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    if value == "-":
        return np.asarray([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)
    if isinstance(value, str):
        if value not in {"up", "down"}:
            raise ValueError("initial_state strings must be 0, 1, +, -, up, or down.")
        return value
    array = np.asarray(value, dtype=np.complex128)
    if array.shape != (2,) or not np.all(np.isfinite(array)) or np.linalg.norm(array) == 0.0:
        raise ValueError("Each initial_state vector must be a finite nonzero two-component vector.")
    return array / np.linalg.norm(array)


def _chain_ground_bitstring(
    ground_bitstring: str | None, order: Sequence[int]
) -> tuple[str | None, str | None]:
    if ground_bitstring is None:
        return None, None
    if len(ground_bitstring) != len(order) or any(bit not in "01" for bit in ground_bitstring):
        raise ValueError("ground_bitstring must contain one binary digit per original qubit.")
    return "".join(ground_bitstring[site] for site in order), ground_bitstring


def _make_product_mps(sites: Sequence[Any], states: Sequence[object]) -> Any:
    from tenpy.networks.mps import MPS

    return MPS.from_product_state(
        list(sites), list(states), bc="finite", dtype=np.complex128, unit_cell_width=len(sites)
    )


def _dense_product_state(states: Sequence[object]) -> np.ndarray:
    vector = np.asarray([1.0 + 0.0j])
    for state in states:
        local = state
        if isinstance(local, str):
            local = np.asarray([1.0, 0.0] if local == "up" else [0.0, 1.0], dtype=np.complex128)
        vector = np.kron(vector, local)
    return vector


def _mps_to_dense_statevector(state: Any) -> np.ndarray:
    theta = state.get_theta(0, int(state.L)).to_ndarray()
    vector = np.asarray(theta, dtype=np.complex128)[0, ..., 0].reshape(-1)
    return complex(state.norm) * vector


def _mps_peak_bond(state: Any) -> int:
    return max((int(value) for value in state.chi), default=1)


def _mpo_bonds(mpo: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in mpo.chi)


def _final_mpo_metrics(state: Any, h1_exact_mpo: Any | None, ground_chain: str | None) -> dict[str, object]:
    metrics: dict[str, object] = {
        "final_energy_status": "not tested",
        "ground_fidelity_status": "not tested" if ground_chain is None else "not_feasible",
    }
    if h1_exact_mpo is not None:
        try:
            energy = sum(
                coefficient * _mps_pauli_expectation(state, label)
                for label, coefficient in h1_exact_mpo._agp_pauli_terms
            )
            if abs(energy.imag) > 1.0e-9:
                raise ValueError("Exact final MPO produced a non-real energy expectation.")
            metrics.update({"final_energy": float(energy.real), "final_energy_status": "ok"})
        except (AttributeError, ValueError):
            metrics["final_energy_status"] = "not_feasible"
    if ground_chain is not None:
        target = _make_product_mps(
            state.sites, ["up" if bit == "0" else "down" for bit in ground_chain]
        )
        try:
            metrics.update(
                {
                    "ground_fidelity": float(abs(_canonical_mps_overlap(state, target)) ** 2),
                    "ground_fidelity_status": "ok",
                }
            )
        except (ValueError, RuntimeError):
            metrics["ground_fidelity_status"] = "not_feasible"
    return metrics


def _effective_finite_mpo_tensors(mpo: Any) -> list[np.ndarray]:
    tensors: list[np.ndarray] = []
    for site in range(mpo.L):
        tensor = mpo.get_W(site)
        labels = tensor.get_leg_labels()
        axes = [labels.index(label) for label in ("wL", "wR", "p", "p*")]
        tensors.append(np.transpose(tensor.to_ndarray(), axes).copy())

    left_boundary = mpo.get_IdL(0)
    right_boundary = mpo.get_IdR(mpo.L - 1)
    if left_boundary is None:
        if tensors[0].shape[0] != 1:
            raise ValueError("Finite MPO has no unambiguous left boundary index.")
        left_boundary = 0
    if right_boundary is None:
        if tensors[-1].shape[1] != 1:
            raise ValueError("Finite MPO has no unambiguous right boundary index.")
        right_boundary = 0
    if not 0 <= int(left_boundary) < tensors[0].shape[0]:
        raise ValueError("Finite MPO left boundary index is outside its bond dimension.")
    if not 0 <= int(right_boundary) < tensors[-1].shape[1]:
        raise ValueError("Finite MPO right boundary index is outside its bond dimension.")
    tensors[0] = tensors[0][int(left_boundary) : int(left_boundary) + 1]
    tensors[-1] = tensors[-1][:, int(right_boundary) : int(right_boundary) + 1]

    for site, tensor in enumerate(tensors):
        if tensor.ndim != 4 or tensor.shape[2:] != (2, 2):
            raise ValueError("MPO tensors must have two-dimensional input and output physical legs.")
        if site and tensors[site - 1].shape[1] != tensor.shape[0]:
            raise ValueError("Adjacent MPO bond dimensions are inconsistent.")
    return tensors


def _mpo_from_effective_tensors(
    MPO: Any,
    npc: Any,
    *,
    sites: Sequence[Any],
    tensors: Sequence[np.ndarray],
) -> Any:
    arrays = []
    for site, tensor in zip(sites, tensors):
        left_leg = npc.LegCharge.from_trivial(
            tensor.shape[0], chargeinfo=site.leg.chinfo, qconj=1
        )
        right_leg = npc.LegCharge.from_trivial(
            tensor.shape[1], chargeinfo=site.leg.chinfo, qconj=-1
        )
        arrays.append(
            npc.Array.from_ndarray(
                tensor,
                [left_leg, right_leg, site.leg, site.leg.conj()],
                labels=["wL", "wR", "p", "p*"],
                qtotal=site.leg.chinfo.make_valid(),
            )
        )
    length = len(arrays)
    return MPO(
        list(sites),
        arrays,
        bc="finite",
        IdL=[0] + [None] * length,
        IdR=[None] * length + [0],
        mps_unit_cell_width=length,
    )


def _mpo_from_pauli_cores(
    MPO: Any,
    npc: Any,
    *,
    sites: Sequence[Any],
    cores: Sequence[np.ndarray],
) -> Any:
    pauli_matrices = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, -1.0j], [1.0j, 0.0]],
            [[1.0, 0.0], [0.0, -1.0]],
        ],
        dtype=np.complex128,
    )
    tensors = [
        np.tensordot(core, pauli_matrices, axes=(1, 0))
        for core in cores
    ]
    return _mpo_from_effective_tensors(MPO, npc, sites=sites, tensors=tensors)


def _encode_pauli_label(label: str) -> int:
    encoded = 0
    for symbol in label:
        encoded = (encoded << 2) | _PAULI_INDEX[symbol]
    return encoded


def _compression_not_feasible(
    diagnostics: dict[str, object],
    *,
    required_workspace_bytes: int,
    peak_workspace_bytes: int,
    failed_bond: int | None,
    reason: str,
) -> tuple[None, dict[str, object]]:
    diagnostics.update(
        {
            "status": "not_feasible",
            "required_workspace_bytes": int(required_workspace_bytes),
            "peak_workspace_bytes": int(peak_workspace_bytes),
            "failed_bond": failed_bond,
            "resource_reason": reason,
        }
    )
    return None, diagnostics


def _mark_exact_identity(
    mpo: Any,
    pauli_terms: Sequence[tuple[str, complex]],
    *,
    maximum_bond: int,
) -> None:
    """Record mutation-detecting identity evidence for a bounded compressed MPO."""
    fingerprint = _bounded_compressed_mpo_fingerprint(mpo, maximum_bond=maximum_bond)
    if fingerprint is None:
        raise ValueError("Exact-identity evidence requires bounded compressed MPO tensors.")
    mpo._agp_exact_identity_certificate = {
        "pauli_terms": tuple(pauli_terms),
        "maximum_bond": int(maximum_bond),
        "fingerprint": fingerprint,
    }


def _exact_identity_certificate_status(
    mpo: Any, pauli_terms: Sequence[tuple[str, complex]]
) -> str:
    """Verify bounded compressed-MPO identity evidence without inspecting an exact graph."""
    certificate = getattr(mpo, "_agp_exact_identity_certificate", None)
    if not isinstance(certificate, dict):
        return "not_established"
    if certificate.get("pauli_terms") != tuple(pauli_terms):
        return "invalidated"
    maximum_bond = certificate.get("maximum_bond")
    fingerprint = certificate.get("fingerprint")
    if not isinstance(maximum_bond, Integral) or not isinstance(fingerprint, str):
        return "invalidated"
    current_fingerprint = _bounded_compressed_mpo_fingerprint(
        mpo, maximum_bond=int(maximum_bond)
    )
    if current_fingerprint is None or current_fingerprint != fingerprint:
        return "invalidated"
    return "verified"


def _bounded_compressed_mpo_fingerprint(mpo: Any, *, maximum_bond: int) -> str | None:
    """Hash bounded compressed tensor payloads; never invoke this on an exact MPO graph."""
    if int(maximum_bond) < 1 or max(int(bond) for bond in mpo.chi) > int(maximum_bond):
        return None
    digest = hashlib.sha256()
    digest.update(
        repr(("agp-mpo-identity-v1", int(mpo.L), tuple(int(bond) for bond in mpo.chi))).encode(
            "ascii"
        )
    )
    for site in range(int(mpo.L)):
        tensor = mpo.get_W(site)
        labels = tuple(tensor.get_leg_labels())
        payload = np.ascontiguousarray(tensor.to_ndarray())
        digest.update(repr((labels, payload.shape, payload.dtype.str)).encode("ascii"))
        digest.update(payload.tobytes(order="C"))
    return digest.hexdigest()


def _svd_retained_rank(
    singular_values: np.ndarray,
    *,
    max_bond: int,
    cutoff: float,
    unfolding_dimension: int | None = None,
) -> tuple[int, float]:
    if singular_values.size == 0:
        raise ValueError("SVD returned no singular values for a nonempty MPO tensor.")
    scale = float(np.max(np.abs(singular_values)))
    if scale == 0.0:
        return 1, 0.0
    relative_squared = np.square(singular_values / scale)
    total = float(np.sum(relative_squared))
    if cutoff == 0.0:
        dimension = max(int(unfolding_dimension or singular_values.size), 1)
        # Singular values are obtained from an eigendecomposition of M M^H.
        # Its roundoff floor is in squared-singular-value units and scales with
        # the unfolding dimension. Retaining that null space makes exact sparse
        # MPOs inflate to the requested bond cap.
        numerical_squared_floor = (
            8.0 * dimension * np.finfo(np.float64).eps
        )
        retained_for_cutoff = int(np.count_nonzero(relative_squared > numerical_squared_floor))
        retained_for_cutoff = max(retained_for_cutoff, 1)
    else:
        retained_for_cutoff = singular_values.size
        for rank in range(1, singular_values.size + 1):
            discarded = float(np.sum(relative_squared[rank:]) / total)
            if discarded <= cutoff:
                retained_for_cutoff = rank
                break
    retained_rank = min(retained_for_cutoff, max_bond)
    retained_rank = max(1, retained_rank)
    discarded_weight = float(np.sum(relative_squared[retained_rank:]) / total)
    return retained_rank, discarded_weight


def _apply_mpo_action(mpo: Any, state: Any, npc: Any) -> Any:
    result = state.copy()
    if mpo.L > 1:
        mpo.apply_naively(result)
        return result

    local_matrix = _effective_finite_mpo_tensors(mpo)[0][0, 0]
    site = mpo.sites[0]
    local_operator = npc.Array.from_ndarray(
        local_matrix,
        [site.leg, site.leg.conj()],
        labels=["p", "p*"],
        qtotal=site.leg.chinfo.make_valid(),
    )
    updated_tensor = npc.tensordot(
        local_operator,
        result.get_B(0, form=None),
        axes=(["p*"], ["p"]),
    )
    result.set_B(0, updated_tensor, result.form[0])
    return result


def _real_overlap(left: Any, right: Any) -> float:
    overlap = left.overlap(right, ignore_form=True)
    result = float(np.real(overlap))
    scale = float(abs(overlap))
    if result < 0.0 and abs(result) <= _gamma_n_bound(scale, operation_estimate=1):
        return 0.0
    if result < 0.0:
        raise ValueError("MPS norm contraction produced a negative value.")
    return result


def _validate_dense_helper_size(n_qubits: int) -> None:
    if n_qubits < 1 or n_qubits > 4:
        raise ValueError("Dense MPO test helpers are restricted to q <= 4 (at most four qubits).")


def _finite_complex(coefficient: complex) -> complex:
    try:
        value = complex(coefficient)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Term coefficients must be finite complex scalars.") from error
    if not np.isfinite(value.real) or not np.isfinite(value.imag):
        raise ValueError("Term coefficients must be finite.")
    return value


def _stable_complex_sum(values: Sequence[complex]) -> complex:
    try:
        result = complex(
            math.fsum(value.real for value in values),
            math.fsum(value.imag for value in values),
        )
    except (OverflowError, ValueError) as error:
        raise ValueError("Combined term coefficients must remain finite.") from error
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValueError("Combined term coefficients must remain finite.")
    return result


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


def _relative_squared_values(values: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return np.zeros(values.shape, dtype=np.float64)
    normalized = values / scale
    return normalized * normalized


def _scaled_column_squared_norms(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
    squared_norms = np.zeros(scales.shape, dtype=np.float64)
    nonzero_columns = scales > 0.0
    normalized = values[:, nonzero_columns] / scales[nonzero_columns]
    squared_norms[nonzero_columns] = np.sum(normalized * normalized, axis=0)
    return squared_norms


def _column_retained_energy_fractions(
    reconstructed: np.ndarray,
    source_scales: np.ndarray,
    source_relative_energies: np.ndarray,
) -> np.ndarray:
    fractions = np.ones(source_scales.shape, dtype=np.float64)
    nonzero_columns = source_scales > 0.0
    reconstructed_relative_energies = _scaled_column_squared_norms(reconstructed, source_scales)
    fractions[nonzero_columns] = (
        reconstructed_relative_energies[nonzero_columns]
        / source_relative_energies[nonzero_columns]
    )
    return fractions


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
    spectrum_scale = float(np.max(np.abs(eigenvalues)))
    if spectrum_scale == 0.0:
        return tuple(range(n_qubits))
    degeneracy_tolerance = _SPECTRAL_EIGENVALUE_RELATIVE_ATOL * spectrum_scale
    if eigenvalues[1] <= degeneracy_tolerance or (
        n_qubits > 2 and eigenvalues[2] - eigenvalues[1] <= degeneracy_tolerance
    ):
        return tuple(range(n_qubits))
    fiedler = eigenvectors[:, 1].copy()
    component_tolerance = _SPECTRAL_COMPONENT_ATOL_SCALE * np.finfo(np.float64).eps * np.sqrt(n_qubits)
    nonzero = np.flatnonzero(np.abs(fiedler) > component_tolerance)
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
