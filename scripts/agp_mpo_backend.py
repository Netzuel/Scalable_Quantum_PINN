"""Temporal factorization and deterministic ordering for MPO AGP evaluation.

This module deliberately depends only on NumPy. TeNPy is imported only by the
later MPO construction and evolution layers so training remains independent of
the optional tensor-network extra.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
import math
from numbers import Integral
import warnings
from typing import Any, Sequence

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
                >= float(retained_norm) - _COLUMN_RETAINED_ENERGY_ATOL
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
            >= float(retained_norm) - _COLUMN_RETAINED_ENERGY_ATOL
        ):
            raise ValueError("Unable to retain the requested energy for every nonzero source coefficient column.")
        rank_increase_reason = (
            None
            if rank == rank_for_retained_norm
            else (
                f"Increased rank from {rank_for_retained_norm} to {rank} so every nonzero source "
                f"coefficient column retains at least {float(retained_norm):.6g} of its squared "
                f"temporal norm within {_COLUMN_RETAINED_ENERGY_ATOL:.3e} floating tolerance."
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


def build_exact_pauli_mpo(
    terms: Sequence[tuple[str, complex]],
    *,
    n_qubits: int,
    order: Sequence[int],
    arithmetic_zero_tolerance: float = 0.0,
) -> tuple[Any, dict[str, object]]:
    """Build a finite TeNPy MPO from every nonzero combined Pauli label.

    Duplicate labels are combined in the original qubit order. A combined
    coefficient is omitted only when its magnitude is no larger than the
    explicitly reported arithmetic-zero tolerance.
    """
    SpinHalfSite, TermList, MPOGraph, _, _ = _require_tenpy()
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

    site = SpinHalfSite(conserve=None)
    site.add_op("X", site.get_op("Sigmax"), hc="X")
    site.add_op("Y", site.get_op("Sigmay"), hc="Y")
    site.add_op("Z", site.get_op("Sigmaz"), hc="Z")
    sites = [site] * n_qubits

    operator_terms: list[list[tuple[str, int]]] = []
    strengths: list[complex] = []
    included_chain_labels: list[str] = []
    included_chain_terms: list[tuple[str, complex]] = []
    for label, coefficient in included_coefficients.items():
        chain_label = permute_pauli_label(label, normalized_order)
        local_terms = [
            (symbol, chain_site)
            for chain_site, symbol in enumerate(chain_label)
            if symbol != "I"
        ]
        if not local_terms:
            local_terms = [("Id", 0)]
        operator_terms.append(local_terms)
        strengths.append(coefficient)
        included_chain_labels.append(chain_label)
        included_chain_terms.append((chain_label, coefficient))

    if not operator_terms:
        operator_terms = [[("Id", 0)]]
        strengths = [0.0]

    term_list = TermList(operator_terms, strengths)
    graph = MPOGraph.from_term_list(
        term_list,
        sites,
        bc="finite",
        insert_all_id=True,
        unit_cell_width=n_qubits,
    )
    mpo = graph.build_MPO()
    mpo._agp_pauli_terms = tuple(sorted(included_chain_terms))
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
    return mpo, metadata


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
    if mpo.L > 32:
        return _compression_not_feasible(
            diagnostics,
            required_workspace_bytes=0,
            peak_workspace_bytes=0,
            failed_bond=None,
            reason="compact Pauli TT encoding supports at most 32 qubits",
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
            }
        )
        return compressed, diagnostics

    initial_required = (
        len(pauli_terms) * (4 * _INDEX_BYTES + 3 * _COMPLEX_BYTES)
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
    cores: list[np.ndarray] = []
    per_bond_cutoff_weights: list[float] = []
    per_bond_discarded_coefficient_norms: list[float] = []
    cutoff_satisfied_by_bond: list[bool] = []
    retained_ranks: list[int] = []
    peak_workspace = initial_required
    peak_explicit_workspace = int(codes.nbytes + values.nbytes)
    required_workspace = initial_required

    for bond in range(mpo.L - 1):
        previous_rank, entry_count = values.shape
        remaining_sites = mpo.L - bond
        shift = 2 * (remaining_sites - 1)
        index_workspace = entry_count * 7 * _INDEX_BYTES
        pre_index_required = (
            sum(core.nbytes for core in cores)
            + codes.nbytes
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

        symbols = np.asarray(codes >> np.uint64(shift), dtype=np.int64)
        if shift:
            suffix_mask = np.uint64((1 << shift) - 1)
            suffixes = codes & suffix_mask
        else:
            suffixes = np.zeros_like(codes)
        unique_suffixes, inverse = np.unique(suffixes, return_inverse=True)
        column_count = unique_suffixes.size
        row_count = previous_rank * 4
        retained_bound = min(hard_max_bond, row_count, column_count)
        matrix_bytes = row_count * column_count * _COMPLEX_BYTES
        gram_bytes = row_count * row_count * _COMPLEX_BYTES
        retained_core_bytes = previous_rank * 4 * retained_bound * _COMPLEX_BYTES
        next_values_bytes = retained_bound * column_count * _COMPLEX_BYTES
        core_bytes = sum(core.nbytes for core in cores)
        conservative_required = (
            core_bytes
            + codes.nbytes
            + values.nbytes
            + symbols.nbytes
            + suffixes.nbytes
            + unique_suffixes.nbytes
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
        )
        retained_ranks.append(retained_rank)
        per_bond_cutoff_weights.append(discarded_weight)
        discarded_values = singular_values[retained_rank:]
        per_bond_discarded_coefficient_norms.append(
            float(np.vdot(discarded_values, discarded_values).real)
        )
        cutoff_satisfied_by_bond.append(
            discarded_weight <= relative_cutoff + 64.0 * np.finfo(np.float64).eps
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
            + codes.nbytes
            + values.nbytes
            + symbols.nbytes
            + suffixes.nbytes
            + unique_suffixes.nbytes
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
        codes.nbytes
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
    output_index_bytes = max(output_count, 1) * _INDEX_BYTES
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

    output_states = np.asarray(sorted(exact_action), dtype=np.uint64)
    exact_amplitudes = np.asarray(
        [exact_action[int(state)] for state in output_states], dtype=np.complex128
    )
    left_boundary = mpo.get_IdL(0)
    right_boundary = mpo.get_IdR(mpo.L - 1)
    if left_boundary is None or right_boundary is None:
        raise ValueError("Compressed finite MPO must expose both boundary indices.")
    query = np.zeros((output_states.size, mpo.chi[0]), dtype=np.complex128)
    query[:, int(left_boundary)] = 1.0
    density = np.zeros((mpo.chi[0], mpo.chi[0]), dtype=np.complex128)
    density[int(left_boundary), int(left_boundary)] = 1.0
    peak_workspace = query.nbytes + density.nbytes
    transfer_operation_estimate = 0

    for site, input_bit in enumerate(input_bits):
        tensor = _mpo_tensor_ndarray(mpo, site)
        next_query = np.zeros((output_states.size, tensor.shape[1]), dtype=np.complex128)
        next_density = np.zeros((tensor.shape[1], tensor.shape[1]), dtype=np.complex128)
        output_bits = (output_states >> np.uint64(mpo.L - site - 1)) & np.uint64(1)
        for output_bit in (0, 1):
            local = tensor[:, :, output_bit, input_bit]
            selected = output_bits == output_bit
            if np.any(selected):
                next_query[selected] = query[selected] @ local
            next_density += local.T @ density @ local.conj()
        left_bond, right_bond = tensor.shape[:2]
        transfer_operation_estimate += (
            4 * left_bond * right_bond * (left_bond + right_bond)
            + 4 * max(output_states.size, 1) * left_bond * right_bond
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
            operation_estimate=transfer_operation_estimate + 2 * output_states.size,
        )
    )
    amplitude_differences = compressed_amplitudes - exact_amplitudes
    direct_difference_norm_squared = float(
        np.vdot(amplitude_differences, amplitude_differences).real
    )
    direct_operation_estimate = max(1, 16 * output_states.size)
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
            direct_operation_estimate + transfer_operation_estimate + 2 * output_states.size
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
) -> tuple[int, float]:
    if singular_values.size == 0:
        raise ValueError("SVD returned no singular values for a nonempty MPO tensor.")
    scale = float(np.max(np.abs(singular_values)))
    if scale == 0.0:
        return 1, 0.0
    relative_squared = np.square(singular_values / scale)
    total = float(np.sum(relative_squared))
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
        raise ValueError("Dense MPO test helpers are restricted to q <= 4.")


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
