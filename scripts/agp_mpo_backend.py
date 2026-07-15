"""Temporal factorization and deterministic ordering for MPO AGP evaluation.

This module deliberately depends only on NumPy. TeNPy is imported only by the
later MPO construction and evolution layers so training remains independent of
the optional tensor-network extra.
"""

from __future__ import annotations

from dataclasses import dataclass
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
) -> tuple[Any, dict[str, object]]:
    """Compress a finite MPO with a left-to-right Hilbert-Schmidt SVD sweep.

    ``cutoff`` is the maximum relative cumulative squared singular weight to
    discard at each bond. ``max_bond`` is a hard cap and can force a larger
    discarded weight; ``cutoff_satisfied_by_bond`` records that condition.
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

    hard_max_bond = int(max_bond)
    relative_cutoff = float(cutoff)
    tensors = _effective_finite_mpo_tensors(mpo)
    pre_bonds = [tensors[0].shape[0]] + [tensor.shape[1] for tensor in tensors]
    _right_canonicalize_mpo_tensors(tensors)
    canonical_bonds = [tensors[0].shape[0]] + [tensor.shape[1] for tensor in tensors]
    source_hilbert_schmidt_norm_squared = float(np.vdot(tensors[0], tensors[0]).real)
    per_bond_cutoff_weights: list[float] = []
    per_bond_discarded_squared_norms: list[float] = []
    cutoff_satisfied_by_bond: list[bool] = []
    retained_ranks: list[int] = []

    for bond in range(mpo.L - 1):
        left_tensor = tensors[bond]
        left_dim, right_dim, physical_out, physical_in = left_tensor.shape
        matrix = left_tensor.transpose(0, 2, 3, 1).reshape(
            left_dim * physical_out * physical_in, right_dim
        )
        left_vectors, singular_values, right_vectors = np.linalg.svd(
            matrix, full_matrices=False
        )
        retained_rank, discarded_weight = _svd_retained_rank(
            singular_values,
            max_bond=hard_max_bond,
            cutoff=relative_cutoff,
        )
        retained_ranks.append(retained_rank)
        per_bond_cutoff_weights.append(discarded_weight)
        discarded_values = singular_values[retained_rank:]
        per_bond_discarded_squared_norms.append(
            float(np.vdot(discarded_values, discarded_values).real)
        )
        cutoff_satisfied_by_bond.append(
            discarded_weight <= relative_cutoff + 64.0 * np.finfo(np.float64).eps
        )

        tensors[bond] = left_vectors[:, :retained_rank].reshape(
            left_dim, physical_out, physical_in, retained_rank
        ).transpose(0, 3, 1, 2)
        remainder = singular_values[:retained_rank, None] * right_vectors[:retained_rank, :]
        tensors[bond + 1] = np.tensordot(remainder, tensors[bond + 1], axes=(1, 0))

    compressed = _mpo_from_effective_tensors(
        MPO,
        npc,
        sites=mpo.sites,
        tensors=tensors,
    )
    total_discarded_squared_norm = float(sum(per_bond_discarded_squared_norms))
    if source_hilbert_schmidt_norm_squared > 0.0:
        per_bond_discarded_weights = [
            weight / source_hilbert_schmidt_norm_squared
            for weight in per_bond_discarded_squared_norms
        ]
    else:
        per_bond_discarded_weights = [0.0] * len(per_bond_discarded_squared_norms)
    discarded_weight = float(sum(per_bond_discarded_weights))
    diagnostics: dict[str, object] = {
        "max_bond": hard_max_bond,
        "cutoff": relative_cutoff,
        "cutoff_semantics": "maximum relative cumulative discarded squared singular weight per bond",
        "source_bonds": list(mpo.chi),
        "pre_bonds": pre_bonds,
        "canonical_bonds": canonical_bonds,
        "post_bonds": list(compressed.chi),
        "retained_ranks": retained_ranks,
        "per_bond_cutoff_weights": per_bond_cutoff_weights,
        "per_bond_discarded_weights": per_bond_discarded_weights,
        "discarded_weight": discarded_weight,
        "per_bond_discarded_squared_norms": per_bond_discarded_squared_norms,
        "total_discarded_squared_norm": total_discarded_squared_norm,
        "source_hilbert_schmidt_norm_squared": source_hilbert_schmidt_norm_squared,
        "relative_hilbert_schmidt_error_squared": (
            discarded_weight
        ),
        "cutoff_satisfied_by_bond": cutoff_satisfied_by_bond,
    }
    return compressed, diagnostics


def probe_mpo_compression(
    exact_mpo: Any,
    compressed_mpo: Any,
    *,
    product_states: Sequence[Sequence[str]] | None = None,
    random_state_count: int = 2,
    random_bond: int = 4,
    seed: int = 0,
) -> dict[str, object]:
    """Measure deterministic exact-versus-compressed MPO action errors on MPS probes."""
    _, _, _, _, npc = _require_tenpy()
    from tenpy.networks.mps import MPS

    if exact_mpo.bc != "finite" or compressed_mpo.bc != "finite":
        raise ValueError("Action probes require finite MPOs.")
    if exact_mpo.L != compressed_mpo.L:
        raise ValueError("Exact and compressed MPO lengths must match.")
    if isinstance(random_state_count, bool) or not isinstance(random_state_count, Integral):
        raise ValueError("random_state_count must be a nonnegative integer.")
    if int(random_state_count) < 0:
        raise ValueError("random_state_count must be a nonnegative integer.")
    if isinstance(random_bond, bool) or not isinstance(random_bond, Integral) or int(random_bond) < 1:
        raise ValueError("random_bond must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer.")

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

    probes: list[dict[str, object]] = []
    states: list[tuple[str, Any]] = [
        (
            f"product_{index}",
            MPS.from_product_state(
                exact_mpo.sites,
                state,
                bc="finite",
                dtype=np.complex128,
                unit_cell_width=exact_mpo.L,
            ),
        )
        for index, state in enumerate(normalized_product_states)
    ]

    random_generator_state = np.random.get_state()
    maximum_finite_bond = 2 ** (exact_mpo.L // 2)
    effective_random_bond = min(int(random_bond), maximum_finite_bond)
    try:
        np.random.seed(int(seed))
        for index in range(int(random_state_count)):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="unit_cell_width is a new argument.*",
                    category=UserWarning,
                    module=r"tenpy\.networks\.mps",
                )
                random_state = MPS.from_random_unitary_evolution(
                    exact_mpo.sites,
                    chi=effective_random_bond,
                    p_state=["up"] * exact_mpo.L,
                    bc="finite",
                    dtype=np.complex128,
                )
            states.append(
                (
                    f"random_{index}",
                    random_state,
                )
            )
    finally:
        np.random.set_state(random_generator_state)

    for name, state in states:
        exact_action = _apply_mpo_action(exact_mpo, state, npc)
        compressed_action = _apply_mpo_action(compressed_mpo, state, npc)
        exact_norm_sq = _real_overlap(exact_action, exact_action)
        if exact_norm_sq == 0.0:
            probes.append(
                {
                    "name": name,
                    "status": "not_tested",
                    "action_norm": 0.0,
                    "relative_action_error": None,
                }
            )
            continue
        compressed_norm_sq = _real_overlap(compressed_action, compressed_action)
        cross_overlap = exact_action.overlap(compressed_action, ignore_form=True)
        difference_norm_sq = exact_norm_sq + compressed_norm_sq - 2.0 * float(
            np.real(cross_overlap)
        )
        roundoff_scale = max(exact_norm_sq, compressed_norm_sq, 1.0)
        if difference_norm_sq < 0.0 and abs(difference_norm_sq) <= (
            256.0 * np.finfo(np.float64).eps * roundoff_scale
        ):
            difference_norm_sq = 0.0
        if difference_norm_sq < 0.0:
            raise ValueError("MPS action-overlap contraction produced a negative squared norm.")
        probes.append(
            {
                "name": name,
                "status": "tested",
                "action_norm": float(np.sqrt(exact_norm_sq)),
                "relative_action_error": float(np.sqrt(difference_norm_sq / exact_norm_sq)),
            }
        )

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
        "tested_probes": len(tested_errors),
        "not_tested_probes": len(probes) - len(tested_errors),
        "max_relative_action_error": max(tested_errors) if tested_errors else None,
        "mean_relative_action_error": (
            float(np.mean(tested_errors)) if tested_errors else None
        ),
        "probes": probes,
    }


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


def _right_canonicalize_mpo_tensors(tensors: list[np.ndarray]) -> None:
    for site in range(len(tensors) - 1, 0, -1):
        tensor = tensors[site]
        left_dim, right_dim, physical_out, physical_in = tensor.shape
        matrix = tensor.transpose(0, 2, 3, 1).reshape(
            left_dim, physical_out * physical_in * right_dim
        )
        right_vectors, transfer = np.linalg.qr(matrix.T, mode="reduced")
        canonical_left_dim = right_vectors.shape[1]
        tensors[site] = right_vectors.T.reshape(
            canonical_left_dim, physical_out, physical_in, right_dim
        ).transpose(0, 3, 1, 2)
        previous = np.tensordot(tensors[site - 1], transfer.T, axes=(1, 0))
        tensors[site - 1] = previous.transpose(0, 3, 1, 2)


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
    scale = max(float(abs(overlap)), 1.0)
    if result < 0.0 and abs(result) <= 256.0 * np.finfo(np.float64).eps * scale:
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
