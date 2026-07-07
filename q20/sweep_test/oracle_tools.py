from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from utils import _commutator_pauli_labels_unchecked, pauli_weight


@dataclass(frozen=True)
class HamiltonianItem:
    label: str
    h0: complex
    h1: complex
    score: float


@dataclass(frozen=True)
class DoubleCommutatorPath:
    residual_index: int
    phase: complex
    left_h_index: int
    right_h_index: int


def ranked_hamiltonian_items(h0, h1, *, top_k: int) -> list[HamiltonianItem]:
    labels = sorted(set(h0.labels) | set(h1.labels))
    items = [
        HamiltonianItem(
            label=label,
            h0=complex(h0.coefficient(label)),
            h1=complex(h1.coefficient(label)),
            score=max(abs(h0.coefficient(label)), abs(h1.coefficient(label))),
        )
        for label in labels
    ]
    return sorted(items, key=lambda item: (item.score, -pauli_weight(item.label), item.label), reverse=True)[:top_k]


def double_commutator_paths(
    candidate_label: str,
    *,
    residual_index: dict[str, int],
    hamiltonian_items: list[HamiltonianItem],
) -> list[DoubleCommutatorPath]:
    """Return projected paths for -[[candidate, H], H].

    The minus sign matches the derivative of

    R(A) = [i dH/dlambda - [A, H], H]

    with respect to the candidate AGP coefficient.
    """

    paths: list[DoubleCommutatorPath] = []
    for left_idx, left_item in enumerate(hamiltonian_items):
        first = _commutator_pauli_labels_unchecked(candidate_label, left_item.label)
        if first is None:
            continue
        first_phase, intermediate_label = first
        for right_idx, right_item in enumerate(hamiltonian_items):
            second = _commutator_pauli_labels_unchecked(intermediate_label, right_item.label)
            if second is None:
                continue
            second_phase, residual_label = second
            out_idx = residual_index.get(residual_label)
            if out_idx is None:
                continue
            paths.append(
                DoubleCommutatorPath(
                    residual_index=out_idx,
                    phase=-complex(first_phase) * complex(second_phase),
                    left_h_index=left_idx,
                    right_h_index=right_idx,
                )
            )
    return paths


def _h_ad_coefficients(hamiltonian_items: list[HamiltonianItem], tau: float) -> np.ndarray:
    lam = np.sin(0.5 * np.pi * float(tau)) ** 2
    return np.asarray([item.h0 + lam * (item.h1 - item.h0) for item in hamiltonian_items], dtype=np.complex128)


def score_candidates_with_omp(
    candidates: list[dict[str, object]],
    *,
    residual_labels: list[str],
    residual_by_tau: np.ndarray,
    tau_values: np.ndarray,
    h0,
    h1,
    hamiltonian_top_k: int,
    max_candidates: int,
    min_score: float = 0.0,
) -> list[dict[str, object]]:
    """Rank candidate AGP strings by projected matching-pursuit score.

    Each candidate is scored by the normalized projection of the current
    residual onto its approximate double-commutator column on the supplied
    residual basis. This is a symbolic Pauli-coordinate diagnostic.
    """

    if not candidates:
        return []
    residual_index = {label: idx for idx, label in enumerate(residual_labels)}
    hamiltonian_items = ranked_hamiltonian_items(h0, h1, top_k=max(int(hamiltonian_top_k), 1))
    limited_candidates = candidates[: max(int(max_candidates), 1)]
    scored: list[dict[str, object]] = []
    h_cache = [_h_ad_coefficients(hamiltonian_items, float(tau)) for tau in tau_values]

    for candidate in limited_candidates:
        label = str(candidate["label"])
        paths = double_commutator_paths(
            label,
            residual_index=residual_index,
            hamiltonian_items=hamiltonian_items,
        )
        if not paths:
            enriched = dict(candidate)
            enriched.update(
                {
                    "omp_score": 0.0,
                    "omp_projection": 0.0,
                    "omp_column_norm": 0.0,
                    "omp_path_count": 0,
                    "score": float(candidate.get("score", 0.0)),
                    "proposal_score_rule": "inverse_commutator_fallback_no_projected_column",
                }
            )
            if enriched["score"] >= min_score:
                scored.append(enriched)
            continue

        score = 0.0
        projection_accumulator = 0.0
        norm_accumulator = 0.0
        for tau_idx, h_coefficients in enumerate(h_cache):
            sparse_column: defaultdict[int, complex] = defaultdict(complex)
            for path in paths:
                sparse_column[path.residual_index] += (
                    path.phase
                    * h_coefficients[path.left_h_index]
                    * h_coefficients[path.right_h_index]
                )
            if not sparse_column:
                continue
            residual_row = residual_by_tau[tau_idx]
            dot = 0.0j
            norm = 0.0
            for out_idx, value in sparse_column.items():
                dot += np.conjugate(value) * residual_row[out_idx]
                norm += float(abs(value) ** 2)
            if norm <= 0.0:
                continue
            projection = float(abs(dot) ** 2)
            score += projection / norm
            projection_accumulator += projection
            norm_accumulator += norm

        score /= max(len(h_cache), 1)
        if score < min_score:
            continue
        enriched = dict(candidate)
        enriched.update(
            {
                "omp_score": float(score),
                "omp_projection": float(projection_accumulator),
                "omp_column_norm": float(norm_accumulator),
                "omp_path_count": len(paths),
                "inverse_commutator_score": float(candidate.get("score", 0.0)),
                "score": float(score),
                "proposal_score_rule": "projected_matching_pursuit_double_commutator",
                "oracle_hamiltonian_terms": len(hamiltonian_items),
            }
        )
        scored.append(enriched)

    return sorted(
        scored,
        key=lambda row: (
            float(row.get("omp_score", 0.0)),
            float(row.get("inverse_commutator_score", 0.0)),
            -int(row.get("order", pauli_weight(str(row["label"])))),
            str(row["label"]),
        ),
        reverse=True,
    )


def projected_residual_matrix(model: torch.nn.Module, t: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        residual = model.euler_lagrange_residual(t)
    return residual.detach().cpu().numpy().astype(np.complex128, copy=False)


def projected_linear_oracle(
    agp_labels: Iterable[str],
    *,
    residual_labels: list[str],
    reference_residual_by_tau: np.ndarray,
    tau_values: np.ndarray,
    h0,
    h1,
    hamiltonian_top_k: int,
    max_agp_terms: int | None = None,
    ridge: float = 1e-10,
) -> dict[str, object]:
    """Approximate best projected residual achievable by a fixed AGP support.

    This solves an independent least-squares problem at each tau using symbolic
    double-commutator columns. It is a support-capacity diagnostic, not a PINN.
    """

    selected_agp_labels = list(agp_labels)
    if max_agp_terms is not None:
        selected_agp_labels = selected_agp_labels[: int(max_agp_terms)]
    residual_index = {label: idx for idx, label in enumerate(residual_labels)}
    hamiltonian_items = ranked_hamiltonian_items(h0, h1, top_k=max(int(hamiltonian_top_k), 1))
    paths_by_label = [
        double_commutator_paths(label, residual_index=residual_index, hamiltonian_items=hamiltonian_items)
        for label in selected_agp_labels
    ]
    reference_norm = float(np.mean(np.sum(np.abs(reference_residual_by_tau) ** 2, axis=1).real))
    residual_norms: list[float] = []
    ranks: list[int] = []
    for tau_idx, tau in enumerate(tau_values):
        h_coefficients = _h_ad_coefficients(hamiltonian_items, float(tau))
        matrix = np.zeros((len(residual_labels), len(selected_agp_labels)), dtype=np.complex128)
        for col_idx, paths in enumerate(paths_by_label):
            for path in paths:
                matrix[path.residual_index, col_idx] += (
                    path.phase
                    * h_coefficients[path.left_h_index]
                    * h_coefficients[path.right_h_index]
                )
        target = reference_residual_by_tau[tau_idx]
        if matrix.size == 0 or not np.any(matrix):
            residual_norms.append(float(np.sum(np.abs(target) ** 2).real))
            ranks.append(0)
            continue
        if ridge > 0.0:
            gram = matrix.conj().T @ matrix
            rhs = matrix.conj().T @ target
            coeffs = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), rhs)
        else:
            coeffs, *_ = np.linalg.lstsq(matrix, target, rcond=None)
        remaining = target - matrix @ coeffs
        residual_norms.append(float(np.sum(np.abs(remaining) ** 2).real))
        ranks.append(int(np.linalg.matrix_rank(matrix)))
    oracle_residual = float(np.mean(residual_norms))
    return {
        "oracle_residual": oracle_residual,
        "reference_residual": reference_norm,
        "oracle_relative_residual": oracle_residual / max(reference_norm, np.finfo(float).eps),
        "agp_terms_used": len(selected_agp_labels),
        "residual_terms": len(residual_labels),
        "hamiltonian_terms": len(hamiltonian_items),
        "tau_points": len(tau_values),
        "mean_column_rank": float(np.mean(ranks)) if ranks else 0.0,
        "max_column_rank": int(max(ranks)) if ranks else 0,
        "ridge": float(ridge),
        "caveat": (
            "Projected symbolic least-squares oracle on a truncated Hamiltonian-term "
            "set. Passing this diagnostic is necessary evidence of support capacity, "
            "not a full-basis proof."
        ),
    }
