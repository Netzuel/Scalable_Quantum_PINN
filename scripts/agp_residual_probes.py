"""Pure helpers for stable unseen AGP residual probes."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class FixedUnseenProbeConfig:
    enabled: bool = False
    active_terms: int = 0
    null_terms: int = 0
    reference_rms_threshold: float = 1.0e-12
    seed: int = 0
    candidate_multiplier: int = 4


def select_fixed_unseen_probes(
    labels: Sequence[str],
    reference_rms: np.ndarray,
    *,
    excluded_labels: Collection[str],
    config: FixedUnseenProbeConfig,
) -> dict[str, object]:
    """Select deterministic active and null probes from an ordered candidate pool."""

    if len(labels) != len(reference_rms):
        raise ValueError("labels and reference_rms must have equal length")

    excluded = set(excluded_labels)
    rows = [
        (str(label), float(value))
        for label, value in zip(labels, reference_rms, strict=True)
        if label not in excluded
    ]
    active = sorted(
        (row for row in rows if row[1] > config.reference_rms_threshold),
        key=lambda row: (-row[1], row[0]),
    )[: config.active_terms]
    null = sorted(
        (row for row in rows if row[1] <= config.reference_rms_threshold),
        key=lambda row: (row[1], row[0]),
    )[: config.null_terms]

    return {
        "active_labels": [label for label, _ in active],
        "null_labels": [label for label, _ in null],
        "active_reference_rms": [value for _, value in active],
        "null_reference_rms": [value for _, value in null],
        "requested_active_terms": config.active_terms,
        "requested_null_terms": config.null_terms,
        "status": (
            "complete"
            if len(active) == config.active_terms and len(null) == config.null_terms
            else "insufficient_candidates"
        ),
    }


def norm_sq_subset(values: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    """Return mean squared norm across the selected final-axis terms."""

    if not indices:
        return torch.zeros((), dtype=values.real.dtype, device=values.device)
    index = torch.tensor(list(indices), dtype=torch.long, device=values.device)
    subset = values.index_select(-1, index)
    return torch.mean(torch.sum(torch.abs(subset) ** 2, dim=-1).real)


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _relative_metric_status(
    *,
    residual: torch.Tensor,
    reference: torch.Tensor,
    term_count: int,
    reference_floor: float,
) -> tuple[float | None, dict[str, object]]:
    residual_value = _scalar(residual)
    reference_value = _scalar(reference)
    if term_count == 0:
        return None, {
            "valid": False,
            "reason": "empty_subset",
            "residual": residual_value,
            "reference_residual": reference_value,
            "reference_floor": float(reference_floor),
        }
    if reference_value <= float(reference_floor):
        return None, {
            "valid": False,
            "reason": "zero_reference",
            "residual": residual_value,
            "reference_residual": reference_value,
            "reference_floor": float(reference_floor),
        }
    return residual_value / reference_value, {
        "valid": True,
        "reason": "finite_reference",
        "residual": residual_value,
        "reference_residual": reference_value,
        "reference_floor": float(reference_floor),
    }


def fixed_unseen_metrics(
    *,
    residual: torch.Tensor,
    reference: torch.Tensor,
    active_indices: Sequence[int],
    null_indices: Sequence[int],
    reference_floor: float,
) -> dict[str, object]:
    """Compute active relative and null leakage metrics for fixed probes."""

    active_term_count = len(active_indices)
    null_term_count = len(null_indices)
    active_residual = norm_sq_subset(residual, active_indices)
    active_reference = norm_sq_subset(reference, active_indices)
    active_relative, active_status = _relative_metric_status(
        residual=active_residual,
        reference=active_reference,
        term_count=active_term_count,
        reference_floor=reference_floor,
    )

    null_residual = norm_sq_subset(residual, null_indices)
    null_absolute_per_term = (
        _scalar(null_residual) / null_term_count if null_term_count else None
    )
    active_reference_per_term = (
        _scalar(active_reference) / active_term_count
        if active_term_count and _scalar(active_reference) > float(reference_floor)
        else None
    )
    null_scaled = (
        null_absolute_per_term / active_reference_per_term
        if null_absolute_per_term is not None and active_reference_per_term is not None
        else None
    )

    return {
        "active_terms": active_term_count,
        "active_residual": _scalar(active_residual),
        "active_reference_residual": _scalar(active_reference),
        "active_relative": active_relative,
        "active_status": active_status,
        "null_terms": null_term_count,
        "null_residual": _scalar(null_residual),
        "null_absolute_per_term": null_absolute_per_term,
        "null_scaled": null_scaled,
    }
