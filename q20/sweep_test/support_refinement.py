from __future__ import annotations

from typing import Iterable

from utils import pauli_weight, sort_pauli_labels


def resolve_active_agp_budget(
    *,
    n_qubits: int,
    cap_qubits: int = 7,
    requested: object = "auto",
) -> int:
    """Resolve the trainable AGP output budget for scalable q experiments."""

    if requested is not None and str(requested).lower() != "auto":
        value = int(requested)
        if value <= 0:
            raise ValueError("The active AGP budget must be positive.")
        return value
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive.")
    if cap_qubits <= 0:
        raise ValueError("cap_qubits must be positive.")
    return 4 ** min(int(n_qubits), int(cap_qubits))


def resolve_exploratory_agp_budget(
    *,
    n_qubits: int,
    active_cap_qubits: int = 7,
    exploratory_cap_qubits: int = 8,
    requested: object = "auto",
) -> int:
    """Resolve the generated candidate-pool budget without making it trainable."""

    active_budget = resolve_active_agp_budget(n_qubits=n_qubits, cap_qubits=active_cap_qubits)
    if requested is not None and str(requested).lower() != "auto":
        value = int(requested)
        if value <= 0:
            raise ValueError("The exploratory AGP budget must be positive.")
        return max(active_budget, value)
    if exploratory_cap_qubits <= 0:
        raise ValueError("exploratory_cap_qubits must be positive.")
    exploratory_budget = 4 ** min(int(n_qubits), int(exploratory_cap_qubits))
    return max(active_budget, exploratory_budget)


def importance_by_label(active_importance_terms: Iterable[dict[str, object]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for row in active_importance_terms:
        label = str(row["label"])
        value = row.get("importance", row.get("rms", 0.0))
        scores[label] = float(value)
    return scores


def fixed_budget_swap_labels(
    *,
    current_agp_labels: Iterable[str],
    candidate_additions: Iterable[dict[str, object]],
    active_importance_terms: Iterable[dict[str, object]],
    swap_terms: int,
    protected_fraction: float = 0.0,
) -> dict[str, object]:
    """Replace weak active AGP terms with stronger outside candidates.

    This preserves a fixed output budget. Active terms are ranked by the learned
    direct counterdiabatic coefficient importance exported by the previous run.
    Missing importance is treated as zero, so newly unmeasured labels are not
    accidentally protected.
    """

    current_labels = sort_pauli_labels(set(str(label) for label in current_agp_labels))
    current_set = set(current_labels)
    requested_swaps = max(int(swap_terms), 0)
    if requested_swaps == 0:
        return {
            "agp_labels": current_labels,
            "added_agp_terms": [],
            "removed_agp_terms": [],
            "active_budget": len(current_labels),
            "requested_swap_terms": requested_swaps,
        }

    additions: list[dict[str, object]] = []
    seen_additions: set[str] = set()
    for candidate in candidate_additions:
        label = str(candidate["label"])
        if label in current_set or label in seen_additions:
            continue
        enriched = dict(candidate)
        enriched["label"] = label
        enriched["order"] = int(enriched.get("order", pauli_weight(label)))
        additions.append(enriched)
        seen_additions.add(label)
        if len(additions) >= requested_swaps:
            break

    if not additions:
        return {
            "agp_labels": current_labels,
            "added_agp_terms": [],
            "removed_agp_terms": [],
            "active_budget": len(current_labels),
            "requested_swap_terms": requested_swaps,
        }

    importance = importance_by_label(active_importance_terms)
    protected_count = int(round(len(current_labels) * min(max(float(protected_fraction), 0.0), 1.0)))
    protected = set(
        sorted(
            current_labels,
            key=lambda label: (importance.get(label, 0.0), -pauli_weight(label), label),
            reverse=True,
        )[:protected_count]
    )
    removable = [
        label
        for label in sorted(
            current_labels,
            key=lambda label: (importance.get(label, 0.0), -pauli_weight(label), label),
        )
        if label not in protected
    ]
    additions = additions[: len(removable)]
    removed = removable[: len(additions)]
    removed_payload = [
        {
            "label": label,
            "importance": float(importance.get(label, 0.0)),
            "order": pauli_weight(label),
        }
        for label in removed
    ]
    next_labels = (current_set - set(removed)) | {str(row["label"]) for row in additions}

    return {
        "agp_labels": sort_pauli_labels(next_labels),
        "added_agp_terms": additions,
        "removed_agp_terms": removed_payload,
        "active_budget": len(current_labels),
        "requested_swap_terms": requested_swaps,
        "protected_active_terms": protected_count,
    }
