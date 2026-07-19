"""Deterministic locality and spatial coverage for ranked AGP candidates."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class StratifiedSelection:
    selected_rows: tuple[dict[str, object], ...]
    provenance: dict[str, object]


def _candidate_score(row: Mapping[str, object]) -> float:
    for key in ("score", "importance", "residual_rms", "rms"):
        if key in row:
            score = float(row[key])
            if not math.isfinite(score):
                raise ValueError(f"Candidate score {key} must be finite.")
            return score
    return 0.0


def _seeded_tie_key(label: str, seed: int) -> str:
    return hashlib.sha256(f"{int(seed)}:{label}".encode("ascii")).hexdigest()


def _locality_matcher(raw_key: object, *, q: int):
    key = str(raw_key).strip()
    if key.endswith("+"):
        lower = int(key[:-1])
        if lower <= 0:
            raise ValueError("Locality lower bounds must be positive.")
        return key, lambda order: order >= lower
    if "-" in key:
        lower_text, upper_text = key.split("-", maxsplit=1)
        lower, upper = int(lower_text), int(upper_text)
        if lower <= 0 or upper < lower or upper > q:
            raise ValueError(f"Invalid locality range {key!r} for q={q}.")
        return key, lambda order: lower <= order <= upper
    order = int(key)
    if order <= 0 or order > q:
        raise ValueError(f"Invalid Pauli locality {key!r} for q={q}.")
    return key, lambda candidate_order: candidate_order == order


def _normalize_quotas(
    locality_quotas: Mapping[object, object] | None,
    *,
    q: int,
) -> tuple[list[str], dict[str, int], dict[str, object]]:
    keys: list[str] = []
    quotas: dict[str, int] = {}
    matchers: dict[str, object] = {}
    for raw_key, raw_quota in (locality_quotas or {}).items():
        key, matcher = _locality_matcher(raw_key, q=q)
        quota = int(raw_quota)
        if isinstance(raw_quota, bool) or quota < 0 or quota != raw_quota:
            raise ValueError(f"Locality quota {key!r} must be a non-negative integer.")
        keys.append(key)
        quotas[key] = quota
        matchers[key] = matcher

    for order in range(1, q + 1):
        matching = [key for key in keys if matchers[key](order)]
        if len(matching) > 1:
            raise ValueError(
                f"Pauli locality {order} matches overlapping quota strata {matching}."
            )
    return keys, quotas, matchers


def _spatial_bin(label: str, bins: int) -> int:
    active = [index for index, symbol in enumerate(label) if symbol != "I"]
    center = sum(active) / len(active)
    return min(bins - 1, int(center * bins / len(label)))


def stratified_ranked_selection(
    candidates: Sequence[Mapping[str, object]],
    budget: int,
    *,
    q: int,
    locality_quotas: Mapping[object, object] | None,
    spatial_bins: int,
    seed: int,
) -> StratifiedSelection:
    """Reserve coverage quotas, then fill by the unchanged global score rank."""

    q = int(q)
    budget = int(budget)
    if q <= 0:
        raise ValueError("q must be positive.")
    if budget < 0:
        raise ValueError("budget must be non-negative.")
    bins = min(max(1, int(spatial_bins)), q)
    locality_keys, quotas, matchers = _normalize_quotas(locality_quotas, q=q)

    unique: dict[str, dict[str, object]] = {}
    for raw_row in candidates:
        row = dict(raw_row)
        label = str(row.get("label", ""))
        if len(label) != q or any(symbol not in "IXYZ" for symbol in label):
            raise ValueError(f"Invalid q={q} Pauli candidate label {label!r}.")
        if set(label) == {"I"}:
            continue
        row["label"] = label
        row["score"] = _candidate_score(row)
        previous = unique.get(label)
        if previous is None or float(row["score"]) > float(previous["score"]):
            unique[label] = row

    ranked = sorted(
        unique.values(),
        key=lambda row: (
            -float(row["score"]),
            _seeded_tie_key(str(row["label"]), seed),
            str(row["label"]),
        ),
    )
    rank = {str(row["label"]): index for index, row in enumerate(ranked)}

    locality_for_label: dict[str, str | None] = {}
    available_counts = {key: 0 for key in locality_keys}
    strata: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in ranked:
        label = str(row["label"])
        order = sum(symbol != "I" for symbol in label)
        matching = [key for key in locality_keys if matchers[key](order)]
        locality = matching[0] if matching else None
        locality_for_label[label] = locality
        if locality is None:
            continue
        available_counts[locality] += 1
        bin_index = _spatial_bin(label, bins)
        strata.setdefault((locality, bin_index), []).append(row)

    quota_sequences: dict[str, list[dict[str, object]]] = {}
    for locality in locality_keys:
        queues = {
            bin_index: list(strata[(locality, bin_index)])
            for bin_index in range(bins)
            if (locality, bin_index) in strata
        }
        sequence: list[dict[str, object]] = []
        while queues and len(sequence) < quotas[locality]:
            ordered_bins = sorted(
                queues,
                key=lambda bin_index: rank[str(queues[bin_index][0]["label"])],
            )
            for bin_index in ordered_bins:
                sequence.append(queues[bin_index].pop(0))
                if not queues[bin_index]:
                    del queues[bin_index]
                if len(sequence) >= quotas[locality]:
                    break
        quota_sequences[locality] = sequence

    selected: list[dict[str, object]] = []
    selected_labels: set[str] = set()
    quota_offsets = {key: 0 for key in locality_keys}
    while len(selected) < budget:
        progress = False
        for locality in locality_keys:
            offset = quota_offsets[locality]
            sequence = quota_sequences[locality]
            if offset >= len(sequence):
                continue
            row = sequence[offset]
            quota_offsets[locality] += 1
            label = str(row["label"])
            if label not in selected_labels:
                selected.append(row)
                selected_labels.add(label)
            progress = True
            if len(selected) >= budget:
                break
        if not progress:
            break

    for row in ranked:
        if len(selected) >= budget:
            break
        label = str(row["label"])
        if label in selected_labels:
            continue
        selected.append(row)
        selected_labels.add(label)

    realized_counts = {key: 0 for key in locality_keys}
    realized_spatial_counts: dict[str, int] = {}
    for row in selected:
        label = str(row["label"])
        locality = locality_for_label[label]
        if locality is None:
            continue
        realized_counts[locality] += 1
        stratum = f"{locality}:{_spatial_bin(label, bins)}"
        realized_spatial_counts[stratum] = realized_spatial_counts.get(stratum, 0) + 1

    provenance = {
        "enabled": True,
        "requested_terms": budget,
        "selected_terms": len(selected),
        "candidate_terms": len(ranked),
        "spatial_bins": bins,
        "seed": int(seed),
        "requested_locality_quotas": quotas,
        "available_locality_counts": available_counts,
        "realized_locality_counts": realized_counts,
        "realized_spatial_counts": realized_spatial_counts,
        "fill_rule": "locality-spatial minimum quotas, then global importance ranking",
    }
    return StratifiedSelection(tuple(selected), provenance)
