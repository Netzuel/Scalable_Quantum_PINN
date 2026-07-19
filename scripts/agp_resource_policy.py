"""Deterministic q-aware resource budgets for sparse AGP experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ResourceBudget:
    name: str
    mode: str
    q: int
    capacity: int
    requested: int
    realized: int
    minimum: int
    maximum: int | None
    per_qubit: float | None
    clipping_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer, not a Boolean.")
    result = int(value)
    if result < 0 or result != value:
        raise ValueError(f"{field} must be a non-negative integer.")
    return result


def _half_up_nonnegative(value: float) -> int:
    return int(math.floor(value + 0.5))


def resolve_resource_budget(
    spec: int | Mapping[str, object],
    *,
    q: int,
    capacity: int,
    name: str,
) -> ResourceBudget:
    """Resolve a fixed or per-qubit count and retain all clipping provenance."""

    q = int(q)
    capacity = int(capacity)
    if q <= 0:
        raise ValueError("q must be positive.")
    if capacity < 0:
        raise ValueError("capacity must be non-negative.")

    per_qubit: float | None = None
    if isinstance(spec, Mapping):
        mode = str(spec.get("mode", "per_qubit")).strip().lower()
        if mode == "per_qubit":
            per_qubit = float(spec.get("per_qubit", float("nan")))
            if not math.isfinite(per_qubit) or per_qubit < 0.0:
                raise ValueError("per_qubit must be finite and non-negative.")
            requested = _half_up_nonnegative(per_qubit * q)
        elif mode == "fixed":
            requested = _nonnegative_integer(spec.get("value"), field="value")
        else:
            raise ValueError(f"Unsupported resource budget mode {mode!r}.")
        minimum = _nonnegative_integer(spec.get("minimum", 0), field="minimum")
        maximum_raw = spec.get("maximum")
        maximum = (
            None
            if maximum_raw is None
            else _nonnegative_integer(maximum_raw, field="maximum")
        )
    elif isinstance(spec, int) and not isinstance(spec, bool):
        mode = "fixed"
        requested = _nonnegative_integer(spec, field=name)
        minimum = 0
        maximum = None
    else:
        raise TypeError(f"{name} must be an integer or a resource-policy mapping.")

    if maximum is not None and minimum > maximum:
        raise ValueError("minimum cannot exceed maximum.")

    realized = requested
    clipping_reasons: list[str] = []
    if realized < minimum:
        realized = minimum
        clipping_reasons.append("minimum")
    if maximum is not None and realized > maximum:
        realized = maximum
        clipping_reasons.append("maximum")
    if realized > capacity:
        realized = capacity
        clipping_reasons.append("capacity")

    return ResourceBudget(
        name=str(name),
        mode=mode,
        q=q,
        capacity=capacity,
        requested=requested,
        realized=realized,
        minimum=minimum,
        maximum=maximum,
        per_qubit=per_qubit,
        clipping_reasons=tuple(clipping_reasons),
    )
