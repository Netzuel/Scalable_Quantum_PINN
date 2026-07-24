#!/usr/bin/env python3
"""Tensor-network dynamical validation for sparse AGP protocols."""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import SparsePauliOperator, load_pauli_hamiltonian_pair  # noqa: E402

from agp_physical_validation import (  # noqa: E402
    learned_term_selection,
    interpolate_coefficients,
    learned_schedule,
    refresh_hcd_connection_summary,
    schedule_sin2,
    subset_learned_terms,
    variational_l1_agp,
)
from agp_validation_identity import (  # noqa: E402
    VALIDATION_IDENTITY_KEYS,
    canonical_hash as _shared_canonical_hash,
    checkpoint_identity as _shared_checkpoint_identity,
    ground_reference_identity as _shared_ground_reference_identity,
    hamiltonian_identity as _shared_hamiltonian_identity,
    schedule_identity as _shared_schedule_identity,
    schedule_parameters_identity,
    validation_identity_from_settings,
)
from scripts.agp_plot_annotations import plot_physical_comparison_table  # noqa: E402

try:
    import quimb.tensor as qtn
except ImportError as exc:  # pragma: no cover - exercised by the CLI error path
    raise ImportError(
        "MPS validation requires the optional dependency: "
        "pip install '.[tensor-network]'"
    ) from exc


_PAULI = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}
_CANONICAL_MPO_PROTOCOLS = ("no_cd", "nested_l1", "learned_sparse_agp")
_MPO_PROTOCOL_ALIASES = {"kipu_dqfm_l1": "nested_l1", "nested_l1": "nested_l1"}
_MPO_OPERATOR_IMPLEMENTATION_VERSION = "positioned-adaptive-joint-time-pauli-tt-v5"
_ELIGIBLE_MPO_IDENTITY_KEYS = (
    "backend",
    "integrator",
    "n_qubits",
    "learned_terms",
    "full_learned_terms",
    "learned_scale",
    "hamiltonian_identity",
    "ground_reference_identity",
    "ground_bitstring",
    "schedule_identity",
    "checkpoint_identity",
    "coefficient_identity",
    "total_time",
    "initial_state",
)
_STATEVECTOR_REFERENCE_IDENTITY_KEYS = VALIDATION_IDENTITY_KEYS


def resolve_validation_backend(validation: Mapping[str, object]) -> dict[str, object]:
    """Return the explicit tensor-network backend configuration.

    The product-formula evaluator remains the backwards-compatible default.
    TeNPy is selected only by an explicit ``mpo_backend.name`` setting.
    """

    raw = validation.get("mpo_backend", {})
    raw = raw if isinstance(raw, Mapping) else {}
    name = str(raw.get("name", "quimb_product_formula"))
    if name not in {"quimb_product_formula", "tenpy_tdvp_mpo"}:
        raise ValueError("mpo_backend.name must be 'quimb_product_formula' or 'tenpy_tdvp_mpo'.")
    candidates = raw.get("qubit_order_candidates", ("native", "spectral"))
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("mpo_backend.qubit_order_candidates must be a sequence of names.")
    resource_caps = raw.get("resource_caps", {})
    resource_caps = resource_caps if isinstance(resource_caps, Mapping) else {}
    integrator = str(raw.get("integrator", "tdvp"))
    if integrator not in {"tdvp", "expm_mpo"}:
        raise ValueError("mpo_backend.integrator must be 'tdvp' or 'expm_mpo'.")
    representation = str(raw.get("representation", "temporal_mode_block_sum"))
    if representation not in {
        "temporal_mode_block_sum",
        "direct_time_full_support",
        "joint_time_full_support",
    }:
        raise ValueError(
            "mpo_backend.representation must be 'temporal_mode_block_sum', "
            "'direct_time_full_support', or 'joint_time_full_support'."
        )
    raw_window_size = raw.get("time_window_size")
    if raw_window_size is not None and (
        isinstance(raw_window_size, bool) or int(raw_window_size) < 1
    ):
        raise ValueError("mpo_backend.time_window_size must be a positive integer or null.")
    raw_time_axis_position = raw.get("time_axis_position", 0)
    if (
        isinstance(raw_time_axis_position, bool)
        or not isinstance(raw_time_axis_position, (int, np.integer))
        or int(raw_time_axis_position) < 0
    ):
        raise ValueError("mpo_backend.time_axis_position must be a nonnegative integer.")
    return {
        "name": name,
        "integrator": integrator,
        "representation": representation,
        "qubit_order_candidates": tuple(str(item) for item in candidates),
        "temporal_grid_points": int(raw.get("temporal_grid_points", 257)),
        "temporal_retained_norm": float(raw.get("temporal_retained_norm", 0.9999)),
        "action_probe_seed": int(raw.get("action_probe_seed", 11)),
        "action_probe_product_states": int(raw.get("action_probe_product_states", 4)),
        "action_probe_random_mps": int(raw.get("action_probe_random_mps", 2)),
        "action_probe_exact_work_cap": int(raw.get("action_probe_exact_work_cap", 10_000_000)),
        "action_probe_dynamic_samples": int(raw.get("action_probe_dynamic_samples", 3)),
        "coefficient_error_max": float(raw.get("coefficient_error_max", 1.0e-3)),
        "action_error_max": float(raw.get("action_error_max", 1.0e-3)),
        "time_window_size": (
            None if raw_window_size is None else int(raw_window_size)
        ),
        "adaptive_time_windows": bool(raw.get("adaptive_time_windows", True)),
        "time_axis_position": int(raw_time_axis_position),
        "resource_caps": dict(resource_caps),
        "mpo_workspace_cap_bytes": int(
            raw.get(
                "mpo_workspace_cap_bytes",
                float(resource_caps.get("max_peak_memory_gb", 24.0)) * (1024**3),
            )
        ),
        "lanczos_max": int(raw.get("lanczos_max", 20)),
        "ablation": bool(raw.get("ablation", False)),
    }


def require_full_learned_support(*, selected_terms: int, available_terms: int, ablation: bool) -> str:
    """Classify a learned deployment without silently certifying a truncation."""

    if int(selected_terms) == int(available_terms):
        return "full_support"
    if ablation:
        return "ablation"
    raise ValueError(
        "tenpy_tdvp_mpo certifiable learned validation requires full learned support; "
        "mark reduced support as ablation explicitly."
    )


def resolve_case_ablation(
    *, cli_ablation: bool, case_ablation: object, backend_ablation: object
) -> bool:
    """Make a command-line reduced-support calibration explicitly noncanonical."""

    return bool(cli_ablation or case_ablation or backend_ablation)


def _checkpoint_identity(path: Path) -> dict[str, object]:
    return _shared_checkpoint_identity(path)


def _canonical_hash(value: object) -> str:
    return _shared_canonical_hash(value)


def _hamiltonian_identity(h0: SparsePauliOperator, h1: SparsePauliOperator) -> str:
    return _shared_hamiltonian_identity(h0, h1)


def _ground_reference_identity(
    validation: Mapping[str, object], *, ground_energy: float, ground_bitstring: str
) -> str:
    reference_path = validation.get("exact_final_ground_reference")
    return _shared_ground_reference_identity(
        ground_energy=ground_energy,
        ground_bitstring=ground_bitstring,
        reference_path=_resolve_path(reference_path, base=ROOT) if reference_path else None,
    )


def _learned_schedule_identity(learned: Mapping[str, object], *, learned_scale: float) -> str:
    return _shared_schedule_identity(learned, learned_scale=learned_scale)


def _kron_paulis(label: str) -> np.ndarray:
    operator = np.array([[1.0]], dtype=np.complex128)
    for symbol in label:
        try:
            operator = np.kron(operator, _PAULI[symbol])
        except KeyError as exc:
            raise ValueError(f"Unsupported Pauli symbol {symbol!r} in {label!r}.") from exc
    return operator


def pauli_rotation_matrix(label: str, angle: float) -> np.ndarray:
    """Return ``exp(-i * angle * P)`` for a Pauli-product label."""

    pauli = _kron_paulis(label)
    return np.cos(angle) * np.eye(pauli.shape[0]) - 1.0j * np.sin(angle) * pauli


def make_product_mps(bitstring: str):
    """Construct an open-boundary computational-basis MPS."""

    if not bitstring or any(bit not in "01" for bit in bitstring):
        raise ValueError("bitstring must be a non-empty string containing only 0 and 1.")
    return qtn.MPS_computational_state(bitstring, dtype="complex128")


def make_plus_mps(n_qubits: int):
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive.")
    plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    return qtn.MPS_product_state([plus.copy() for _ in range(n_qubits)])


def apply_pauli_rotation_mps(
    state,
    label: str,
    angle: float,
    *,
    cutoff: float,
    max_bond: int,
) -> None:
    """Apply a local Pauli-product exponential while preserving MPS form."""

    occupied = [site for site, symbol in enumerate(label) if symbol != "I"]
    if not occupied or angle == 0.0:
        return
    first, last = occupied[0], occupied[-1]
    local_label = label[first : last + 1]
    where = tuple(range(first, last + 1))
    if len(where) == 1:
        state.gate_(pauli_rotation_matrix(local_label, float(angle)), where=where[0], contract=True)
        return

    identity_mpo = qtn.MPO_product_operator(
        [_PAULI["I"] for _ in local_label],
        sites=where,
        L=state.L,
    )
    pauli_mpo = qtn.MPO_product_operator(
        [_PAULI[symbol] for symbol in local_label],
        sites=where,
        L=state.L,
    )
    gate_mpo = np.cos(angle) * identity_mpo - 1.0j * np.sin(angle) * pauli_mpo
    state.gate_with_submpo_(
        gate_mpo,
        where=where,
        method="direct",
        cutoff=cutoff,
        max_bond=max_bond,
    )


@functools.lru_cache(maxsize=None)
def _local_pauli_matrix(symbols: str) -> np.ndarray:
    return _kron_paulis(symbols)


def group_hamiltonian_terms_by_support(
    terms: list[tuple[str, float]],
) -> dict[tuple[int, ...], np.ndarray]:
    """Combine all Pauli coefficients acting on the same occupied sites."""

    grouped: dict[tuple[int, ...], np.ndarray] = {}
    for label, coefficient in terms:
        support = tuple(site for site, symbol in enumerate(label) if symbol != "I")
        if not support or coefficient == 0.0:
            continue
        symbols = "".join(label[site] for site in support)
        contribution = float(coefficient) * _local_pauli_matrix(symbols)
        if support in grouped:
            grouped[support] += contribution
        else:
            grouped[support] = contribution.copy()
    return grouped


def apply_grouped_hamiltonian_rotation_mps(
    state,
    terms: list[tuple[str, float]],
    angle: float,
    *,
    cutoff: float,
    max_bond: int,
) -> dict[str, int]:
    """Apply grouped local exponentials while retaining every Pauli term."""

    grouped_terms: defaultdict[tuple[int, ...], list[tuple[str, float]]] = defaultdict(list)
    for label, coefficient in terms:
        support = tuple(site for site, symbol in enumerate(label) if symbol != "I")
        if support and coefficient != 0.0:
            grouped_terms[support].append((label, coefficient))

    applied_groups = 0
    for support, support_terms in grouped_terms.items():
        if len(support_terms) == 1:
            label, coefficient = support_terms[0]
            apply_pauli_rotation_mps(
                state,
                label,
                float(angle) * float(coefficient),
                cutoff=cutoff,
                max_bond=max_bond,
            )
            applied_groups += 1
            continue

        hamiltonian = group_hamiltonian_terms_by_support(support_terms)[support]
        if not np.allclose(hamiltonian, hamiltonian.conj().T, atol=1.0e-10):
            raise ValueError(f"Non-Hermitian grouped Hamiltonian on support {support}.")
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        gate = (eigenvectors * np.exp(-1.0j * float(angle) * eigenvalues)) @ eigenvectors.conj().T
        if len(support) == 1:
            state.gate_(gate, where=support[0], contract=True)
        else:
            state.gate_nonlocal_(
                gate,
                where=support,
                method="direct",
                cutoff=cutoff,
                max_bond=max_bond,
            )
        applied_groups += 1
    return {"pauli_terms": len(terms), "support_groups": applied_groups}


def _combined_hamiltonian_terms(
    *,
    protocol: str,
    h0: SparsePauliOperator,
    h1: SparsePauliOperator,
    learned: Mapping[str, object] | None,
    t: float,
    total_time: float,
    learned_scale: float,
    coefficient_threshold: float,
) -> list[tuple[str, float]]:
    if protocol == "learned_sparse_agp" and learned is not None:
        lam, _ = learned_schedule(learned, t, total_time)
    else:
        lam, dlam_dt = schedule_sin2(t, total_time)

    terms: defaultdict[str, complex] = defaultdict(complex)
    for label, coefficient in h0.terms.items():
        terms[label] += (1.0 - lam) * coefficient
    for label, coefficient in h1.terms.items():
        terms[label] += lam * coefficient

    if protocol in {"nested_l1", "kipu_dqfm_l1"}:
        agp = variational_l1_agp(h0, h1, lam)
        for label, coefficient in agp.terms.items():
            terms[label] += dlam_dt * coefficient
    elif protocol == "learned_sparse_agp":
        if learned is None:
            raise ValueError("learned payload is required for learned_sparse_agp.")
        tau = t / total_time
        coefficients = interpolate_coefficients(
            np.asarray(learned["tau"], dtype=np.float64),
            np.asarray(learned["coefficients"], dtype=np.float64),
            tau,
        )
        for label, coefficient in zip(learned["labels"], coefficients, strict=True):
            terms[str(label)] += learned_scale * float(coefficient)
    elif protocol != "no_cd":
        raise ValueError(f"Unsupported protocol: {protocol!r}.")

    real_terms: list[tuple[str, float]] = []
    for label, coefficient in terms.items():
        if abs(coefficient.imag) > 1.0e-10:
            raise ValueError(f"Non-Hermitian coefficient for {label}: {coefficient}.")
        value = float(coefficient.real)
        if abs(value) > coefficient_threshold:
            real_terms.append((label, value))
    return real_terms


def evolve_protocol_mps(
    *,
    protocol: str,
    h0: SparsePauliOperator,
    h1: SparsePauliOperator,
    learned: Mapping[str, object] | None,
    total_time: float,
    steps: int,
    cutoff: float,
    max_bond: int,
    learned_scale: float = 1.0,
    coefficient_threshold: float = 0.0,
    operator_grouping: str = "pauli_term",
    progress: bool = False,
):
    """Evolve ``|+>`` with a symmetric Pauli-product formula and MPS compression."""

    if h0.n_qubits != h1.n_qubits:
        raise ValueError("h0 and h1 must act on the same number of qubits.")
    if steps < 1 or total_time <= 0.0:
        raise ValueError("steps and total_time must be positive.")
    if operator_grouping not in {"pauli_term", "support"}:
        raise ValueError("operator_grouping must be 'pauli_term' or 'support'.")
    state = make_plus_mps(h0.n_qubits)
    dt = float(total_time) / int(steps)
    gate_count = 0
    pauli_term_applications = 0
    peak_bond = int(state.max_bond() or 1)

    for step in range(int(steps)):
        midpoint = (step + 0.5) * dt
        terms = _combined_hamiltonian_terms(
            protocol=protocol,
            h0=h0,
            h1=h1,
            learned=learned,
            t=midpoint,
            total_time=total_time,
            learned_scale=learned_scale,
            coefficient_threshold=coefficient_threshold,
        )
        for sequence in (terms, list(reversed(terms))):
            if operator_grouping == "support":
                counts = apply_grouped_hamiltonian_rotation_mps(
                    state,
                    sequence,
                    0.5 * dt,
                    cutoff=cutoff,
                    max_bond=max_bond,
                )
                gate_count += counts["support_groups"]
                pauli_term_applications += counts["pauli_terms"]
            else:
                for label, coefficient in sequence:
                    apply_pauli_rotation_mps(
                        state,
                        label,
                        0.5 * dt * coefficient,
                        cutoff=cutoff,
                        max_bond=max_bond,
                    )
                    gate_count += 1
                    pauli_term_applications += 1
        state.normalize()
        current_bond = int(state.max_bond() or 1)
        peak_bond = max(peak_bond, current_bond)
        if progress:
            print(
                f"mps_protocol={protocol} step={step + 1}/{steps} "
                f"terms={len(terms)} grouping={operator_grouping} max_bond={current_bond}",
                flush=True,
            )

    diagnostics = {
        "gate_count": gate_count,
        "pauli_term_applications": pauli_term_applications,
        "operator_grouping": operator_grouping,
        "peak_bond": peak_bond,
        "final_bond": int(state.max_bond() or 1),
        "steps": int(steps),
        "cutoff": float(cutoff),
        "max_bond": int(max_bond),
        "coefficient_threshold": float(coefficient_threshold),
    }
    return state, diagnostics


def diagonal_pauli_mps_metrics(
    state,
    final_terms: Mapping[str, float],
    *,
    exact_ground_energy: float,
    ground_bitstring: str | None = None,
) -> dict[str, float]:
    """Evaluate a diagonal Pauli objective against a product ground state."""

    if ground_bitstring is None:
        ground_bitstring = "0" * state.L
    if len(ground_bitstring) != state.L or any(bit not in "01" for bit in ground_bitstring):
        raise ValueError(f"ground_bitstring must contain exactly {state.L} binary digits.")
    target_z = np.asarray([1.0 if bit == "0" else -1.0 for bit in ground_bitstring])
    target_zz = target_z[:-1] * target_z[1:]

    energy = 0.0
    orthogonality = {"cur_orthog": "calc"}
    for label, coefficient in final_terms.items():
        if any(symbol not in "IZ" for symbol in label):
            raise ValueError(f"Final Hamiltonian term {label!r} is not diagonal in the Z basis.")
        occupied = [site for site, symbol in enumerate(label) if symbol != "I"]
        if not occupied:
            energy += float(coefficient)
            continue
        local_label = "".join(label[site] for site in occupied)
        expectation = state.local_expectation_canonical(
            _kron_paulis(local_label),
            where=tuple(occupied),
            normalized=True,
            info=orthogonality,
        )
        coefficient = complex(coefficient)
        if abs(coefficient.imag) > 1.0e-10:
            raise ValueError(f"Final Hamiltonian coefficient for {label} is not real: {coefficient}.")
        energy += float(coefficient.real) * float(np.real(expectation))

    fidelity = abs(complex(state.amplitude(ground_bitstring))) ** 2
    norm = float(np.real(state.norm()))
    z_values = np.asarray(
        [
            float(
                np.real(
                    state.local_expectation_canonical(
                        _PAULI["Z"],
                        where=(site,),
                        normalized=True,
                        info=orthogonality,
                    )
                )
            )
            for site in range(state.L)
        ],
        dtype=np.float64,
    )
    zz = np.kron(_PAULI["Z"], _PAULI["Z"])
    zz_values = np.asarray(
        [
            float(
                np.real(
                    state.local_expectation_canonical(
                        zz,
                        where=(site, site + 1),
                        normalized=True,
                        info=orthogonality,
                    )
                )
            )
            for site in range(state.L - 1)
        ],
        dtype=np.float64,
    )
    return {
        "final_energy": energy,
        "ground_energy": float(exact_ground_energy),
        "energy_error": energy - float(exact_ground_energy),
        "ground_fidelity": fidelity,
        "ground_state_fidelity": fidelity,
        "excitation_probability": 1.0 - fidelity,
        "z_rmse": float(np.sqrt(np.mean((z_values - target_z) ** 2))),
        "nearest_neighbor_zz_rmse": (
            float(np.sqrt(np.mean((zz_values - target_zz) ** 2))) if zz_values.size else 0.0
        ),
        "state_norm": norm,
    }


# Compatibility name retained for existing Ising benchmark imports.
diagonal_ising_mps_metrics = diagonal_pauli_mps_metrics


def run_mps_case(
    *,
    h0: SparsePauliOperator,
    h1: SparsePauliOperator,
    learned: Mapping[str, object] | None,
    exact_ground_energy: float,
    ground_bitstring: str,
    protocols: tuple[str, ...],
    total_time: float,
    steps: int,
    cutoff: float,
    max_bond: int,
    coefficient_threshold: float,
    learned_scale: float = 1.0,
    operator_grouping: str = "pauli_term",
    progress: bool = False,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for protocol in protocols:
        start = time.perf_counter()
        state, diagnostics = evolve_protocol_mps(
            protocol=protocol,
            h0=h0,
            h1=h1,
            learned=learned,
            total_time=total_time,
            steps=steps,
            cutoff=cutoff,
            max_bond=max_bond,
            learned_scale=learned_scale,
            coefficient_threshold=coefficient_threshold,
            operator_grouping=operator_grouping,
            progress=progress,
        )
        row: dict[str, object] = diagonal_pauli_mps_metrics(
            state,
            h1.terms,
            exact_ground_energy=exact_ground_energy,
            ground_bitstring=ground_bitstring,
        )
        diagnostics["runtime_seconds"] = time.perf_counter() - start
        row["mps_diagnostics"] = diagnostics
        results[protocol] = row
    return results


def _learned_mpo_factorization(
    learned: Mapping[str, object],
    *,
    learned_scale: float,
    retained_norm: float,
    temporal_grid_points: int,
):
    from scripts.agp_mpo_backend import factor_direct_cd_coefficients

    if int(temporal_grid_points) < 2:
        raise ValueError("temporal_grid_points must be at least two.")
    source_tau = np.asarray(learned["tau"], dtype=np.float64)
    source_coefficients = float(learned_scale) * np.asarray(learned["coefficients"], dtype=np.float64)
    tau = np.linspace(0.0, 1.0, int(temporal_grid_points), dtype=np.float64)
    coefficients = np.column_stack(
        [np.interp(tau, source_tau, source_coefficients[:, column]) for column in range(source_coefficients.shape[1])]
    )
    return factor_direct_cd_coefficients(tau, coefficients, retained_norm=float(retained_norm))


def _mpo_schedule(learned: Mapping[str, object] | None):
    if learned is None:
        return None

    def schedule(tau: float, total_time: float) -> tuple[float, float]:
        return learned_schedule(learned, float(tau) * float(total_time), float(total_time))

    return schedule


def _static_mpo_compression_summary(value: object) -> tuple[int | None, float | None]:
    """Flatten the backend's per-component compression diagnostics for a resolution record."""

    bonds: list[int] = []
    discarded_weights: list[float] = []

    def visit(node: object) -> None:
        if isinstance(node, Mapping):
            compression = node.get("compression")
            if isinstance(compression, Mapping):
                raw_bonds = compression.get("post_bonds", [])
                if isinstance(raw_bonds, Sequence) and not isinstance(raw_bonds, (str, bytes)):
                    bonds.extend(int(item) for item in raw_bonds)
                discarded = compression.get("discarded_weight")
                if discarded is not None:
                    discarded_weights.append(float(discarded))
            for child in node.values():
                visit(child)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for child in node:
                visit(child)

    visit(value)
    return (max(bonds) if bonds else None, sum(discarded_weights) if discarded_weights else None)


def _aggregate_mpo_action_probes(diagnostics: Mapping[str, object]) -> dict[str, object]:
    probes: list[Mapping[str, object]] = []
    static = diagnostics.get("static_mpo_action_probes", {})
    if isinstance(static, Mapping):
        probes.extend(item for item in static.values() if isinstance(item, Mapping))
    dynamic = diagnostics.get("dynamic_mpo_action_probes", [])
    if isinstance(dynamic, Sequence) and not isinstance(dynamic, (str, bytes)):
        probes.extend(item for item in dynamic if isinstance(item, Mapping))
    statuses = [str(item.get("status", "not_tested")) for item in probes]
    finite_error_intervals = [
        interval
        for item in probes
        for interval in item.get("finite_error_intervals", [])
        if isinstance(interval, Mapping)
    ]
    bounds = [
        float(item["max_relative_action_error_upper_bound"])
        for item in probes
        if item.get("max_relative_action_error_upper_bound") is not None
        and np.isfinite(float(item["max_relative_action_error_upper_bound"]))
    ]
    if "not_feasible" in statuses:
        status = "not_feasible"
    elif any(item in {"not_tested", "numerically_unresolved"} for item in statuses):
        status = "numerically_unresolved" if "numerically_unresolved" in statuses else "not_tested"
    elif probes and len(bounds) == len(probes):
        status = "pass"
    else:
        status = "not_tested"
    return {
        "status": status,
        "probe_count": len(probes),
        "max_relative_action_error_upper_bound": max(bounds) if bounds else None,
        "finite_error_intervals": finite_error_intervals,
        "probes": probes,
    }


def _mps_z_observables(state: Any, ground_bitstring: str) -> dict[str, object]:
    target_z = np.asarray([1.0 if bit == "0" else -1.0 for bit in ground_bitstring])
    try:
        z_values = np.asarray(
            [float(np.real(state.expectation_value_term([("Z", site)]))) for site in range(state.L)],
            dtype=np.float64,
        )
        result: dict[str, object] = {
            "z_rmse": float(np.sqrt(np.mean((z_values - target_z) ** 2))),
            "z_observables_status": "ok",
        }
        if state.L < 2:
            result.update({"nearest_neighbor_zz_rmse": None, "nearest_neighbor_zz_status": "not_applicable"})
            return result
        target_zz = target_z[:-1] * target_z[1:]
        zz_values = np.asarray(
            [
                float(np.real(state.expectation_value_term([("Z", site), ("Z", site + 1)])))
                for site in range(state.L - 1)
            ],
            dtype=np.float64,
        )
        result.update(
            {
                "nearest_neighbor_zz_rmse": float(np.sqrt(np.mean((zz_values - target_zz) ** 2))),
                "nearest_neighbor_zz_status": "ok",
            }
        )
        return result
    except (RuntimeError, TypeError, ValueError):
        return {
            "z_rmse": None,
            "z_observables_status": "not_tested",
            "nearest_neighbor_zz_rmse": None,
            "nearest_neighbor_zz_status": "not_tested",
        }


def _completed_mpo_result(diagnostics: Mapping[str, object]) -> bool:
    return (
        str(diagnostics.get("status", "unresolved_error")) == "ok"
        and int(diagnostics.get("completed_steps", -1)) == int(diagnostics.get("steps", -2))
    )


def _canonical_mpo_protocol(protocol: object) -> str:
    name = str(protocol)
    return _MPO_PROTOCOL_ALIASES.get(name, name)


def _canonical_mpo_results(results: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    canonical: dict[str, Mapping[str, object]] = {}
    for protocol, row in results.items():
        name = _canonical_mpo_protocol(protocol)
        if name in _CANONICAL_MPO_PROTOCOLS and isinstance(row, Mapping):
            canonical[name] = row
    return canonical


def statevector_comparison_results(
    *,
    gate_resolution: Mapping[str, object] | None,
    final_results: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    """Select a canonical gate row or an explicitly noncanonical calibration row."""

    source = (
        gate_resolution.get("results", {})
        if gate_resolution is not None
        else final_results
    )
    return _canonical_mpo_results(source if isinstance(source, Mapping) else {})


def _eligible_mpo_resolution_identity(
    case: Mapping[str, object],
    *,
    required_protocols: Sequence[str] = _CANONICAL_MPO_PROTOCOLS,
) -> tuple[object, ...] | None:
    """Return the shared physical identity for a completed canonical MPO resolution."""

    if bool(case.get("ablation", False)) or str(case.get("learned_support", "")) != "full_support":
        return None
    settings = case.get("settings", {})
    results = case.get("results", {})
    if not isinstance(settings, Mapping) or not isinstance(results, Mapping):
        return None
    if str(settings.get("backend", "")) != "tenpy_tdvp_mpo":
        return None
    if int(settings.get("learned_terms", -1)) != int(settings.get("full_learned_terms", -2)):
        return None
    if any(key not in settings for key in _ELIGIBLE_MPO_IDENTITY_KEYS):
        return None
    canonical = _canonical_mpo_results(results)
    required = {_canonical_mpo_protocol(protocol) for protocol in required_protocols}
    if not required or not required.issubset(canonical):
        return None
    if not all(
        isinstance(row.get("mps_diagnostics"), Mapping)
        and _completed_mpo_result(row["mps_diagnostics"])
        for protocol, row in canonical.items()
        if protocol in required
    ):
        return None
    return (
        *(
            _canonical_cache_value(settings[key])
            for key in _ELIGIBLE_MPO_IDENTITY_KEYS
        ),
        _canonical_cache_value(
            settings.get("operator_representation", "temporal_mode_block_sum")
        ),
        _canonical_cache_value(
            settings.get("operator_implementation_version", "legacy")
        ),
    )


def eligible_mpo_resolution_ladder(
    resolutions: Sequence[Mapping[str, object]],
    *,
    required_protocols: Sequence[str] = _CANONICAL_MPO_PROTOCOLS,
) -> list[Mapping[str, object]]:
    """Select only full-support canonical resolutions with one common physical identity."""

    candidates = [
        (
            case,
            _eligible_mpo_resolution_identity(
                case,
                required_protocols=required_protocols,
            ),
        )
        for case in resolutions
        if isinstance(case, Mapping)
    ]
    candidates = [(case, identity) for case, identity in candidates if identity is not None]
    if not candidates:
        return []
    reference_identity = candidates[-1][1]
    return [case for case, identity in candidates if identity == reference_identity]


def final_eligible_mpo_resolution(
    resolutions: Sequence[Mapping[str, object]],
    *,
    required_protocols: Sequence[str] = _CANONICAL_MPO_PROTOCOLS,
) -> Mapping[str, object] | None:
    ladder = eligible_mpo_resolution_ladder(
        resolutions,
        required_protocols=required_protocols,
    )
    return ladder[-1] if ladder else None


def publish_final_eligible_mpo_results(
    payload: dict[str, object],
    resolutions: Sequence[Mapping[str, object]],
    *,
    required_protocols: Sequence[str] = _CANONICAL_MPO_PROTOCOLS,
) -> Mapping[str, object] | None:
    """Publish top-level certified metrics only from the final eligible MPO row."""

    resolution = final_eligible_mpo_resolution(
        resolutions,
        required_protocols=required_protocols,
    )
    if resolution is None:
        return None
    settings = resolution.get("settings", {})
    results = resolution.get("results", {})
    if not isinstance(settings, Mapping) or not isinstance(results, Mapping):
        return None
    payload["results"] = dict(results)
    payload["full_learned_terms"] = settings["full_learned_terms"]
    payload["certification_resolution"] = {
        "name": resolution.get("name"),
        "validation_identity": _statevector_reference_identity(settings),
    }
    return resolution


def _completed_comparable_mpo_resolution_count(
    resolutions: Sequence[Mapping[str, object]],
) -> int:
    """Count only the shared-identity canonical full-support MPO ladder."""

    return len(eligible_mpo_resolution_ladder(resolutions))


def run_mpo_case(
    *,
    h0: SparsePauliOperator,
    h1: SparsePauliOperator,
    learned: Mapping[str, object] | None,
    exact_ground_energy: float,
    ground_bitstring: str,
    protocols: tuple[str, ...],
    total_time: float,
    settings: Mapping[str, object],
    backend: Mapping[str, object],
    learned_scale: float = 1.0,
) -> dict[str, dict[str, object]]:
    """Run the full-support TeNPy MPO backend and retain its diagnostics verbatim."""

    from scripts.agp_mpo_backend import (
        build_full_support_identity,
        evolve_protocol_direct_tdvp,
        evolve_protocol_expm_mpo,
        evolve_protocol_time_tensor_tdvp,
        evolve_protocol_tdvp,
        select_qubit_order,
    )

    h0_terms = list(h0.terms.items())
    h1_terms = list(h1.terms.items())
    results: dict[str, dict[str, object]] = {}
    for protocol in protocols:
        try:
            factorization = None
            labels: tuple[str, ...] = ()
            direct_coefficients: np.ndarray | None = None
            direct_identity = None
            if protocol == "learned_sparse_agp":
                if learned is None:
                    raise ValueError("learned payload is required for learned_sparse_agp.")
                labels = tuple(str(label) for label in learned["labels"])
                direct_coefficients = float(learned_scale) * np.asarray(
                    learned["coefficients"], dtype=np.float64
                )
                direct_identity = build_full_support_identity(labels, direct_coefficients)
                if str(
                    settings.get(
                        "operator_representation",
                        backend.get("representation", "temporal_mode_block_sum"),
                    )
                ) == "temporal_mode_block_sum":
                    factorization = _learned_mpo_factorization(
                        learned,
                        learned_scale=learned_scale,
                        retained_norm=float(settings["temporal_retained_norm"]),
                        temporal_grid_points=int(settings["temporal_grid_points"]),
                    )
            support_terms = [*h0_terms, *h1_terms]
            if learned is not None:
                rms = np.sqrt(np.mean(np.asarray(learned["coefficients"], dtype=np.float64) ** 2, axis=0))
                support_terms.extend(zip((str(label) for label in learned["labels"]), rms, strict=True))
            order = select_qubit_order(
                support_terms,
                n_qubits=h0.n_qubits,
                candidates=tuple(backend["qubit_order_candidates"]),
            )
            direct_learned = (
                protocol == "learned_sparse_agp"
                and str(
                    settings.get(
                        "operator_representation",
                        backend.get("representation", "temporal_mode_block_sum"),
                    )
                )
                == "direct_time_full_support"
            )
            joint_learned = (
                protocol == "learned_sparse_agp"
                and str(
                    settings.get(
                        "operator_representation",
                        backend.get("representation", "temporal_mode_block_sum"),
                    )
                )
                == "joint_time_full_support"
            )
            engine = evolve_protocol_tdvp if str(settings["integrator"]) == "tdvp" else evolve_protocol_expm_mpo
            mapped_protocol = {
                "no_cd": "no_cd",
                "kipu_dqfm_l1": "nested_l1",
                "nested_l1": "nested_l1",
                "learned_sparse_agp": "learned",
            }.get(protocol)
            if mapped_protocol is None:
                raise ValueError(f"Unsupported protocol: {protocol!r}.")
            start = time.perf_counter()
            common_evolution = {
                "h0_terms": h0_terms,
                "h1_terms": h1_terms,
                "schedule": _mpo_schedule(
                    learned if protocol == "learned_sparse_agp" else None
                ),
                "total_time": total_time,
                "steps": int(settings["steps"]),
                "order": order.order,
                "ground_bitstring": ground_bitstring,
                "mps_max_bond": int(settings["mps_max_bond"]),
                "mps_cutoff": float(settings["mps_cutoff"]),
                "mpo_max_bond": int(settings["mpo_max_bond"]),
                "mpo_cutoff": float(settings["mpo_cutoff"]),
                "lanczos_max": int(settings["lanczos_max"]),
                "mpo_workspace_cap_bytes": int(settings["mpo_workspace_cap_bytes"]),
                "action_probe_product_states": int(
                    settings.get("action_probe_product_states", 0)
                ),
                "action_probe_seed": int(settings.get("action_probe_seed", 0)),
                "action_probe_exact_work_cap": int(
                    settings.get("action_probe_exact_work_cap", 10_000_000)
                ),
            }
            if direct_learned:
                if str(settings["integrator"]) != "tdvp":
                    raise ValueError(
                        "direct_time_full_support currently requires the TDVP integrator."
                    )
                assert learned is not None and direct_coefficients is not None
                assert direct_identity is not None
                state, diagnostics = evolve_protocol_direct_tdvp(
                    **common_evolution,
                    learned_tau=np.asarray(learned["tau"], dtype=np.float64),
                    learned_direct_cd_coefficients=direct_coefficients,
                    learned_labels=labels,
                    full_support_identity=direct_identity,
                    action_error_max=float(backend["action_error_max"]),
                )
            elif joint_learned:
                if str(settings["integrator"]) != "tdvp":
                    raise ValueError(
                        "joint_time_full_support currently requires the TDVP integrator."
                    )
                assert learned is not None and direct_coefficients is not None
                assert direct_identity is not None

                def report_joint_progress(event: Mapping[str, object]) -> None:
                    print(
                        "mps_progress="
                        + json.dumps(
                            {"protocol": protocol, **dict(event)},
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )

                state, diagnostics = evolve_protocol_time_tensor_tdvp(
                    **common_evolution,
                    learned_tau=np.asarray(learned["tau"], dtype=np.float64),
                    learned_direct_cd_coefficients=direct_coefficients,
                    learned_labels=labels,
                    full_support_identity=direct_identity,
                    coefficient_error_max=float(backend["coefficient_error_max"]),
                    action_error_max=float(backend["action_error_max"]),
                    action_probe_time_samples=int(
                        settings.get("action_probe_dynamic_samples", 3)
                    ),
                    time_window_size=settings.get("time_window_size"),
                    adaptive_time_windows=bool(
                        settings.get("adaptive_time_windows", True)
                    ),
                    time_axis_position=int(settings.get("time_axis_position", 0)),
                    progress_callback=report_joint_progress,
                )
            else:
                state, diagnostics = engine(
                    **common_evolution,
                    cd_factorization=factorization,
                    cd_labels=labels,
                    protocol=mapped_protocol,
                    action_probe_random_mps=int(
                        settings.get("action_probe_random_mps", 0)
                    ),
                    action_probe_dynamic_samples=int(
                        settings.get(
                            "action_probe_dynamic_samples",
                            1
                            if int(settings.get("action_probe_product_states", 0))
                            or int(settings.get("action_probe_random_mps", 0))
                            else 0,
                        )
                    ),
                )
            diagnostics["runtime_seconds"] = time.perf_counter() - start
            max_build_seconds = settings.get("max_build_seconds")
            if max_build_seconds is not None and diagnostics["runtime_seconds"] > float(max_build_seconds):
                diagnostics["status"] = "not_feasible"
                diagnostics["resource_reason"] = "runtime exceeded configured max_build_seconds"
            diagnostics["qubit_order_candidate"] = order.candidate
            diagnostics["qubit_order"] = list(order.order)
            if direct_learned or joint_learned:
                diagnostics["temporal_retained_norm"] = None
                diagnostics["temporal_reconstruction_error"] = (
                    diagnostics.get("operator_certificate", {}).get(
                        "max_relative_coefficient_error_upper_bound"
                    )
                    if joint_learned
                    else 0.0
                )
                diagnostics["static_mpo_max_bond"] = None
                diagnostics["static_mpo_discarded_weight"] = None
                if joint_learned:
                    operator_certificate = diagnostics.get("operator_certificate", {})
                    operator_certificate = (
                        operator_certificate
                        if isinstance(operator_certificate, Mapping)
                        else {}
                    )
                    diagnostics["mpo_action_error"] = operator_certificate.get(
                        "max_relative_action_error_upper_bound"
                    )
                    diagnostics["mpo_action_status"] = operator_certificate.get(
                        "action_status", "not_tested"
                    )
                else:
                    errors = [
                        float(row["max_relative_action_error_upper_bound"])
                        for row in diagnostics.get("operator_certificates", [])
                        if row.get("max_relative_action_error_upper_bound") is not None
                    ]
                    diagnostics["mpo_action_error"] = max(errors) if errors else None
                    diagnostics["mpo_action_status"] = diagnostics.get(
                        "operator_gate_status", "not_tested"
                    )
                diagnostics["mpo_action_diagnostics"] = {
                    "status": diagnostics["mpo_action_status"],
                    "max_relative_action_error_upper_bound": diagnostics["mpo_action_error"],
                }
            else:
                diagnostics["temporal_retained_norm"] = (
                    None
                    if factorization is None
                    else float(factorization.retained_norm_fraction)
                )
                diagnostics["temporal_reconstruction_error"] = (
                    None if factorization is None else float(factorization.max_abs_error)
                )
                static_max_bond, static_discarded_weight = _static_mpo_compression_summary(
                    diagnostics.get("static_mpo_compression")
                )
                diagnostics["static_mpo_max_bond"] = static_max_bond
                diagnostics["static_mpo_discarded_weight"] = static_discarded_weight
                diagnostics["mpo_action_diagnostics"] = _aggregate_mpo_action_probes(
                    diagnostics
                )
                diagnostics["mpo_action_error"] = diagnostics[
                    "mpo_action_diagnostics"
                ]["max_relative_action_error_upper_bound"]
                diagnostics["mpo_action_status"] = diagnostics[
                    "mpo_action_diagnostics"
                ]["status"]
            diagnostics["dynamic_mpo_max_bond"] = diagnostics.get("dynamic_mpo_peak_bond")
            diagnostics["mps_max_bond"] = int(settings["mps_max_bond"])
            diagnostics["mps_cutoff"] = float(settings["mps_cutoff"])
            diagnostics["timestep"] = float(settings["timestep"])
            complete = _completed_mpo_result(diagnostics)
            final_energy = diagnostics.get("final_energy") if complete else None
            ground_fidelity = diagnostics.get("ground_fidelity") if complete else None
            observables = _mps_z_observables(state, ground_bitstring) if complete else {
                "z_rmse": None,
                "z_observables_status": "not_tested",
                "nearest_neighbor_zz_rmse": None,
                "nearest_neighbor_zz_status": "not_tested",
            }
            row: dict[str, object] = {
                "final_energy": final_energy,
                "ground_energy": float(exact_ground_energy),
                "ground_state_fidelity": ground_fidelity,
                "energy_error": (
                    float(final_energy) - float(exact_ground_energy)
                    if complete and diagnostics.get("final_energy_status") == "ok"
                    else None
                ),
                "excitation_probability": 1.0 - float(ground_fidelity) if ground_fidelity is not None else None,
                **observables,
                "mps_diagnostics": diagnostics,
            }
            results[protocol] = row
        except (MemoryError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
            results[protocol] = {
                "final_energy": None,
                "ground_energy": float(exact_ground_energy),
                "ground_state_fidelity": None,
                "energy_error": None,
                "excitation_probability": None,
                "z_rmse": None,
                "z_observables_status": "not_tested",
                "nearest_neighbor_zz_rmse": None,
                "nearest_neighbor_zz_status": "not_tested",
                "mps_diagnostics": {
                    "status": "unresolved_error",
                    "reason": str(exc),
                    "runtime_seconds": 0.0,
                    "completed_steps": 0,
                    "steps": int(settings["steps"]),
                    "final_energy_status": "not_tested",
                    "ground_fidelity_status": "not_tested",
                },
            }
    return results


def assess_mps_convergence(
    coarse: Mapping[str, Mapping[str, float]],
    fine: Mapping[str, Mapping[str, float]],
    *,
    energy_atol: float,
    fidelity_atol: float,
    required_protocols: Sequence[str] | None = None,
) -> dict[str, object]:
    """Require successive-resolution agreement for every retained protocol."""

    protocols: dict[str, dict[str, float | bool]] = {}
    incomplete = False
    compared_protocols = tuple(required_protocols) if required_protocols is not None else tuple(fine)
    for protocol in compared_protocols:
        if protocol not in coarse or protocol not in fine:
            incomplete = True
            continue
        try:
            energy_delta = abs(float(fine[protocol]["final_energy"]) - float(coarse[protocol]["final_energy"]))
            fidelity_delta = abs(
                float(fine[protocol]["ground_state_fidelity"])
                - float(coarse[protocol]["ground_state_fidelity"])
            )
        except (KeyError, TypeError, ValueError):
            incomplete = True
            continue
        protocols[protocol] = {
            "energy_delta": energy_delta,
            "fidelity_delta": fidelity_delta,
            "pass": energy_delta <= energy_atol and fidelity_delta <= fidelity_atol,
        }
    passed = bool(protocols) and not incomplete and all(bool(row["pass"]) for row in protocols.values())
    return {
        "status": "pass" if passed else ("not_tested" if incomplete else "fail"),
        "energy_atol": float(energy_atol),
        "fidelity_atol": float(fidelity_atol),
        "protocols": protocols,
    }


def assess_timestep_convergence(
    coarse: Mapping[str, object],
    fine: Mapping[str, object],
) -> dict[str, object]:
    """Require a completed eligible ladder to actually refine its timestep."""

    coarse_settings = coarse.get("settings", {})
    fine_settings = fine.get("settings", {})
    if not isinstance(coarse_settings, Mapping) or not isinstance(fine_settings, Mapping):
        return {"status": "not_tested", "reason": "Resolution settings are unavailable."}
    try:
        coarse_timestep = float(coarse_settings["timestep"])
        fine_timestep = float(fine_settings["timestep"])
    except (KeyError, TypeError, ValueError):
        return {"status": "not_tested", "reason": "Timestep metadata is unavailable."}
    if not np.isfinite(coarse_timestep) or not np.isfinite(fine_timestep) or fine_timestep <= 0.0:
        return {"status": "not_tested", "reason": "Timestep metadata is invalid."}
    return {
        "status": "pass" if fine_timestep < coarse_timestep else "not_tested",
        "coarse_timestep": coarse_timestep,
        "fine_timestep": fine_timestep,
    }


def assess_independent_mpo_convergence(
    resolutions: Sequence[Mapping[str, object]],
    *,
    convergence_pairs: Mapping[str, Sequence[str]],
    energy_atol: float,
    fidelity_atol: float,
    required_protocols: Sequence[str] = _CANONICAL_MPO_PROTOCOLS,
) -> dict[str, object]:
    """Assess timestep and MPS refinement on explicit, unconfounded pairs."""

    by_name = {
        str(case.get("name")): case
        for case in resolutions
        if isinstance(case, Mapping) and case.get("name") is not None
    }
    operator_keys = (
        "mpo_max_bond",
        "mpo_cutoff",
        "time_window_size",
        "time_axis_position",
        "operator_representation",
        "operator_implementation_version",
        "lanczos_max",
        "coefficient_error_max",
        "action_error_max",
        "adaptive_time_windows",
        "mpo_workspace_cap_bytes",
    )
    state_keys = ("mps_max_bond", "mps_cutoff")

    def pair(axis: str) -> tuple[Mapping[str, object], Mapping[str, object]] | None:
        names = convergence_pairs.get(axis)
        if (
            not isinstance(names, Sequence)
            or isinstance(names, (str, bytes))
            or len(names) != 2
        ):
            return None
        coarse = by_name.get(str(names[0]))
        fine = by_name.get(str(names[1]))
        return (coarse, fine) if coarse is not None and fine is not None else None

    def same(settings_a: Mapping[str, object], settings_b: Mapping[str, object], keys: Sequence[str]) -> bool:
        return all(
            (
                key not in settings_a
                and key not in settings_b
            )
            or (
                key in settings_a
                and key in settings_b
                and _canonical_cache_value(settings_a[key])
                == _canonical_cache_value(settings_b[key])
            )
            for key in keys
        )

    def metrics(
        coarse: Mapping[str, object], fine: Mapping[str, object]
    ) -> dict[str, object]:
        coarse_results = coarse.get("results", {})
        fine_results = fine.get("results", {})
        if not isinstance(coarse_results, Mapping) or not isinstance(fine_results, Mapping):
            return {"status": "not_tested", "reason": "Resolution results are unavailable."}
        return assess_mps_convergence(
            _canonical_mpo_results(coarse_results),
            _canonical_mpo_results(fine_results),
            energy_atol=float(energy_atol),
            fidelity_atol=float(fidelity_atol),
            required_protocols=required_protocols,
        )

    timestep_pair = pair("timestep")
    if timestep_pair is None:
        timestep: dict[str, object] = {
            "status": "not_tested",
            "reason": "The named timestep convergence pair is unavailable.",
        }
    else:
        coarse, fine = timestep_pair
        coarse_settings = coarse.get("settings", {})
        fine_settings = fine.get("settings", {})
        if not isinstance(coarse_settings, Mapping) or not isinstance(fine_settings, Mapping):
            timestep = {"status": "not_tested", "reason": "Resolution settings are unavailable."}
        elif not same(coarse_settings, fine_settings, (*operator_keys, *state_keys)):
            timestep = {
                "status": "not_comparable",
                "reason": "The timestep pair also changes an MPO or MPS setting.",
            }
        else:
            refinement = assess_timestep_convergence(coarse, fine)
            comparison = metrics(coarse, fine)
            timestep = {
                **comparison,
                "axis_refinement": refinement,
                "status": (
                    str(comparison.get("status"))
                    if refinement.get("status") == "pass"
                    else "not_comparable"
                ),
                "coarse": coarse.get("name"),
                "fine": fine.get("name"),
            }

    state_pair = pair("state")
    if state_pair is None:
        state: dict[str, object] = {
            "status": "not_tested",
            "reason": "The named state convergence pair is unavailable.",
        }
    else:
        coarse, fine = state_pair
        coarse_settings = coarse.get("settings", {})
        fine_settings = fine.get("settings", {})
        fixed_keys = (*operator_keys, "steps", "timestep")
        if not isinstance(coarse_settings, Mapping) or not isinstance(fine_settings, Mapping):
            state = {"status": "not_tested", "reason": "Resolution settings are unavailable."}
        elif not same(coarse_settings, fine_settings, fixed_keys):
            state = {
                "status": "not_comparable",
                "reason": "The state pair also changes a timestep or MPO setting.",
            }
        else:
            try:
                coarse_bond = int(coarse_settings["mps_max_bond"])
                fine_bond = int(fine_settings["mps_max_bond"])
                coarse_cutoff = float(coarse_settings["mps_cutoff"])
                fine_cutoff = float(fine_settings["mps_cutoff"])
                refined = (
                    fine_bond >= coarse_bond
                    and fine_cutoff <= coarse_cutoff
                    and (fine_bond > coarse_bond or fine_cutoff < coarse_cutoff)
                )
            except (KeyError, TypeError, ValueError):
                refined = False
            comparison = metrics(coarse, fine)
            state = {
                **comparison,
                "axis_refinement": {
                    "status": "pass" if refined else "not_comparable",
                    "coarse_mps_max_bond": coarse_settings.get("mps_max_bond"),
                    "fine_mps_max_bond": fine_settings.get("mps_max_bond"),
                    "coarse_mps_cutoff": coarse_settings.get("mps_cutoff"),
                    "fine_mps_cutoff": fine_settings.get("mps_cutoff"),
                },
                "status": str(comparison.get("status")) if refined else "not_comparable",
                "coarse": coarse.get("name"),
                "fine": fine.get("name"),
            }

    statuses = (str(timestep.get("status")), str(state.get("status")))
    if all(status == "pass" for status in statuses):
        status = "pass"
    elif any(status in {"not_tested", "not_comparable"} for status in statuses):
        status = "not_tested"
    else:
        status = "fail"
    return {"status": status, "timestep": timestep, "state": state}


def assess_statevector_agreement(
    mps_results: Mapping[str, Mapping[str, object]],
    statevector_results: Mapping[str, Mapping[str, object]],
    *,
    energy_atol: float,
    fidelity_atol: float,
    require_all_protocols: bool = False,
    required_protocols: Sequence[str] = _CANONICAL_MPO_PROTOCOLS,
) -> dict[str, object]:
    protocols: dict[str, dict[str, float | bool]] = {}
    incomplete = False
    compared_protocols = tuple(required_protocols) if require_all_protocols else tuple(mps_results)
    if require_all_protocols and (
        not set(required_protocols).issubset(mps_results)
        or not set(required_protocols).issubset(statevector_results)
    ):
        incomplete = True
    for protocol in compared_protocols:
        mps_row = mps_results.get(protocol)
        reference = statevector_results.get(protocol)
        if mps_row is None or reference is None:
            incomplete = True
            continue
        try:
            energy_delta = abs(float(mps_row["final_energy"]) - float(reference["final_energy"]))
            fidelity_delta = abs(
                float(mps_row["ground_state_fidelity"])
                - float(reference["ground_state_fidelity"])
            )
        except (KeyError, TypeError, ValueError):
            incomplete = True
            continue
        protocols[protocol] = {
            "energy_delta": energy_delta,
            "fidelity_delta": fidelity_delta,
            "pass": energy_delta <= energy_atol and fidelity_delta <= fidelity_atol,
        }
    passed = bool(protocols) and not (require_all_protocols and incomplete) and all(bool(row["pass"]) for row in protocols.values())
    return {
        "status": "pass" if passed else ("not_tested" if incomplete else "fail"),
        "energy_atol": float(energy_atol),
        "fidelity_atol": float(fidelity_atol),
        "protocols": protocols,
    }


def assess_mpo_compression(
    results: Mapping[str, Mapping[str, object]],
    *,
    action_error_max: float,
) -> dict[str, object]:
    """Require explicit temporal, MPO, and action evidence before certification."""

    protocols: dict[str, dict[str, object]] = {}
    statuses: list[str] = []
    for protocol, row in results.items():
        diagnostics = row.get("mps_diagnostics", {})
        diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
        representation = str(
            diagnostics.get("representation", "temporal_mode_block_sum")
        )
        direct_full_support = representation in {
            "direct_time_full_support",
            "joint_time_full_support",
        }
        joint_full_support = representation == "joint_time_full_support"
        evolution_status = str(diagnostics.get("status", "unresolved_error"))
        temporal_status = "pass"
        if protocol == "learned_sparse_agp":
            if direct_full_support:
                temporal_status = "not_applicable"
            else:
                retained = diagnostics.get("temporal_retained_norm")
                rank = diagnostics.get("temporal_rank")
                temporal_status = (
                    "pass"
                    if retained is not None and float(retained) > 0.0 and int(rank or 0) > 0
                    else "not_tested"
                )
        static_status = (
            "not_applicable"
            if direct_full_support
            else ("pass" if diagnostics.get("static_mpo_compression") else "not_tested")
        )
        if protocol != "learned_sparse_agp" or not direct_full_support:
            source_status = "not_applicable"
        elif diagnostics.get("source_completeness_status") is not None:
            source_status = str(diagnostics["source_completeness_status"])
        else:
            certificates = diagnostics.get("operator_certificates", [])
            certificates = (
                certificates
                if isinstance(certificates, Sequence)
                and not isinstance(certificates, (str, bytes))
                else []
            )
            source_status = (
                "pass"
                if certificates
                and all(
                    isinstance(certificate, Mapping)
                    and certificate.get("source_completeness_status") == "pass"
                    for certificate in certificates
                )
                else "not_tested"
            )
        dynamic_resource_key = (
            "dynamic_mpo_slicing" if joint_full_support else "dynamic_mpo_assembly"
        )
        dynamic_value = str(
            diagnostics.get("resource_statuses", {}).get(
                dynamic_resource_key, "not_tested"
            )
        )
        dynamic_status = "pass" if dynamic_value == "ok" else dynamic_value
        action_diagnostics = (
            diagnostics.get("operator_certificate", {})
            if joint_full_support
            else diagnostics.get("mpo_action_diagnostics", {})
        )
        action_diagnostics = action_diagnostics if isinstance(action_diagnostics, Mapping) else {}
        action_error = action_diagnostics.get(
            "max_relative_action_error_upper_bound", diagnostics.get("mpo_action_error")
        )
        measured_status = str(
            action_diagnostics.get(
                "action_status" if joint_full_support else "status",
                diagnostics.get("mpo_action_status", "not_tested"),
            )
        )
        if measured_status in {
            "not_feasible",
            "not_tested",
            "not_comparable",
            "unresolved_error",
        }:
            action_status = "not_tested"
        elif measured_status == "fail":
            action_status = "fail"
        elif action_error is None or not np.isfinite(float(action_error)):
            action_status = "not_tested"
        elif float(action_error) > float(action_error_max):
            action_status = "fail"
        else:
            action_status = "pass"
        gate_statuses = [
            evolution_status,
            temporal_status,
            static_status,
            dynamic_status,
            action_status,
            source_status,
        ]
        if any(
            item in {"not_feasible", "not_tested", "not_comparable", "unresolved_error", "numerically_unresolved"}
            for item in gate_statuses
        ):
            status = "not_tested"
        elif all(item in {"pass", "ok", "not_applicable"} for item in gate_statuses):
            status = "pass"
        else:
            status = "fail"
        protocols[protocol] = {
            "status": status,
            "temporal": temporal_status,
            "static_mpo": static_status,
            "dynamic_mpo": dynamic_status,
            "mpo_action": action_status,
            "mpo_action_error": action_error,
            "source_completeness": source_status,
            "representation": representation,
        }
        statuses.append(status)
    if not statuses:
        status = "not_tested"
    elif "not_tested" in statuses:
        status = "not_tested"
    elif all(item == "pass" for item in statuses):
        status = "pass"
    else:
        status = "fail"
    return {"status": status, "action_error_max": float(action_error_max), "protocols": protocols}


def statevector_results_for_learned_terms(
    payload: Mapping[str, object],
    *,
    learned_terms: int,
    learned_scale: float,
    require_matching_learned_terms: bool = False,
    required_identity: Mapping[str, object] | None = None,
) -> dict[str, Mapping[str, object]]:
    if required_identity is not None:
        identities = [payload.get("validation_identity", {})]
        variant_identities = payload.get("learned_variant_validation_identities", {})
        if isinstance(variant_identities, Mapping):
            identities.extend(value for value in variant_identities.values() if isinstance(value, Mapping))
        matches_identity = any(
            isinstance(identity, Mapping)
            and all(key in required_identity and key in identity for key in _STATEVECTOR_REFERENCE_IDENTITY_KEYS)
            and all(
                _canonical_cache_value(identity[key])
                == _canonical_cache_value(required_identity[key])
                for key in _STATEVECTOR_REFERENCE_IDENTITY_KEYS
            )
            for identity in identities
        )
        if not matches_identity:
            return {}
    raw_results = payload.get("results", {})
    if not isinstance(raw_results, dict):
        raise TypeError("statevector reference results must be a JSON object.")
    results: dict[str, Mapping[str, object]] = {
        str(name): row
        for name, row in raw_results.items()
        if isinstance(row, dict)
    }
    learned_default = results.get("learned_sparse_agp")
    if require_matching_learned_terms and (
        not isinstance(learned_default, Mapping)
        or int(learned_default.get("learned_terms", -1)) != int(learned_terms)
    ):
        results.pop("learned_sparse_agp", None)
    variants = payload.get("learned_variant_results", {})
    if isinstance(variants, dict):
        for row in variants.values():
            if not isinstance(row, dict):
                continue
            if int(row.get("learned_terms", -1)) != int(learned_terms):
                continue
            if not np.isclose(float(row.get("learned_scale", 1.0)), learned_scale, atol=1.0e-12, rtol=0.0):
                continue
            results["learned_sparse_agp"] = row
            break
    return results


def _statevector_reference_identity(settings: Mapping[str, object]) -> dict[str, object]:
    return validation_identity_from_settings(settings)


def validation_certification(
    *,
    convergence: Mapping[str, object],
    statevector_agreement: Mapping[str, object],
    require_convergence: bool,
    require_statevector: bool,
    compression: Mapping[str, object] | None = None,
    require_compression: bool = False,
    timestep_convergence: Mapping[str, object] | None = None,
    require_timestep: bool = False,
    state_convergence: Mapping[str, object] | None = None,
    require_state_convergence: bool = False,
    ablation: bool = False,
    completed_comparable_resolutions: int | None = None,
) -> dict[str, object]:
    gates: list[tuple[str, Mapping[str, object]]] = []
    if require_convergence:
        gates.append(("mps_convergence", convergence))
    if require_compression:
        gates.append(("mpo_compression", compression or {"status": "not_tested"}))
    if require_timestep:
        gates.append(("timestep_convergence", timestep_convergence or {"status": "not_tested"}))
    if require_state_convergence:
        gates.append(("state_convergence", state_convergence or {"status": "not_tested"}))
    if require_statevector:
        gates.append(("statevector_agreement", statevector_agreement))
    if ablation:
        return {
            "status": "not_tested",
            "required_gates": [name for name, _ in gates],
            "reason": "ablation deployments cannot certify learned physical validation.",
        }
    if completed_comparable_resolutions is not None and int(completed_comparable_resolutions) < 2:
        return {
            "status": "not_tested",
            "required_gates": [name for name, _ in gates],
            "reason": "MPO certification requires two completed comparable resolutions.",
        }
    statuses = [str(gate.get("status", "not_tested")) for _, gate in gates]
    unresolved = {"not_feasible", "not_tested", "not_comparable", "unresolved_error", "numerically_unresolved"}
    if not statuses or any(status in unresolved for status in statuses):
        status = "not_tested"
    elif all(item == "pass" for item in statuses):
        status = "pass"
    else:
        status = "fail"
    return {
        "status": status,
        "required_gates": [name for name, _ in gates],
    }


def cached_protocol_result(
    previous_resolutions: object,
    *,
    settings: Mapping[str, object],
    protocol: str,
) -> dict[str, object] | None:
    if not isinstance(previous_resolutions, list):
        return None
    for case in previous_resolutions:
        if not isinstance(case, dict):
            continue
        previous_settings = case.get("settings", {})
        if not isinstance(previous_settings, dict):
            continue
        if _canonical_cache_value(previous_settings) != _canonical_cache_value(settings):
            continue
        results = case.get("results", {})
        if isinstance(results, dict) and isinstance(results.get(protocol), dict):
            result = dict(results[protocol])
            diagnostics_key = (
                "mps_diagnostics"
                if isinstance(result.get("mps_diagnostics"), Mapping)
                else "diagnostics"
            )
            diagnostics_value = result.get(diagnostics_key)
            if not isinstance(diagnostics_value, Mapping):
                return result
            diagnostics = dict(diagnostics_value)
            if diagnostics.get("representation") != "joint_time_full_support":
                return result
            certificate_value = diagnostics.get("operator_certificate")
            if not isinstance(certificate_value, Mapping):
                return result
            action_status = certificate_value.get("action_status")
            if action_status is None and certificate_value.get("status") == "pass":
                action_status = "pass"
            if action_status not in {"pass", "fail"}:
                return result
            action_error = certificate_value.get(
                "max_relative_action_error_upper_bound"
            )
            diagnostics["mpo_action_status"] = action_status
            diagnostics["mpo_action_error"] = action_error
            diagnostics["mpo_action_diagnostics"] = {
                "status": action_status,
                "method": "adaptive_windowed_joint_time_pauli_tt",
                "max_relative_action_error_upper_bound": action_error,
            }
            result[diagnostics_key] = diagnostics
            return result
    return None


def _canonical_cache_value(value: object) -> object:
    """Canonicalize all settings axes so a changed numerical contract cannot reuse a run."""

    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _canonical_cache_value(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_cache_value(item) for item in value)
    if isinstance(value, (np.floating, float)):
        return ("float", float(value).hex())
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return ("int", int(value))
    return value


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return payload


def _resolve_path(raw: object, *, base: Path) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else base / path


def load_optional_statevector_reference(
    raw: object,
    *,
    base: Path,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    """Load an optional statevector oracle without discarding completed TN results."""

    path = _resolve_path(raw, base=base)
    if not path.is_file():
        return None, {
            "status": "not_tested",
            "reason": f"Configured statevector reference does not exist: {path}",
            "reference_path": str(path),
        }
    return _load_json(path), {
        "status": "not_tested",
        "reason": "Statevector reference was loaded but has not yet been compared.",
        "reference_path": str(path),
    }


def _ground_reference(
    validation: Mapping[str, object],
    *,
    n_qubits: int,
) -> tuple[float, str]:
    energy = validation.get("ground_energy")
    bitstring = validation.get("ground_bitstring")
    reference_path = validation.get("exact_final_ground_reference")
    if reference_path:
        payload = _load_json(_resolve_path(reference_path, base=ROOT))
        solutions = payload.get("solutions", [])
        if isinstance(solutions, list):
            for row in solutions:
                if isinstance(row, dict) and int(row.get("q", -1)) == n_qubits:
                    energy = row.get("ground_energy", energy)
                    bitstrings = row.get("ground_bitstrings", row.get("ground_bitstring", bitstring))
                    if isinstance(bitstrings, list) and bitstrings:
                        bitstring = bitstrings[0]
                    elif bitstrings is not None:
                        bitstring = bitstrings
                    break
    if energy is None or bitstring is None:
        raise ValueError("tensor_network_validation needs an exact ground energy and bitstring reference.")
    bitstring = str(bitstring)
    if len(bitstring) != n_qubits:
        raise ValueError(f"Ground bitstring has length {len(bitstring)}, expected {n_qubits}.")
    return float(energy), bitstring


def _add_baseline_quotients(results: dict[str, dict[str, object]]) -> None:
    baseline = results.get("no_cd")
    if baseline is None:
        return
    for row in results.values():
        for metric in ("energy_error", "excitation_probability", "z_rmse", "nearest_neighbor_zz_rmse"):
            value = row.get(metric)
            reference = baseline.get(metric)
            if value is None or reference is None:
                row[f"{metric}_quotient_vs_no_cd"] = None
                continue
            try:
                row[f"{metric}_quotient_vs_no_cd"] = float(value) / max(float(reference), 1.0e-15)
            except (TypeError, ValueError):
                row[f"{metric}_quotient_vs_no_cd"] = None


def _save_progress(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def finalize_validation_artifacts(
    *,
    summary_path: Path,
    images_dir: Path,
    payload: Mapping[str, object],
    trained_run: Path,
    refresh_training_hcd: bool = True,
) -> None:
    """Publish the TN summary/table and refresh the matching training figure."""

    _save_progress(summary_path, payload)
    images_dir.mkdir(parents=True, exist_ok=True)
    plot_physical_comparison_table(images_dir, payload)
    if refresh_training_hcd:
        refresh_hcd_connection_summary(trained_run, trained_run)


def select_validation_cases(
    configured_cases: object,
    *,
    preflight_only: bool,
) -> list[dict[str, object]]:
    """Separate diagnostic preflights from the certifiable resolution ladder."""

    cases = (
        [dict(case) for case in configured_cases if isinstance(case, Mapping)]
        if isinstance(configured_cases, list)
        else []
    )
    for case in cases:
        marker = case.get("preflight_only", False)
        if not isinstance(marker, bool):
            raise ValueError("resolution preflight_only must be a Boolean.")
    if not cases:
        if preflight_only:
            raise ValueError("--preflight-only requires a resolution with preflight_only=true.")
        return [{}]
    selected = [
        case
        for case in cases
        if case.get("preflight_only", False) is preflight_only
    ]
    if not selected:
        mode = "preflight_only=true" if preflight_only else "preflight_only=false"
        raise ValueError(f"No tensor-network validation resolution is configured with {mode}.")
    return selected


def apply_case_override_mode(
    configured_cases: list[dict[str, object]],
    *,
    preflight_only: bool,
    override_requested: bool,
) -> list[dict[str, object]]:
    """Keep preflight safety limits while preserving legacy validation overrides."""

    if override_requested and not preflight_only:
        return [{}]
    return configured_cases


def execution_output_dir(output_dir: Path, *, preflight_only: bool) -> Path:
    """Isolate diagnostic preflight artifacts from canonical validation outputs."""

    return output_dir / "preflight" if preflight_only else output_dir


def preflight_gate_status_payload(
    payload: Mapping[str, object],
    *,
    action_error_max: float,
    preflight_summary_path: Path,
) -> dict[str, object]:
    """Publish a canonical not-tested record without promoting preflight metrics."""

    results = payload.get("results", {})
    results = results if isinstance(results, Mapping) else {}
    learned = results.get("learned_sparse_agp", {})
    learned = learned if isinstance(learned, Mapping) else {}
    diagnostics = learned.get("mps_diagnostics", {})
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    raw_error = diagnostics.get("mpo_action_error")
    try:
        action_error = float(raw_error) if raw_error is not None else None
    except (TypeError, ValueError):
        action_error = None
    action_status = str(diagnostics.get("mpo_action_status", "not_tested"))
    if action_error is not None and np.isfinite(action_error) and action_error > action_error_max:
        gate_status = "fail"
        reason = (
            "Canonical full-support TDVP validation was not run because the diagnostic "
            f"preflight MPO action error was {action_error:.6g}, above the "
            f"configured {action_error_max:.6g} limit."
        )
    elif (
        action_error is not None
        and np.isfinite(action_error)
        and action_error <= action_error_max
        and action_status in {"measured", "pass", "ok"}
    ):
        gate_status = "pass"
        reason = (
            "The diagnostic full-support TDVP preflight passed the MPO action-error "
            "gate; canonical multi-resolution dynamics have not yet been run."
        )
    else:
        gate_status = "not_tested"
        reason = (
            "Canonical full-support TDVP validation was not run because the diagnostic "
            "preflight did not establish the MPO action-error gate."
        )

    retained_keys = (
        "description",
        "backend",
        "backend_configuration",
        "n_qubits",
        "total_time",
        "trained_run",
        "coefficient_path",
        "ground_energy",
        "ground_bitstring",
        "protocols",
        "full_learned_terms",
    )
    status_payload = {key: payload[key] for key in retained_keys if key in payload}
    status_payload.update(
        {
            "execution_mode": "validation_status",
            "results": {},
            "resolution_results": [],
            "availability_note": reason,
            "preflight_gate": {
                "status": gate_status,
                "mpo_action_error": action_error,
                "mpo_action_error_max": float(action_error_max),
                "mpo_action_status": action_status,
                "summary_path": str(preflight_summary_path),
            },
            "certification": {
                "status": "not_tested",
                "reason": reason,
            },
        }
    )
    return status_payload


def should_publish_preflight_status(summary_path: Path) -> bool:
    """Never replace an existing canonical validation with a diagnostic status."""

    summary_path = Path(summary_path)
    if not summary_path.is_file():
        return True
    try:
        payload = _load_json(summary_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return payload.get("execution_mode") == "validation_status"


def previous_payload_matches_execution(
    payload: Mapping[str, object],
    *,
    n_qubits: int,
    coefficient_path: Path,
    execution_mode: str,
) -> bool:
    """Require immutable execution provenance before reusing cached protocols."""

    return (
        int(payload.get("n_qubits", -1)) == int(n_qubits)
        and str(payload.get("coefficient_path", "")) == str(coefficient_path)
        and payload.get("execution_mode") == execution_mode
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scalable MPS validation of sparse counterdiabatic protocols.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trained-run", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--evaluation-duration",
        type=float,
        default=None,
        help=(
            "Explicit physical duration for normalized-time reparameterization. "
            "The learned schedule is kept as a function of tau and its velocity is "
            "rescaled by d_lambda/dt=(1/T)d_lambda/dtau."
        ),
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max-bond", type=int, default=None)
    parser.add_argument("--mpo-max-bond", type=int, default=None)
    parser.add_argument("--mps-max-bond", type=int, default=None)
    parser.add_argument("--learned-terms", type=int, default=None)
    parser.add_argument("--cutoff", type=float, default=None)
    parser.add_argument("--mpo-cutoff", type=float, default=None)
    parser.add_argument("--mps-cutoff", type=float, default=None)
    parser.add_argument("--coefficient-threshold", type=float, default=None)
    parser.add_argument("--operator-grouping", choices=("pauli_term", "support"), default=None)
    parser.add_argument("--protocols", type=str, default=None)
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Mark an explicit reduced-support CLI run as a noncanonical calibration.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run only resolutions marked preflight_only=true; never certify them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    run_root = config_path.parent
    config = _load_json(config_path)
    physical = config.get("physical", {})
    parameters = physical.get("parameters", {}) if isinstance(physical, dict) else {}
    validation = config.get("tensor_network_validation", {})
    if not isinstance(validation, dict) or not validation.get("enabled", False):
        raise ValueError("tensor_network_validation.enabled must be true in the benchmark config.")
    backend = resolve_validation_backend(validation)

    n_qubits = int(parameters.get("num_qubits", 0))
    source_total_time = float(parameters.get("T", 1.0))
    total_time = float(
        args.evaluation_duration
        if args.evaluation_duration is not None
        else source_total_time
    )
    if total_time <= 0.0:
        raise ValueError("Tensor-network evaluation duration must be positive.")
    hamiltonian_source = _resolve_path(
        parameters.get("hamiltonian_source", "Hamiltonians_to_use/pauli_decompositions/index.json"),
        base=ROOT,
    )
    h0, h1 = load_pauli_hamiltonian_pair(
        hamiltonian_source,
        system=str(parameters.get("system", "TransverseIsingDriverProblem")),
        n_qubits=n_qubits,
        distance=str(parameters.get("distance", "1_0")),
    )
    ground_energy, ground_bitstring = _ground_reference(validation, n_qubits=n_qubits)
    hamiltonian_identity = _hamiltonian_identity(h0, h1)
    ground_reference_identity = _ground_reference_identity(
        validation, ground_energy=ground_energy, ground_bitstring=ground_bitstring
    )

    trained_run_raw = args.trained_run or validation.get("trained_run")
    if trained_run_raw is None:
        raise ValueError("tensor_network_validation.trained_run must identify the retained checkpoint directory.")
    trained_run = _resolve_path(trained_run_raw, base=run_root)
    coefficient_path = trained_run / "Models_Data" / "final_agp_coefficients.pt"
    if not coefficient_path.is_file():
        raise FileNotFoundError(f"Missing trained AGP coefficients: {coefficient_path}")

    configured_cases = select_validation_cases(
        validation.get("resolutions", []),
        preflight_only=bool(args.preflight_only),
    )
    configured_cases = apply_case_override_mode(
        configured_cases,
        preflight_only=bool(args.preflight_only),
        override_requested=any(
            value is not None
            for value in (
                args.steps,
                args.max_bond,
                args.mpo_max_bond,
                args.mps_max_bond,
                args.learned_terms,
                args.cutoff,
                args.mpo_cutoff,
                args.mps_cutoff,
            )
        )
        or bool(args.ablation),
    )
    protocols_raw = args.protocols or validation.get(
        "protocols", ["no_cd", "kipu_dqfm_l1", "learned_sparse_agp"]
    )
    if isinstance(protocols_raw, str):
        protocols = tuple(item.strip() for item in protocols_raw.split(",") if item.strip())
    else:
        protocols = tuple(str(item) for item in protocols_raw)
    required_canonical_protocols = tuple(
        _MPO_PROTOCOL_ALIASES.get(protocol, protocol) for protocol in protocols
    )
    max_requested_terms = max(
        int(args.learned_terms or row.get("learned_terms", validation.get("learned_terms", 256)))
        for row in configured_cases
        if isinstance(row, dict)
    )
    selection_limit = sys.maxsize if backend["name"] == "tenpy_tdvp_mpo" else max_requested_terms
    learned_full = learned_term_selection(
        coefficient_path,
        selection_limit,
        expected_total_time=total_time,
        allow_duration_reparameterization=args.evaluation_duration is not None,
    )

    output_dir = args.output_dir or _resolve_path(
        validation.get("output_dir", "mps_validation"),
        base=trained_run,
    )
    if not output_dir.is_absolute():
        output_dir = run_root / output_dir
    canonical_output_dir = output_dir
    execution_mode = "preflight_only" if args.preflight_only else "validation"
    output_dir = execution_output_dir(output_dir, preflight_only=bool(args.preflight_only))
    data_dir = output_dir / "Models_Data"
    images_dir = output_dir / "Images"
    summary_path = data_dir / "mps_physical_validation_summary.json"
    previous_resolutions: object = []
    if summary_path.is_file():
        previous_payload = _load_json(summary_path)
        if previous_payload_matches_execution(
            previous_payload,
            n_qubits=n_qubits,
            coefficient_path=coefficient_path,
            execution_mode=execution_mode,
        ):
            previous_resolutions = previous_payload.get("resolution_results", [])
    payload: dict[str, object] = {
        "description": (
            "Tensor-network dynamical validation with an explicit legacy product-formula backend or "
            "full-support compressed-MPO TDVP, exact final-energy contractions, and product-state overlap."
        ),
        "backend": backend["name"],
        "backend_configuration": backend,
        "n_qubits": n_qubits,
        "total_time": total_time,
        "source_total_time": source_total_time,
        "duration_reparameterized": bool(args.evaluation_duration is not None),
        "trained_run": str(trained_run),
        "coefficient_path": str(coefficient_path),
        "ground_energy": ground_energy,
        "ground_bitstring": ground_bitstring,
        "protocols": list(protocols),
        "resolution_results": [],
        "execution_mode": execution_mode,
        "certification": {"status": "not_tested"},
    }

    resolution_results: list[dict[str, object]] = []
    for case_index, raw_case in enumerate(configured_cases):
        case = raw_case if isinstance(raw_case, dict) else {}
        steps = int(args.steps or case.get("steps", validation.get("steps", 48)))
        max_bond = int(args.max_bond or case.get("max_bond", validation.get("max_bond", 64)))
        learned_terms = int(
            args.learned_terms or case.get("learned_terms", validation.get("learned_terms", 256))
        )
        cutoff = float(args.cutoff or case.get("cutoff", validation.get("cutoff", 1.0e-10)))
        coefficient_threshold = float(
            args.coefficient_threshold
            if args.coefficient_threshold is not None
            else case.get("coefficient_threshold", validation.get("coefficient_threshold", 0.0))
        )
        operator_grouping = str(
            args.operator_grouping
            or case.get("operator_grouping", validation.get("operator_grouping", "pauli_term"))
        )
        learned = subset_learned_terms(learned_full, learned_terms)
        is_mpo_backend = backend["name"] == "tenpy_tdvp_mpo"
        ablation = resolve_case_ablation(
            cli_ablation=bool(args.ablation),
            case_ablation=case.get("ablation", False),
            backend_ablation=backend["ablation"],
        )
        support_class = "legacy"
        if is_mpo_backend and "learned_sparse_agp" in protocols:
            support_class = require_full_learned_support(
                selected_terms=int(learned["selected_terms"]),
                available_terms=int(learned["available_terms"]),
                ablation=ablation,
            )
        mpo_max_bond = int(
            args.mpo_max_bond
            if args.mpo_max_bond is not None
            else case.get("mpo_max_bond", case.get("max_bond", validation.get("max_bond", 64)))
        )
        mpo_cutoff = float(
            args.mpo_cutoff
            if args.mpo_cutoff is not None
            else case.get("mpo_cutoff", case.get("cutoff", validation.get("cutoff", 1.0e-10)))
        )
        mps_max_bond = int(
            args.mps_max_bond
            if args.mps_max_bond is not None
            else case.get("mps_max_bond", case.get("max_bond", validation.get("max_bond", 64)))
        )
        mps_cutoff = float(
            args.mps_cutoff
            if args.mps_cutoff is not None
            else case.get("mps_cutoff", case.get("cutoff", validation.get("cutoff", 1.0e-10)))
        )
        resource_caps = backend["resource_caps"]
        resource_caps = resource_caps if isinstance(resource_caps, Mapping) else {}
        case_payload: dict[str, object] = {
            "name": str(case.get("name", f"resolution_{case_index + 1}")),
            "full_learned_terms": int(learned["available_terms"]),
            "learned_support": support_class,
            "ablation": ablation,
            "settings": {
                "backend": backend["name"],
                "integrator": str(case.get("integrator", backend["integrator"])),
                "operator_representation": str(
                    case.get("operator_representation", backend["representation"])
                ),
                "operator_implementation_version": _MPO_OPERATOR_IMPLEMENTATION_VERSION,
                "n_qubits": n_qubits,
                "total_time": total_time,
                "initial_state": "+" * n_qubits,
                "steps": steps,
                "timestep": total_time / steps,
                "max_bond": max_bond,
                "learned_terms": int(learned["selected_terms"]),
                "full_learned_terms": int(learned["available_terms"]),
                "retained_rms_norm_fraction": float(learned["retained_rms_norm_fraction"]),
                "cutoff": cutoff,
                "coefficient_threshold": coefficient_threshold,
                "operator_grouping": operator_grouping,
                "temporal_grid_points": int(case.get("temporal_grid_points", backend["temporal_grid_points"])),
                "temporal_retained_norm": float(case.get("temporal_retained_norm", backend["temporal_retained_norm"])),
                "mpo_max_bond": mpo_max_bond,
                "mpo_cutoff": mpo_cutoff,
                "dynamic_mpo_max_bond": int(case.get("dynamic_mpo_max_bond", mpo_max_bond)),
                "dynamic_mpo_cutoff": float(case.get("dynamic_mpo_cutoff", mpo_cutoff)),
                "mps_max_bond": mps_max_bond,
                "mps_cutoff": mps_cutoff,
                "lanczos_max": int(case.get("lanczos_max", backend["lanczos_max"])),
                "qubit_order_candidates": list(backend["qubit_order_candidates"]),
                "action_probe_seed": int(backend["action_probe_seed"]),
                "action_probe_product_states": int(backend["action_probe_product_states"]),
                "action_probe_random_mps": int(backend["action_probe_random_mps"]),
                "action_probe_exact_work_cap": int(
                    case.get("action_probe_exact_work_cap", backend["action_probe_exact_work_cap"])
                ),
                "action_probe_dynamic_samples": int(
                    case.get("action_probe_dynamic_samples", backend["action_probe_dynamic_samples"])
                ),
                "coefficient_error_max": float(backend["coefficient_error_max"]),
                "action_error_max": float(backend["action_error_max"]),
                "time_window_size": case.get(
                    "time_window_size", backend["time_window_size"]
                ),
                "adaptive_time_windows": bool(
                    case.get(
                        "adaptive_time_windows", backend["adaptive_time_windows"]
                    )
                ),
                "time_axis_position": int(
                    case.get("time_axis_position", backend["time_axis_position"])
                ),
                "mpo_workspace_cap_bytes": int(case.get("mpo_workspace_cap_bytes", backend["mpo_workspace_cap_bytes"])),
                "max_build_seconds": case.get("max_build_seconds", resource_caps.get("max_build_seconds")),
                "max_peak_memory_gb": case.get("max_peak_memory_gb", resource_caps.get("max_peak_memory_gb")),
                "checkpoint_identity": _checkpoint_identity(coefficient_path),
                "coefficient_identity": _checkpoint_identity(coefficient_path),
                "learned_scale": float(validation.get("learned_scale", 1.0)),
                "hamiltonian_identity": hamiltonian_identity,
                "ground_reference_identity": ground_reference_identity,
                "ground_bitstring": ground_bitstring,
                "schedule_identity": _learned_schedule_identity(
                    learned, learned_scale=float(validation.get("learned_scale", 1.0))
                ),
                "schedule_parameters_identity": schedule_parameters_identity(learned),
                "statevector_integrator": "rk4_renormalized",
            },
            "results": {},
        }
        resolution_results.append(case_payload)
        payload["resolution_results"] = resolution_results
        for protocol in protocols:
            cached = cached_protocol_result(
                previous_resolutions,
                settings=case_payload["settings"],  # type: ignore[arg-type]
                protocol=protocol,
            )
            if cached is not None:
                print(f"mps_case={case_payload['name']} protocol={protocol} reuse=checkpoint", flush=True)
                case_payload["results"][protocol] = cached  # type: ignore[index]
                _save_progress(summary_path, payload)
                continue
            print(
                f"mps_case={case_payload['name']} protocol={protocol} q={n_qubits} "
                f"steps={steps} max_bond={max_bond} learned_terms={learned['selected_terms']} "
                f"grouping={operator_grouping}",
                flush=True,
            )
            if is_mpo_backend:
                result = run_mpo_case(
                    h0=h0,
                    h1=h1,
                    learned=learned,
                    exact_ground_energy=ground_energy,
                    ground_bitstring=ground_bitstring,
                    protocols=(protocol,),
                    total_time=total_time,
                    settings=case_payload["settings"],  # type: ignore[arg-type]
                    backend=backend,
                    learned_scale=float(validation.get("learned_scale", 1.0)),
                )
            else:
                result = run_mps_case(
                    h0=h0,
                    h1=h1,
                    learned=learned,
                    exact_ground_energy=ground_energy,
                    ground_bitstring=ground_bitstring,
                    protocols=(protocol,),
                    total_time=total_time,
                    steps=steps,
                    cutoff=cutoff,
                    max_bond=max_bond,
                    coefficient_threshold=coefficient_threshold,
                    learned_scale=float(validation.get("learned_scale", 1.0)),
                    operator_grouping=operator_grouping,
                    progress=bool(validation.get("progress", True)),
                )
            case_payload["results"].update(result)  # type: ignore[union-attr]
            _save_progress(summary_path, payload)
        _add_baseline_quotients(case_payload["results"])  # type: ignore[arg-type]
        _save_progress(summary_path, payload)

    final_results = resolution_results[-1]["results"]
    convergence: dict[str, object] = {"status": "not_tested", "reason": "Only one resolution was run."}
    timestep_convergence: dict[str, object] = {
        "status": "not_tested",
        "reason": "Only one eligible resolution was run.",
    }
    state_convergence: dict[str, object] = {
        "status": "not_tested",
        "reason": "No independent state-bond pair was configured.",
    }
    eligible_ladder: list[Mapping[str, object]] = []
    if backend["name"] == "tenpy_tdvp_mpo":
        eligible_ladder = eligible_mpo_resolution_ladder(
            resolution_results,
            required_protocols=required_canonical_protocols,
        )
        payload["eligible_resolution_count"] = len(eligible_ladder)
    convergence_pairs = validation.get("convergence_pairs", {})
    if (
        backend["name"] == "tenpy_tdvp_mpo"
        and isinstance(convergence_pairs, Mapping)
        and convergence_pairs
    ):
        independent = assess_independent_mpo_convergence(
            eligible_ladder,
            convergence_pairs=convergence_pairs,  # type: ignore[arg-type]
            energy_atol=float(validation.get("convergence_energy_atol", 0.05)),
            fidelity_atol=float(validation.get("convergence_fidelity_atol", 0.01)),
            required_protocols=required_canonical_protocols,
        )
        timestep_value = independent.get("timestep", {})
        state_value = independent.get("state", {})
        timestep_convergence = (
            dict(timestep_value) if isinstance(timestep_value, Mapping) else {"status": "not_tested"}
        )
        state_convergence = (
            dict(state_value) if isinstance(state_value, Mapping) else {"status": "not_tested"}
        )
        convergence = {
            "status": independent.get("status", "not_tested"),
            "method": "independent_named_timestep_and_state_pairs",
            "timestep": timestep_convergence,
            "state": state_convergence,
        }
    elif backend["name"] == "tenpy_tdvp_mpo" and len(eligible_ladder) >= 2:
        coarse_results = _canonical_mpo_results(eligible_ladder[-2]["results"])  # type: ignore[arg-type]
        fine_results = _canonical_mpo_results(eligible_ladder[-1]["results"])  # type: ignore[arg-type]
        convergence = assess_mps_convergence(
            coarse_results,
            fine_results,
            energy_atol=float(validation.get("convergence_energy_atol", 0.05)),
            fidelity_atol=float(validation.get("convergence_fidelity_atol", 0.01)),
            required_protocols=required_canonical_protocols,
        )
        timestep_convergence = assess_timestep_convergence(eligible_ladder[-2], eligible_ladder[-1])
    elif backend["name"] != "tenpy_tdvp_mpo" and len(resolution_results) >= 2:
        convergence = assess_mps_convergence(
            resolution_results[-2]["results"],  # type: ignore[arg-type]
            final_results,  # type: ignore[arg-type]
            energy_atol=float(validation.get("convergence_energy_atol", 0.05)),
            fidelity_atol=float(validation.get("convergence_fidelity_atol", 0.01)),
        )
    payload["convergence"] = convergence
    payload["timestep_convergence"] = timestep_convergence
    payload["state_convergence"] = state_convergence

    compression: dict[str, object] = {"status": "not_tested", "reason": "Legacy product-formula backend."}
    gate_resolution = publish_final_eligible_mpo_results(
        payload,
        resolution_results,
        required_protocols=required_canonical_protocols,
    )
    if backend["name"] == "tenpy_tdvp_mpo" and gate_resolution is not None:
        final_results = gate_resolution["results"]  # type: ignore[index]
    else:
        payload["results"] = final_results
    gate_results = (
        _canonical_mpo_results(gate_resolution["results"])  # type: ignore[arg-type]
        if gate_resolution is not None
        else {}
    )
    statevector_gate_results = statevector_comparison_results(
        gate_resolution=gate_resolution,
        final_results=final_results,  # type: ignore[arg-type]
    )
    if backend["name"] == "tenpy_tdvp_mpo" and gate_resolution is not None:
        compression = assess_mpo_compression(
            gate_results,
            action_error_max=float(backend["action_error_max"]),
        )
    payload["compression"] = compression

    statevector_agreement: dict[str, object] = {
        "status": "not_tested",
        "reason": "No statevector reference was configured.",
    }
    statevector_reference = validation.get("statevector_reference")
    if statevector_reference:
        reference_payload, statevector_agreement = load_optional_statevector_reference(
            statevector_reference,
            base=run_root,
        )
        if reference_payload is not None:
            final_settings = (
                gate_resolution["settings"]  # type: ignore[index]
                if backend["name"] == "tenpy_tdvp_mpo" and gate_resolution is not None
                else resolution_results[-1]["settings"]
            )
            reference_results = statevector_results_for_learned_terms(
                reference_payload,
                learned_terms=int(final_settings["learned_terms"]),  # type: ignore[index]
                learned_scale=float(validation.get("learned_scale", 1.0)),
                require_matching_learned_terms=backend["name"] == "tenpy_tdvp_mpo",
                required_identity=(
                    _statevector_reference_identity(final_settings)  # type: ignore[arg-type]
                    if backend["name"] == "tenpy_tdvp_mpo"
                    else None
                ),
            )
            statevector_agreement = assess_statevector_agreement(
                statevector_gate_results if backend["name"] == "tenpy_tdvp_mpo" else final_results,  # type: ignore[arg-type]
                (
                    _canonical_mpo_results(reference_results)
                    if backend["name"] == "tenpy_tdvp_mpo"
                    else reference_results
                ),
                energy_atol=float(validation.get("statevector_energy_atol", 0.05)),
                fidelity_atol=float(validation.get("statevector_fidelity_atol", 0.01)),
                require_all_protocols=backend["name"] == "tenpy_tdvp_mpo",
                required_protocols=required_canonical_protocols,
            )
    payload["statevector_agreement"] = statevector_agreement
    payload["certification"] = validation_certification(
        convergence=convergence,
        compression=compression,
        statevector_agreement=statevector_agreement,
        require_convergence=backend["name"] == "tenpy_tdvp_mpo" or len(resolution_results) >= 2,
        require_compression=backend["name"] == "tenpy_tdvp_mpo",
        timestep_convergence=timestep_convergence,
        require_timestep=backend["name"] == "tenpy_tdvp_mpo",
        state_convergence=state_convergence,
        require_state_convergence=backend["name"] == "tenpy_tdvp_mpo",
        require_statevector=(
            n_qubits <= 15 and backend["name"] == "tenpy_tdvp_mpo" and gate_resolution is not None
        ) or bool(statevector_reference),
        ablation=bool(
            gate_resolution.get("ablation", False)
            if gate_resolution is not None
            else resolution_results[-1]["ablation"]
        ),
        completed_comparable_resolutions=(
            len(eligible_ladder)
            if backend["name"] == "tenpy_tdvp_mpo"
            else None
        ),
    )
    if args.preflight_only:
        payload["certification"] = {
            "status": "not_tested",
            "reason": "Diagnostic preflight cannot certify physical dynamics.",
            "required_gates": [],
        }
    finalize_validation_artifacts(
        summary_path=summary_path,
        images_dir=images_dir,
        payload=payload,
        trained_run=trained_run,
        refresh_training_hcd=not bool(args.preflight_only),
    )
    if args.preflight_only:
        status_payload = preflight_gate_status_payload(
            payload,
            action_error_max=float(backend["action_error_max"]),
            preflight_summary_path=summary_path,
        )
        canonical_data_dir = canonical_output_dir / "Models_Data"
        canonical_images_dir = canonical_output_dir / "Images"
        canonical_summary_path = canonical_data_dir / "mps_physical_validation_summary.json"
        if should_publish_preflight_status(canonical_summary_path):
            _save_progress(canonical_summary_path, status_payload)
            canonical_images_dir.mkdir(parents=True, exist_ok=True)
            plot_physical_comparison_table(canonical_images_dir, status_payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
