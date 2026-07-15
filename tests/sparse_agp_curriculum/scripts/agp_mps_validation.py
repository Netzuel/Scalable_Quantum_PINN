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
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import SparsePauliOperator, load_pauli_hamiltonian_pair  # noqa: E402

from agp_physical_validation import (  # noqa: E402
    learned_term_selection,
    interpolate_coefficients,
    learned_schedule,
    schedule_sin2,
    subset_learned_terms,
    variational_l1_agp,
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

    grouped = group_hamiltonian_terms_by_support(terms)
    applied_groups = 0
    for support, hamiltonian in grouped.items():
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


def diagonal_ising_mps_metrics(
    state,
    final_terms: Mapping[str, float],
    *,
    exact_ground_energy: float,
    ground_bitstring: str | None = None,
) -> dict[str, float]:
    """Evaluate diagonal Pauli energy and product-ground-state fidelity."""

    energy = 0.0
    orthogonality = {"cur_orthog": "calc"}
    for label, coefficient in final_terms.items():
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

    if ground_bitstring is None:
        ground_bitstring = "0" * state.L
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
        "z_rmse": float(np.sqrt(np.mean((z_values - 1.0) ** 2))),
        "nearest_neighbor_zz_rmse": (
            float(np.sqrt(np.mean((zz_values - 1.0) ** 2))) if zz_values.size else 0.0
        ),
        "state_norm": norm,
    }


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
        row: dict[str, object] = diagonal_ising_mps_metrics(
            state,
            h1.terms,
            exact_ground_energy=exact_ground_energy,
            ground_bitstring=ground_bitstring,
        )
        diagnostics["runtime_seconds"] = time.perf_counter() - start
        row["mps_diagnostics"] = diagnostics
        results[protocol] = row
    return results


def assess_mps_convergence(
    coarse: Mapping[str, Mapping[str, float]],
    fine: Mapping[str, Mapping[str, float]],
    *,
    energy_atol: float,
    fidelity_atol: float,
) -> dict[str, object]:
    """Require successive-resolution agreement for every retained protocol."""

    protocols: dict[str, dict[str, float | bool]] = {}
    for protocol in fine:
        if protocol not in coarse:
            continue
        energy_delta = abs(float(fine[protocol]["final_energy"]) - float(coarse[protocol]["final_energy"]))
        fidelity_delta = abs(
            float(fine[protocol]["ground_state_fidelity"])
            - float(coarse[protocol]["ground_state_fidelity"])
        )
        protocols[protocol] = {
            "energy_delta": energy_delta,
            "fidelity_delta": fidelity_delta,
            "pass": energy_delta <= energy_atol and fidelity_delta <= fidelity_atol,
        }
    passed = bool(protocols) and all(bool(row["pass"]) for row in protocols.values())
    return {
        "status": "pass" if passed else "fail",
        "energy_atol": float(energy_atol),
        "fidelity_atol": float(fidelity_atol),
        "protocols": protocols,
    }


def assess_statevector_agreement(
    mps_results: Mapping[str, Mapping[str, object]],
    statevector_results: Mapping[str, Mapping[str, object]],
    *,
    energy_atol: float,
    fidelity_atol: float,
) -> dict[str, object]:
    protocols: dict[str, dict[str, float | bool]] = {}
    for protocol, mps_row in mps_results.items():
        reference = statevector_results.get(protocol)
        if reference is None:
            continue
        energy_delta = abs(float(mps_row["final_energy"]) - float(reference["final_energy"]))
        fidelity_delta = abs(
            float(mps_row["ground_state_fidelity"])
            - float(reference["ground_state_fidelity"])
        )
        protocols[protocol] = {
            "energy_delta": energy_delta,
            "fidelity_delta": fidelity_delta,
            "pass": energy_delta <= energy_atol and fidelity_delta <= fidelity_atol,
        }
    passed = bool(protocols) and all(bool(row["pass"]) for row in protocols.values())
    return {
        "status": "pass" if passed else "fail",
        "energy_atol": float(energy_atol),
        "fidelity_atol": float(fidelity_atol),
        "protocols": protocols,
    }


def statevector_results_for_learned_terms(
    payload: Mapping[str, object],
    *,
    learned_terms: int,
    learned_scale: float,
) -> dict[str, Mapping[str, object]]:
    raw_results = payload.get("results", {})
    if not isinstance(raw_results, dict):
        raise TypeError("statevector reference results must be a JSON object.")
    results = {
        str(name): row
        for name, row in raw_results.items()
        if isinstance(row, dict)
    }
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


def validation_certification(
    *,
    convergence: Mapping[str, object],
    statevector_agreement: Mapping[str, object],
    require_convergence: bool,
    require_statevector: bool,
) -> dict[str, object]:
    gates: list[tuple[str, Mapping[str, object]]] = []
    if require_convergence:
        gates.append(("mps_convergence", convergence))
    if require_statevector:
        gates.append(("statevector_agreement", statevector_agreement))
    statuses = [str(gate.get("status", "not_tested")) for _, gate in gates]
    if not statuses or "not_tested" in statuses:
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
    integer_keys = ("steps", "max_bond", "learned_terms")
    float_keys = ("cutoff", "coefficient_threshold")
    string_keys = ("operator_grouping",)
    for case in previous_resolutions:
        if not isinstance(case, dict):
            continue
        previous_settings = case.get("settings", {})
        if not isinstance(previous_settings, dict):
            continue
        if any(int(previous_settings.get(key, -1)) != int(settings[key]) for key in integer_keys):
            continue
        if any(
            not np.isclose(
                float(previous_settings.get(key, np.nan)),
                float(settings[key]),
                rtol=0.0,
                atol=max(1.0e-16, abs(float(settings[key])) * 1.0e-12),
            )
            for key in float_keys
        ):
            continue
        if any(str(previous_settings.get(key, "")) != str(settings[key]) for key in string_keys):
            continue
        results = case.get("results", {})
        if isinstance(results, dict) and isinstance(results.get(protocol), dict):
            return dict(results[protocol])
    return None


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return payload


def _resolve_path(raw: object, *, base: Path) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else base / path


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
            row[f"{metric}_quotient_vs_no_cd"] = float(row[metric]) / max(float(baseline[metric]), 1.0e-15)


def _save_progress(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scalable MPS validation of sparse counterdiabatic protocols.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trained-run", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max-bond", type=int, default=None)
    parser.add_argument("--learned-terms", type=int, default=None)
    parser.add_argument("--cutoff", type=float, default=None)
    parser.add_argument("--coefficient-threshold", type=float, default=None)
    parser.add_argument("--operator-grouping", choices=("pauli_term", "support"), default=None)
    parser.add_argument("--protocols", type=str, default=None)
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

    n_qubits = int(parameters.get("num_qubits", 0))
    total_time = float(parameters.get("T", 1.0))
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

    trained_run_raw = args.trained_run or validation.get("trained_run")
    if trained_run_raw is None:
        raise ValueError("tensor_network_validation.trained_run must identify the retained checkpoint directory.")
    trained_run = _resolve_path(trained_run_raw, base=run_root)
    coefficient_path = trained_run / "Models_Data" / "final_agp_coefficients.pt"
    if not coefficient_path.is_file():
        raise FileNotFoundError(f"Missing trained AGP coefficients: {coefficient_path}")

    configured_cases = validation.get("resolutions", [])
    if not isinstance(configured_cases, list) or not configured_cases:
        configured_cases = [{}]
    if any(value is not None for value in (args.steps, args.max_bond, args.learned_terms, args.cutoff)):
        configured_cases = [{}]
    protocols_raw = args.protocols or validation.get(
        "protocols", ["no_cd", "kipu_dqfm_l1", "learned_sparse_agp"]
    )
    if isinstance(protocols_raw, str):
        protocols = tuple(item.strip() for item in protocols_raw.split(",") if item.strip())
    else:
        protocols = tuple(str(item) for item in protocols_raw)
    max_requested_terms = max(
        int(args.learned_terms or row.get("learned_terms", validation.get("learned_terms", 256)))
        for row in configured_cases
        if isinstance(row, dict)
    )
    learned_full = learned_term_selection(coefficient_path, max_requested_terms)

    output_dir = args.output_dir or _resolve_path(validation.get("output_dir", "mps_validation"), base=trained_run)
    if not output_dir.is_absolute():
        output_dir = run_root / output_dir
    data_dir = output_dir / "Models_Data"
    images_dir = output_dir / "Images"
    summary_path = data_dir / "mps_physical_validation_summary.json"
    previous_resolutions: object = []
    if summary_path.is_file():
        previous_payload = _load_json(summary_path)
        if (
            int(previous_payload.get("n_qubits", -1)) == n_qubits
            and str(previous_payload.get("coefficient_path", "")) == str(coefficient_path)
        ):
            previous_resolutions = previous_payload.get("resolution_results", [])
    payload: dict[str, object] = {
        "description": (
            "Matrix-product-state dynamical validation using a symmetric Pauli-product formula, "
            "bounded-bond compression, exact diagonal final-energy contractions, and exact product-state overlap."
        ),
        "backend": "quimb_mps",
        "n_qubits": n_qubits,
        "total_time": total_time,
        "trained_run": str(trained_run),
        "coefficient_path": str(coefficient_path),
        "ground_energy": ground_energy,
        "ground_bitstring": ground_bitstring,
        "protocols": list(protocols),
        "resolution_results": [],
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
        case_payload: dict[str, object] = {
            "name": str(case.get("name", f"resolution_{case_index + 1}")),
            "settings": {
                "steps": steps,
                "max_bond": max_bond,
                "learned_terms": int(learned["selected_terms"]),
                "retained_rms_norm_fraction": float(learned["retained_rms_norm_fraction"]),
                "cutoff": cutoff,
                "coefficient_threshold": coefficient_threshold,
                "operator_grouping": operator_grouping,
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
    payload["results"] = final_results
    convergence: dict[str, object] = {"status": "not_tested", "reason": "Only one resolution was run."}
    if len(resolution_results) >= 2:
        convergence = assess_mps_convergence(
            resolution_results[-2]["results"],  # type: ignore[arg-type]
            final_results,  # type: ignore[arg-type]
            energy_atol=float(validation.get("convergence_energy_atol", 0.05)),
            fidelity_atol=float(validation.get("convergence_fidelity_atol", 0.01)),
        )
    payload["convergence"] = convergence

    statevector_agreement: dict[str, object] = {
        "status": "not_tested",
        "reason": "No statevector reference was configured.",
    }
    statevector_reference = validation.get("statevector_reference")
    if statevector_reference:
        reference_payload = _load_json(_resolve_path(statevector_reference, base=run_root))
        final_settings = resolution_results[-1]["settings"]
        reference_results = statevector_results_for_learned_terms(
            reference_payload,
            learned_terms=int(final_settings["learned_terms"]),  # type: ignore[index]
            learned_scale=float(validation.get("learned_scale", 1.0)),
        )
        statevector_agreement = assess_statevector_agreement(
            final_results,  # type: ignore[arg-type]
            reference_results,  # type: ignore[arg-type]
            energy_atol=float(validation.get("statevector_energy_atol", 0.05)),
            fidelity_atol=float(validation.get("statevector_fidelity_atol", 0.01)),
        )
    payload["statevector_agreement"] = statevector_agreement
    payload["certification"] = validation_certification(
        convergence=convergence,
        statevector_agreement=statevector_agreement,
        require_convergence=len(resolution_results) >= 2,
        require_statevector=bool(statevector_reference),
    )
    _save_progress(summary_path, payload)
    images_dir.mkdir(parents=True, exist_ok=True)
    plot_physical_comparison_table(images_dir, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
