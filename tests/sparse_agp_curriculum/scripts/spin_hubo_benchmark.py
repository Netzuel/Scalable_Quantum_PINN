#!/usr/bin/env python3
"""Utilities for transverse-field to diagonal spin-HUBO benchmarks."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import SparsePauliOperator  # noqa: E402


SpinPolynomial = Mapping[tuple[int, ...], float]


@dataclass(frozen=True)
class SpinGroundReference:
    """Exact computational-basis ground reference for a spin polynomial."""

    ground_energy: float
    ground_bitstrings: tuple[str, ...]
    ground_state_degeneracy: int
    method: str = "exact_walsh_hadamard_enumeration"


def _validated_polynomial(
    polynomial: SpinPolynomial,
    *,
    n_qubits: int | None = None,
) -> dict[tuple[int, ...], float]:
    if n_qubits is not None and n_qubits < 1:
        raise ValueError("n_qubits must be positive.")
    validated: dict[tuple[int, ...], float] = {}
    for raw_support, raw_coefficient in polynomial.items():
        support = tuple(raw_support)
        if any(not isinstance(index, int) or isinstance(index, bool) for index in support):
            raise TypeError(f"Spin support indices must be integers: {raw_support!r}.")
        if any(left >= right for left, right in zip(support, support[1:])):
            raise ValueError(f"Spin support must be strictly increasing: {support!r}.")
        if support and support[0] < 0:
            raise ValueError(f"Spin support contains a negative index: {support!r}.")
        if n_qubits is not None and support and support[-1] >= n_qubits:
            raise ValueError(f"Spin support {support!r} is outside n_qubits={n_qubits}.")
        coefficient = float(raw_coefficient)
        if not math.isfinite(coefficient):
            raise ValueError(f"Spin coefficient for {support!r} must be finite.")
        validated[support] = coefficient
    if not validated:
        raise ValueError("Spin polynomial must contain at least one term.")
    return validated


def load_spin_polynomial(path: str | Path) -> dict[tuple[int, ...], float]:
    """Load the tuple-keyed JSON convention used by ``HAMILTONIANS_SPIN``."""

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    polynomial: dict[tuple[int, ...], float] = {}
    for raw_support, coefficient in payload.items():
        try:
            support = ast.literal_eval(raw_support)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid spin support key {raw_support!r} in {path}.") from exc
        if not isinstance(support, tuple):
            raise ValueError(f"Spin support key must decode to a tuple: {raw_support!r}.")
        polynomial[support] = coefficient
    return _validated_polynomial(polynomial)


def spin_polynomial_to_pauli_pair(
    polynomial: SpinPolynomial,
    *,
    n_qubits: int,
    transverse_field: float = 1.0,
) -> tuple[SparsePauliOperator, SparsePauliOperator]:
    """Map ``s_i`` directly to the computational-basis eigenvalue of ``Z_i``."""

    terms = _validated_polynomial(polynomial, n_qubits=n_qubits)
    transverse_field = float(transverse_field)
    if not math.isfinite(transverse_field) or transverse_field <= 0.0:
        raise ValueError("transverse_field must be finite and positive.")

    initial: dict[str, float] = {}
    for site in range(n_qubits):
        label = ["I"] * n_qubits
        label[site] = "X"
        initial["".join(label)] = -transverse_field

    final: dict[str, float] = {}
    for support, coefficient in terms.items():
        label = ["I"] * n_qubits
        for site in support:
            label[site] = "Z"
        final["".join(label)] = coefficient
    return (
        SparsePauliOperator(initial, n_qubits=n_qubits),
        SparsePauliOperator(final, n_qubits=n_qubits),
    )


def evaluate_spin_energy(polynomial: SpinPolynomial, spins: Sequence[int]) -> float:
    """Evaluate ``sum_S c_S product_(i in S) s_i`` in the spin convention."""

    terms = _validated_polynomial(polynomial, n_qubits=len(spins))
    if any(spin not in (-1, 1) for spin in spins):
        raise ValueError("spins must contain only -1 and +1.")
    return float(
        sum(
            coefficient * math.prod(spins[index] for index in support)
            for support, coefficient in terms.items()
        )
    )


def exact_walsh_ground_reference(
    polynomial: SpinPolynomial,
    *,
    n_qubits: int,
    max_qubits: int = 24,
    atol: float = 1.0e-11,
) -> SpinGroundReference:
    """Evaluate every diagonal energy with an in-place Walsh-Hadamard transform.

    The explicit ``max_qubits`` bound keeps this test-side exact oracle within
    predictable memory. It never constructs a dense quantum Hamiltonian.
    """

    if n_qubits > max_qubits:
        raise ValueError(f"Exact Walsh enumeration supports at most {max_qubits} qubits.")
    terms = _validated_polynomial(polynomial, n_qubits=n_qubits)
    size = 1 << n_qubits
    energies = np.zeros(size, dtype=np.float64)
    for support, coefficient in terms.items():
        mask = sum(1 << site for site in support)
        energies[mask] += coefficient

    block = 1
    while block < size:
        view = energies.reshape(-1, 2 * block)
        left = view[:, :block].copy()
        right = view[:, block:].copy()
        view[:, :block] = left + right
        view[:, block:] = left - right
        block *= 2

    ground_energy = float(np.min(energies))
    ground_indices = np.flatnonzero(np.isclose(energies, ground_energy, rtol=0.0, atol=atol))
    bitstrings = tuple(
        "".join(str((int(index) >> site) & 1) for site in range(n_qubits))
        for index in ground_indices
    )
    return SpinGroundReference(
        ground_energy=ground_energy,
        ground_bitstrings=bitstrings,
        ground_state_degeneracy=len(bitstrings),
    )


def _complex_terms_payload(terms: Mapping[str, complex]) -> dict[str, list[float]]:
    return {
        label: [float(complex(coefficient).real), float(complex(coefficient).imag)]
        for label, coefficient in sorted(terms.items())
    }


def _terms_by_order(terms: SpinPolynomial) -> dict[str, int]:
    counts: dict[str, int] = {}
    for support in terms:
        key = str(len(support))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_benchmark_assets(
    *,
    source_instance: str | Path,
    output_root: str | Path,
    system: str,
    distance: str,
) -> dict[str, Path]:
    """Create a self-contained sparse-Pauli benchmark from one spin instance."""

    source_instance = Path(source_instance).resolve()
    output_root = Path(output_root).resolve()
    source_poly = source_instance / "poly.json"
    source_meta = source_instance / "meta.json"
    if not source_poly.is_file() or not source_meta.is_file():
        raise FileNotFoundError("source_instance must contain poly.json and meta.json.")

    meta = json.loads(source_meta.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise TypeError(f"{source_meta} must contain a JSON object.")
    n_qubits = int(meta.get("n_qubits", 0))
    polynomial = load_spin_polynomial(source_poly)
    h0, h1 = spin_polynomial_to_pauli_pair(polynomial, n_qubits=n_qubits)
    reference = exact_walsh_ground_reference(polynomial, n_qubits=n_qubits)

    problem_data = output_root / "problem_data"
    snapshot_poly = problem_data / "source_spin_poly.json"
    snapshot_meta = problem_data / "source_meta.json"
    problem_data.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_poly, snapshot_poly)
    shutil.copyfile(source_meta, snapshot_meta)

    source_hash = hashlib.sha256(source_poly.read_bytes()).hexdigest()
    supplied_minimum = meta.get("min_energy")
    supplied_matches_exact = (
        supplied_minimum is not None
        and math.isclose(
            float(supplied_minimum),
            reference.ground_energy,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        )
    )
    manifest_path = problem_data / "source_manifest.json"
    manifest = {
        "format": "spin_hubo_source_manifest_v1",
        "source_instance": str(source_instance),
        "source_name": meta.get("name", source_instance.name),
        "source_run": meta.get("source_run"),
        "source_hamiltonian_convention": meta.get("hamiltonian_convention", "spin_pm1"),
        "source_bitstring_convention": meta.get("bitstring_convention", "spin_pm1"),
        "source_energy_convention": meta.get("energy_convention", "spin_pm1"),
        "source_poly_sha256": source_hash,
        "source_reported_min_energy": supplied_minimum,
        "source_reported_minimum_matches_exact": supplied_matches_exact,
        "n_qubits": n_qubits,
        "n_terms": len(polynomial),
        "terms_by_order": _terms_by_order(polynomial),
        "conversion": "s_i maps directly to the Z_i eigenvalue; +1 maps to |0> and -1 maps to |1>",
    }
    _write_json(manifest_path, manifest)

    pair_path = (
        output_root
        / "Hamiltonians_to_use"
        / "pauli_decompositions"
        / system
        / f"{n_qubits}_qubits"
        / f"distance_{distance}.json"
    )
    pair_payload = {
        "format": "pauli_hamiltonian_pair_v1",
        "system": system,
        "n_qubits": n_qubits,
        "distance": distance,
        "drop_tol": 1.0e-14,
        "basis": {
            "alphabet": ["I", "X", "Y", "Z"],
            "max_order": n_qubits,
            "ordering": "matrix tensor-product order; label character i is qubit i",
            "size": 4**n_qubits,
        },
        "coefficient_convention": {
            "operator_expansion": "H = sum_P C_P P",
            "spin_mapping": "s_i = eigenvalue(Z_i)",
            "stored_terms": "Only source and transverse-driver nonzero terms are stored.",
        },
        "source": {
            "backend": "spin_hubo_json_to_sparse_pauli",
            "generator": "tests/sparse_agp_curriculum/scripts/spin_hubo_benchmark.py",
            "source_poly_sha256": source_hash,
            "source_manifest": str(manifest_path.relative_to(output_root)),
            "dense_usage": "None; sparse Pauli labels are exported directly.",
        },
        "hamiltonians": {
            "initial": {
                "role": "H_initial",
                "source_key": "analytic_transverse_driver_sparse_pauli",
                "tau": 0.0,
                "term_count": len(h0.terms),
                "terms_by_order": {"1": len(h0.terms)},
                "terms": _complex_terms_payload(h0.terms),
            },
            "final": {
                "role": "H_final",
                "source_key": "tracked_spin_hubo_source_snapshot",
                "tau": 1.0,
                "term_count": len(h1.terms),
                "terms_by_order": _terms_by_order(polynomial),
                "terms": _complex_terms_payload(h1.terms),
            },
        },
    }
    _write_json(pair_path, pair_payload)

    reference_path = problem_data / "ground_reference.json"
    reference_payload = {
        "format": "diagonal_spin_ground_reference_v1",
        "description": "Exact exhaustive computational-basis reference via Walsh-Hadamard evaluation.",
        "solutions": [
            {
                "q": n_qubits,
                "method": reference.method,
                "ground_energy": reference.ground_energy,
                "ground_bitstrings": list(reference.ground_bitstrings),
                "ground_state_degeneracy": reference.ground_state_degeneracy,
                "ground_bitstrings_truncated": False,
                "exact": True,
            }
        ],
    }
    _write_json(reference_path, reference_payload)
    return {
        "hamiltonian_pair": pair_path,
        "ground_reference": reference_path,
        "source_manifest": manifest_path,
        "source_poly_snapshot": snapshot_poly,
        "source_meta_snapshot": snapshot_meta,
    }
