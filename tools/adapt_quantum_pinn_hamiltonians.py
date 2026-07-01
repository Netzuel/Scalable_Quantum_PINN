"""Convert retained Quantum_PINN dense Hamiltonians to Pauli coefficients.

This is a one-time data adapter for the copied source file
``Hamiltonians_to_use/Hamiltonians.h5``. Training code should consume the JSON
outputs from this script, not the dense source matrices.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from itertools import product
from pathlib import Path

import h5py
import numpy as np


PAULI_MATRICES = {
    "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}
KEY_RE = re.compile(r"(?P<system>.+)_(?P<n_qubits>\d+)_qubits_(?P<role>H_hf|H_prob)_(?P<distance>\d+_\d+)$")
ROLE_TO_ENDPOINT = {"H_hf": "initial", "H_prob": "final"}
ENDPOINT_TAU = {"initial": 0.0, "final": 1.0}
COEFFICIENT_CONVENTION = {
    "operator_expansion": "H = sum_P C_P P",
    "coefficient_formula": "C_P = Tr(P H) / 2**n_qubits",
    "pauli_alphabet": ["I", "X", "Y", "Z"],
    "pauli_basis": "{I, X, Y, Z}^{tensor n_qubits}",
    "stored_terms": "Only coefficients with abs(C_P) > drop_tol are stored.",
}


def all_pauli_labels(n_qubits: int) -> list[str]:
    return ["".join(item) for item in product(("I", "X", "Y", "Z"), repeat=n_qubits)]


def pauli_matrix(label: str) -> np.ndarray:
    out = PAULI_MATRICES[label[0]]
    for symbol in label[1:]:
        out = np.kron(out, PAULI_MATRICES[symbol])
    return out


def decompose_dense_matrix(matrix: np.ndarray, n_qubits: int, drop_tol: float) -> dict[str, list[float]]:
    dim = 2**n_qubits
    if matrix.shape != (dim, dim):
        raise ValueError(f"Expected shape {(dim, dim)} for {n_qubits} qubits, got {matrix.shape}.")
    matrix = np.asarray(matrix, dtype=np.complex128)
    terms: dict[str, list[float]] = {}
    for label in all_pauli_labels(n_qubits):
        basis_matrix = pauli_matrix(label)
        coeff = np.trace(basis_matrix @ matrix) / dim
        if abs(coeff) > drop_tol:
            terms[label] = [float(np.real(coeff)), float(np.imag(coeff))]
    return terms


def _terms_by_order(terms: dict[str, list[float]], n_qubits: int) -> dict[str, int]:
    counts = {str(order): 0 for order in range(n_qubits + 1)}
    for label in terms:
        counts[str(sum(symbol != "I" for symbol in label))] += 1
    return {order: count for order, count in counts.items() if count > 0}


def _pair_file(system: str, n_qubits: int, distance: str) -> Path:
    return Path(system) / f"{n_qubits}_qubits" / f"distance_{distance}.json"


def _write_legacy_aggregate(records: dict[str, dict[str, object]], input_path: Path, output_path: Path, drop_tol: float) -> None:
    payload = {
        "format": "sparse_pauli_hamiltonians_v1",
        "source": str(input_path),
        "drop_tol": drop_tol,
        "coefficient_convention": COEFFICIENT_CONVENTION,
        "datasets": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_pair_files(
    records: dict[str, dict[str, object]],
    organized_dir: Path,
    input_path: Path,
    drop_tol: float,
) -> None:
    grouped: dict[tuple[str, int, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for key, record in records.items():
        grouped[(str(record["system"]), int(record["n_qubits"]), str(record["distance"]))][str(record["role"])] = {
            "key": key,
            **record,
        }

    index_pairs: dict[str, dict[str, object]] = {}
    for (system, n_qubits, distance), roles in sorted(grouped.items()):
        missing = sorted(set(ROLE_TO_ENDPOINT) - set(roles))
        if missing:
            raise ValueError(f"Missing endpoint role(s) {missing} for {system}, {n_qubits} qubits, distance {distance}.")
        relative_file = _pair_file(system, n_qubits, distance)
        pair_path = organized_dir / relative_file
        pair_path.parent.mkdir(parents=True, exist_ok=True)
        hamiltonians: dict[str, dict[str, object]] = {}
        source_keys: dict[str, str] = {}
        source_shapes: dict[str, list[int]] = {}
        for role, endpoint in ROLE_TO_ENDPOINT.items():
            record = roles[role]
            terms = record["terms"]
            source_keys[endpoint] = str(record["key"])
            source_shapes[endpoint] = list(record["source_shape"])  # type: ignore[arg-type]
            hamiltonians[endpoint] = {
                "role": role,
                "tau": ENDPOINT_TAU[endpoint],
                "source_key": record["key"],
                "terms": terms,
                "term_count": len(terms),  # type: ignore[arg-type]
                "terms_by_order": _terms_by_order(terms, n_qubits),  # type: ignore[arg-type]
            }
        pair_payload = {
            "format": "pauli_hamiltonian_pair_v1",
            "system": system,
            "n_qubits": n_qubits,
            "distance": distance,
            "basis": {
                "size": 4**n_qubits,
                "max_order": n_qubits,
                "ordering": "lexicographic product over I, X, Y, Z",
                "alphabet": ["I", "X", "Y", "Z"],
            },
            "coefficient_convention": COEFFICIENT_CONVENTION,
            "source": {
                "dense_hdf5": str(input_path),
                "source_keys": source_keys,
                "source_shapes": source_shapes,
                "dense_dtype": "float32",
                "dense_usage": "Used only by the one-time adapter; training consumes this Pauli-coordinate JSON.",
            },
            "drop_tol": drop_tol,
            "hamiltonians": hamiltonians,
        }
        with pair_path.open("w", encoding="utf-8") as handle:
            json.dump(pair_payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        pair_id = f"{system}_{n_qubits}_qubits_{distance}"
        index_pairs[pair_id] = {
            "system": system,
            "n_qubits": n_qubits,
            "distance": distance,
            "file": str(relative_file),
            "basis_size": 4**n_qubits,
            "initial_key": source_keys["initial"],
            "final_key": source_keys["final"],
            "initial_term_count": hamiltonians["initial"]["term_count"],
            "final_term_count": hamiltonians["final"]["term_count"],
        }

    index_payload = {
        "format": "pauli_hamiltonian_index_v1",
        "source_dense_hdf5": str(input_path),
        "drop_tol": drop_tol,
        "coefficient_convention": COEFFICIENT_CONVENTION,
        "pair_count": len(index_pairs),
        "pairs": index_pairs,
    }
    organized_dir.mkdir(parents=True, exist_ok=True)
    with (organized_dir / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(index_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def convert(input_path: Path, output_path: Path, organized_dir: Path, drop_tol: float) -> None:
    records: dict[str, dict[str, object]] = {}
    with h5py.File(input_path, "r") as h5:
        for key in sorted(h5.keys()):
            match = KEY_RE.match(key)
            if match is None:
                raise ValueError(f"Unexpected Hamiltonian key format: {key}")
            n_qubits = int(match.group("n_qubits"))
            terms = decompose_dense_matrix(np.array(h5[key]), n_qubits, drop_tol)
            records[key] = {
                "system": match.group("system"),
                "n_qubits": n_qubits,
                "role": match.group("role"),
                "distance": match.group("distance"),
                "source_shape": list(h5[key].shape),
                "terms": terms,
            }
            print(f"{key}: {len(terms)} Pauli terms")

    _write_legacy_aggregate(records, input_path, output_path, drop_tol)
    _write_pair_files(records, organized_dir, input_path, drop_tol)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("Hamiltonians_to_use/Hamiltonians.h5"))
    parser.add_argument("--output", type=Path, default=Path("Hamiltonians_to_use/Hamiltonians_pauli.json"))
    parser.add_argument(
        "--organized-dir",
        type=Path,
        default=Path("Hamiltonians_to_use/pauli_decompositions"),
        help="Directory for the Pauli-coordinate index and per-system pair files.",
    )
    parser.add_argument("--drop-tol", type=float, default=1e-8)
    args = parser.parse_args()
    convert(args.input, args.output, args.organized_dir, args.drop_tol)


if __name__ == "__main__":
    main()
