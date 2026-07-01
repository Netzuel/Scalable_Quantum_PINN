"""Generate Pauli-coordinate Hamiltonian pairs from Qiskit operators.

The training code consumes sparse Pauli dictionaries, not dense matrices. This
tool creates the same ``pauli_hamiltonian_pair_v1`` JSON format from a Qiskit
``SparsePauliOp`` produced by Qiskit Nature.

Chemistry example, when ``qiskit-nature`` and ``pyscf`` are installed:

    conda run -n torch-mps python tools/generate_qiskit_pauli_hamiltonian.py \
        chemistry \
        --system Hidrogen \
        --distance 1.0 \
        --atom "H 0 0 0; H 0 0 1.0" \
        --basis sto3g \
        --mapper parity \
        --taper \
        --target-qubits 2 \
        --include-nuclear-repulsion \
        --update-index

For a molecular problem, the number of qubits is determined by the active
space, fermion-to-qubit mapper, and optional symmetry tapering. The
``--target-qubits`` option validates the result; it does not invent a new
physical Hamiltonian if the chosen chemistry settings produce a different
qubit count.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


PAULI_ALPHABET = ("I", "X", "Y", "Z")
DIAGONAL_ALPHABET = {"I", "Z"}
DEFAULT_ORGANIZED_DIR = Path("Hamiltonians_to_use/pauli_decompositions")
COEFFICIENT_CONVENTION = {
    "operator_expansion": "H = sum_P C_P P",
    "coefficient_formula": "Qiskit SparsePauliOp coefficients, with labels in matrix tensor-product order",
    "pauli_alphabet": list(PAULI_ALPHABET),
    "pauli_basis": "{I, X, Y, Z}^{tensor n_qubits}",
    "stored_terms": "Only coefficients with abs(C_P) > drop_tol are stored.",
}


def distance_token(distance: str | float) -> str:
    return str(distance).replace(".", "_")


def pauli_order(label: str) -> int:
    return sum(symbol != "I" for symbol in label)


def sort_terms(terms: Mapping[str, complex]) -> dict[str, complex]:
    return dict(sorted(terms.items(), key=lambda item: (pauli_order(item[0]), item[0])))


def infer_n_qubits(terms: Mapping[str, complex]) -> int:
    lengths = {len(label) for label in terms}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent Pauli-string lengths: {sorted(lengths)}")
    return lengths.pop()


def encode_complex(value: complex) -> list[float]:
    value = complex(value)
    return [float(value.real), float(value.imag)]


def encode_terms(terms: Mapping[str, complex]) -> dict[str, list[float]]:
    return {label: encode_complex(coeff) for label, coeff in sort_terms(terms).items()}


def terms_by_order(terms: Mapping[str, complex], n_qubits: int) -> dict[str, int]:
    counts = {str(order): 0 for order in range(n_qubits + 1)}
    for label in terms:
        counts[str(pauli_order(label))] += 1
    return {order: count for order, count in counts.items() if count > 0}


def sparse_pauli_op_to_terms(operator: Any, drop_tol: float) -> dict[str, complex]:
    """Convert a Qiskit SparsePauliOp-like object to a sparse term dict."""

    if hasattr(operator, "simplify"):
        operator = operator.simplify(atol=drop_tol)
    if not hasattr(operator, "to_list"):
        raise TypeError("Expected a Qiskit SparsePauliOp-like object with a to_list() method.")

    terms: defaultdict[str, complex] = defaultdict(complex)
    for label, coeff in operator.to_list():
        label = str(label).upper()
        invalid = sorted(set(label) - set(PAULI_ALPHABET))
        if invalid:
            raise ValueError(f"Invalid Pauli symbols {invalid} in Qiskit label {label!r}.")
        terms[label] += complex(coeff)

    return sort_terms({label: coeff for label, coeff in terms.items() if abs(coeff) > drop_tol})


def diagonal_projection_terms(terms: Mapping[str, complex]) -> dict[str, complex]:
    """Return the computational-basis diagonal projection.

    In Pauli coordinates this is exactly the subset containing only ``I`` and
    ``Z`` factors. This is the sparse analogue of taking the diagonal of the
    endpoint matrix, without constructing that matrix.
    """

    return sort_terms({label: coeff for label, coeff in terms.items() if set(label) <= DIAGONAL_ALPHABET})


def add_identity_shift(terms: Mapping[str, complex], n_qubits: int, shift: float, drop_tol: float) -> dict[str, complex]:
    if abs(shift) <= drop_tol:
        return sort_terms(dict(terms))
    shifted: defaultdict[str, complex] = defaultdict(complex)
    shifted.update(terms)
    shifted["I" * n_qubits] += complex(float(shift), 0.0)
    return sort_terms({label: coeff for label, coeff in shifted.items() if abs(coeff) > drop_tol})


def endpoint_payload(
    *,
    role: str,
    tau: float,
    source_key: str,
    terms: Mapping[str, complex],
    n_qubits: int,
) -> dict[str, object]:
    return {
        "role": role,
        "tau": tau,
        "source_key": source_key,
        "terms": encode_terms(terms),
        "term_count": len(terms),
        "terms_by_order": terms_by_order(terms, n_qubits),
    }


def build_pair_payload(
    *,
    system: str,
    n_qubits: int,
    distance: str,
    initial_terms: Mapping[str, complex],
    final_terms: Mapping[str, complex],
    source: Mapping[str, object],
    drop_tol: float,
) -> dict[str, object]:
    return {
        "format": "pauli_hamiltonian_pair_v1",
        "system": system,
        "n_qubits": n_qubits,
        "distance": distance,
        "basis": {
            "size": 4**n_qubits,
            "max_order": n_qubits,
            "ordering": "lexicographic product over I, X, Y, Z",
            "alphabet": list(PAULI_ALPHABET),
        },
        "coefficient_convention": COEFFICIENT_CONVENTION,
        "source": dict(source),
        "drop_tol": drop_tol,
        "hamiltonians": {
            "initial": endpoint_payload(
                role="H_diagonal",
                tau=0.0,
                source_key="diagonal_projection_of_final",
                terms=initial_terms,
                n_qubits=n_qubits,
            ),
            "final": endpoint_payload(
                role="H_problem",
                tau=1.0,
                source_key="qiskit_sparse_pauli_op",
                terms=final_terms,
                n_qubits=n_qubits,
            ),
        },
    }


def pair_output_path(organized_dir: Path, system: str, n_qubits: int, distance: str) -> Path:
    return organized_dir / system / f"{n_qubits}_qubits" / f"distance_{distance}.json"


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def update_index(index_path: Path, pair_payload: Mapping[str, object], pair_path: Path, drop_tol: float) -> None:
    organized_dir = index_path.parent
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
    else:
        index = {
            "format": "pauli_hamiltonian_index_v1",
            "drop_tol": drop_tol,
            "coefficient_convention": COEFFICIENT_CONVENTION,
            "pair_count": 0,
            "pairs": {},
        }

    system = str(pair_payload["system"])
    n_qubits = int(pair_payload["n_qubits"])
    distance = str(pair_payload["distance"])
    pair_id = f"{system}_{n_qubits}_qubits_{distance}"
    hamiltonians = pair_payload["hamiltonians"]
    if not isinstance(hamiltonians, Mapping):
        raise TypeError("Pair payload has an invalid hamiltonians block.")
    initial = hamiltonians["initial"]
    final = hamiltonians["final"]
    if not isinstance(initial, Mapping) or not isinstance(final, Mapping):
        raise TypeError("Pair payload has invalid endpoint blocks.")
    try:
        relative_file = str(pair_path.relative_to(organized_dir))
    except ValueError:
        relative_file = str(pair_path)

    pairs = index.setdefault("pairs", {})
    pairs[pair_id] = {
        "system": system,
        "n_qubits": n_qubits,
        "distance": distance,
        "file": relative_file,
        "basis_size": 4**n_qubits,
        "initial_key": initial["source_key"],
        "final_key": final["source_key"],
        "initial_term_count": initial["term_count"],
        "final_term_count": final["term_count"],
    }
    index["pair_count"] = len(pairs)
    write_json(index_path, index)


def parse_active_electrons(value: str | None) -> int | tuple[int, int] | None:
    if value is None:
        return None
    if "," in value:
        left, right = value.split(",", maxsplit=1)
        return (int(left), int(right))
    return int(value)


def mapper_from_name(name: str) -> Any:
    try:
        from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper, ParityMapper
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("Install qiskit-nature to generate chemistry Hamiltonians.") from exc

    normalized = name.lower().replace("-", "_")
    if normalized in {"jw", "jordan_wigner", "jordanwigner"}:
        return JordanWignerMapper()
    if normalized == "parity":
        return ParityMapper()
    if normalized in {"bk", "bravyi_kitaev", "bravyikitaev"}:
        return BravyiKitaevMapper()
    raise ValueError(f"Unsupported mapper {name!r}. Use jordan_wigner, parity, or bravyi_kitaev.")


def build_qiskit_nature_operator(args: argparse.Namespace) -> tuple[Any, dict[str, object], float]:
    try:
        from qiskit_nature.second_q.drivers import PySCFDriver
        from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
        from qiskit_nature.units import DistanceUnit
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError(
            "Chemistry generation requires optional dependencies: qiskit-nature and pyscf. "
            "Install the repository's chemistry extra or install them in torch-mps."
        ) from exc

    driver = PySCFDriver(
        atom=args.atom,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        unit=DistanceUnit.ANGSTROM,
    )
    problem = driver.run()

    active_electrons = parse_active_electrons(args.active_electrons)
    active_spatial_orbitals = args.active_spatial_orbitals
    if active_spatial_orbitals is None and args.target_qubits is not None and active_electrons is not None and not args.taper:
        if args.target_qubits % 2 != 0:
            raise ValueError("Without tapering, a molecular active space needs an even target qubit count.")
        active_spatial_orbitals = args.target_qubits // 2
    if active_spatial_orbitals is not None:
        if active_electrons is None:
            raise ValueError("--active-electrons is required when an active space is requested.")
        transformer = ActiveSpaceTransformer(active_electrons, active_spatial_orbitals)
        problem = transformer.transform(problem)

    second_q_op = problem.hamiltonian.second_q_op()
    mapper = mapper_from_name(args.mapper)
    if args.taper:
        mapper = problem.get_tapered_mapper(mapper)
    qubit_op = mapper.map(second_q_op)

    nuclear_repulsion = getattr(problem.hamiltonian, "nuclear_repulsion_energy", None)
    nuclear_shift = float(nuclear_repulsion) if args.include_nuclear_repulsion and nuclear_repulsion is not None else 0.0
    source = {
        "generator": "tools/generate_qiskit_pauli_hamiltonian.py",
        "backend": "qiskit_nature_pyscf",
        "atom": args.atom,
        "basis": args.basis,
        "charge": args.charge,
        "spin": args.spin,
        "mapper": args.mapper,
        "tapered": bool(args.taper),
        "active_electrons": active_electrons,
        "active_spatial_orbitals": active_spatial_orbitals,
        "nuclear_repulsion_included": bool(args.include_nuclear_repulsion),
        "nuclear_repulsion_shift": nuclear_shift,
        "initial_construction": "computational-basis diagonal projection, keeping only I/Z Pauli strings",
        "dense_usage": "None; Qiskit SparsePauliOp terms are exported directly.",
    }
    return qubit_op, source, nuclear_shift


def run_chemistry(args: argparse.Namespace) -> None:
    qubit_op, source, nuclear_shift = build_qiskit_nature_operator(args)
    final_terms = sparse_pauli_op_to_terms(qubit_op, args.drop_tol)
    n_qubits = infer_n_qubits(final_terms)
    final_terms = add_identity_shift(final_terms, n_qubits, nuclear_shift, args.drop_tol)
    initial_terms = diagonal_projection_terms(final_terms)

    if args.target_qubits is not None and n_qubits != args.target_qubits:
        raise ValueError(
            f"Generated {n_qubits} qubits, but --target-qubits requested {args.target_qubits}. "
            "Change the active space, mapper, or tapering settings."
        )

    distance = distance_token(args.distance)
    payload = build_pair_payload(
        system=args.system,
        n_qubits=n_qubits,
        distance=distance,
        initial_terms=initial_terms,
        final_terms=final_terms,
        source=source,
        drop_tol=args.drop_tol,
    )
    output_path = args.output or pair_output_path(args.organized_dir, args.system, n_qubits, distance)
    write_json(output_path, payload)
    if args.update_index:
        update_index(args.organized_dir / "index.json", payload, output_path, args.drop_tol)
    print(
        f"wrote {output_path} with {len(initial_terms)} initial terms, "
        f"{len(final_terms)} final terms, n_qubits={n_qubits}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    chemistry = subparsers.add_parser("chemistry", help="Generate a molecular Hamiltonian with Qiskit Nature.")
    chemistry.add_argument("--system", required=True, help="System name stored in the output JSON, e.g. Hidrogen.")
    chemistry.add_argument("--distance", required=True, help="Distance token/value stored in the output JSON.")
    chemistry.add_argument("--atom", required=True, help='PySCF atom string, e.g. "H 0 0 0; H 0 0 1.0".')
    chemistry.add_argument("--basis", default="sto3g", help="Basis set passed to PySCFDriver.")
    chemistry.add_argument("--charge", type=int, default=0)
    chemistry.add_argument("--spin", type=int, default=0, help="2S value passed to PySCFDriver; singlet is 0.")
    chemistry.add_argument("--mapper", default="parity", help="jordan_wigner, parity, or bravyi_kitaev.")
    chemistry.add_argument("--taper", action="store_true", help="Use problem.get_tapered_mapper(mapper).")
    chemistry.add_argument("--target-qubits", type=int, default=None, help="Validate the generated qubit count.")
    chemistry.add_argument("--active-electrons", default=None, help='Active electrons, e.g. "2" or "1,1".')
    chemistry.add_argument("--active-spatial-orbitals", type=int, default=None)
    chemistry.add_argument("--include-nuclear-repulsion", action="store_true", help="Add nuclear repulsion to II.")
    chemistry.add_argument("--drop-tol", type=float, default=1e-10)
    chemistry.add_argument("--organized-dir", type=Path, default=DEFAULT_ORGANIZED_DIR)
    chemistry.add_argument("--output", type=Path, default=None)
    chemistry.add_argument("--update-index", action="store_true")
    chemistry.set_defaults(func=run_chemistry)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.exit(status=2, message=f"error: {exc}\n")


if __name__ == "__main__":
    main()
