from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from generate_qiskit_pauli_hamiltonian import (  # noqa: E402
    DEFAULT_ORGANIZED_DIR,
    build_pair_payload,
    distance_token,
    pair_output_path,
    pauli_label,
    scaled_coefficient,
    sort_terms,
    update_index,
    write_json,
)


def driver_terms(*, n_qubits: int, x_field: float, drop_tol: float) -> dict[str, complex]:
    terms: defaultdict[str, complex] = defaultdict(complex)
    for site in range(n_qubits):
        coeff = -float(x_field)
        if abs(coeff) > drop_tol:
            terms[pauli_label(n_qubits, {site: "X"})] += complex(coeff, 0.0)
    return sort_terms({label: coeff for label, coeff in terms.items() if abs(coeff) > drop_tol})


def problem_terms(
    *,
    n_qubits: int,
    z_field: float,
    zz_coupling: float,
    field_gradient: float,
    coupling_gradient: float,
    periodic: bool,
    drop_tol: float,
) -> dict[str, complex]:
    terms: defaultdict[str, complex] = defaultdict(complex)
    for site in range(n_qubits):
        coeff = -scaled_coefficient(z_field, field_gradient, site, n_qubits)
        if abs(coeff) > drop_tol:
            terms[pauli_label(n_qubits, {site: "Z"})] += complex(coeff, 0.0)

    edges = [(site, site + 1) for site in range(n_qubits - 1)]
    if periodic:
        edges.append((n_qubits - 1, 0))
    for edge_index, (left, right) in enumerate(edges):
        coeff = -scaled_coefficient(zz_coupling, coupling_gradient, edge_index, len(edges))
        if abs(coeff) > drop_tol:
            terms[pauli_label(n_qubits, {left: "Z", right: "Z"})] += complex(coeff, 0.0)
    return sort_terms({label: coeff for label, coeff in terms.items() if abs(coeff) > drop_tol})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a driver-to-problem transverse-Ising Hamiltonian pair.")
    parser.add_argument("--system", default="TransverseIsingDriverProblem")
    parser.add_argument("--num-qubits", type=int, default=15)
    parser.add_argument("--distance", default="1.0")
    parser.add_argument("--x-field", type=float, default=1.0)
    parser.add_argument("--z-field", type=float, default=0.35)
    parser.add_argument("--zz-coupling", type=float, default=1.0)
    parser.add_argument("--field-gradient", type=float, default=0.15)
    parser.add_argument("--coupling-gradient", type=float, default=0.10)
    parser.add_argument("--periodic", action="store_true")
    parser.add_argument("--drop-tol", type=float, default=1e-10)
    parser.add_argument("--organized-dir", type=Path, default=DEFAULT_ORGANIZED_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--update-index", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_qubits < 2:
        raise ValueError("Use at least two qubits.")

    distance = distance_token(args.distance)
    initial_terms = driver_terms(
        n_qubits=args.num_qubits,
        x_field=args.x_field,
        drop_tol=args.drop_tol,
    )
    final_terms = problem_terms(
        n_qubits=args.num_qubits,
        z_field=args.z_field,
        zz_coupling=args.zz_coupling,
        field_gradient=args.field_gradient,
        coupling_gradient=args.coupling_gradient,
        periodic=bool(args.periodic),
        drop_tol=args.drop_tol,
    )
    source = {
        "generator": "scripts/build_driver_problem_hamiltonian.py",
        "backend": "analytic_driver_to_problem_transverse_ising",
        "num_qubits": args.num_qubits,
        "x_field": args.x_field,
        "z_field": args.z_field,
        "zz_coupling": args.zz_coupling,
        "field_gradient": args.field_gradient,
        "coupling_gradient": args.coupling_gradient,
        "periodic": bool(args.periodic),
        "initial_construction": "analytic transverse-field driver H_initial = -sum_i X_i",
        "final_construction": "analytic diagonal Ising problem with Z_i and nearest-neighbor Z_i Z_j terms",
        "dense_usage": "None; analytic Pauli strings are exported directly.",
    }
    payload = build_pair_payload(
        system=args.system,
        n_qubits=args.num_qubits,
        distance=distance,
        initial_terms=initial_terms,
        final_terms=final_terms,
        source=source,
        drop_tol=args.drop_tol,
        initial_source_key="analytic_transverse_driver_sparse_pauli",
        final_source_key="analytic_diagonal_ising_problem_sparse_pauli",
    )
    output_path = args.output or pair_output_path(args.organized_dir, args.system, args.num_qubits, distance)
    write_json(output_path, payload)
    if args.update_index:
        update_index(args.organized_dir / "index.json", payload, output_path, args.drop_tol)
    print(
        f"wrote {output_path} with {len(initial_terms)} initial terms, "
        f"{len(final_terms)} final terms, n_qubits={args.num_qubits}"
    )


if __name__ == "__main__":
    main()
