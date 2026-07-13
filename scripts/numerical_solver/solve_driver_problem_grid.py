from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.numerical_solver.ising_ground_state import (  # noqa: E402
    DEFAULT_ATOL,
    DiagonalIsingProblem,
    GroundStateSolution,
    build_driver_problem_ising,
    ferromagnetic_closed_form,
    load_final_ising_problem,
    solve_brute_force,
    solve_dynamic_programming,
    to_qubo,
)


DEFAULT_HAMILTONIAN_ROOT = (
    ROOT / "Hamiltonians_to_use" / "pauli_decompositions" / "TransverseIsingDriverProblem"
)
DEFAULT_OUTPUT_DIR = ROOT / "tests" / "sparse_agp_curriculum" / "ground_truth" / "diagonal_ising"


def _problem_hash(problem: DiagonalIsingProblem) -> str:
    serialized = json.dumps(problem.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _problems_match(left: DiagonalIsingProblem, right: DiagonalIsingProblem, atol: float) -> bool:
    if left.num_qubits != right.num_qubits or len(left.couplings) != len(right.couplings):
        return False
    if not math.isclose(left.constant, right.constant, rel_tol=0.0, abs_tol=atol):
        return False
    if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=atol) for a, b in zip(left.fields, right.fields)):
        return False
    return all(
        edge_a[:2] == edge_b[:2] and math.isclose(edge_a[2], edge_b[2], rel_tol=0.0, abs_tol=atol)
        for edge_a, edge_b in zip(left.couplings, right.couplings)
    )


def _ground_states_match(left: GroundStateSolution, right: GroundStateSolution, atol: float) -> bool:
    return (
        math.isclose(left.ground_energy, right.ground_energy, rel_tol=0.0, abs_tol=atol)
        and left.ground_state_degeneracy == right.ground_state_degeneracy
        and set(left.ground_bitstrings) == set(right.ground_bitstrings)
    )


def _optional_float_matches(left: float | None, right: float | None, atol: float) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=0.0, abs_tol=atol)


def _solutions_match(left: GroundStateSolution, right: GroundStateSolution, atol: float) -> bool:
    return (
        _ground_states_match(left, right, atol)
        and _optional_float_matches(left.first_excited_energy, right.first_excited_energy, atol)
        and _optional_float_matches(left.spectral_gap, right.spectral_gap, atol)
    )


def _clean_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(f"{value:.15g}")


def _load_or_build_problem(
    num_qubits: int,
    *,
    hamiltonian_root: Path,
    atol: float,
) -> tuple[DiagonalIsingProblem, str, bool]:
    analytic = build_driver_problem_ising(num_qubits=num_qubits)
    source_path = hamiltonian_root / f"{num_qubits}_qubits" / "distance_1_0.json"
    if not source_path.is_file():
        return analytic, "analytic_parameter_construction", True
    loaded = load_final_ising_problem(source_path)
    return loaded, str(source_path.relative_to(ROOT) if source_path.is_relative_to(ROOT) else source_path), _problems_match(
        loaded, analytic, atol
    )


def _solution_row(
    num_qubits: int,
    *,
    problem: DiagonalIsingProblem,
    source: str,
    source_matches_analytic: bool,
    dynamic: GroundStateSolution,
    closed_form: GroundStateSolution | None,
    brute_force: GroundStateSolution | None,
    atol: float,
) -> dict[str, object]:
    qubo = to_qubo(problem)
    qubo_ground_energies = [qubo.energy(bitstring) for bitstring in dynamic.ground_bitstrings]
    qubo_passed = bool(qubo_ground_energies) and all(
        math.isclose(value, dynamic.ground_energy, rel_tol=0.0, abs_tol=atol)
        for value in qubo_ground_energies
    )
    closed_form_passed = closed_form is not None and _ground_states_match(dynamic, closed_form, atol)
    brute_force_passed = None if brute_force is None else _solutions_match(dynamic, brute_force, atol)
    return {
        "q": num_qubits,
        "source": source,
        "problem_sha256": _problem_hash(problem),
        "field_term_count": sum(abs(value) > atol for value in problem.fields),
        "coupling_term_count": sum(abs(value) > atol for _, _, value in problem.couplings),
        "ground_energy": _clean_float(dynamic.ground_energy),
        "ground_bitstrings": list(dynamic.ground_bitstrings),
        "ground_state_kets": [f"|{bitstring}>" for bitstring in dynamic.ground_bitstrings],
        "ground_state_degeneracy": dynamic.ground_state_degeneracy,
        "ground_bitstrings_truncated": dynamic.ground_bitstrings_truncated,
        "first_excited_energy": _clean_float(dynamic.first_excited_energy),
        "spectral_gap": _clean_float(dynamic.spectral_gap),
        "solver": dynamic.method,
        "qubo": {
            "constant": _clean_float(qubo.constant),
            "linear": [_clean_float(value) for value in qubo.linear],
            "quadratic": [
                [left, right, _clean_float(value)] for left, right, value in qubo.quadratic
            ],
        },
        "validation": {
            "source_matches_analytic_construction": source_matches_analytic,
            "termwise_closed_form_performed": closed_form is not None,
            "termwise_closed_form_passed": closed_form_passed,
            "exhaustive_enumeration_performed": brute_force is not None,
            "exhaustive_enumeration_passed": brute_force_passed,
            "qubo_ground_energy_passed": qubo_passed,
        },
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "q",
        "ground_energy",
        "ground_bitstrings",
        "ground_state_degeneracy",
        "first_excited_energy",
        "spectral_gap",
        "solver",
        "source",
        "problem_sha256",
        "source_matches_analytic_construction",
        "termwise_closed_form_passed",
        "exhaustive_enumeration_performed",
        "exhaustive_enumeration_passed",
        "qubo_ground_energy_passed",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            validation = row["validation"]
            assert isinstance(validation, Mapping)
            writer.writerow(
                {
                    "q": row["q"],
                    "ground_energy": f"{float(row['ground_energy']):.15g}",
                    "ground_bitstrings": ";".join(str(value) for value in row["ground_bitstrings"]),
                    "ground_state_degeneracy": row["ground_state_degeneracy"],
                    "first_excited_energy": (
                        "" if row["first_excited_energy"] is None else f"{float(row['first_excited_energy']):.15g}"
                    ),
                    "spectral_gap": "" if row["spectral_gap"] is None else f"{float(row['spectral_gap']):.15g}",
                    "solver": row["solver"],
                    "source": row["source"],
                    "problem_sha256": row["problem_sha256"],
                    "source_matches_analytic_construction": validation["source_matches_analytic_construction"],
                    "termwise_closed_form_passed": validation["termwise_closed_form_passed"],
                    "exhaustive_enumeration_performed": validation["exhaustive_enumeration_performed"],
                    "exhaustive_enumeration_passed": validation["exhaustive_enumeration_passed"],
                    "qubo_ground_energy_passed": validation["qubo_ground_energy_passed"],
                }
            )


def run_ground_truth_grid(
    *,
    min_qubits: int,
    max_qubits: int,
    validation_max_qubits: int,
    hamiltonian_root: Path,
    output_dir: Path,
    max_bitstrings: int = 256,
    atol: float = DEFAULT_ATOL,
) -> dict[str, object]:
    if min_qubits < 2 or max_qubits < min_qubits:
        raise ValueError("Require 2 <= min_qubits <= max_qubits.")
    validation_end = min(max_qubits, validation_max_qubits)
    rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    for num_qubits in range(min_qubits, max_qubits + 1):
        problem, source, source_matches_analytic = _load_or_build_problem(
            num_qubits,
            hamiltonian_root=hamiltonian_root,
            atol=atol,
        )
        dynamic = solve_dynamic_programming(problem, max_bitstrings=max_bitstrings, atol=atol)
        closed_form = ferromagnetic_closed_form(problem, atol=atol)
        brute_force = (
            solve_brute_force(problem, max_bitstrings=max_bitstrings, atol=atol)
            if num_qubits <= validation_end
            else None
        )
        row = _solution_row(
            num_qubits,
            problem=problem,
            source=source,
            source_matches_analytic=source_matches_analytic,
            dynamic=dynamic,
            closed_form=closed_form,
            brute_force=brute_force,
            atol=atol,
        )
        validation = row["validation"]
        assert isinstance(validation, Mapping)
        required_checks = [
            validation["source_matches_analytic_construction"],
            validation["termwise_closed_form_passed"],
            validation["qubo_ground_energy_passed"],
        ]
        if brute_force is not None:
            required_checks.append(validation["exhaustive_enumeration_passed"])
            validation_rows.append(row)
        if not all(bool(value) for value in required_checks):
            raise RuntimeError(f"Ground-state validation failed for q={num_qubits}: {validation}")
        rows.append(row)
        print(
            f"q={num_qubits:3d} E0={dynamic.ground_energy:.12g} "
            f"gap={dynamic.spectral_gap:.12g} bitstring={dynamic.ground_bitstrings[0]}"
        )

    payload: dict[str, object] = {
        "format": "diagonal_ising_ground_truth_v1",
        "system": "TransverseIsingDriverProblem",
        "hamiltonian": {
            "initial": "H_initial = -sum_i X_i",
            "final": "H_final = -sum_i h_i Z_i - sum_i J_i Z_i Z_{i+1}",
            "boundary": "open",
            "z_field": 0.35,
            "zz_coupling": 1.0,
            "field_gradient": 0.15,
            "coupling_gradient": 0.10,
        },
        "bitstring_convention": (
            "Characters follow stored Pauli-label order; bit 0 has Z eigenvalue +1 and bit 1 has Z eigenvalue -1."
        ),
        "exactness": {
            "primary_solver": "Exact dynamic programming on a path, O(q) time and O(q) retained-path memory.",
            "closed_form": "All h_i and J_i are positive, so every final-Hamiltonian term is minimized by |0...0>.",
            "ground_energy_formula": "E0(q) = -sum_i h_i - sum_i J_i = 1 - 1.35*q",
            "small_q_oracle": f"All 2**q bitstrings enumerated independently for q={min_qubits}..{validation_end}.",
        },
        "commercial_solver_context": {
            "gurobi": "Supports binary variables and quadratic objectives through gurobipy.",
            "cplex": "Supports mixed-integer quadratic optimization through the CPLEX Python API.",
            "selection": (
                "Neither proprietary package is required: the current path-graph Ising family has a stronger, "
                "dependency-free exact O(q) solver. The exported QUBO is directly representable in either solver."
            ),
        },
        "validation_summary": {
            "validation_min_qubits": min_qubits,
            "validation_max_qubits": validation_end,
            "validated_size_count": len(validation_rows),
            "all_passed": True,
        },
        "solutions": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"ground_states_q{min_qubits}_q{max_qubits}.json"
    csv_path = output_dir / f"ground_states_q{min_qubits}_q{max_qubits}.csv"
    validation_path = output_dir / f"validation_q{min_qubits}_q{validation_end}.json"
    _write_json(json_path, payload)
    _write_csv(csv_path, rows)
    _write_json(
        validation_path,
        {
            "format": "diagonal_ising_ground_truth_validation_v1",
            "system": payload["system"],
            "validation_summary": payload["validation_summary"],
            "solutions": validation_rows,
        },
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve the diagonal-Ising H_final exactly and export ground-state validation data."
    )
    parser.add_argument("--min-qubits", type=int, default=2)
    parser.add_argument("--max-qubits", type=int, default=156)
    parser.add_argument("--validation-max-qubits", type=int, default=20)
    parser.add_argument("--hamiltonian-root", type=Path, default=DEFAULT_HAMILTONIAN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-bitstrings", type=int, default=256)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_ground_truth_grid(
        min_qubits=args.min_qubits,
        max_qubits=args.max_qubits,
        validation_max_qubits=args.validation_max_qubits,
        hamiltonian_root=args.hamiltonian_root,
        output_dir=args.output_dir,
        max_bitstrings=args.max_bitstrings,
        atol=args.atol,
    )
    summary = payload["validation_summary"]
    print(
        f"wrote {len(payload['solutions'])} exact solutions to {args.output_dir}; "
        f"exhaustive_validation={summary['validation_min_qubits']}..{summary['validation_max_qubits']}"
    )


if __name__ == "__main__":
    main()
