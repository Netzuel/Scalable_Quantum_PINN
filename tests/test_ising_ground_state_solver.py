from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from scripts.numerical_solver.ising_ground_state import (  # noqa: E402
    DiagonalIsingProblem,
    GroundStateSolution,
    build_driver_problem_ising,
    energy_of_bitstring,
    ferromagnetic_closed_form,
    load_final_ising_problem,
    solve_brute_force,
    solve_dynamic_programming,
    to_qubo,
)
from scripts.numerical_solver.solve_driver_problem_grid import (  # noqa: E402
    _solutions_match,
    run_ground_truth_grid,
)


class TestDiagonalIsingGroundStateSolver(unittest.TestCase):
    def test_loads_q2_final_hamiltonian_and_preserves_label_order(self) -> None:
        path = (
            ROOT
            / "Hamiltonians_to_use"
            / "pauli_decompositions"
            / "TransverseIsingDriverProblem"
            / "2_qubits"
            / "distance_1_0.json"
        )

        problem = load_final_ising_problem(path)

        self.assertEqual(problem.num_qubits, 2)
        self.assertEqual(problem.fields, (-0.32375, -0.37625))
        self.assertEqual(problem.couplings, ((0, 1, -1.0),))
        self.assertAlmostEqual(energy_of_bitstring(problem, "00"), -1.7)
        self.assertAlmostEqual(energy_of_bitstring(problem, "01"), 1.0525)

    def test_qubo_matches_ising_energy_for_every_q3_bitstring(self) -> None:
        problem = DiagonalIsingProblem(
            num_qubits=3,
            constant=0.25,
            fields=(-0.4, 0.2, -0.1),
            couplings=((0, 1, -0.7), (1, 2, 0.3)),
        )
        qubo = to_qubo(problem)

        for bits in itertools.product("01", repeat=problem.num_qubits):
            bitstring = "".join(bits)
            self.assertAlmostEqual(qubo.energy(bitstring), energy_of_bitstring(problem, bitstring))

    def test_dynamic_programming_recovers_q2_ground_state_and_gap(self) -> None:
        problem = build_driver_problem_ising(num_qubits=2)

        solution = solve_dynamic_programming(problem)

        self.assertAlmostEqual(solution.ground_energy, -1.7)
        self.assertEqual(solution.ground_bitstrings, ("00",))
        self.assertEqual(solution.ground_state_degeneracy, 1)
        self.assertAlmostEqual(solution.first_excited_energy, -0.3)
        self.assertAlmostEqual(solution.spectral_gap, 1.4)

    def test_dynamic_programming_matches_brute_force_for_signed_path(self) -> None:
        problem = DiagonalIsingProblem(
            num_qubits=5,
            constant=-0.125,
            fields=(-0.4, 0.3, -0.2, 0.1, -0.05),
            couplings=((0, 1, 0.7), (1, 2, -0.8), (2, 3, 0.25), (3, 4, -0.6)),
        )

        dynamic = solve_dynamic_programming(problem)
        brute_force = solve_brute_force(problem)

        self.assertAlmostEqual(dynamic.ground_energy, brute_force.ground_energy)
        self.assertEqual(dynamic.ground_state_degeneracy, brute_force.ground_state_degeneracy)
        self.assertEqual(set(dynamic.ground_bitstrings), set(brute_force.ground_bitstrings))
        self.assertAlmostEqual(dynamic.first_excited_energy, brute_force.first_excited_energy)

    def test_dynamic_programming_supports_periodic_edge(self) -> None:
        problem = DiagonalIsingProblem(
            num_qubits=4,
            constant=0.0,
            fields=(0.2, -0.1, 0.05, -0.3),
            couplings=((0, 1, -0.5), (1, 2, 0.4), (2, 3, -0.7), (0, 3, 0.25)),
        )

        dynamic = solve_dynamic_programming(problem)
        brute_force = solve_brute_force(problem)

        self.assertAlmostEqual(dynamic.ground_energy, brute_force.ground_energy)
        self.assertEqual(set(dynamic.ground_bitstrings), set(brute_force.ground_bitstrings))
        self.assertAlmostEqual(dynamic.first_excited_energy, brute_force.first_excited_energy)

    def test_solution_comparison_includes_excitation_and_gap(self) -> None:
        reference = GroundStateSolution(
            method="reference",
            ground_energy=-2.0,
            ground_bitstrings=("00",),
            ground_state_degeneracy=1,
            ground_bitstrings_truncated=False,
            first_excited_energy=-1.0,
            spectral_gap=1.0,
        )

        self.assertFalse(_solutions_match(reference, replace(reference, first_excited_energy=-0.9), atol=1e-10))
        self.assertFalse(_solutions_match(reference, replace(reference, spectral_gap=1.1), atol=1e-10))

    def test_ferromagnetic_closed_form_matches_current_family(self) -> None:
        for num_qubits in (2, 7, 20, 156):
            problem = build_driver_problem_ising(num_qubits=num_qubits)
            closed_form = ferromagnetic_closed_form(problem)
            self.assertIsNotNone(closed_form)
            assert closed_form is not None
            self.assertEqual(closed_form.ground_bitstrings, ("0" * num_qubits,))
            self.assertAlmostEqual(
                closed_form.ground_energy,
                problem.constant + sum(problem.fields) + sum(value for _, _, value in problem.couplings),
            )

    def test_rejects_non_diagonal_final_hamiltonian(self) -> None:
        payload = {
            "n_qubits": 2,
            "hamiltonians": {"final": {"terms": {"XI": [-1.0, 0.0]}}},
        }
        path = ROOT / "tests" / "_temporary_non_diagonal_hamiltonian.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.addCleanup(path.unlink, missing_ok=True)

        with self.assertRaisesRegex(ValueError, "diagonal"):
            load_final_ising_problem(path)

    def test_grid_export_writes_validated_json_and_csv(self) -> None:
        hamiltonian_root = (
            ROOT
            / "Hamiltonians_to_use"
            / "pauli_decompositions"
            / "TransverseIsingDriverProblem"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            payload = run_ground_truth_grid(
                min_qubits=2,
                max_qubits=5,
                validation_max_qubits=4,
                hamiltonian_root=hamiltonian_root,
                output_dir=output_dir,
            )

            self.assertTrue(payload["validation_summary"]["all_passed"])
            self.assertEqual(len(payload["solutions"]), 4)
            self.assertEqual(payload["solutions"][0]["first_excited_energy"], -0.3)
            self.assertEqual(payload["solutions"][0]["qubo"]["constant"], -1.7)
            self.assertEqual(payload["solutions"][0]["qubo"]["linear"], [2.6475, 2.7525])
            self.assertEqual(payload["solutions"][0]["qubo"]["quadratic"], [[0, 1, -4.0]])
            self.assertEqual(payload["solutions"][-1]["ground_bitstrings"], ["00000"])
            self.assertFalse(payload["solutions"][-1]["validation"]["exhaustive_enumeration_performed"])
            self.assertIsNone(payload["solutions"][-1]["validation"]["exhaustive_enumeration_passed"])
            self.assertTrue((output_dir / "ground_states_q2_q5.json").is_file())
            self.assertTrue((output_dir / "ground_states_q2_q5.csv").is_file())
            self.assertTrue((output_dir / "validation_q2_q4.json").is_file())


if __name__ == "__main__":
    unittest.main()
