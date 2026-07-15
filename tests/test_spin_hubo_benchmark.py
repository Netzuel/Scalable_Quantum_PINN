from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_SCRIPTS = ROOT / "tests" / "sparse_agp_curriculum" / "scripts"
if str(FRAMEWORK_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_SCRIPTS))

from spin_hubo_benchmark import (  # noqa: E402
    build_benchmark_assets,
    evaluate_spin_energy,
    exact_walsh_ground_reference,
    load_spin_polynomial,
    spin_polynomial_to_pauli_pair,
)


class SpinHUBOBenchmarkTests(unittest.TestCase):
    def test_loads_structured_spin_supports(self) -> None:
        payload = {"()": 0.25, "(0,)": -0.5, "(0, 2)": 1.25}
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "poly.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            polynomial = load_spin_polynomial(path)

        self.assertEqual(polynomial, {(): 0.25, (0,): -0.5, (0, 2): 1.25})

    def test_maps_spin_polynomial_to_transverse_driver_and_z_terms(self) -> None:
        h0, h1 = spin_polynomial_to_pauli_pair(
            {(): 0.25, (0,): -0.5, (0, 2): 1.25},
            n_qubits=3,
        )

        self.assertEqual(h0.terms, {"XII": -1.0, "IXI": -1.0, "IIX": -1.0})
        self.assertEqual(h1.terms, {"III": 0.25, "ZII": -0.5, "ZIZ": 1.25})

    def test_evaluates_spin_energy_without_changing_convention(self) -> None:
        polynomial = {(): 0.25, (0,): -0.5, (0, 2): 1.25}

        energy = evaluate_spin_energy(polynomial, spins=(1, -1, -1))

        self.assertAlmostEqual(energy, -1.5)

    def test_walsh_oracle_preserves_qubit_to_bitstring_order(self) -> None:
        reference = exact_walsh_ground_reference(
            {(0,): 1.0, (1,): -2.0},
            n_qubits=2,
        )

        self.assertAlmostEqual(reference.ground_energy, -3.0)
        self.assertEqual(reference.ground_bitstrings, ("10",))
        self.assertEqual(reference.ground_state_degeneracy, 1)

    def test_walsh_oracle_reports_degenerate_ground_space(self) -> None:
        reference = exact_walsh_ground_reference({(0, 1): -1.0}, n_qubits=2)

        self.assertAlmostEqual(reference.ground_energy, -1.0)
        self.assertEqual(reference.ground_bitstrings, ("00", "11"))
        self.assertEqual(reference.ground_state_degeneracy, 2)

    def test_rejects_duplicate_and_out_of_range_support_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            spin_polynomial_to_pauli_pair({(0, 0): 1.0}, n_qubits=2)
        with self.assertRaisesRegex(ValueError, "outside"):
            spin_polynomial_to_pauli_pair({(2,): 1.0}, n_qubits=2)

    def test_exact_oracle_enforces_documented_qubit_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most 24"):
            exact_walsh_ground_reference({(0,): -1.0}, n_qubits=25)

    def test_builds_self_contained_pauli_pair_and_ground_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "poly.json").write_text(
                json.dumps({"(0,)": 1.0, "(1,)": -2.0}),
                encoding="utf-8",
            )
            (source / "meta.json").write_text(
                json.dumps({"name": "toy", "n_qubits": 2, "min_energy": -3.0}),
                encoding="utf-8",
            )

            assets = build_benchmark_assets(
                source_instance=source,
                output_root=root / "benchmark",
                system="TransverseFieldSpinHUBO",
                distance="toy",
            )

            pair = json.loads(assets["hamiltonian_pair"].read_text(encoding="utf-8"))
            reference = json.loads(assets["ground_reference"].read_text(encoding="utf-8"))
            manifest = json.loads(assets["source_manifest"].read_text(encoding="utf-8"))

        self.assertEqual(pair["format"], "pauli_hamiltonian_pair_v1")
        self.assertEqual(pair["hamiltonians"]["initial"]["term_count"], 2)
        self.assertEqual(pair["hamiltonians"]["final"]["terms"], {"IZ": [-2.0, 0.0], "ZI": [1.0, 0.0]})
        self.assertEqual(reference["solutions"][0]["ground_bitstrings"], ["10"])
        self.assertEqual(reference["solutions"][0]["ground_state_degeneracy"], 1)
        self.assertEqual(len(manifest["source_poly_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
