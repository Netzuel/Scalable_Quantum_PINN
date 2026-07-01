import unittest

from tools.generate_qiskit_pauli_hamiltonian import (
    build_pair_payload,
    diagonal_projection_terms,
    sparse_pauli_op_to_terms,
)


class QiskitHamiltonianGeneratorTests(unittest.TestCase):
    def test_pair_payload_uses_sparse_diagonal_initial_hamiltonian(self):
        final_terms = {
            "II": -0.5 + 0.0j,
            "IZ": 0.25 + 0.0j,
            "XX": 0.125 + 0.0j,
            "XY": 0.0 + 0.0625j,
        }
        initial_terms = diagonal_projection_terms(final_terms)
        payload = build_pair_payload(
            system="Toy",
            n_qubits=2,
            distance="1_0",
            initial_terms=initial_terms,
            final_terms=final_terms,
            source={"generator": "unit-test"},
            drop_tol=1e-10,
        )

        self.assertEqual(set(initial_terms), {"II", "IZ"})
        self.assertEqual(payload["format"], "pauli_hamiltonian_pair_v1")
        self.assertEqual(payload["hamiltonians"]["initial"]["term_count"], 2)
        self.assertEqual(payload["hamiltonians"]["final"]["term_count"], 4)
        self.assertEqual(payload["hamiltonians"]["final"]["terms_by_order"], {"0": 1, "1": 1, "2": 2})

    def test_sparse_pauli_op_to_terms_reads_qiskit_labels(self):
        try:
            from qiskit.quantum_info import SparsePauliOp
        except ImportError:
            self.skipTest("qiskit is not installed")

        operator = SparsePauliOp(["ZI", "XX", "XX"], coeffs=[1.0, 0.25, -0.25])
        terms = sparse_pauli_op_to_terms(operator, drop_tol=1e-12)
        self.assertEqual(terms, {"ZI": 1.0 + 0.0j})


if __name__ == "__main__":
    unittest.main()
