import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_SCRIPTS_DIR = ROOT / "tests" / "sparse_agp_curriculum" / "scripts"
if str(FRAMEWORK_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_SCRIPTS_DIR))

from agp_mps_validation import (
    apply_grouped_hamiltonian_rotation_mps,
    assess_mps_convergence,
    assess_statevector_agreement,
    apply_pauli_rotation_mps,
    diagonal_ising_mps_metrics,
    evolve_protocol_mps,
    make_product_mps,
    pauli_rotation_matrix,
    run_mps_case,
    statevector_results_for_learned_terms,
    validation_certification,
    cached_protocol_result,
    group_hamiltonian_terms_by_support,
)
from utils import SparsePauliOperator


class AGPMPSValidationTests(unittest.TestCase):
    def test_group_hamiltonian_terms_preserves_every_pauli_coefficient(self):
        grouped = group_hamiltonian_terms_by_support(
            [("XI", 0.25), ("YI", -0.5), ("XZ", 0.75)]
        )

        x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
        z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        np.testing.assert_allclose(grouped[(0,)], 0.25 * x - 0.5 * y, atol=1.0e-12)
        np.testing.assert_allclose(grouped[(0, 1)], 0.75 * np.kron(x, z), atol=1.0e-12)

    def test_grouped_hamiltonian_rotation_matches_dense_exponential(self):
        state = make_product_mps("00")
        terms = [("XI", 0.25), ("YI", -0.5), ("XZ", 0.75)]
        angle = 0.31

        diagnostics = apply_grouped_hamiltonian_rotation_mps(
            state,
            terms,
            angle,
            cutoff=1.0e-14,
            max_bond=16,
        )

        x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
        z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        identity = np.eye(2, dtype=np.complex128)
        local_hamiltonian = 0.25 * x - 0.5 * y
        eigenvalues, eigenvectors = np.linalg.eigh(local_hamiltonian)
        local_gate = (eigenvectors * np.exp(-1.0j * angle * eigenvalues)) @ eigenvectors.conj().T
        nonlocal_gate = np.cos(0.75 * angle) * np.eye(4) - 1.0j * np.sin(0.75 * angle) * np.kron(x, z)
        expected = nonlocal_gate @ np.kron(local_gate, identity) @ np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.complex128
        )

        np.testing.assert_allclose(np.asarray(state.to_dense()).reshape(-1), expected, atol=1.0e-11)
        self.assertEqual(diagnostics["pauli_terms"], 3)
        self.assertEqual(diagnostics["support_groups"], 2)

    def test_protocol_cache_requires_identical_resolution_settings(self):
        previous = [
            {
                "settings": {
                    "steps": 96,
                    "max_bond": 128,
                    "learned_terms": 1024,
                    "cutoff": 1.0e-12,
                    "coefficient_threshold": 1.0e-10,
                    "operator_grouping": "support",
                },
                "results": {"learned_sparse_agp": {"final_energy": -18.99}},
            }
        ]
        settings = dict(previous[0]["settings"])

        cached = cached_protocol_result(previous, settings=settings, protocol="learned_sparse_agp")

        self.assertEqual(cached["final_energy"], -18.99)
        self.assertIsNone(cached_protocol_result(previous, settings={**settings, "steps": 48}, protocol="learned_sparse_agp"))
        self.assertIsNone(
            cached_protocol_result(
                previous,
                settings={**settings, "operator_grouping": "pauli_term"},
                protocol="learned_sparse_agp",
            )
        )

    def test_certification_requires_only_configured_validation_gates(self):
        certification = validation_certification(
            convergence={"status": "not_tested"},
            statevector_agreement={"status": "pass"},
            require_convergence=False,
            require_statevector=True,
        )

        self.assertEqual(certification["status"], "pass")
        self.assertEqual(certification["required_gates"], ["statevector_agreement"])

    def test_statevector_reference_selects_matching_learned_support(self):
        payload = {
            "results": {
                "no_cd": {"final_energy": -2.0},
                "learned_sparse_agp": {"final_energy": -4.0, "learned_terms": 2048},
            },
            "learned_variant_results": {
                "learned_sparse_agp_terms_1024_scale_1": {
                    "final_energy": -3.5,
                    "learned_terms": 1024,
                    "learned_scale": 1.0,
                }
            },
        }

        selected = statevector_results_for_learned_terms(payload, learned_terms=1024, learned_scale=1.0)

        self.assertEqual(selected["no_cd"]["final_energy"], -2.0)
        self.assertEqual(selected["learned_sparse_agp"]["final_energy"], -3.5)

    def test_statevector_agreement_gate_compares_same_protocols(self):
        mps = {
            "no_cd": {"final_energy": -2.001, "ground_state_fidelity": 0.201},
            "learned_sparse_agp": {"final_energy": -4.004, "ground_state_fidelity": 0.804},
        }
        statevector = {
            "no_cd": {"final_energy": -2.0, "ground_state_fidelity": 0.20},
            "learned_sparse_agp": {"final_energy": -4.0, "ground_state_fidelity": 0.80},
        }

        assessment = assess_statevector_agreement(
            mps,
            statevector,
            energy_atol=0.005,
            fidelity_atol=0.005,
        )

        self.assertEqual(assessment["status"], "pass")
        self.assertEqual(set(assessment["protocols"]), set(mps))

    def test_validation_case_emits_physical_metrics_and_mps_diagnostics(self):
        h0 = SparsePauliOperator({"X": -1.0}, n_qubits=1)
        h1 = SparsePauliOperator({"Z": -1.0}, n_qubits=1)

        result = run_mps_case(
            h0=h0,
            h1=h1,
            learned=None,
            exact_ground_energy=-1.0,
            ground_bitstring="0",
            protocols=("no_cd",),
            total_time=1.0,
            steps=8,
            cutoff=1.0e-12,
            max_bond=4,
            coefficient_threshold=0.0,
        )

        self.assertEqual(set(result), {"no_cd"})
        self.assertIn("final_energy", result["no_cd"])
        self.assertIn("ground_state_fidelity", result["no_cd"])
        self.assertEqual(result["no_cd"]["mps_diagnostics"]["steps"], 8)

    def test_convergence_requires_all_protocol_deltas_within_tolerance(self):
        coarse = {
            "no_cd": {"final_energy": -2.0, "ground_state_fidelity": 0.20},
            "kipu_dqfm_l1": {"final_energy": -3.0, "ground_state_fidelity": 0.30},
            "learned_sparse_agp": {"final_energy": -4.0, "ground_state_fidelity": 0.80},
        }
        fine = {
            "no_cd": {"final_energy": -2.005, "ground_state_fidelity": 0.201},
            "kipu_dqfm_l1": {"final_energy": -3.008, "ground_state_fidelity": 0.302},
            "learned_sparse_agp": {"final_energy": -4.009, "ground_state_fidelity": 0.804},
        }

        assessment = assess_mps_convergence(
            coarse,
            fine,
            energy_atol=0.01,
            fidelity_atol=0.005,
        )

        self.assertEqual(assessment["status"], "pass")
        self.assertTrue(all(row["pass"] for row in assessment["protocols"].values()))

    def test_no_cd_evolution_matches_independent_rk4_reference(self):
        h0 = SparsePauliOperator({"X": -1.0}, n_qubits=1)
        h1 = SparsePauliOperator({"Z": -1.0}, n_qubits=1)
        total_time = 1.0
        state, diagnostics = evolve_protocol_mps(
            protocol="no_cd",
            h0=h0,
            h1=h1,
            learned=None,
            total_time=total_time,
            steps=256,
            cutoff=1.0e-14,
            max_bond=8,
        )

        x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        psi = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
        dt = total_time / 4096

        def rhs(t, vector):
            lam = np.sin(0.5 * np.pi * t / total_time) ** 2
            hamiltonian = -(1.0 - lam) * x - lam * z
            return -1.0j * hamiltonian @ vector

        for step in range(4096):
            t = step * dt
            k1 = rhs(t, psi)
            k2 = rhs(t + 0.5 * dt, psi + 0.5 * dt * k1)
            k3 = rhs(t + 0.5 * dt, psi + 0.5 * dt * k2)
            k4 = rhs(t + dt, psi + dt * k3)
            psi += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            psi /= np.linalg.norm(psi)

        mps_vector = np.asarray(state.to_dense()).reshape(-1)
        overlap = abs(np.vdot(psi, mps_vector)) ** 2
        self.assertGreater(overlap, 1.0 - 2.0e-8)
        self.assertEqual(diagnostics["gate_count"], 1024)

    def test_support_grouped_evolution_records_full_term_and_group_counts(self):
        h0 = SparsePauliOperator({"XI": -1.0, "IX": -1.0}, n_qubits=2)
        h1 = SparsePauliOperator({"ZI": -0.4, "IZ": -0.5, "ZZ": -1.0}, n_qubits=2)

        _, diagnostics = evolve_protocol_mps(
            protocol="no_cd",
            h0=h0,
            h1=h1,
            learned=None,
            total_time=1.0,
            steps=4,
            cutoff=1.0e-12,
            max_bond=8,
            operator_grouping="support",
        )

        self.assertEqual(diagnostics["operator_grouping"], "support")
        self.assertEqual(diagnostics["pauli_term_applications"], 40)
        self.assertEqual(diagnostics["gate_count"], 24)

    def test_support_grouping_converges_to_legacy_pauli_term_evolution(self):
        h0 = SparsePauliOperator({"XI": -1.0, "IX": -1.0}, n_qubits=2)
        h1 = SparsePauliOperator({"ZI": -0.4, "IZ": -0.5, "ZZ": -1.0}, n_qubits=2)
        common = {
            "protocol": "no_cd",
            "h0": h0,
            "h1": h1,
            "learned": None,
            "total_time": 1.0,
            "steps": 128,
            "cutoff": 1.0e-14,
            "max_bond": 16,
        }

        legacy, _ = evolve_protocol_mps(**common, operator_grouping="pauli_term")
        grouped, _ = evolve_protocol_mps(**common, operator_grouping="support")

        legacy_vector = np.asarray(legacy.to_dense()).reshape(-1)
        grouped_vector = np.asarray(grouped.to_dense()).reshape(-1)
        self.assertGreater(abs(np.vdot(legacy_vector, grouped_vector)) ** 2, 1.0 - 1.0e-8)

    def test_pauli_rotation_matches_dense_definition(self):
        angle = 0.371
        x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
        pauli = np.kron(x, y)
        expected = np.cos(angle) * np.eye(4) - 1.0j * np.sin(angle) * pauli

        np.testing.assert_allclose(pauli_rotation_matrix("XY", angle), expected, atol=1.0e-12)

    def test_pauli_rotation_updates_product_mps(self):
        state = make_product_mps("00")

        apply_pauli_rotation_mps(state, "XI", np.pi / 2.0, cutoff=1.0e-14, max_bond=8)

        self.assertAlmostEqual(abs(complex(state.amplitude("10"))) ** 2, 1.0, places=12)
        self.assertAlmostEqual(abs(complex(state.amplitude("00"))) ** 2, 0.0, places=12)

    def test_diagonal_metrics_recover_known_ground_product_state(self):
        state = make_product_mps("000")
        terms = {
            "ZII": -0.3,
            "IZI": -0.4,
            "IIZ": -0.5,
            "ZZI": -1.0,
            "IZZ": -1.1,
        }

        metrics = diagonal_ising_mps_metrics(state, terms, exact_ground_energy=-3.3)

        self.assertAlmostEqual(metrics["final_energy"], -3.3, places=12)
        self.assertAlmostEqual(metrics["energy_error"], 0.0, places=12)
        self.assertAlmostEqual(metrics["ground_fidelity"], 1.0, places=12)
        self.assertAlmostEqual(metrics["state_norm"], 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
