import unittest
from pathlib import Path

import torch

from models import FullPauliAGPPINN, PadeActivation, QuadraticMLP, TrainableSiLU
from utils import fixed_sinusoidal_schedule, load_pauli_hamiltonian_pair


ROOT = Path(__file__).resolve().parents[1]
HAMILTONIANS = ROOT / "Hamiltonians_to_use" / "pauli_decompositions" / "index.json"
LEGACY_HAMILTONIANS = ROOT / "Hamiltonians_to_use" / "Hamiltonians_pauli.json"


class FullPauliPINNTests(unittest.TestCase):
    def test_trainable_silu_activation_has_learned_slope(self):
        activation = TrainableSiLU(initial_beta=1.25)
        x = torch.linspace(-2.0, 2.0, 7)
        y = activation(x)
        y.sum().backward()

        self.assertEqual(y.shape, x.shape)
        self.assertIsNotNone(activation.beta.grad)
        self.assertTrue(torch.isfinite(activation.beta.grad))

    def test_pade_activation_has_trainable_rational_coefficients(self):
        activation = PadeActivation()
        x = torch.linspace(-2.0, 2.0, 7)
        y = activation(x)
        y.sum().backward()

        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())
        self.assertIsNotNone(activation.numerator.grad)
        self.assertIsNotNone(activation.denominator.grad)

    def test_quadratic_mlp_accepts_trainable_activation_names(self):
        for name in ("trainable_silu", "pau"):
            with self.subTest(activation=name):
                model = QuadraticMLP(1, 3, hidden_width=4, hidden_layers=1, activation=name)
                x = torch.linspace(0.0, 1.0, 5).view(-1, 1)
                y = model(x)
                loss = y.square().mean()
                loss.backward()

                self.assertEqual(y.shape, (5, 3))
                self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_fixed_schedule_endpoints(self):
        t = torch.tensor([[0.0], [0.5], [1.0]])
        lam, d_lambda_dt = fixed_sinusoidal_schedule(t)
        self.assertTrue(torch.allclose(lam[[0, -1]], torch.tensor([[0.0], [1.0]]), atol=1e-7))
        self.assertTrue(torch.allclose(d_lambda_dt[[0, -1]], torch.zeros(2, 1), atol=1e-6))
        t_long = torch.tensor([[0.0], [1.0], [2.0]])
        lam_long, d_lambda_dt_long = fixed_sinusoidal_schedule(t_long, t_min=0.0, t_max=2.0)
        self.assertTrue(torch.allclose(lam_long, lam, atol=1e-7))
        self.assertTrue(torch.allclose(d_lambda_dt_long[1], 0.5 * d_lambda_dt[1], atol=1e-6))

    def test_full_output_and_symbolic_loss_for_copied_hamiltonians(self):
        cases = [
            ("Hidrogen", 2, "1_0"),
            ("Li", 4, "1_0"),
            ("Hidrogen", 6, "1_0"),
        ]
        for system, n_qubits, distance in cases:
            with self.subTest(system=system, n_qubits=n_qubits):
                h0, h1 = load_pauli_hamiltonian_pair(
                    HAMILTONIANS,
                    system=system,
                    n_qubits=n_qubits,
                    distance=distance,
                )
                model = FullPauliAGPPINN(h0, h1, hidden_width=8, hidden_layers=1)
                self.assertEqual(model.output_terms, 4**n_qubits)
                self.assertIsInstance(model.body, QuadraticMLP)
                t = torch.linspace(0.0, 1.0, 3).view(-1, 1)
                output = model(t)
                self.assertEqual(output["agp_coefficients"].shape, (3, 4**n_qubits))
                loss, diagnostics = model.loss(t)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertIn("action", diagnostics)
                self.assertIn("reference_residual", diagnostics)
                self.assertIn("relative_residual", diagnostics)
                self.assertNotIn("boundary", diagnostics)
                self.assertNotIn("velocity", diagnostics)
                self.assertEqual(int(diagnostics["agp_terms"].item()), 4**n_qubits)
                self.assertGreater(int(diagnostics["hamiltonian_terms"].item()), 0)
                self.assertTrue(any(param.grad is not None for param in model.parameters()))

    def test_counterdiabatic_hamiltonian_matches_endpoints(self):
        h0, h1 = load_pauli_hamiltonian_pair(
            HAMILTONIANS,
            system="Hidrogen",
            n_qubits=2,
            distance="1_0",
        )
        model = FullPauliAGPPINN(h0, h1, hidden_width=8, hidden_layers=1)
        endpoints = torch.tensor([[model.t_min], [model.t_max]])
        h_cd = model.counterdiabatic_hamiltonian(endpoints).detach().cpu()
        h0_full = model.embed_hamiltonian(model.h_initial_sparse.unsqueeze(0)).detach().cpu()[0]
        h1_full = model.embed_hamiltonian(model.h_final_sparse.unsqueeze(0)).detach().cpu()[0]
        self.assertTrue(torch.allclose(h_cd[0], h0_full, atol=1e-7))
        self.assertTrue(torch.allclose(h_cd[1], h1_full, atol=1e-7))

    def test_organized_hamiltonian_index_matches_legacy_aggregate(self):
        h0, h1 = load_pauli_hamiltonian_pair(
            HAMILTONIANS,
            system="Hidrogen",
            n_qubits=2,
            distance="1_0",
        )
        legacy_h0, legacy_h1 = load_pauli_hamiltonian_pair(
            LEGACY_HAMILTONIANS,
            system="Hidrogen",
            n_qubits=2,
            distance="1_0",
        )
        self.assertEqual(h0.terms, legacy_h0.terms)
        self.assertEqual(h1.terms, legacy_h1.terms)

if __name__ == "__main__":
    unittest.main()
