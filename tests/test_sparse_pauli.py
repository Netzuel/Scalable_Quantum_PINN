import unittest

from pathlib import Path

import torch

from models import ProjectedSparseAGPPINN, ScalableAGPPINN
from utils import (
    ProjectedCommutator,
    SparsePauliOperator,
    all_local_pauli_labels,
    commutator_pauli_labels,
    transverse_field_ising_problem,
)


ROOT = Path(__file__).resolve().parents[1]


class SparsePauliTests(unittest.TestCase):
    def test_single_qubit_commutator(self):
        phase, label = commutator_pauli_labels("X", "Y")
        self.assertEqual(label, "Z")
        self.assertEqual(phase, 2j)

    def test_sparse_operator_commutator(self):
        x = SparsePauliOperator({"X": 1.0})
        y = SparsePauliOperator({"Y": 1.0})
        commutator = x.commutator(y)
        self.assertEqual(commutator.terms, {"Z": 2j})

    def test_projected_commutator(self):
        commutator = ProjectedCommutator(["X"], ["Y"], ["Z"])
        left = torch.tensor([[1.0]])
        right = torch.tensor([[2.0]])
        out = commutator.commutator(left, right)
        self.assertTrue(torch.allclose(out, torch.tensor([[0.0 + 4.0j]])))

    def test_pinn_loss_is_differentiable(self):
        h0, h1 = transverse_field_ising_problem(2)
        agp_labels = all_local_pauli_labels(2, max_weight=2)
        model = ScalableAGPPINN(
            h0,
            h1,
            agp_labels,
            hidden_width=8,
            hidden_layers=1,
            max_closure_weight=2,
        )
        t = torch.linspace(0.0, 1.0, 5)[:, None]
        loss, diagnostics = model.loss(t)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertLessEqual(diagnostics["basis_size"].item(), 16)
        self.assertTrue(any(param.grad is not None for param in model.parameters()))

    def test_projected_sparse_pinn_loss_is_differentiable(self):
        h0, h1 = transverse_field_ising_problem(2)
        labels = all_local_pauli_labels(2, max_weight=2)
        model = ProjectedSparseAGPPINN(
            h0,
            h1,
            agp_labels=labels[:4],
            intermediate_labels=labels,
            residual_labels=labels,
            hidden_width=8,
            hidden_layers=1,
        )
        t = torch.linspace(0.0, 1.0, 5)[:, None]
        loss, diagnostics = model.loss(t)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(int(diagnostics["agp_terms"].item()), 4)
        self.assertTrue(any(param.grad is not None for param in model.parameters()))

    def test_sparse_q20_training_folder_exists(self):
        run_dir = ROOT / "tests" / "sparse_agp_20_qubits"
        self.assertTrue((run_dir / "training_script.py").is_file())
        self.assertTrue((run_dir / "restart_folders.py").is_file())
        self.assertTrue((run_dir / "config.json").is_file())


if __name__ == "__main__":
    unittest.main()
