import unittest

import torch

from models import ScalableAGPPINN
from utils import (
    SparsePauliOperator,
    all_local_pauli_labels,
    commutator_pauli_labels,
    transverse_field_ising_problem,
)


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


if __name__ == "__main__":
    unittest.main()
