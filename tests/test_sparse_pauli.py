import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import sys

import torch

TESTS_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
for path in (SCRIPTS_DIR, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models import (
    ProjectedSparseAGPPINN,
    ScalableAGPPINN,
    TrainableSiLU,
    projected_residual_objective,
)
from projected_sparse_training_common import (
    build_projected_support,
    export_results,
    load_body_state_compatible,
    split_epochs,
)
from utils import (
    FULL_PAULI_EXACT_MAX_QUBITS,
    ProjectedCommutator,
    SparsePauliOperator,
    all_local_pauli_labels,
    commutator_pauli_labels,
    pauli_training_regime,
    transverse_field_ising_problem,
)

class SparsePauliTests(unittest.TestCase):
    def test_reference_normalized_residual_is_invariant_to_common_scale(self):
        base = projected_residual_objective(
            torch.tensor(2.0),
            torch.tensor(8.0),
            mode="reference_normalized",
        )
        scaled = projected_residual_objective(
            torch.tensor(18.0),
            torch.tensor(72.0),
            mode="reference_normalized",
        )

        torch.testing.assert_close(base, torch.tensor(0.25))
        torch.testing.assert_close(scaled, base)

    def test_reference_normalized_residual_stops_reference_gradient(self):
        residual = torch.tensor(2.0, requires_grad=True)
        reference = torch.tensor(8.0, requires_grad=True)

        objective = projected_residual_objective(
            residual,
            reference,
            mode="reference_normalized",
        )
        objective.backward()

        torch.testing.assert_close(residual.grad, torch.tensor(0.125))
        self.assertIsNone(reference.grad)

    def test_projected_residual_objective_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "residual objective"):
            projected_residual_objective(
                torch.tensor(1.0),
                torch.tensor(1.0),
                mode="ground_truth_guided",
            )

    def test_body_state_can_warm_start_trainable_activation_from_fixed_silu(self):
        source = torch.nn.Sequential(
            torch.nn.Linear(1, 4),
            torch.nn.SiLU(),
            torch.nn.Linear(4, 2),
        )
        target = torch.nn.Sequential(
            torch.nn.Linear(1, 4),
            TrainableSiLU(),
            torch.nn.Linear(4, 2),
        )

        load_body_state_compatible(target, source.state_dict())

        self.assertIsNotNone(target[1].beta)

    def test_single_qubit_commutator(self):
        phase, label = commutator_pauli_labels("X", "Y")
        self.assertEqual(label, "Z")
        self.assertEqual(phase, 2j)

    def test_sparse_operator_commutator(self):
        x = SparsePauliOperator({"X": 1.0})
        y = SparsePauliOperator({"Y": 1.0})
        commutator = x.commutator(y)
        self.assertEqual(commutator.terms, {"Z": 2j})

    def test_default_training_regime_split(self):
        self.assertEqual(FULL_PAULI_EXACT_MAX_QUBITS, 8)
        self.assertEqual(pauli_training_regime(8), "full_pauli_exact")
        self.assertEqual(pauli_training_regime(9), "adaptive_projected_sparse")

    def test_projected_commutator(self):
        commutator = ProjectedCommutator(["X"], ["Y"], ["Z"])
        left = torch.tensor([[1.0]])
        right = torch.tensor([[2.0]])
        out = commutator.commutator(left, right)
        self.assertTrue(torch.allclose(out, torch.tensor([[0.0 + 4.0j]])))

    def test_generated_projected_support_can_be_expanded(self):
        h0, h1 = transverse_field_ising_problem(4)
        initial = build_projected_support(
            h0,
            h1,
            agp_top_k=2,
            intermediate_top_k=16,
            residual_top_k=16,
        )
        metadata = initial["metadata"]
        self.assertEqual(metadata["strategy"], "adaptive_generated_commutator_projected_residual")
        self.assertGreaterEqual(metadata["generated_residual_candidate_terms"], metadata["endpoint_commutator_terms"])
        self.assertEqual(len(initial["agp_labels"]), 2)

        expanded_labels = list(initial["agp_labels"]) + list(initial["residual_labels"])[:2]
        expanded = build_projected_support(
            h0,
            h1,
            agp_top_k=2,
            intermediate_top_k=16,
            residual_top_k=16,
            agp_labels=expanded_labels,
            stage=1,
        )
        self.assertGreaterEqual(len(expanded["agp_labels"]), len(initial["agp_labels"]))
        self.assertEqual(expanded["metadata"]["stage"], 1)

    def test_adaptive_epoch_split_preserves_total_budget(self):
        self.assertEqual(split_epochs(5, 2), [3, 2])
        self.assertEqual(sum(split_epochs(7, 4)), 7)

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
        self.assertIn("reference_residual", diagnostics)
        self.assertIn("relative_residual", diagnostics)
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
        self.assertIn("reference_residual", diagnostics)
        self.assertIn("relative_residual", diagnostics)
        self.assertTrue(any(param.grad is not None for param in model.parameters()))

    def test_projected_export_always_writes_connectivity_plots(self):
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
        tau = torch.linspace(0.0, 1.0, 4).view(-1, 1)
        history = [
            {"epoch": 0.0, "total": 1.0, "residual": 1.0, "relative_residual": 1.0},
            {"epoch": 1.0, "total": 0.5, "residual": 0.5, "relative_residual": 0.5},
        ]
        metadata = {
            "n_qubits": 2,
            "agp_terms": len(model.agp_labels),
            "intermediate_terms": len(model.intermediate_labels),
            "residual_terms": len(model.residual_labels),
            "hamiltonian_terms": len(model.hamiltonian_labels),
        }

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "Images"
            data_dir = root / "Models_Data"
            images_dir.mkdir()
            data_dir.mkdir()
            context_lines = ["H0", "H1", "E0", "EPINN"]
            with (
                patch(
                    "projected_sparse_training_common.hcd_context_lines_for_images_dir",
                    return_value=context_lines,
                ) as context_mock,
                patch("projected_sparse_training_common.draw_physical_footer") as draw_mock,
            ):
                export_results(model, tau, tau, images_dir, data_dir, metadata, history, top_k=4)

            context_mock.assert_called_once_with(images_dir)
            draw_mock.assert_called_once()
            self.assertEqual(draw_mock.call_args.args[1], context_lines)

            self.assertTrue((images_dir / "hcd_coefficient_support_map.pdf").is_file())
            self.assertTrue((images_dir / "hcd_least_important_coefficient_support_map.pdf").is_file())
            self.assertTrue((images_dir / "hcd_connection_summary.pdf").is_file())
            self.assertTrue((data_dir / "coefficient_plot_data.json").is_file())


if __name__ == "__main__":
    unittest.main()
