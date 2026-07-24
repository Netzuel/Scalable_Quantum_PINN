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
    HamiltonianPauliGraphCoefficientNetwork,
    HamiltonianPauliFactorGraphCoefficientNetwork,
    ProjectedSparseAGPPINN,
    ProjectedSparseLossWeights,
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
    PauliAlgebra,
    ProjectedCommutator,
    SparsePauliOperator,
    all_local_pauli_labels,
    commutator_pauli_labels,
    pauli_training_regime,
    transverse_field_ising_problem,
    hamiltonian_pauli_factor_graph_data,
)

class SparsePauliTests(unittest.TestCase):
    def test_variational_action_is_normalized_to_zero_agp_reference(self):
        model = ProjectedSparseAGPPINN(
            SparsePauliOperator({"X": -1.0}),
            SparsePauliOperator({"Z": -1.0}),
            ["Y"],
            ["X", "Y", "Z"],
            ["X", "Z"],
            hidden_width=4,
            hidden_layers=1,
            layer_type="linear",
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        t = torch.tensor([[0.5]], dtype=torch.float32)

        base_total, base = model.loss(
            t,
            weights=ProjectedSparseLossWeights(
                residual=1.0,
                agp_l2=0.0,
                variational_action=0.0,
            ),
        )
        weighted_total, weighted = model.loss(
            t,
            weights=ProjectedSparseLossWeights(
                residual=1.0,
                agp_l2=0.0,
                variational_action=0.25,
            ),
        )

        torch.testing.assert_close(
            base["variational_action"],
            base["reference_variational_action"],
        )
        torch.testing.assert_close(
            base["relative_variational_action"],
            torch.ones_like(base["relative_variational_action"]),
        )
        torch.testing.assert_close(
            weighted_total - base_total,
            torch.tensor(0.25),
        )
        torch.testing.assert_close(
            weighted["relative_variational_action"],
            base["relative_variational_action"],
        )

    def test_projected_residual_order_blocks_have_equal_total_weight(self):
        model = ProjectedSparseAGPPINN(
            SparsePauliOperator({"XII": -1.0}),
            SparsePauliOperator({"ZII": 1.0}),
            ["YII"],
            ["XII", "YII", "ZII"],
            ["YII", "IYI", "IIY", "YYY"],
            hidden_width=4,
            hidden_layers=1,
            layer_type="linear",
        )

        weights = model.residual_block_weights
        order_one = weights[:3].sum()
        order_three = weights[3:].sum()

        torch.testing.assert_close(order_one, torch.tensor(0.5))
        torch.testing.assert_close(order_three, torch.tensor(0.5))

    @staticmethod
    def _trainable_parameter_count(module: torch.nn.Module) -> int:
        return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)

    def test_graph_coefficient_network_parameter_count_is_independent_of_q_and_k(self):
        h0_small, h1_small = transverse_field_ising_problem(2)
        h0_large, h1_large = transverse_field_ising_problem(5)
        small = HamiltonianPauliGraphCoefficientNetwork(
            h0_small,
            h1_small,
            ["YI", "IY"],
            hidden_width=12,
            hidden_layers=1,
            activation="silu",
            layer_type="linear",
            node_width=10,
            message_layers=2,
            latent_rank=6,
        )
        large = HamiltonianPauliGraphCoefficientNetwork(
            h0_large,
            h1_large,
            [
                "YIIII",
                "IYIII",
                "IIYII",
                "IIIYI",
                "IIIIY",
                "YZIII",
                "IYZII",
            ],
            hidden_width=12,
            hidden_layers=1,
            activation="silu",
            layer_type="linear",
            node_width=10,
            message_layers=2,
            latent_rank=6,
        )

        self.assertEqual(
            self._trainable_parameter_count(small),
            self._trainable_parameter_count(large),
        )
        self.assertEqual(small(torch.linspace(0.0, 1.0, 4).view(-1, 1)).shape, (4, 2))
        self.assertEqual(large(torch.linspace(0.0, 1.0, 4).view(-1, 1)).shape, (4, 7))

    def test_factor_graph_coefficient_network_parameter_count_is_independent_of_q_and_k(self):
        h0_small, h1_small = transverse_field_ising_problem(2)
        h0_large, h1_large = transverse_field_ising_problem(5)
        kwargs = dict(
            hidden_width=16,
            hidden_layers=1,
            activation="silu",
            layer_type="linear",
            node_width=12,
            message_layers=2,
            term_width=20,
            latent_rank=8,
            time_fourier_order=2,
        )
        small = HamiltonianPauliFactorGraphCoefficientNetwork(
            h0_small, h1_small, ["YI", "IY"], **kwargs
        )
        large = HamiltonianPauliFactorGraphCoefficientNetwork(
            h0_large,
            h1_large,
            ["YIIII", "IYIII", "IIYII", "IIIYI", "IIIIY", "YZIII", "IYZII"],
            **kwargs,
        )

        self.assertEqual(
            self._trainable_parameter_count(small),
            self._trainable_parameter_count(large),
        )
        self.assertEqual(small(torch.linspace(0.0, 1.0, 4).view(-1, 1)).shape, (4, 2))
        self.assertEqual(large(torch.linspace(0.0, 1.0, 4).view(-1, 1)).shape, (4, 7))

    def test_factor_graph_data_preserves_coefficient_sign_and_higher_body_identity(self):
        h0 = SparsePauliOperator({"XII": -1.0})
        positive = SparsePauliOperator({"ZZZ": 0.7})
        negative = SparsePauliOperator({"ZZZ": -0.7})
        pair_only = SparsePauliOperator({"ZZI": 0.7, "IZZ": 0.7, "ZIZ": 0.7})
        labels = ["YII", "IYI", "IIY", "YZZ"]

        positive_data = hamiltonian_pauli_factor_graph_data(h0, positive, labels)
        negative_data = hamiltonian_pauli_factor_graph_data(h0, negative, labels)
        pair_data = hamiltonian_pauli_factor_graph_data(h0, pair_only, labels)

        self.assertFalse(torch.equal(positive_data.factor_features, negative_data.factor_features))
        self.assertNotEqual(positive_data.factor_features.shape[0], pair_data.factor_features.shape[0])
        self.assertEqual(positive_data.factor_indices.unique().numel(), 2)

    def test_factor_graph_coefficient_network_is_qubit_permutation_equivariant(self):
        permutation = (2, 0, 1)

        def permute_label(label: str) -> str:
            output = ["I"] * len(label)
            for old_site, new_site in enumerate(permutation):
                output[new_site] = label[old_site]
            return "".join(output)

        h0 = SparsePauliOperator({"XII": -0.7, "IZI": 0.2, "YIX": -0.4j})
        h1 = SparsePauliOperator({"ZZZ": -1.1, "IXX": 0.35, "YIZ": 0.6j})
        labels = ["YII", "IXZ", "ZYI", "XYZ"]
        permuted_h0 = SparsePauliOperator(
            {permute_label(label): coefficient for label, coefficient in h0.terms.items()}
        )
        permuted_h1 = SparsePauliOperator(
            {permute_label(label): coefficient for label, coefficient in h1.terms.items()}
        )
        kwargs = dict(
            hidden_width=12,
            hidden_layers=1,
            activation="silu",
            layer_type="linear",
            node_width=10,
            message_layers=2,
            term_width=16,
            latent_rank=8,
            time_fourier_order=2,
        )
        model = HamiltonianPauliFactorGraphCoefficientNetwork(h0, h1, labels, **kwargs)
        permuted = HamiltonianPauliFactorGraphCoefficientNetwork(
            permuted_h0,
            permuted_h1,
            [permute_label(label) for label in labels],
            **kwargs,
        )
        permuted.load_state_dict(model.state_dict())

        time = torch.linspace(0.0, 1.0, 4).view(-1, 1)
        original_output = model(time)
        permuted_output = permuted(time)
        permuted_index = {label: index for index, label in enumerate(permuted.agp_labels)}
        reordered = torch.stack(
            [permuted_output[:, permuted_index[permute_label(label)]] for label in model.agp_labels],
            dim=1,
        )

        torch.testing.assert_close(original_output, reordered, rtol=1e-6, atol=1e-6)

    def test_graph_coefficient_network_is_differentiable_and_support_buffers_are_nonpersistent(self):
        h0, h1 = transverse_field_ising_problem(3)
        model = HamiltonianPauliGraphCoefficientNetwork(
            h0,
            h1,
            ["YII", "IYI", "IIY", "YZI"],
            hidden_width=8,
            hidden_layers=1,
            activation="silu",
            layer_type="linear",
            node_width=8,
            message_layers=1,
            latent_rank=4,
        )

        coefficients = model(torch.linspace(0.0, 1.0, 3).view(-1, 1))
        coefficients.square().mean().backward()

        self.assertTrue(torch.isfinite(coefficients).all())
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        state_keys = set(model.state_dict())
        self.assertFalse(any(key.startswith("node_features") for key in state_keys))
        self.assertFalse(any(key.startswith("term_") and key.endswith("indices") for key in state_keys))

    def test_graph_coefficient_network_is_qubit_permutation_equivariant(self):
        permutation = (2, 0, 1)

        def permute_label(label: str) -> str:
            output = ["I"] * len(label)
            for old_site, new_site in enumerate(permutation):
                output[new_site] = label[old_site]
            return "".join(output)

        h0 = SparsePauliOperator({"XII": -0.7, "IZI": 0.2, "YIX": -0.4j})
        h1 = SparsePauliOperator({"ZII": -1.1, "IXX": 0.35, "YIZ": 0.6j})
        labels = ["YII", "IXZ", "ZYI", "XYZ"]
        permuted_h0 = SparsePauliOperator(
            {permute_label(label): coefficient for label, coefficient in h0.terms.items()}
        )
        permuted_h1 = SparsePauliOperator(
            {permute_label(label): coefficient for label, coefficient in h1.terms.items()}
        )
        model = HamiltonianPauliGraphCoefficientNetwork(
            h0,
            h1,
            labels,
            hidden_width=8,
            hidden_layers=1,
            activation="silu",
            layer_type="linear",
            node_width=8,
            message_layers=2,
            latent_rank=5,
        )
        permuted = HamiltonianPauliGraphCoefficientNetwork(
            permuted_h0,
            permuted_h1,
            [permute_label(label) for label in labels],
            hidden_width=8,
            hidden_layers=1,
            activation="silu",
            layer_type="linear",
            node_width=8,
            message_layers=2,
            latent_rank=5,
        )
        permuted.load_state_dict(model.state_dict())

        time = torch.linspace(0.0, 1.0, 4).view(-1, 1)
        original_output = model(time)
        permuted_output = permuted(time)
        permuted_index = {label: index for index, label in enumerate(permuted.agp_labels)}
        reordered = torch.stack(
            [permuted_output[:, permuted_index[permute_label(label)]] for label in model.agp_labels],
            dim=1,
        )

        torch.testing.assert_close(original_output, reordered, rtol=1e-6, atol=1e-6)

    def test_projected_graph_pinn_preserves_complete_k_output_contract(self):
        h0, h1 = transverse_field_ising_problem(2)
        labels = all_local_pauli_labels(2, max_weight=2)
        agp_labels = labels[:5]
        model = ProjectedSparseAGPPINN(
            h0,
            h1,
            agp_labels=agp_labels,
            intermediate_labels=labels,
            residual_labels=labels,
            hidden_width=8,
            hidden_layers=1,
            layer_type="linear",
            coefficient_architecture="hamiltonian_pauli_graph",
            graph_node_width=8,
            graph_message_layers=1,
            graph_latent_rank=4,
        )

        prediction = model(torch.linspace(0.0, 1.0, 3).view(-1, 1))

        self.assertEqual(model.coefficient_architecture, "hamiltonian_pauli_graph")
        self.assertEqual(prediction["agp_coefficients"].shape, (3, len(agp_labels)))

    def test_projected_factor_graph_pinn_preserves_complete_k_output_contract(self):
        h0, h1 = transverse_field_ising_problem(2)
        labels = all_local_pauli_labels(2, max_weight=2)
        agp_labels = labels[:5]
        model = ProjectedSparseAGPPINN(
            h0,
            h1,
            agp_labels=agp_labels,
            intermediate_labels=labels,
            residual_labels=labels,
            hidden_width=12,
            hidden_layers=1,
            layer_type="linear",
            coefficient_architecture="hamiltonian_pauli_factor_graph",
            graph_node_width=10,
            graph_message_layers=2,
            graph_term_width=16,
            graph_latent_rank=8,
            graph_time_fourier_order=2,
        )

        prediction = model(torch.linspace(0.0, 1.0, 3).view(-1, 1))
        prediction["agp_coefficients"].square().mean().backward()

        self.assertEqual(model.coefficient_architecture, "hamiltonian_pauli_factor_graph")
        self.assertEqual(prediction["agp_coefficients"].shape, (3, len(agp_labels)))
        self.assertTrue(all(parameter.grad is not None for parameter in model.body.parameters()))
        self.assertFalse(any(key.startswith("body.factor_features") for key in model.state_dict()))

    def test_projected_loss_reuses_one_coefficient_forward_pass(self):
        h0, h1 = transverse_field_ising_problem(2)
        labels = all_local_pauli_labels(2, max_weight=2)
        model = ProjectedSparseAGPPINN(
            h0,
            h1,
            agp_labels=labels[:5],
            intermediate_labels=labels,
            residual_labels=labels,
            hidden_width=8,
            hidden_layers=1,
            layer_type="linear",
        )
        calls = []
        handle = model.body.register_forward_hook(lambda *_: calls.append(1))
        try:
            loss, diagnostics = model.loss(torch.linspace(0.0, 1.0, 4).view(-1, 1))
        finally:
            handle.remove()

        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(diagnostics["residual"]))

        direct_residual = PauliAlgebra.norm_sq(model.euler_lagrange_residual(
            torch.linspace(0.0, 1.0, 4).view(-1, 1)
        ))
        direct_reference = PauliAlgebra.norm_sq(model.euler_lagrange_reference_residual(
            torch.linspace(0.0, 1.0, 4).view(-1, 1)
        ))
        torch.testing.assert_close(diagnostics["residual"], direct_residual)
        torch.testing.assert_close(diagnostics["reference_residual"], direct_reference)

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
