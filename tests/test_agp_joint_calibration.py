import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
for path in (SCRIPTS_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models import ProjectedSparseLossWeights, calibration_budget_penalty
from agp_residual_calibration import build_gate_initial_logits, calibrated_agp_coefficients
from projected_sparse_training_common import (
    ProjectedRunSettings,
    ProjectedTrainingConfig,
    build_projected_support,
    default_config_payload,
    enable_projected_agp_calibration,
    enable_projected_trainable_schedule,
    make_optimizer,
    make_projected_model,
    projected_trainable_state,
    restore_projected_trainable_state,
    settings_from_payload,
)
from agp_baseline_train import settings_for_support
from agp_evaluate_holdout import load_body_weights as load_evaluate_holdout_weights
from agp_holdout_study import load_body_weights as load_holdout_study_weights
from utils import SparsePauliOperator


class AGPJointCalibrationTests(unittest.TestCase):
    def test_calibration_budget_can_normalize_by_target(self):
        active_sum = torch.tensor(6.0)

        support_normalized = calibration_budget_penalty(
            active_sum,
            target_active_terms=2,
            support_terms=8,
            mode="support",
        )
        target_normalized = calibration_budget_penalty(
            active_sum,
            target_active_terms=2,
            support_terms=8,
            mode="target",
        )

        torch.testing.assert_close(support_normalized, torch.tensor(0.25))
        torch.testing.assert_close(target_normalized, torch.tensor(4.0))

    def test_normalization_modes_are_configurable_and_backward_compatible(self):
        fallback = ProjectedTrainingConfig(system="unit", n_qubits=9)
        default_settings = settings_from_payload(default_config_payload(fallback), fallback)

        self.assertEqual(default_settings.residual_objective, "absolute")
        self.assertEqual(default_settings.calibration_budget_normalization, "support")

        payload = default_config_payload(fallback)
        payload["training"]["loss"]["residual_objective"] = "reference_normalized"
        payload["agp_calibration"]["budget_normalization"] = "target"
        configured = settings_from_payload(payload, fallback)

        self.assertEqual(configured.residual_objective, "reference_normalized")
        self.assertEqual(configured.calibration_budget_normalization, "target")

    def test_baseline_settings_preserve_normalized_objective_modes(self):
        fallback = ProjectedTrainingConfig(system="unit", n_qubits=9)
        payload = default_config_payload(fallback)
        payload["training"]["loss"]["residual_objective"] = "reference_normalized"
        payload["agp_calibration"]["budget_normalization"] = "target"

        settings = settings_for_support(payload, agp_terms=32)

        self.assertEqual(settings.residual_objective, "reference_normalized")
        self.assertEqual(settings.calibration_budget_normalization, "target")

    def test_per_qubit_active_budget_is_resolved_against_model_capacity(self):
        model, settings = self._model_and_settings()
        settings = ProjectedRunSettings(
            **{
                **settings.__dict__,
                "calibration_target_active_terms": {
                    "mode": "per_qubit",
                    "per_qubit": 3.0,
                    "minimum": 1,
                },
            }
        )

        enable_projected_agp_calibration(model, settings)

        self.assertEqual(model.agp_target_active_terms, len(model.agp_labels))
        self.assertEqual(model.agp_target_active_budget["requested"], 6)
        self.assertEqual(model.agp_target_active_budget["realized"], len(model.agp_labels))
        self.assertEqual(model.agp_target_active_budget["clipping_reasons"], ("capacity",))

    def test_config_preserves_per_qubit_active_budget_spec(self):
        fallback = ProjectedTrainingConfig(system="unit", n_qubits=9)
        payload = default_config_payload(fallback)
        spec = {
            "mode": "per_qubit",
            "per_qubit": 102.4,
            "minimum": 2048,
            "maximum": 32768,
        }
        payload["agp_calibration"]["target_active_terms"] = spec

        settings = settings_from_payload(payload, fallback)

        self.assertEqual(settings.calibration_target_active_terms, spec)

    def _model_and_settings(self):
        h0 = SparsePauliOperator({"XI": -1.0, "IX": -1.0}, n_qubits=2)
        h1 = SparsePauliOperator({"ZI": -1.0, "IZ": -1.0, "ZZ": -0.5}, n_qubits=2)
        support = build_projected_support(
            h0,
            h1,
            agp_top_k=4,
            intermediate_top_k=16,
            residual_top_k=16,
        )
        config = ProjectedTrainingConfig(
            system="unit",
            n_qubits=2,
            hidden_layers=1,
            hidden_width=8,
            layer_type="linear",
        )
        settings = ProjectedRunSettings(
            model=config,
            optimizer="AdamW",
            lr=1e-3,
            calibration_enabled=True,
            calibration_target_active_terms=2,
            calibration_gamma_lr=0.03,
            calibration_gate_lr=0.04,
            calibration_active_logit=3.0,
            calibration_inactive_logit=-5.0,
            calibration_budget_weight=1.0,
            calibration_binary_weight=0.1,
            calibration_scale_l2_weight=0.2,
        )
        model = make_projected_model(h0, h1, support, config, torch.device("cpu"))
        return model, settings

    def test_joint_calibration_parameters_train_inside_projected_loss(self):
        model, settings = self._model_and_settings()

        enable_projected_agp_calibration(
            model,
            settings,
            preferred_active_labels=model.agp_labels[: settings.calibration_target_active_terms],
        )
        optimizer, info = make_optimizer(model, settings)
        t = torch.linspace(0.0, 1.0, 4).view(-1, 1)
        weights = ProjectedSparseLossWeights(
            residual=settings.residual_weight,
            agp_l2=settings.agp_l2_weight,
            calibration_budget=settings.calibration_budget_weight,
            calibration_binary=settings.calibration_binary_weight,
            calibration_scale_l2=settings.calibration_scale_l2_weight,
        )

        loss, diagnostics = model.loss(t, weights=weights)
        loss.backward()

        self.assertEqual(info["calibration"], "joint_trainable")
        self.assertIn(settings.calibration_gamma_lr, {group["lr"] for group in optimizer.param_groups})
        self.assertIn(settings.calibration_gate_lr, {group["lr"] for group in optimizer.param_groups})
        self.assertIn("calibration_gamma", diagnostics)
        self.assertIn("calibration_active_gate_sum", diagnostics)
        self.assertIsNotNone(model.agp_log_gamma.grad)
        self.assertIsNotNone(model.agp_gate_logits.grad)

    def test_joint_calibration_state_restores_between_rounds(self):
        model, settings = self._model_and_settings()
        enable_projected_agp_calibration(model, settings, preferred_active_labels=model.agp_labels[:2])
        with torch.no_grad():
            model.agp_log_gamma.fill_(0.25)
            model.agp_gate_logits[0] = 1.5
        state = projected_trainable_state(model)
        restored, _ = self._model_and_settings()

        restore_projected_trainable_state(restored, state, settings=settings)

        self.assertTrue(hasattr(restored, "agp_log_gamma"))
        self.assertAlmostEqual(float(restored.agp_log_gamma.detach()), 0.25, places=6)
        self.assertAlmostEqual(float(restored.agp_gate_logits.detach()[0]), 1.5, places=6)

    def test_trainable_schedule_preserves_endpoint_constraints_and_trains(self):
        model, settings = self._model_and_settings()
        settings = ProjectedRunSettings(
            **{
                **settings.__dict__,
                "schedule_trainable_enabled": True,
                "schedule_lr": 0.02,
                "schedule_hidden_width": 6,
                "schedule_hidden_layers": 1,
                "schedule_correction_amplitude": 2.4,
            }
        )

        enable_projected_trainable_schedule(model, settings)
        optimizer, info = make_optimizer(model, settings)
        t = torch.linspace(0.0, 1.0, 5).view(-1, 1)
        prediction = model(t)
        loss, diagnostics = model.loss(t, weights=ProjectedSparseLossWeights(schedule_monotonic=1.0))
        loss.backward()

        self.assertEqual(info["schedule"], "joint_trainable")
        self.assertIn(settings.schedule_lr, {group["lr"] for group in optimizer.param_groups})
        self.assertTrue(torch.allclose(prediction["lambda"][[0, -1]].flatten(), torch.tensor([0.0, 1.0]), atol=1e-6))
        self.assertTrue(torch.allclose(prediction["d_lambda_dt"][[0, -1]], torch.zeros(2, 1), atol=1e-5))
        self.assertGreaterEqual(float(torch.min(prediction["lambda"]).detach()), -1e-6)
        self.assertLessEqual(float(torch.max(prediction["lambda"]).detach()), 1.0 + 1e-6)
        self.assertIn("schedule_monotonic", diagnostics)
        schedule_grads = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith("schedule_body.")
        ]
        self.assertTrue(any(grad is not None for grad in schedule_grads))

    def test_holdout_checkpoint_loaders_restore_the_trainable_schedule(self):
        source, settings = self._model_and_settings()
        settings = ProjectedRunSettings(
            **{
                **settings.__dict__,
                "schedule_trainable_enabled": True,
                "schedule_hidden_width": 6,
                "schedule_hidden_layers": 1,
                "schedule_activation": "tanh",
                "schedule_correction_amplitude": 2.4,
            }
        )
        enable_projected_trainable_schedule(source, settings)
        with torch.no_grad():
            for parameter in source.schedule_body.parameters():
                parameter.fill_(0.125)

        checkpoint = {
            "model_state_dict": source.state_dict(),
            "agp_labels": list(source.agp_labels),
            "config": {
                "training": {
                    "schedule_trainable_enabled": True,
                    "schedule_hidden_width": settings.schedule_hidden_width,
                    "schedule_hidden_layers": settings.schedule_hidden_layers,
                    "schedule_activation": settings.schedule_activation,
                    "schedule_base": settings.schedule_base,
                    "schedule_correction_amplitude": settings.schedule_correction_amplitude,
                }
            },
        }
        sample_t = torch.tensor([[0.5]])
        expected_lambda = source.schedule(sample_t)[0]

        for loader in (load_holdout_study_weights, load_evaluate_holdout_weights):
            with self.subTest(loader=loader.__module__):
                restored, _ = self._model_and_settings()
                loader(restored, checkpoint)

                self.assertTrue(restored.has_trainable_schedule())
                torch.testing.assert_close(restored.schedule(sample_t)[0], expected_lambda)
                for key, expected in source.schedule_body.state_dict().items():
                    torch.testing.assert_close(restored.schedule_body.state_dict()[key], expected)

    def test_gate_initial_logits_select_top_rms_terms(self):
        logits = build_gate_initial_logits(
            torch.tensor([0.1, 2.0, 0.5, 3.0]),
            target_active_terms=2,
            active_logit=4.0,
            inactive_logit=-4.0,
        )

        self.assertEqual(logits.tolist(), [-4.0, 4.0, -4.0, 4.0])

    def test_calibrated_agp_coefficients_apply_gamma_and_soft_gates(self):
        raw = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        log_gamma = torch.log(torch.tensor(2.0))
        gate_logits = torch.tensor([0.0, 20.0])

        calibrated, gamma, gates = calibrated_agp_coefficients(
            raw,
            log_gamma=log_gamma,
            gate_logits=gate_logits,
            gate_temperature=1.0,
        )

        self.assertAlmostEqual(float(gamma), 2.0, places=6)
        self.assertAlmostEqual(float(gates[0]), 0.5, places=6)
        self.assertAlmostEqual(float(gates[1]), 1.0, places=6)
        torch.testing.assert_close(calibrated, torch.tensor([[1.0, 4.0]]))


if __name__ == "__main__":
    unittest.main()
