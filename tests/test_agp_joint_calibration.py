import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
for path in (SCRIPTS_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models import ProjectedSparseLossWeights
from agp_residual_calibration import build_gate_initial_logits, calibrated_agp_coefficients
from projected_sparse_training_common import (
    ProjectedRunSettings,
    ProjectedTrainingConfig,
    build_projected_support,
    enable_projected_agp_calibration,
    enable_projected_trainable_schedule,
    make_optimizer,
    make_projected_model,
    projected_trainable_state,
    restore_projected_trainable_state,
)
from utils import SparsePauliOperator


class AGPJointCalibrationTests(unittest.TestCase):
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
