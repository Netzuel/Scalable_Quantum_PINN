import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
for path in (SCRIPTS_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agp_residual_calibration import build_gate_initial_logits, calibrated_agp_coefficients


class AGPResidualCalibrationTests(unittest.TestCase):
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
