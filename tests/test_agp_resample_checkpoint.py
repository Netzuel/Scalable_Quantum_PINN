import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
DIAGNOSTICS_DIR = SCRIPTS_DIR / "diagnostics"
for path in (DIAGNOSTICS_DIR, SCRIPTS_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agp_resample_checkpoint import physical_export_sha256, resample_checkpoint_export
from projected_sparse_training_common import (
    ProjectedTrainingConfig,
    build_projected_support,
    default_config_payload,
    enable_projected_agp_calibration,
    enable_projected_trainable_schedule,
    make_projected_model,
    make_projected_export_model,
    projected_trainable_state,
    restore_projected_trainable_state,
    settings_from_payload,
)
from utils import load_pauli_hamiltonian_pair


class AGPCheckpointResamplingTests(unittest.TestCase):
    def test_factor_graph_checkpoint_state_rebuilds_deterministic_export_surface(self):
        model_config = ProjectedTrainingConfig(
            system="TransverseIsingDriverProblem",
            n_qubits=2,
            hidden_layers=1,
            hidden_width=12,
            activation="silu",
            layer_type="linear",
            coefficient_architecture="hamiltonian_pauli_factor_graph",
            graph_node_width=10,
            graph_message_layers=2,
            graph_term_width=16,
            graph_latent_rank=8,
            graph_time_fourier_order=2,
            graph_term_chunk_size=2,
        )
        payload = default_config_payload(model_config)
        settings = settings_from_payload(payload, model_config)
        h0, h1 = load_pauli_hamiltonian_pair(
            ROOT / model_config.hamiltonian_source,
            system=model_config.system,
            n_qubits=model_config.n_qubits,
            distance=model_config.distance,
        )
        support = build_projected_support(
            h0,
            h1,
            agp_top_k=4,
            intermediate_top_k=16,
            residual_top_k=16,
        )
        trained = make_projected_model(h0, h1, support, model_config, torch.device("cpu"))
        state = projected_trainable_state(trained)
        exported = make_projected_export_model(
            model_config,
            trained.agp_labels,
            torch.device("cpu"),
        )

        restore_projected_trainable_state(exported, state, settings=settings)
        times = torch.linspace(0.0, 1.0, 5).view(-1, 1)

        self.assertEqual(state["coefficient_architecture"], "hamiltonian_pauli_factor_graph")
        torch.testing.assert_close(exported(times)["agp_coefficients"], trained(times)["agp_coefficients"])

    def test_graph_checkpoint_state_rebuilds_deterministic_export_surface(self):
        model_config = ProjectedTrainingConfig(
            system="TransverseIsingDriverProblem",
            n_qubits=2,
            hidden_layers=1,
            hidden_width=8,
            activation="silu",
            layer_type="linear",
            coefficient_architecture="hamiltonian_pauli_graph",
            graph_node_width=8,
            graph_message_layers=1,
            graph_latent_rank=4,
            graph_term_chunk_size=2,
        )
        payload = default_config_payload(model_config)
        settings = settings_from_payload(payload, model_config)
        h0, h1 = load_pauli_hamiltonian_pair(
            ROOT / model_config.hamiltonian_source,
            system=model_config.system,
            n_qubits=model_config.n_qubits,
            distance=model_config.distance,
        )
        support = build_projected_support(
            h0,
            h1,
            agp_top_k=4,
            intermediate_top_k=16,
            residual_top_k=16,
        )
        trained = make_projected_model(h0, h1, support, model_config, torch.device("cpu"))
        state = projected_trainable_state(trained)
        exported = make_projected_export_model(
            model_config,
            trained.agp_labels,
            torch.device("cpu"),
        )

        restore_projected_trainable_state(exported, state, settings=settings)
        times = torch.linspace(0.0, 1.0, 5).view(-1, 1)

        self.assertEqual(state["coefficient_architecture"], "hamiltonian_pauli_graph")
        torch.testing.assert_close(exported(times)["agp_coefficients"], trained(times)["agp_coefficients"])

    def test_resampling_preserves_model_and_schedule_semantics(self):
        model_config = ProjectedTrainingConfig(
            system="TransverseIsingDriverProblem",
            n_qubits=2,
            hidden_layers=1,
            hidden_width=8,
            activation="silu",
            layer_type="linear",
        )
        payload = default_config_payload(model_config)
        payload["support"].update(
            {
                "agp_top_k": 4,
                "intermediate_top_k": 16,
                "residual_top_k": 16,
            }
        )
        payload["agp_calibration"].update(
            {
                "enabled": True,
                "target_active_terms": 2,
                "gate_temperature": 1.0,
            }
        )
        payload["schedule_optimization"].update(
            {
                "enabled": True,
                "base": "sinusoidal_sin2",
                "correction_amplitude": 2.4,
                "hidden_width": 6,
                "hidden_layers": 1,
                "activation": "tanh",
            }
        )
        settings = settings_from_payload(payload, model_config)
        h0, h1 = load_pauli_hamiltonian_pair(
            ROOT / model_config.hamiltonian_source,
            system=model_config.system,
            n_qubits=model_config.n_qubits,
            distance=model_config.distance,
        )
        support = build_projected_support(
            h0,
            h1,
            agp_top_k=settings.agp_top_k,
            intermediate_top_k=settings.intermediate_top_k,
            residual_top_k=settings.residual_top_k,
        )
        model = make_projected_model(h0, h1, support, model_config, torch.device("cpu"))
        enable_projected_trainable_schedule(model, settings)
        enable_projected_agp_calibration(model, settings, preferred_active_labels=model.agp_labels[:2])
        with torch.no_grad():
            model.agp_log_gamma.fill_(0.25)
            model.agp_gate_logits.copy_(torch.linspace(-2.0, 2.0, len(model.agp_labels)))
            for parameter in model.schedule_body.parameters():
                parameter.add_(0.01)
        model.eval()

        tau = torch.linspace(0.0, 1.0, 9).view(-1, 1)
        t = model_config.t_initial + model_config.physical_time * tau
        expected = model(t)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            trained_run = root / "trained"
            data_dir = trained_run / "Models_Data"
            data_dir.mkdir(parents=True)
            checkpoint_config = {
                "physical": asdict(model_config),
                "training": asdict(settings),
                "support": {
                    "agp_calibration": {
                        "target_active_terms": 2,
                        "gate_temperature": 1.0,
                    }
                },
                "source_config": payload,
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": checkpoint_config,
                    "agp_labels": model.agp_labels,
                    "intermediate_labels": model.intermediate_labels,
                    "residual_labels": model.residual_labels,
                    "hamiltonian_labels": model.hamiltonian_labels,
                },
                data_dir / "training_checkpoint.pt",
            )

            output_path = resample_checkpoint_export(
                config_path=config_path,
                trained_run=trained_run,
                output_dir=root / "resampled",
                num_points=9,
                device="cpu",
            )
            exported = torch.load(output_path, map_location="cpu")
            manifest = json.loads((output_path.parent / "resampling_manifest.json").read_text(encoding="utf-8"))
            resolved_source = json.loads(
                (output_path.parent / "resolved_source_config.json").read_text(encoding="utf-8")
            )

        self.assertEqual(exported["pauli_labels"], model.agp_labels)
        self.assertEqual(tuple(exported["t"].shape), (9, 1))
        torch.testing.assert_close(exported["counterdiabatic_coefficients"], expected["d_lambda_dt"] * expected["agp_coefficients"])
        torch.testing.assert_close(exported["d_lambda_d_tau"], expected["d_lambda_d_tau"])
        torch.testing.assert_close(
            exported["d_lambda_dt"],
            exported["d_lambda_d_tau"] / model_config.physical_time,
        )
        torch.testing.assert_close(exported["calibration_gates"], model.agp_calibration_gates())
        self.assertAlmostEqual(
            exported["calibration_gamma"],
            float(model.agp_calibration_gamma().detach()),
            places=6,
        )
        torch.testing.assert_close(exported["lambda"][[0, -1]].flatten(), torch.tensor([0.0, 1.0]), atol=1e-6, rtol=0.0)
        torch.testing.assert_close(exported["d_lambda_dt"][[0, -1]], torch.zeros(2, 1), atol=1e-5, rtol=0.0)
        self.assertEqual(manifest["num_points"], 9)
        self.assertEqual(manifest["uses_ground_truth_observables"], False)
        self.assertEqual(manifest["source_checkpoint_sha256"], exported["resampling_provenance"]["source_checkpoint_sha256"])
        self.assertEqual(manifest["source_config_origin"], "embedded_checkpoint")
        self.assertEqual(len(manifest["source_config_sha256"]), 64)
        self.assertEqual(
            manifest["physical_export_sha256"],
            physical_export_sha256(exported),
        )
        self.assertEqual(
            exported["resampling_provenance"]["physical_export_sha256"],
            manifest["physical_export_sha256"],
        )
        self.assertEqual(resolved_source, payload)


if __name__ == "__main__":
    unittest.main()
