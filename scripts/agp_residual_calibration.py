from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agp_baseline_train import settings_for_support  # noqa: E402
from agp_holdout_feedback import make_support_with_residual_labels  # noqa: E402
from agp_physical_validation import configure_run_dir as configure_physical_run_dir  # noqa: E402
from agp_physical_validation import final_run_from_summary  # noqa: E402
from projected_sparse_training_common import (  # noqa: E402
    make_projected_model,
    projected_trainable_state_from_checkpoint,
    rank_coefficients,
    restore_projected_trainable_state,
    select_device,
)
from utils import load_pauli_hamiltonian_pair  # noqa: E402


RUN_DIR = Path.cwd()
DEFAULT_CONFIG = Path("config.json")


def configure_run_dir(config_path: Path) -> None:
    global RUN_DIR
    RUN_DIR = config_path.resolve().parent
    configure_physical_run_dir(config_path)


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return payload


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def build_gate_initial_logits(
    rms: torch.Tensor,
    *,
    target_active_terms: int,
    active_logit: float,
    inactive_logit: float,
) -> torch.Tensor:
    if rms.ndim != 1:
        raise ValueError("rms must be a one-dimensional tensor.")
    target = max(1, min(int(target_active_terms), int(rms.numel())))
    logits = torch.full_like(rms, float(inactive_logit), dtype=torch.float32)
    top_indices = torch.topk(rms.detach().float().cpu(), k=target).indices
    logits[top_indices] = float(active_logit)
    return logits


def calibrated_agp_coefficients(
    raw_coefficients: torch.Tensor,
    *,
    log_gamma: torch.Tensor,
    gate_logits: torch.Tensor,
    gate_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gamma = torch.exp(log_gamma)
    gates = torch.sigmoid(gate_logits / float(gate_temperature)).to(raw_coefficients.device)
    return gamma * raw_coefficients * gates, gamma, gates


def calibrated_residual_loss(
    model,
    t: torch.Tensor,
    *,
    log_gamma: torch.Tensor,
    gate_logits: torch.Tensor,
    gate_temperature: float,
    residual_block_normalization: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    t = model._time_column(t)
    prediction = model.forward(t)
    calibrated_coefficients, gamma, gates = calibrated_agp_coefficients(
        prediction["agp_coefficients"],
        log_gamma=log_gamma,
        gate_logits=gate_logits,
        gate_temperature=gate_temperature,
    )
    lam = prediction["lambda"].to(model.h_initial_sparse.dtype)
    h0 = model.h_initial_sparse.to(t.device)
    h1 = model.h_final_sparse.to(t.device)
    h_ad_sparse = (1.0 - lam) * h0 + lam * h1
    d_h_d_lambda = model.h_delta_intermediate.to(t.device).expand(
        calibrated_coefficients.shape[:-1] + model.h_delta_intermediate.shape
    )
    commutator_1 = model.first_commutator.commutator(calibrated_coefficients, h_ad_sparse)
    generator = 1.0j * d_h_d_lambda - commutator_1
    residual = model.second_commutator.commutator(generator, h_ad_sparse)
    reference_generator = 1.0j * d_h_d_lambda
    reference = model.second_commutator.commutator(reference_generator, h_ad_sparse)
    if str(residual_block_normalization).lower() in {"pauli_order", "order", "by_order"}:
        block_weights = model.residual_block_weights.to(device=residual.device, dtype=residual.real.dtype)
        residual_loss = torch.mean(torch.sum(torch.abs(residual) ** 2 * block_weights, dim=-1).real)
        reference_loss = torch.mean(torch.sum(torch.abs(reference) ** 2 * block_weights, dim=-1).real)
    else:
        residual_loss = torch.mean(torch.sum(torch.abs(residual) ** 2, dim=-1).real)
        reference_loss = torch.mean(torch.sum(torch.abs(reference) ** 2, dim=-1).real)
    eps = torch.finfo(residual_loss.dtype).eps
    relative = residual_loss / torch.clamp(reference_loss, min=eps)
    return relative, {
        "residual": residual_loss.detach(),
        "reference_residual": reference_loss.detach(),
        "relative_residual": relative.detach(),
        "gamma": gamma.detach(),
        "active_gate_sum": torch.sum(gates.detach()),
        "mean_gate": torch.mean(gates.detach()),
    }


def output_run_for_source(source_run: Path, suffix: str) -> Path:
    return source_run.with_name(f"{source_run.name}_{suffix}")


def load_trained_projected_model(
    *,
    payload: dict[str, object],
    checkpoint_path: Path,
    device: torch.device,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    agp_labels = [str(label) for label in checkpoint["agp_labels"]]
    residual_labels = [str(label) for label in checkpoint["residual_labels"]]
    settings = settings_for_support(payload, len(agp_labels))
    config = settings.model
    hamiltonian_path = Path(config.hamiltonian_source)
    if not hamiltonian_path.is_absolute():
        hamiltonian_path = ROOT / hamiltonian_path
    h0, h1 = load_pauli_hamiltonian_pair(
        hamiltonian_path,
        system=config.system,
        n_qubits=config.n_qubits,
        distance=config.distance,
    )
    support = make_support_with_residual_labels(
        h0=h0,
        h1=h1,
        settings=settings,
        agp_labels=agp_labels,
        residual_labels=residual_labels,
        stage=0,
    )
    model = make_projected_model(h0, h1, support, config, device)
    restore_projected_trainable_state(
        model,
        projected_trainable_state_from_checkpoint(checkpoint_path),
        settings=settings,
    )
    return model, settings, checkpoint


def export_calibrated_coefficients(
    *,
    model,
    t: torch.Tensor,
    tau: torch.Tensor,
    output_dir: Path,
    metadata: dict[str, object],
    history: list[dict[str, float]],
    log_gamma: torch.Tensor,
    gate_logits: torch.Tensor,
    gate_temperature: float,
    top_k: int,
) -> None:
    data_dir = output_dir / "Models_Data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        prediction = model(t)
        agp_coefficients, gamma, gates = calibrated_agp_coefficients(
            prediction["agp_coefficients"],
            log_gamma=log_gamma,
            gate_logits=gate_logits,
            gate_temperature=gate_temperature,
        )
        d_lambda_dt = prediction["d_lambda_dt"].detach().cpu()
        hcd_coefficients = d_lambda_dt * agp_coefficients.detach().cpu()
        tau_cpu = tau.detach().cpu()
        t_cpu = t.detach().cpu()
        agp_cpu = agp_coefficients.detach().cpu()
        gates_cpu = gates.detach().cpu()

    ranked = rank_coefficients(hcd_coefficients, model.agp_labels)
    torch.save(
        {
            "t": t_cpu,
            "tau": tau_cpu,
            "pauli_labels": model.agp_labels,
            "agp_coefficients": agp_cpu,
            "counterdiabatic_coefficients": hcd_coefficients,
            "counterdiabatic_coefficient_definition": "d_lambda_dt * gamma * g_P * C_P(t)",
            "lambda": prediction["lambda"].detach().cpu(),
            "d_lambda_dt": d_lambda_dt,
            "calibration_gamma": float(gamma.detach().cpu()),
            "calibration_gates": gates_cpu,
            "support_metadata": metadata,
        },
        data_dir / "final_agp_coefficients.pt",
    )
    with (data_dir / "loss_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
        handle.write("\n")
    with (data_dir / "support_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    with (data_dir / "coefficient_importance.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "coefficient_kind": "residual_calibrated_projected_counterdiabatic_hamiltonian",
                "coefficient_definition": "d_lambda_dt * gamma * g_P * C_P(t)",
                "ranking_metric": "rms_over_time",
                "all_terms": ranked,
                "top_terms": ranked[:top_k],
            },
            handle,
            indent=2,
        )
        handle.write("\n")


def run_calibration(
    *,
    payload: dict[str, object],
    source_run: Path,
    output_run: Path,
    calibration: Mapping[str, object],
) -> dict[str, object]:
    checkpoint_path = source_run / "Models_Data" / "training_checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing source checkpoint: {checkpoint_path}")
    device = select_device(str(calibration.get("device", payload.get("training", {}).get("device", "auto") if isinstance(payload.get("training"), dict) else "auto")))
    model, settings, checkpoint = load_trained_projected_model(
        payload=payload,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    fine_tune_body = bool(calibration.get("fine_tune_body", False))
    model.train(mode=fine_tune_body)
    for parameter in model.parameters():
        parameter.requires_grad_(fine_tune_body)

    num_points = int(calibration.get("num_points", settings.num_points))
    tau = torch.linspace(0.0, 1.0, num_points, device=device).view(-1, 1)
    t = settings.model.t_initial + settings.model.physical_time * tau
    with torch.no_grad():
        prediction = model(t)
        hcd = prediction["d_lambda_dt"] * prediction["agp_coefficients"]
        rms = torch.sqrt(torch.mean(hcd.detach().float().cpu() ** 2, dim=0))

    target_active_terms = int(calibration.get("target_active_terms", min(len(model.agp_labels), 1024)))
    gate_logits = torch.nn.Parameter(
        build_gate_initial_logits(
            rms,
            target_active_terms=target_active_terms,
            active_logit=float(calibration.get("active_logit", 4.0)),
            inactive_logit=float(calibration.get("inactive_logit", -8.0)),
        ).to(device)
    )
    initial_gamma = float(calibration.get("initial_gamma", 1.0))
    log_gamma = torch.nn.Parameter(torch.tensor(math.log(max(initial_gamma, 1e-8)), dtype=torch.float32, device=device))
    parameters: list[dict[str, object]] = [
        {"params": [log_gamma], "lr": float(calibration.get("gamma_lr", calibration.get("lr", 1e-2)))},
        {"params": [gate_logits], "lr": float(calibration.get("gate_lr", calibration.get("lr", 1e-2)))},
    ]
    body_lr = float(calibration.get("body_lr", 0.0))
    if fine_tune_body and body_lr > 0.0:
        parameters.append({"params": model.parameters(), "lr": body_lr})
    optimizer = torch.optim.AdamW(parameters, weight_decay=float(calibration.get("weight_decay", 0.0)))
    epochs = int(calibration.get("epochs", 1000))
    gate_temperature = float(calibration.get("gate_temperature", 1.0))
    budget_weight = float(calibration.get("budget_weight", 1.0))
    binary_weight = float(calibration.get("binary_weight", 1e-3))
    scale_l2_weight = float(calibration.get("scale_l2_weight", 1e-4))
    residual_block_normalization = str(calibration.get("residual_block_normalization", settings.residual_block_normalization))
    history: list[dict[str, float]] = []
    log_every = max(1, int(calibration.get("log_every", max(epochs // 10, 1))))

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        relative, diagnostics = calibrated_residual_loss(
            model,
            t,
            log_gamma=log_gamma,
            gate_logits=gate_logits,
            gate_temperature=gate_temperature,
            residual_block_normalization=residual_block_normalization,
        )
        gates = torch.sigmoid(gate_logits / gate_temperature)
        budget_loss = ((torch.sum(gates) - float(target_active_terms)) / max(len(model.agp_labels), 1)) ** 2
        binary_loss = torch.mean(gates * (1.0 - gates))
        scale_l2 = (torch.exp(log_gamma) - 1.0) ** 2
        total = relative + budget_weight * budget_loss + binary_weight * binary_loss + scale_l2_weight * scale_l2
        total.backward()
        optimizer.step()
        row = {
            "epoch": epoch,
            "total": scalar(total),
            "relative_residual": scalar(relative),
            "residual": scalar(diagnostics["residual"]),
            "reference_residual": scalar(diagnostics["reference_residual"]),
            "gamma": scalar(torch.exp(log_gamma.detach())),
            "active_gate_sum": scalar(torch.sum(torch.sigmoid(gate_logits.detach() / gate_temperature))),
            "budget_loss": scalar(budget_loss),
            "binary_loss": scalar(binary_loss),
            "scale_l2": scalar(scale_l2),
        }
        history.append(row)
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % log_every == 0:
            print(
                f"calibration_epoch={epoch:05d} total={row['total']:.6e} "
                f"relative={row['relative_residual']:.6e} gamma={row['gamma']:.6f} "
                f"active_gate_sum={row['active_gate_sum']:.2f}"
            )

    metadata = dict(checkpoint.get("config", {}).get("support", {}))
    metadata.update(
        {
            "regime": "residual_calibrated_projected_sparse_agp",
            "source_run": str(source_run.relative_to(RUN_DIR) if source_run.is_relative_to(RUN_DIR) else source_run),
            "target_active_terms": target_active_terms,
            "calibration_gamma": history[-1]["gamma"],
            "calibration_active_gate_sum": history[-1]["active_gate_sum"],
            "calibration_epochs": epochs,
            "calibration_loss": history[-1],
            "calibration_uses_ground_truth_observables": False,
            "calibration_objective": "projected_euler_lagrange_residual_only",
        }
    )
    export_calibrated_coefficients(
        model=model,
        t=t,
        tau=tau,
        output_dir=output_run,
        metadata=metadata,
        history=history,
        log_gamma=log_gamma,
        gate_logits=gate_logits,
        gate_temperature=gate_temperature,
        top_k=int(calibration.get("top_coefficients", settings.top_coefficients)),
    )
    summary = {
        "source_run": metadata["source_run"],
        "output_run": str(output_run.relative_to(RUN_DIR) if output_run.is_relative_to(RUN_DIR) else output_run),
        "settings": dict(calibration),
        "final": history[-1],
        "metadata": metadata,
    }
    data_dir = output_run / "Models_Data"
    with (data_dir / "residual_calibration_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def compact_summary(summary: Mapping[str, object]) -> dict[str, object]:
    final = summary.get("final", {})
    metadata = summary.get("metadata", {})
    final = final if isinstance(final, Mapping) else {}
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return {
        "source_run": summary.get("source_run"),
        "output_run": summary.get("output_run"),
        "final": {
            "relative_residual": final.get("relative_residual"),
            "gamma": final.get("gamma"),
            "active_gate_sum": final.get("active_gate_sum"),
            "target_active_terms": metadata.get("target_active_terms"),
        },
        "calibration_objective": metadata.get("calibration_objective"),
        "calibration_uses_ground_truth_observables": metadata.get("calibration_uses_ground_truth_observables"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residual-only calibration of a trained sparse AGP.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-run", type=Path, default=None)
    parser.add_argument("--output-run", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    configure_run_dir(config_path)
    payload = load_json(config_path)
    calibration = payload.get("residual_calibration", {})
    calibration = calibration if isinstance(calibration, dict) else {}
    source_run = args.source_run or final_run_from_summary(payload)
    if not source_run.is_absolute():
        source_run = RUN_DIR / source_run
    suffix = str(calibration.get("output_suffix", "residual_calibrated"))
    output_run = args.output_run or output_run_for_source(source_run, suffix)
    if not output_run.is_absolute():
        output_run = RUN_DIR / output_run
    output_run.mkdir(parents=True, exist_ok=True)
    summary = run_calibration(
        payload=payload,
        source_run=source_run,
        output_run=output_run,
        calibration=calibration,
    )
    print(json.dumps(compact_summary(summary), indent=2))


if __name__ == "__main__":
    main()
