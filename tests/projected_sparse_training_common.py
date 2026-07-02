"""Shared runner for projected sparse-AGP experiments."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import ProjectedSparseAGPPINN, ProjectedSparseLossWeights
from utils import (
    FULL_PAULI_EXACT_MAX_QUBITS,
    SafeMPSSOAP,
    SparsePauliOperator,
    _commutator_pauli_codes_unchecked,
    _encode_pauli_label,
    load_pauli_hamiltonian_pair,
    pauli_weight,
    pytorch_optimizer,
    sort_pauli_labels,
)


OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]
TITLE_FS = 13
LABEL_FS = 12
TICK_FS = 10
LEGEND_FS = 8
LINE_WIDTH = 1.5
TICK_LENGTH = 4.0
TICK_WIDTH = 0.8


@dataclass(frozen=True)
class ProjectedTrainingConfig:
    system: str
    n_qubits: int
    distance: str = "1_0"
    hamiltonian_source: str = "Hamiltonians_to_use/pauli_decompositions/index.json"
    t_initial: float = 0.0
    physical_time: float = 1.0
    hidden_layers: int = 3
    hidden_width: int = 56
    activation: str = "silu"
    layer_type: str = "quadratic"

    @property
    def t_final(self) -> float:
        return self.t_initial + self.physical_time


@dataclass(frozen=True)
class ProjectedRunSettings:
    model: ProjectedTrainingConfig
    epochs: int = 25
    num_points: int = 16
    lr: float = 1e-4
    optimizer: str = "SOAP"
    device: str = "auto"
    seed: int = 11
    agp_top_k: int = 32
    intermediate_top_k: int = 128
    residual_top_k: int = 256
    allow_low_q_projected: bool = False
    adaptive_enabled: bool = True
    adaptive_stages: int = 2
    adaptive_growth_per_stage: int = 32
    adaptive_min_rms: float = 0.0
    adaptive_max_agp_terms: int | None = None
    top_coefficients: int = 8
    residual_weight: float = 1.0
    agp_l2_weight: float = 1e-8
    path_images: str = "Images/"
    path_data: str = "Models_Data/"


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            try:
                torch.empty(1, device="mps")
                return torch.device("mps")
            except Exception:
                pass
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requests device='cuda', but torch.cuda.is_available() is false.")
    if device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Config requests device='mps', but torch.backends.mps.is_available() is false.")
        torch.empty(1, device=device)
    return device


def deep_update(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


def default_config_payload(config: ProjectedTrainingConfig) -> dict[str, object]:
    return {
        "physical": {
            "parameters": {
                "system": config.system,
                "num_qubits": config.n_qubits,
                "distance": config.distance,
                "hamiltonian_source": config.hamiltonian_source,
                "t_initial": config.t_initial,
                "T": config.physical_time,
                "tau_range": [0.0, 1.0],
                "schedule": "sinusoidal_sin2",
            }
        },
        "neural": {
            "model": "ProjectedSparseAGPPINN",
            "general": {
                "n_inputs": 1,
                "n_outputs": "agp_top_k",
                "n_hidden": config.hidden_layers,
                "n_neurons": config.hidden_width,
                "activation": config.activation,
                "layer_type": config.layer_type,
            },
        },
        "support": {
            "strategy": "adaptive_sparse_support",
            "agp_top_k": 32,
            "intermediate_top_k": 128,
            "residual_top_k": 256,
            "residual_projection": "largest_generated_commutator_terms",
            "allow_low_q_projected": False,
            "adaptive": {
                "enabled": True,
                "stages": 2,
                "growth_terms_per_stage": 32,
                "min_rms": 0.0,
                "max_agp_terms": None,
            },
        },
        "training": {
            "device": "auto",
            "optimizer": "SOAP",
            "parameters": {
                "epochs": 25,
                "num_points": 16,
                "lr": 1e-4,
                "random_seed": 11,
            },
            "loss": {
                "residual": 1.0,
                "agp_l2": 1e-8,
            },
            "export": {
                "path_images": "Images/",
                "path_data": "Models_Data/",
                "top_coefficients": 8,
                "plot_quantity": "d_lambda_dt_times_C_P",
                "format": "pdf",
            },
        },
    }


def load_config_payload(path: Path, fallback: ProjectedTrainingConfig) -> dict[str, object]:
    payload = default_config_payload(fallback)
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            payload = deep_update(payload, json.load(handle))
    return payload


def settings_from_payload(payload: dict[str, object], fallback: ProjectedTrainingConfig) -> ProjectedRunSettings:
    physical = payload.get("physical", {})
    physical_parameters = physical.get("parameters", {}) if isinstance(physical, dict) else {}
    neural = payload.get("neural", {})
    neural_general = neural.get("general", {}) if isinstance(neural, dict) else {}
    support = payload.get("support", {})
    training = payload.get("training", {})
    training_parameters = training.get("parameters", {}) if isinstance(training, dict) else {}
    training_loss = training.get("loss", {}) if isinstance(training, dict) else {}
    training_export = training.get("export", {}) if isinstance(training, dict) else {}
    support_adaptive = support.get("adaptive", {}) if isinstance(support, dict) else {}

    model_name = neural.get("model", "ProjectedSparseAGPPINN") if isinstance(neural, dict) else "ProjectedSparseAGPPINN"
    if model_name != "ProjectedSparseAGPPINN":
        raise ValueError(f"Unsupported model {model_name!r}; this runner expects ProjectedSparseAGPPINN.")

    config = ProjectedTrainingConfig(
        system=str(physical_parameters.get("system", fallback.system)),
        n_qubits=int(physical_parameters.get("num_qubits", fallback.n_qubits)),
        distance=str(physical_parameters.get("distance", fallback.distance)),
        hamiltonian_source=str(physical_parameters.get("hamiltonian_source", fallback.hamiltonian_source)),
        t_initial=float(physical_parameters.get("t_initial", fallback.t_initial)),
        physical_time=float(physical_parameters.get("T", fallback.physical_time)),
        hidden_layers=int(neural_general.get("n_hidden", fallback.hidden_layers)),
        hidden_width=int(neural_general.get("n_neurons", fallback.hidden_width)),
        activation=str(neural_general.get("activation", fallback.activation)),
        layer_type=str(neural_general.get("layer_type", fallback.layer_type)),
    )
    if config.physical_time <= 0.0:
        raise ValueError("The physical time T must be positive.")

    max_agp_terms_raw = support_adaptive.get("max_agp_terms") if isinstance(support_adaptive, dict) else None
    adaptive_default = config.n_qubits > FULL_PAULI_EXACT_MAX_QUBITS

    return ProjectedRunSettings(
        model=config,
        epochs=int(training_parameters.get("epochs", 25)),
        num_points=int(training_parameters.get("num_points", 16)),
        lr=float(training_parameters.get("lr", 1e-4)),
        optimizer=str(training.get("optimizer", "SOAP")) if isinstance(training, dict) else "SOAP",
        device=str(training.get("device", "auto")) if isinstance(training, dict) else "auto",
        seed=int(training_parameters.get("random_seed", 11)),
        agp_top_k=int(support.get("agp_top_k", 32)) if isinstance(support, dict) else 32,
        intermediate_top_k=int(support.get("intermediate_top_k", 128)) if isinstance(support, dict) else 128,
        residual_top_k=int(support.get("residual_top_k", 256)) if isinstance(support, dict) else 256,
        allow_low_q_projected=bool(support.get("allow_low_q_projected", False)) if isinstance(support, dict) else False,
        adaptive_enabled=(
            bool(support_adaptive.get("enabled", adaptive_default)) if isinstance(support_adaptive, dict) else adaptive_default
        ),
        adaptive_stages=int(support_adaptive.get("stages", 2)) if isinstance(support_adaptive, dict) else 2,
        adaptive_growth_per_stage=(
            int(support_adaptive.get("growth_terms_per_stage", 32)) if isinstance(support_adaptive, dict) else 32
        ),
        adaptive_min_rms=float(support_adaptive.get("min_rms", 0.0)) if isinstance(support_adaptive, dict) else 0.0,
        adaptive_max_agp_terms=int(max_agp_terms_raw) if max_agp_terms_raw is not None else None,
        top_coefficients=int(training_export.get("top_coefficients", 8)),
        residual_weight=float(training_loss.get("residual", 1.0)),
        agp_l2_weight=float(training_loss.get("agp_l2", 1e-8)),
        path_images=str(training_export.get("path_images", "Images/")),
        path_data=str(training_export.get("path_data", "Models_Data/")),
    )


def make_optimizer(model: torch.nn.Module, settings: ProjectedRunSettings) -> tuple[torch.optim.Optimizer, dict[str, object]]:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    name = settings.optimizer
    optimizer = name.lower()
    if optimizer == "adam":
        instance = torch.optim.Adam(params, lr=settings.lr)
        return instance, {"requested": name, "actual": "Adam", "class": type(instance).__name__}
    if optimizer == "adamw":
        instance = torch.optim.AdamW(params, lr=settings.lr)
        return instance, {"requested": name, "actual": "AdamW", "class": type(instance).__name__}
    if optimizer in {"soap", "safe_mps_soap", "safempssoap"}:
        if pytorch_optimizer is None:
            instance = torch.optim.AdamW(params, lr=settings.lr)
            return instance, {"requested": name, "actual": "AdamW", "class": type(instance).__name__, "fallback": True}
        optimizer_cls = SafeMPSSOAP if any(parameter.device.type == "mps" for parameter in params) else pytorch_optimizer.SOAP
        instance = optimizer_cls(params, lr=settings.lr)
        return instance, {"requested": name, "actual": "SOAP", "class": optimizer_cls.__name__, "fallback": False}
    raise ValueError(f"Unsupported optimizer {name!r}.")


def hamiltonian_importance(h0: SparsePauliOperator, h1: SparsePauliOperator) -> dict[str, float]:
    labels = set(h0.labels) | set(h1.labels)
    return {label: max(abs(h0.coefficient(label)), abs(h1.coefficient(label))) for label in labels}


def operator_importance(operator: SparsePauliOperator) -> dict[str, float]:
    return {label: abs(coeff) for label, coeff in operator.terms.items()}


def ranked_label_scores(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (item[1], -pauli_weight(item[0]), item[0]), reverse=True)


def merge_scores(*score_maps: dict[str, float]) -> dict[str, float]:
    merged: defaultdict[str, float] = defaultdict(float)
    for score_map in score_maps:
        for label, score in score_map.items():
            merged[label] += float(score)
    return dict(merged)


def commutator_generated_scores(
    left_scores: dict[str, float],
    right_scores: dict[str, float],
) -> dict[str, float]:
    scores: defaultdict[str, float] = defaultdict(float)
    left_items = [
        (left_label, float(left_score), _encode_pauli_label(left_label))
        for left_label, left_score in left_scores.items()
        if left_score > 0.0
    ]
    right_items = [
        (right_label, float(right_score), _encode_pauli_label(right_label))
        for right_label, right_score in right_scores.items()
        if right_score > 0.0
    ]
    for _, left_score, left_code in left_items:
        if left_score <= 0.0:
            continue
        for _, right_score, right_code in right_items:
            if right_score <= 0.0:
                continue
            item = _commutator_pauli_codes_unchecked(left_code, right_code)
            if item is None:
                continue
            phase, out_label = item
            scores[out_label] += abs(phase) * left_score * right_score
    return dict(scores)


def build_projected_support(
    h0: SparsePauliOperator,
    h1: SparsePauliOperator,
    *,
    agp_top_k: int,
    intermediate_top_k: int,
    residual_top_k: int,
    agp_labels: list[str] | None = None,
    residual_labels: list[str] | None = None,
    stage: int = 0,
) -> dict[str, object]:
    h_labels = sort_pauli_labels(set(h0.labels) | set(h1.labels))
    h_score = hamiltonian_importance(h0, h1)
    delta_score = operator_importance(h1 - h0)
    commutator = h0.commutator(h1)
    ranked_commutator = sorted(commutator.terms.items(), key=lambda item: abs(item[1]), reverse=True)
    if not ranked_commutator:
        raise RuntimeError("Cannot derive AGP support from a zero endpoint commutator.")
    endpoint_score = {label: abs(coeff) for label, coeff in ranked_commutator}
    if agp_labels is None:
        agp_pairs = ranked_commutator[:agp_top_k]
        resolved_agp_labels = [label for label, _ in agp_pairs]
    else:
        resolved_agp_labels = sort_pauli_labels(agp_labels)
        agp_pairs = [(label, endpoint_score.get(label, 0.0)) for label in resolved_agp_labels]
    agp_score = {label: endpoint_score.get(label, 1.0) for label in resolved_agp_labels}
    endpoint_commutator_l1 = float(sum(abs(coeff) for _, coeff in ranked_commutator))
    endpoint_commutator_l2 = float(sum(abs(coeff) ** 2 for _, coeff in ranked_commutator))
    selected_endpoint_commutator_l1 = float(sum(abs(coeff) for _, coeff in agp_pairs))
    selected_endpoint_commutator_l2 = float(sum(abs(coeff) ** 2 for _, coeff in agp_pairs))

    intermediate_scores = commutator_generated_scores(agp_score, h_score)
    bounded_intermediate_pairs = ranked_label_scores(intermediate_scores)[:intermediate_top_k]
    bounded_intermediate_scores = {label: score for label, score in bounded_intermediate_pairs}
    intermediate_extra = [label for label, _ in bounded_intermediate_pairs]
    intermediate_labels = sort_pauli_labels(set(h_labels) | set(resolved_agp_labels) | set(intermediate_extra))
    if residual_labels is None:
        generator_scores = merge_scores(delta_score, bounded_intermediate_scores)
        residual_scores = merge_scores(endpoint_score, commutator_generated_scores(generator_scores, h_score))
        residual_pairs = ranked_label_scores(residual_scores)[:residual_top_k]
        resolved_residual_labels = [label for label, _ in residual_pairs]
        residual_selection_rule = "ranked_generated_residual_scores"
        generated_residual_candidate_terms: int | None = len(residual_scores)
    else:
        resolved_residual_labels = sort_pauli_labels(residual_labels)
        residual_selection_rule = "explicit_residual_labels"
        generated_residual_candidate_terms = None

    return {
        "agp_labels": resolved_agp_labels,
        "intermediate_labels": intermediate_labels,
        "residual_labels": sort_pauli_labels(resolved_residual_labels),
        "metadata": {
            "strategy": "adaptive_generated_commutator_projected_residual",
            "selection_rule": (
                "Initial AGP labels are chosen from the largest [H_initial, H_final] terms; "
                "adaptive stages may add high-residual labels generated by commutators of "
                "the current AGP and Hamiltonian supports."
            ),
            "selection_caveat": "This is a projected sparse support, not evidence that the unrestricted AGP has only these terms.",
            "stage": stage,
            "endpoint_commutator_terms": len(commutator.terms),
            "generated_intermediate_candidate_terms": len(intermediate_scores),
            "generated_residual_candidate_terms": generated_residual_candidate_terms,
            "residual_selection_rule": residual_selection_rule,
            "endpoint_commutator_l1": endpoint_commutator_l1,
            "endpoint_commutator_l2": endpoint_commutator_l2,
            "selected_endpoint_commutator_l1": selected_endpoint_commutator_l1,
            "selected_endpoint_commutator_l2": selected_endpoint_commutator_l2,
            "selected_endpoint_commutator_l1_fraction": selected_endpoint_commutator_l1
            / endpoint_commutator_l1
            if endpoint_commutator_l1 > 0.0
            else 0.0,
            "selected_endpoint_commutator_l2_fraction": selected_endpoint_commutator_l2
            / endpoint_commutator_l2
            if endpoint_commutator_l2 > 0.0
            else 0.0,
            "hamiltonian_terms": len(h_labels),
            "agp_terms": len(resolved_agp_labels),
            "intermediate_terms": len(intermediate_labels),
            "residual_terms": len(resolved_residual_labels),
            "agp_top_k": agp_top_k,
            "intermediate_top_k": intermediate_top_k,
            "residual_top_k": residual_top_k,
            "agp_weight_counts": {
                str(weight): sum(pauli_weight(label) == weight for label in resolved_agp_labels)
                for weight in sorted({pauli_weight(label) for label in resolved_agp_labels})
            },
            "top_agp_candidates": [
                {
                    "label": label,
                    "endpoint_commutator_abs": float(abs(coeff)),
                    "order": pauli_weight(label),
                }
                for label, coeff in agp_pairs
            ],
        },
    }


def split_epochs(total_epochs: int, stages: int) -> list[int]:
    if total_epochs < 1:
        raise ValueError("epochs must be positive.")
    stages = max(1, min(int(stages), total_epochs))
    base = total_epochs // stages
    remainder = total_epochs % stages
    return [base + (1 if idx < remainder else 0) for idx in range(stages)]


def make_projected_model(
    h0: SparsePauliOperator,
    h1: SparsePauliOperator,
    support: dict[str, object],
    config: ProjectedTrainingConfig,
    device: torch.device,
) -> ProjectedSparseAGPPINN:
    return ProjectedSparseAGPPINN(
        h0,
        h1,
        support["agp_labels"],  # type: ignore[arg-type]
        support["intermediate_labels"],  # type: ignore[arg-type]
        support["residual_labels"],  # type: ignore[arg-type]
        hidden_layers=config.hidden_layers,
        hidden_width=config.hidden_width,
        activation=config.activation,
        layer_type=config.layer_type,
        t_min=config.t_initial,
        t_max=config.t_final,
    ).to(device)


def transfer_output_rows(
    old_layer: torch.nn.Module,
    new_layer: torch.nn.Module,
    old_labels: list[str],
    new_labels: list[str],
) -> None:
    old_index = {label: idx for idx, label in enumerate(old_labels)}
    pairs = [(new_idx, old_index[label]) for new_idx, label in enumerate(new_labels) if label in old_index]
    if not pairs:
        return
    for branch in ("linear", "quad_left", "quad_right"):
        old_module = getattr(old_layer, branch, None)
        new_module = getattr(new_layer, branch, None)
        if old_module is None or new_module is None:
            continue
        new_indices = torch.tensor([new_idx for new_idx, _ in pairs], dtype=torch.long, device=new_module.weight.device)
        old_indices = torch.tensor([old_idx for _, old_idx in pairs], dtype=torch.long, device=old_module.weight.device)
        if old_module.weight.shape[1:] == new_module.weight.shape[1:]:
            new_module.weight.data[new_indices] = old_module.weight.data[old_indices].to(new_module.weight.device)
        if old_module.bias.shape == new_module.bias.shape:
            new_module.bias.data.copy_(old_module.bias.data.to(new_module.bias.device))
        else:
            new_module.bias.data[new_indices] = old_module.bias.data[old_indices].to(new_module.bias.device)


def transfer_projected_weights(old_model: ProjectedSparseAGPPINN, new_model: ProjectedSparseAGPPINN) -> None:
    """Transfer shared network state when adaptive support grows."""

    old_state = old_model.body.state_dict()
    new_state = new_model.body.state_dict()
    for key, value in old_state.items():
        if key in new_state and new_state[key].shape == value.shape:
            new_state[key].copy_(value.to(new_state[key].device))
    new_model.body.load_state_dict(new_state)

    old_body = old_model.body
    new_body = new_model.body
    with torch.no_grad():
        if hasattr(old_body, "layers") and hasattr(new_body, "layers"):
            transfer_output_rows(old_body.layers[-1], new_body.layers[-1], old_model.agp_labels, new_model.agp_labels)
        elif hasattr(old_body, "network") and hasattr(new_body, "network"):
            old_linear = [module for module in old_body.network if isinstance(module, torch.nn.Linear)][-1]
            new_linear = [module for module in new_body.network if isinstance(module, torch.nn.Linear)][-1]
            old_index = {label: idx for idx, label in enumerate(old_model.agp_labels)}
            pairs = [(new_idx, old_index[label]) for new_idx, label in enumerate(new_model.agp_labels) if label in old_index]
            if pairs and old_linear.weight.shape[1:] == new_linear.weight.shape[1:]:
                new_indices = torch.tensor(
                    [new_idx for new_idx, _ in pairs],
                    dtype=torch.long,
                    device=new_linear.weight.device,
                )
                old_indices = torch.tensor(
                    [old_idx for _, old_idx in pairs],
                    dtype=torch.long,
                    device=old_linear.weight.device,
                )
                new_linear.weight.data[new_indices] = old_linear.weight.data[old_indices].to(new_linear.weight.device)
                new_linear.bias.data[new_indices] = old_linear.bias.data[old_indices].to(new_linear.bias.device)


def rank_projected_residual(
    model: ProjectedSparseAGPPINN,
    t: torch.Tensor,
) -> list[dict[str, object]]:
    with torch.no_grad():
        residual = model.euler_lagrange_residual(t)
        scores = torch.sqrt(torch.mean(torch.abs(residual) ** 2, dim=0).real).detach().cpu().numpy()
    return sorted(
        [
            {
                "label": label,
                "index": idx,
                "order": pauli_weight(label),
                "rms": float(scores[idx]),
            }
            for idx, label in enumerate(model.residual_labels)
        ],
        key=lambda row: (row["rms"], row["order"]),
        reverse=True,
    )


def adaptive_agp_additions(
    model: ProjectedSparseAGPPINN,
    t: torch.Tensor,
    *,
    growth_terms: int,
    min_rms: float,
    max_agp_terms: int | None,
) -> tuple[list[str], list[dict[str, object]], list[dict[str, object]]]:
    ranked = rank_projected_residual(model, t)
    existing = set(model.agp_labels)
    identity = "I" * model.n_qubits
    if max_agp_terms is not None:
        growth_terms = min(growth_terms, max(max_agp_terms - len(existing), 0))
    additions: list[dict[str, object]] = []
    for row in ranked:
        label = str(row["label"])
        if len(additions) >= growth_terms:
            break
        if label == identity or label in existing:
            continue
        if float(row["rms"]) <= min_rms:
            continue
        additions.append(row)
    expanded = sort_pauli_labels(existing | {str(row["label"]) for row in additions})
    return expanded, additions, ranked


def train_stage(
    model: ProjectedSparseAGPPINN,
    optimizer: torch.optim.Optimizer,
    loss_weights: ProjectedSparseLossWeights,
    t: torch.Tensor,
    *,
    stage: int,
    epochs: int,
    global_epoch: int,
    history: list[dict[str, float]],
) -> int:
    for local_epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = model.loss(t, weights=loss_weights)
        loss.backward()
        optimizer.step()
        row = {"epoch": float(global_epoch), "stage": float(stage), "stage_epoch": float(local_epoch)}
        row.update({key: float(value.detach().cpu().item()) for key, value in diagnostics.items()})
        history.append(row)
        if global_epoch == 0 or local_epoch == epochs - 1:
            print(
                f"stage={stage:02d} epoch={global_epoch:04d} loss={row['total']:.6e} "
                f"residual={row['residual']:.6e} agp_terms={int(row['agp_terms'])} "
                f"residual_terms={int(row['residual_terms'])}"
            )
        global_epoch += 1
    return global_epoch


def set_paper_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "mathtext.rm": "stix",
            "mathtext.it": "stix:italic",
            "mathtext.bf": "stix:bold",
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )


def save_pdf(fig, images_dir: Path, stem: str) -> None:
    fig.savefig(images_dir / f"{stem}.pdf", format="pdf")


def plot_loss_history(history: list[dict[str, float]], images_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_paper_style(plt)
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.semilogy([row["epoch"] for row in history], [row["total"] for row in history], label="total")
    ax.semilogy([row["epoch"] for row in history], [row["residual"] for row in history], label="projected residual")
    if "relative_residual" in history[-1]:
        ax.semilogy(
            [row["epoch"] for row in history],
            [row["relative_residual"] for row in history],
            label="relative residual",
            linestyle="--",
        )
    ax.set_xlabel("epoch", fontsize=LABEL_FS)
    ax.set_ylabel("loss", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS, frameon=False)
    fig.subplots_adjust(top=0.92, left=0.14, right=0.98, bottom=0.16)
    save_pdf(fig, images_dir, "losses")
    plt.close(fig)


def compact_pauli_label(label: str, *, max_sites: int = 6) -> str:
    """Return a readable mathtext label for long Pauli strings."""

    factors = [(idx + 1, symbol) for idx, symbol in enumerate(label) if symbol != "I"]
    if not factors:
        return r"$I$"
    shown = factors[:max_sites]
    body = "".join(fr"{symbol}_{{{site}}}" for site, symbol in shown)
    if len(factors) > max_sites:
        body += rf"\cdots ({len(factors)})"
    return f"${body}$"


def qubit_ticks(n_qubits: int, *, max_ticks: int = 32) -> tuple[np.ndarray, list[str]]:
    if n_qubits <= max_ticks:
        sites = list(range(n_qubits))
    else:
        step = int(np.ceil(n_qubits / max_ticks))
        sites = list(range(0, n_qubits, step))
        if sites[-1] != n_qubits - 1:
            sites.append(n_qubits - 1)
    return np.asarray(sites, dtype=float) + 0.5, [rf"$q_{{{idx}}}$" for idx in sites]


def rank_coefficients(coefficients: torch.Tensor, labels: list[str]) -> list[dict[str, object]]:
    values = coefficients.detach().cpu().numpy()
    rms = np.sqrt(np.mean(values * values, axis=0))
    mean_abs = np.mean(np.abs(values), axis=0)
    max_abs = np.max(np.abs(values), axis=0)
    return sorted(
        [
            {
                "label": label,
                "index": idx,
                "order": pauli_weight(label),
                "rms": float(rms[idx]),
                "mean_abs": float(mean_abs[idx]),
                "max_abs": float(max_abs[idx]),
            }
            for idx, label in enumerate(labels)
        ],
        key=lambda row: (row["rms"], row["max_abs"]),
        reverse=True,
    )


def plot_top_coefficients(
    tau: torch.Tensor,
    coefficients: torch.Tensor,
    labels: list[str],
    ranked: list[dict[str, object]],
    images_dir: Path,
    *,
    top_k: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    from matplotlib.ticker import ScalarFormatter

    set_paper_style(plt)
    tau_np = tau.detach().cpu().view(-1).numpy()
    coeff_np = coefficients.detach().cpu().numpy()
    selected = ranked[: min(top_k, len(ranked))]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for plot_idx, row in enumerate(selected):
        color = OKABE_ITO[plot_idx % len(OKABE_ITO)]
        values = coeff_np[:, int(row["index"])]
        ax.plot(tau_np, values, color=color, linewidth=LINE_WIDTH)
        label_idx = int(np.argmax(np.abs(values)))
        text = ax.annotate(
            compact_pauli_label(str(row["label"])),
            xy=(tau_np[label_idx], values[label_idx]),
            xytext=(4, 7 if plot_idx % 2 == 0 else -9),
            textcoords="offset points",
            color=color,
            fontsize=LEGEND_FS,
            ha="left",
            va="center",
            annotation_clip=False,
        )
        text.set_path_effects([path_effects.withStroke(linewidth=2.4, foreground="white")])
    ax.axhline(0.0, color="0.3", linewidth=0.7)
    ax.margins(x=0.04, y=0.16)
    ax.set_xlabel(r"$\tau=t/T$", fontsize=LABEL_FS)
    ax.set_ylabel(r"$\dot{\lambda}(t) C_P(t)$", fontsize=LABEL_FS)
    ax.set_title("Largest projected counterdiabatic coefficients", fontsize=TITLE_FS)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.tick_params(axis="both", labelsize=TICK_FS)
    fig.subplots_adjust(top=0.88, left=0.11, right=0.98, bottom=0.17)
    save_pdf(fig, images_dir, "top_projected_hcd_coefficients")
    plt.close(fig)


def summarize_connections(ranked_terms: list[dict[str, object]], n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    pair_matrix = np.zeros((n_qubits, n_qubits), dtype=float)
    order_totals = np.zeros(n_qubits + 1, dtype=float)
    for row in ranked_terms:
        order = int(row["order"])
        if order == 0:
            continue
        importance = float(row["rms"])
        order_totals[order] += importance
        active = [idx for idx, symbol in enumerate(str(row["label"])) if symbol != "I"]
        for pos, left in enumerate(active):
            for right in active[pos + 1 :]:
                pair_matrix[left, right] += importance
                pair_matrix[right, left] += importance
    return pair_matrix, order_totals


def export_plot_numerics(
    tau: torch.Tensor,
    t: torch.Tensor,
    d_lambda_dt: torch.Tensor,
    coefficients: torch.Tensor,
    labels: list[str],
    top_terms: list[dict[str, object]],
    least_terms: list[dict[str, object]],
    ranked_terms: list[dict[str, object]],
    data_dir: Path,
) -> None:
    tau_values = [float(value) for value in tau.detach().cpu().view(-1).numpy()]
    t_values = [float(value) for value in t.detach().cpu().view(-1).numpy()]
    velocity_values = [float(value) for value in d_lambda_dt.detach().cpu().view(-1).numpy()]
    coeff_np = coefficients.detach().cpu().numpy()
    n_qubits = len(labels[0])
    pair_matrix, order_totals = summarize_connections(ranked_terms, n_qubits)
    top_payload = []
    for row in top_terms:
        idx = int(row["index"])
        top_payload.append(
            {
                "label": row["label"],
                "compact_label": compact_pauli_label(str(row["label"])),
                "index": idx,
                "order": row["order"],
                "rms": row["rms"],
                "mean_abs": row["mean_abs"],
                "max_abs": row["max_abs"],
                "values": [float(value) for value in coeff_np[:, idx]],
            }
        )
    least_payload = []
    for row in least_terms:
        idx = int(row["index"])
        least_payload.append(
            {
                "label": row["label"],
                "compact_label": compact_pauli_label(str(row["label"])),
                "index": idx,
                "order": row["order"],
                "rms": row["rms"],
                "mean_abs": row["mean_abs"],
                "max_abs": row["max_abs"],
                "values": [float(value) for value in coeff_np[:, idx]],
            }
        )
    payload = {
        "coefficient_kind": "projected_counterdiabatic_hamiltonian",
        "coefficient_definition": "d_lambda_dt * C_P(t)",
        "tau": tau_values,
        "physical_time": t_values,
        "d_lambda_dt": velocity_values,
        "ranking_metric": "rms_over_time",
        "top_coefficients": top_payload,
        "support_map_terms": top_payload[: min(len(top_payload), 16)],
        "least_coefficients": least_payload,
        "least_support_map_terms": least_payload[: min(len(least_payload), 16)],
        "least_terms_note": "Identity is excluded because it is gauge-trivial in the AGP commutator loss.",
        "pairwise_participation": pair_matrix.tolist(),
        "importance_by_order": {
            str(order): float(value)
            for order, value in enumerate(order_totals)
            if order > 0
        },
    }
    with (data_dir / "coefficient_plot_data.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def plot_support_map(
    terms: list[dict[str, object]],
    n_qubits: int,
    images_dir: Path,
    *,
    title: str = "Top projected operator hyperedges",
    stem: str = "hcd_coefficient_support_map",
    bar_color: str = "#0072B2",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import ScalarFormatter

    set_paper_style(plt)
    selected = terms[: min(len(terms), 16)]
    if not selected:
        return

    code = {"I": 0, "X": 1, "Y": 2, "Z": 3}
    matrix = np.array([[code[symbol] for symbol in str(row["label"])] for row in selected], dtype=float)
    importance = np.array([float(row["rms"]) for row in selected])
    labels = [str(row["label"]) for row in selected]
    tick_positions, tick_labels = qubit_ticks(n_qubits)
    fig_width = min(max(8.2, 0.26 * n_qubits + 3.2), 34.0)
    fig, (ax_map, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(fig_width, max(3.3, 0.30 * len(selected) + 1.2)),
        gridspec_kw={"width_ratios": [max(2.4, 0.23 * n_qubits), 1.35], "wspace": 0.26},
    )
    cmap = ListedColormap(["#f7f7f7", "#0072B2", "#D55E00", "#009E73"])
    ax_map.pcolormesh(
        np.arange(n_qubits + 1),
        np.arange(len(selected) + 1),
        matrix,
        cmap=cmap,
        vmin=-0.5,
        vmax=3.5,
        edgecolors="white",
        linewidth=0.35,
        rasterized=False,
    )
    ax_map.invert_yaxis()
    ax_map.set_xticks(tick_positions)
    ax_map.set_xticklabels(tick_labels, fontsize=TICK_FS, rotation=90)
    ax_map.set_yticks(np.arange(len(selected)) + 0.5)
    ax_map.set_yticklabels([compact_pauli_label(label) for label in labels], fontsize=TICK_FS)
    ax_map.set_xlabel("qubit", fontsize=LABEL_FS)
    ax_map.set_title(title, fontsize=TITLE_FS)
    ax_map.tick_params(axis="both", length=TICK_LENGTH, width=TICK_WIDTH)
    for row_idx, label in enumerate(labels):
        for qubit_idx, symbol in enumerate(label):
            if symbol == "I":
                continue
            ax_map.text(
                qubit_idx + 0.5,
                row_idx + 0.5,
                symbol,
                ha="center",
                va="center",
                fontsize=max(6, TICK_FS - 2),
                color="0.15",
            )

    y = np.arange(len(selected))
    ax_bar.barh(y, importance, color=bar_color)
    ax_bar.invert_yaxis()
    ax_bar.set_yticks([])
    ax_bar.set_xlabel(r"$\mathrm{RMS}_\tau(\dot{\lambda}C_P)$", fontsize=LABEL_FS)
    ax_bar.set_title("importance", fontsize=TITLE_FS)
    ax_bar.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax_bar.tick_params(axis="x", labelsize=TICK_FS, length=TICK_LENGTH, width=TICK_WIDTH)
    fig.subplots_adjust(top=0.88, left=0.20, right=0.98, bottom=0.20, wspace=0.26)
    save_pdf(fig, images_dir, stem)
    plt.close(fig)


def plot_connection_summary(
    ranked_terms: list[dict[str, object]],
    n_qubits: int,
    images_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.ticker import ScalarFormatter

    set_paper_style(plt)
    pair_matrix, order_totals = summarize_connections(ranked_terms, n_qubits)
    tick_positions, tick_labels = qubit_ticks(n_qubits)
    fig, (ax_heat, ax_order) = plt.subplots(
        1,
        2,
        figsize=(min(max(8.0, 0.12 * n_qubits + 5.8), 24.0), 3.5),
        gridspec_kw={"width_ratios": [1.18, 1.0], "wspace": 0.58},
    )
    cmap = mpl.colormaps["viridis"]
    ax_heat.pcolormesh(
        np.arange(n_qubits + 1),
        np.arange(n_qubits + 1),
        pair_matrix,
        cmap=cmap,
        edgecolors="white",
        linewidth=0.25,
        rasterized=False,
    )
    ax_heat.invert_yaxis()
    ax_heat.set_xticks(tick_positions)
    ax_heat.set_yticks(tick_positions)
    ax_heat.set_xticklabels(tick_labels, fontsize=TICK_FS, rotation=90)
    ax_heat.set_yticklabels(tick_labels, fontsize=TICK_FS)
    ax_heat.set_xlabel("qubit", fontsize=LABEL_FS)
    ax_heat.set_ylabel("qubit", fontsize=LABEL_FS)
    ax_heat.set_title("Pairwise participation", fontsize=TITLE_FS)
    ax_heat.tick_params(axis="both", length=TICK_LENGTH, width=TICK_WIDTH)
    cax = ax_heat.inset_axes([1.06, 0.0, 0.06, 1.0])
    vmin = float(np.nanmin(pair_matrix))
    vmax = float(np.nanmax(pair_matrix))
    if np.isclose(vmax, vmin):
        vmax = vmin + 1.0
    scale = np.linspace(vmin, vmax, 32).reshape(-1, 1)
    cax.pcolormesh(
        [0.0, 1.0],
        np.linspace(vmin, vmax, scale.shape[0] + 1),
        scale,
        cmap=cmap,
        shading="flat",
        rasterized=False,
    )
    cax.set_xticks([])
    cax.set_yticks([vmin, vmax])
    cax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    cax.tick_params(axis="y", labelsize=TICK_FS, length=TICK_LENGTH, width=TICK_WIDTH)

    orders = np.arange(1, n_qubits + 1)
    ax_order.bar(orders, order_totals[1:], color="#D55E00")
    ax_order.set_xlabel("Pauli-string order", fontsize=LABEL_FS)
    ax_order.set_ylabel(r"$\sum_P \mathrm{RMS}_\tau(\dot{\lambda}C_P)$", fontsize=LABEL_FS)
    ax_order.yaxis.set_label_position("right")
    ax_order.yaxis.tick_right()
    ax_order.set_title("Importance by order", fontsize=TITLE_FS)
    ax_order.set_xticks(orders)
    ax_order.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax_order.tick_params(axis="both", labelsize=TICK_FS, length=TICK_LENGTH, width=TICK_WIDTH)
    fig.subplots_adjust(top=0.84, left=0.09, right=0.90, bottom=0.20, wspace=0.58)
    save_pdf(fig, images_dir, "hcd_connection_summary")
    plt.close(fig)


def plot_support_summary(metadata: dict[str, object], images_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    set_paper_style(plt)
    labels = ["AGP", "intermediate", "residual", "Hamiltonian"]
    values = [
        float(metadata["agp_terms"]),
        float(metadata["intermediate_terms"]),
        float(metadata["residual_terms"]),
        float(metadata["hamiltonian_terms"]),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.bar(labels, values, color=["#0072B2", "#D55E00", "#009E73", "#CC79A7"])
    ax.set_ylabel("Pauli strings", fontsize=LABEL_FS)
    n_qubits = metadata.get("n_qubits")
    title = f"Projected q={n_qubits} sparse supports" if n_qubits is not None else "Projected sparse supports"
    ax.set_title(title, fontsize=TITLE_FS)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.tick_params(axis="both", labelsize=TICK_FS)
    fig.subplots_adjust(top=0.86, left=0.15, right=0.98, bottom=0.18)
    save_pdf(fig, images_dir, "projected_support_summary")
    plt.close(fig)


def export_results(
    model: ProjectedSparseAGPPINN,
    tau: torch.Tensor,
    t: torch.Tensor,
    images_dir: Path,
    data_dir: Path,
    metadata: dict[str, object],
    history: list[dict[str, float]],
    *,
    top_k: int,
) -> None:
    with torch.no_grad():
        prediction = model(t)
        agp_coefficients = prediction["agp_coefficients"].detach().cpu()
        d_lambda_dt = prediction["d_lambda_dt"].detach().cpu()
        hcd_coefficients = d_lambda_dt * agp_coefficients
        tau_cpu = tau.detach().cpu()
        t_cpu = t.detach().cpu()

    ranked = rank_coefficients(hcd_coefficients, model.agp_labels)
    least_terms = list(reversed([row for row in ranked if row["order"] > 0]))[:top_k]
    importance_payload = {
        "coefficient_kind": "projected_counterdiabatic_hamiltonian",
        "coefficient_definition": "d_lambda_dt * C_P(t)",
        "ranking_metric": "rms_over_time",
        "all_terms": ranked,
        "top_terms": ranked[:top_k],
        "least_terms": least_terms,
        "least_terms_note": "Identity is excluded because it is gauge-trivial in the AGP commutator loss.",
    }
    with (data_dir / "coefficient_importance.json").open("w", encoding="utf-8") as handle:
        json.dump(importance_payload, handle, indent=2)
        handle.write("\n")

    torch.save(
        {
            "t": t_cpu,
            "tau": tau_cpu,
            "pauli_labels": model.agp_labels,
            "agp_coefficients": agp_coefficients,
            "counterdiabatic_coefficients": hcd_coefficients,
            "counterdiabatic_coefficient_definition": "d_lambda_dt * C_P(t)",
            "lambda": prediction["lambda"].detach().cpu(),
            "d_lambda_dt": d_lambda_dt,
            "support_metadata": metadata,
        },
        data_dir / "final_agp_coefficients.pt",
    )

    plot_loss_history(history, images_dir)
    export_plot_numerics(
        tau_cpu,
        t_cpu,
        d_lambda_dt,
        hcd_coefficients,
        model.agp_labels,
        ranked[:top_k],
        least_terms,
        ranked,
        data_dir,
    )
    plot_top_coefficients(tau_cpu, hcd_coefficients, model.agp_labels, ranked, images_dir, top_k=top_k)
    plot_support_map(ranked[:top_k], model.n_qubits, images_dir)
    plot_support_map(
        least_terms,
        model.n_qubits,
        images_dir,
        title="Least important projected operator hyperedges",
        stem="hcd_least_important_coefficient_support_map",
        bar_color="#999999",
    )
    plot_connection_summary(ranked, model.n_qubits, images_dir)
    plot_support_summary(metadata, images_dir)


def regenerate_projected_plots_from_saved_run(run_dir: Path, *, top_k: int | None = None) -> None:
    config_path = run_dir / "config.json"
    path_images = "Images/"
    path_data = "Models_Data/"
    configured_top_k = 8
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        export = payload.get("training", {}).get("export", {})
        path_images = str(export.get("path_images", path_images))
        path_data = str(export.get("path_data", path_data))
        configured_top_k = int(export.get("top_coefficients", configured_top_k))
    top_k = configured_top_k if top_k is None else top_k

    images_dir = run_dir / path_images
    data_dir = run_dir / path_data
    images_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    coefficient_path = data_dir / "final_agp_coefficients.pt"
    history_path = data_dir / "loss_history.json"
    metadata_path = data_dir / "support_metadata.json"
    if not coefficient_path.is_file():
        raise FileNotFoundError(f"Cannot regenerate plots; missing {coefficient_path}.")
    if not history_path.is_file():
        raise FileNotFoundError(f"Cannot regenerate plots; missing {history_path}.")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Cannot regenerate plots; missing {metadata_path}.")

    payload = torch.load(coefficient_path, map_location="cpu")
    history = json.loads(history_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    tau = payload["tau"]
    t = payload["t"]
    d_lambda_dt = payload["d_lambda_dt"]
    coefficients = payload["counterdiabatic_coefficients"]
    labels = list(payload["pauli_labels"])
    ranked = rank_coefficients(coefficients, labels)
    least_terms = list(reversed([row for row in ranked if row["order"] > 0]))[:top_k]

    importance_payload = {
        "coefficient_kind": "projected_counterdiabatic_hamiltonian",
        "coefficient_definition": "d_lambda_dt * C_P(t)",
        "ranking_metric": "rms_over_time",
        "all_terms": ranked,
        "top_terms": ranked[:top_k],
        "least_terms": least_terms,
        "least_terms_note": "Identity is excluded because it is gauge-trivial in the AGP commutator loss.",
    }
    with (data_dir / "coefficient_importance.json").open("w", encoding="utf-8") as handle:
        json.dump(importance_payload, handle, indent=2)
        handle.write("\n")

    plot_loss_history(history, images_dir)
    export_plot_numerics(tau, t, d_lambda_dt, coefficients, labels, ranked[:top_k], least_terms, ranked, data_dir)
    plot_top_coefficients(tau, coefficients, labels, ranked, images_dir, top_k=top_k)
    plot_support_map(ranked[:top_k], len(labels[0]), images_dir)
    plot_support_map(
        least_terms,
        len(labels[0]),
        images_dir,
        title="Least important projected operator hyperedges",
        stem="hcd_least_important_coefficient_support_map",
        bar_color="#999999",
    )
    plot_connection_summary(ranked, len(labels[0]), images_dir)
    plot_support_summary(metadata, images_dir)


def run_training(settings: ProjectedRunSettings, run_dir: Path) -> dict[str, float]:
    config = settings.model
    if config.n_qubits <= FULL_PAULI_EXACT_MAX_QUBITS and not settings.allow_low_q_projected:
        raise ValueError(
            f"q={config.n_qubits} is in the exact low-size regime. "
            f"Use FullPauliAGPPINN for q <= {FULL_PAULI_EXACT_MAX_QUBITS}, or set "
            "support.allow_low_q_projected=true for a deliberate projected diagnostic."
        )
    torch.manual_seed(settings.seed)
    device = select_device(settings.device)
    hamiltonian_path = Path(config.hamiltonian_source)
    if not hamiltonian_path.is_absolute():
        hamiltonian_path = ROOT / hamiltonian_path
    h0, h1 = load_pauli_hamiltonian_pair(
        hamiltonian_path,
        system=config.system,
        n_qubits=config.n_qubits,
        distance=config.distance,
    )
    loss_weights = ProjectedSparseLossWeights(residual=settings.residual_weight, agp_l2=settings.agp_l2_weight)
    tau = torch.linspace(0.0, 1.0, settings.num_points, device=device).view(-1, 1)
    t = config.t_initial + config.physical_time * tau
    history: list[dict[str, float]] = []
    adaptive_active = (
        settings.adaptive_enabled
        and config.n_qubits > FULL_PAULI_EXACT_MAX_QUBITS
        and settings.adaptive_stages > 1
        and settings.adaptive_growth_per_stage > 0
    )
    stage_epochs = split_epochs(settings.epochs, settings.adaptive_stages if adaptive_active else 1)
    current_agp_labels: list[str] | None = None
    previous_model: ProjectedSparseAGPPINN | None = None
    model: ProjectedSparseAGPPINN | None = None
    metadata: dict[str, object] = {}
    optimizer: torch.optim.Optimizer | None = None
    optimizer_info: dict[str, object] = {}
    optimizer_stages: list[dict[str, object]] = []
    adaptive_history: list[dict[str, object]] = []
    global_epoch = 0

    for stage_idx, epochs_this_stage in enumerate(stage_epochs):
        support = build_projected_support(
            h0,
            h1,
            agp_top_k=settings.agp_top_k,
            intermediate_top_k=settings.intermediate_top_k,
            residual_top_k=settings.residual_top_k,
            agp_labels=current_agp_labels,
            stage=stage_idx,
        )
        metadata = dict(support["metadata"])
        metadata["n_qubits"] = config.n_qubits
        metadata["device"] = str(device)
        metadata["full_pauli_basis_size"] = 4**config.n_qubits
        metadata["regime"] = "adaptive_projected_sparse"
        metadata["adaptive_enabled"] = adaptive_active
        metadata["adaptive_requested_stages"] = settings.adaptive_stages
        metadata["adaptive_growth_per_stage"] = settings.adaptive_growth_per_stage
        metadata["adaptive_min_rms"] = settings.adaptive_min_rms
        metadata["adaptive_max_agp_terms"] = settings.adaptive_max_agp_terms

        model = make_projected_model(h0, h1, support, config, device)
        if previous_model is not None:
            transfer_projected_weights(previous_model, model)
        metadata["first_commutator_nnz"] = model.first_commutator.nnz
        metadata["second_commutator_nnz"] = model.second_commutator.nnz

        optimizer, optimizer_info = make_optimizer(model, settings)
        optimizer_stage_info = dict(optimizer_info)
        optimizer_stage_info["stage"] = stage_idx
        optimizer_stage_info["agp_terms"] = len(model.agp_labels)
        optimizer_stages.append(optimizer_stage_info)

        global_epoch = train_stage(
            model,
            optimizer,
            loss_weights,
            t,
            stage=stage_idx,
            epochs=epochs_this_stage,
            global_epoch=global_epoch,
            history=history,
        )

        expanded_labels, additions, residual_ranking = adaptive_agp_additions(
            model,
            t,
            growth_terms=settings.adaptive_growth_per_stage if stage_idx < len(stage_epochs) - 1 else 0,
            min_rms=settings.adaptive_min_rms,
            max_agp_terms=settings.adaptive_max_agp_terms,
        )
        adaptive_history.append(
            {
                "stage": stage_idx,
                "epochs": epochs_this_stage,
                "agp_terms_before_growth": len(model.agp_labels),
                "added_terms": additions,
                "top_residual_terms": residual_ranking[: min(len(residual_ranking), 32)],
                "agp_terms_after_growth": len(expanded_labels),
            }
        )
        current_agp_labels = expanded_labels
        previous_model = model

    if model is None or optimizer is None:
        raise RuntimeError("Projected sparse training did not instantiate a model.")
    metadata["adaptive_history"] = adaptive_history
    metadata["adaptive_completed_stages"] = len(stage_epochs)
    metadata["final_agp_terms"] = len(model.agp_labels)
    metadata["final_intermediate_terms"] = len(model.intermediate_labels)
    metadata["final_residual_terms"] = len(model.residual_labels)

    images_dir = run_dir / settings.path_images
    data_dir = run_dir / settings.path_data
    images_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    with (data_dir / "loss_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
        handle.write("\n")
    with (data_dir / "support_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    with (data_dir / "adaptive_support_history.json").open("w", encoding="utf-8") as handle:
        json.dump(adaptive_history, handle, indent=2)
        handle.write("\n")
    run_metadata = {
        "physical": asdict(config),
        "training": asdict(settings),
        "support": metadata,
        "optimizer": optimizer_info,
        "optimizer_stages": optimizer_stages,
    }
    with (data_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2)
        handle.write("\n")
    torch.save(model.state_dict(), data_dir / "model_weights.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": run_metadata,
            "final_diagnostics": history[-1],
            "agp_labels": model.agp_labels,
            "intermediate_labels": model.intermediate_labels,
            "residual_labels": model.residual_labels,
            "hamiltonian_labels": model.hamiltonian_labels,
        },
        data_dir / "training_checkpoint.pt",
    )
    export_results(
        model,
        tau,
        t,
        images_dir,
        data_dir,
        metadata,
        history,
        top_k=settings.top_coefficients,
    )
    return history[-1]


def main_for_config(config: ProjectedTrainingConfig, run_dir: Path) -> None:
    parser = argparse.ArgumentParser(description="Train a projected sparse AGP PINN.")
    parser.add_argument("--config", type=Path, default=run_dir / "config.json")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--agp-top-k", type=int, default=None)
    parser.add_argument("--intermediate-top-k", type=int, default=None)
    parser.add_argument("--residual-top-k", type=int, default=None)
    parser.add_argument("--adaptive-stages", type=int, default=None)
    parser.add_argument("--adaptive-growth", type=int, default=None)
    parser.add_argument("--disable-adaptive", action="store_true")
    parser.add_argument("--plots-only", action="store_true", help="Regenerate PDFs from Models_Data without training.")
    parser.add_argument(
        "--allow-low-q-projected",
        action="store_true",
        help="Allow projected sparse training in the q<=8 exact regime for diagnostics.",
    )
    args = parser.parse_args()
    settings = settings_from_payload(load_config_payload(args.config, config), config)
    settings = replace(
        settings,
        epochs=args.epochs if args.epochs is not None else settings.epochs,
        num_points=args.num_points if args.num_points is not None else settings.num_points,
        lr=args.lr if args.lr is not None else settings.lr,
        optimizer=args.optimizer if args.optimizer is not None else settings.optimizer,
        device=args.device if args.device is not None else settings.device,
        seed=args.seed if args.seed is not None else settings.seed,
        agp_top_k=args.agp_top_k if args.agp_top_k is not None else settings.agp_top_k,
        intermediate_top_k=args.intermediate_top_k
        if args.intermediate_top_k is not None
        else settings.intermediate_top_k,
        residual_top_k=args.residual_top_k if args.residual_top_k is not None else settings.residual_top_k,
        adaptive_enabled=False if args.disable_adaptive else settings.adaptive_enabled,
        adaptive_stages=args.adaptive_stages if args.adaptive_stages is not None else settings.adaptive_stages,
        adaptive_growth_per_stage=args.adaptive_growth
        if args.adaptive_growth is not None
        else settings.adaptive_growth_per_stage,
        allow_low_q_projected=args.allow_low_q_projected or settings.allow_low_q_projected,
    )
    if args.plots_only:
        regenerate_projected_plots_from_saved_run(run_dir, top_k=settings.top_coefficients)
        print("regenerated_plots=true")
        return
    final = run_training(settings, run_dir)
    print(f"final_loss={final['total']:.6e} final_residual={final['residual']:.6e}")
