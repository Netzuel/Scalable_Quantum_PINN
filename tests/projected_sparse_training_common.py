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
    SafeMPSSOAP,
    SparsePauliOperator,
    commutator_pauli_labels,
    fixed_sinusoidal_schedule,
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
            "strategy": "top_commutator_projected",
            "agp_top_k": 32,
            "intermediate_top_k": 128,
            "residual_top_k": 256,
            "residual_projection": "largest_endpoint_commutator_terms",
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


def build_projected_support(
    h0: SparsePauliOperator,
    h1: SparsePauliOperator,
    *,
    agp_top_k: int,
    intermediate_top_k: int,
    residual_top_k: int,
) -> dict[str, object]:
    h_labels = sort_pauli_labels(set(h0.labels) | set(h1.labels))
    h_score = hamiltonian_importance(h0, h1)
    commutator = h0.commutator(h1)
    ranked_commutator = sorted(commutator.terms.items(), key=lambda item: abs(item[1]), reverse=True)
    if not ranked_commutator:
        raise RuntimeError("Cannot derive AGP support from a zero endpoint commutator.")
    agp_pairs = ranked_commutator[:agp_top_k]
    agp_labels = [label for label, _ in agp_pairs]
    agp_score = {label: abs(coeff) for label, coeff in agp_pairs}

    intermediate_scores: defaultdict[str, float] = defaultdict(float)
    for agp_label in agp_labels:
        for h_label in h_labels:
            item = commutator_pauli_labels(agp_label, h_label)
            if item is None:
                continue
            _, out_label = item
            intermediate_scores[out_label] += agp_score[agp_label] * h_score[h_label]
    intermediate_extra = [
        label
        for label, _ in sorted(intermediate_scores.items(), key=lambda item: item[1], reverse=True)[:intermediate_top_k]
    ]
    intermediate_labels = sort_pauli_labels(set(h_labels) | set(agp_labels) | set(intermediate_extra))
    residual_labels = [label for label, _ in ranked_commutator[:residual_top_k]]

    return {
        "agp_labels": agp_labels,
        "intermediate_labels": intermediate_labels,
        "residual_labels": sort_pauli_labels(residual_labels),
        "metadata": {
            "strategy": "top_endpoint_commutator_projected_residual",
            "endpoint_commutator_terms": len(commutator.terms),
            "hamiltonian_terms": len(h_labels),
            "agp_terms": len(agp_labels),
            "intermediate_terms": len(intermediate_labels),
            "residual_terms": len(residual_labels),
            "agp_top_k": agp_top_k,
            "intermediate_top_k": intermediate_top_k,
            "residual_top_k": residual_top_k,
            "agp_weight_counts": {
                str(weight): sum(pauli_weight(label) == weight for label in agp_labels)
                for weight in sorted({pauli_weight(label) for label in agp_labels})
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
    ax.set_xlabel("epoch", fontsize=LABEL_FS)
    ax.set_ylabel("loss", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS, frameon=False)
    fig.subplots_adjust(top=0.92, left=0.14, right=0.98, bottom=0.16)
    save_pdf(fig, images_dir, "losses")
    plt.close(fig)


def rank_coefficients(coefficients: torch.Tensor, labels: list[str]) -> list[dict[str, object]]:
    values = coefficients.detach().cpu().numpy()
    rms = np.sqrt(np.mean(values * values, axis=0))
    max_abs = np.max(np.abs(values), axis=0)
    return sorted(
        [
            {
                "label": label,
                "index": idx,
                "order": pauli_weight(label),
                "rms": float(rms[idx]),
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
            str(row["label"]),
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
    ax.set_title("Projected q=20 sparse supports", fontsize=TITLE_FS)
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
    importance_payload = {
        "coefficient_kind": "projected_counterdiabatic_hamiltonian",
        "coefficient_definition": "d_lambda_dt * C_P(t)",
        "ranking_metric": "rms_over_time",
        "all_terms": ranked,
        "top_terms": ranked[:top_k],
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
    plot_top_coefficients(tau_cpu, hcd_coefficients, model.agp_labels, ranked, images_dir, top_k=top_k)
    plot_support_summary(metadata, images_dir)


def run_training(settings: ProjectedRunSettings, run_dir: Path) -> dict[str, float]:
    config = settings.model
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
    support = build_projected_support(
        h0,
        h1,
        agp_top_k=settings.agp_top_k,
        intermediate_top_k=settings.intermediate_top_k,
        residual_top_k=settings.residual_top_k,
    )
    metadata = dict(support["metadata"])
    model = ProjectedSparseAGPPINN(
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
    metadata["first_commutator_nnz"] = model.first_commutator.nnz
    metadata["second_commutator_nnz"] = model.second_commutator.nnz
    metadata["device"] = str(device)
    metadata["full_pauli_basis_size"] = 4**config.n_qubits

    optimizer, optimizer_info = make_optimizer(model, settings)
    loss_weights = ProjectedSparseLossWeights(residual=settings.residual_weight, agp_l2=settings.agp_l2_weight)
    tau = torch.linspace(0.0, 1.0, settings.num_points, device=device).view(-1, 1)
    t = config.t_initial + config.physical_time * tau
    history: list[dict[str, float]] = []

    for epoch in range(settings.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = model.loss(t, weights=loss_weights)
        loss.backward()
        optimizer.step()
        row = {"epoch": float(epoch)}
        row.update({key: float(value.detach().cpu().item()) for key, value in diagnostics.items()})
        history.append(row)
        if epoch == 0 or epoch == settings.epochs - 1:
            print(
                f"epoch={epoch:04d} loss={row['total']:.6e} residual={row['residual']:.6e} "
                f"agp_terms={int(row['agp_terms'])} residual_terms={int(row['residual_terms'])}"
            )

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
    run_metadata = {
        "physical": asdict(config),
        "training": asdict(settings),
        "support": metadata,
        "optimizer": optimizer_info,
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
    )
    final = run_training(settings, run_dir)
    print(f"final_loss={final['total']:.6e} final_residual={final['residual']:.6e}")
