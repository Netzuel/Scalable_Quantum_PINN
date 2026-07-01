"""Shared runner for the self-contained full-Pauli AGP experiments."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import FullPauliAGPPINN, FullPauliLossWeights
from utils import SafeMPSSOAP, load_pauli_hamiltonian_pair, pytorch_optimizer


OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]
TITLE_FS = 13
LABEL_FS = 12
TICK_FS = 10
LEGEND_FS = 9
LINE_WIDTH = 1.6
TICK_LENGTH = 4.0
TICK_WIDTH = 0.8
MAX_LABELED_TRACES = 8


@dataclass(frozen=True)
class TrainingConfig:
    system: str
    n_qubits: int
    distance: str = "1_0"
    hamiltonian_source: str = "Hamiltonians_to_use/pauli_decompositions/index.json"
    t_initial: float = 0.0
    physical_time: float = 1.0
    schedule: str = "sinusoidal_sin2"
    hidden_layers: int = 3
    hidden_width: int = 56
    activation: str = "silu"
    layer_type: str = "quadratic"

    @property
    def t_final(self) -> float:
        return self.t_initial + self.physical_time


@dataclass(frozen=True)
class RunSettings:
    model: TrainingConfig
    epochs: int = 10
    num_points: int = 32
    lr: float = 1e-4
    optimizer: str = "SOAP"
    device: str = "auto"
    seed: int = 7
    top_coefficients: int = 12
    path_images: str = "Images/"
    path_data: str = "Models_Data/"
    residual_weight: float = 1.0
    agp_l2_weight: float = 0.0
    soap_precondition_frequency: int | None = None
    soap_max_precondition_dim: int | None = None
    soap_weight_decay: float | None = None
    soap_normalize_gradient: bool | None = None


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


def default_config_payload(config: TrainingConfig) -> dict[str, object]:
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
                "schedule": config.schedule,
            }
        },
        "neural": {
            "model": "FullPauliAGPPINN",
            "general": {
                "n_inputs": 1,
                "n_outputs": f"4**{config.n_qubits}",
                "n_hidden": config.hidden_layers,
                "n_neurons": config.hidden_width,
                "activation": config.activation,
                "layer_type": config.layer_type,
            },
        },
        "training": {
            "device": "auto",
            "optimizer": "SOAP",
            "parameters": {
                "epochs": 10,
                "num_points": 32,
                "lr": 1e-4,
                "random_seed": 7,
            },
            "loss": {
                "residual": 1.0,
                "agp_l2": 0.0,
            },
            "export": {
                "path_images": "Images/",
                "path_data": "Models_Data/",
                "top_coefficients": 12,
                "plot_quantity": "d_lambda_dt_times_C_P",
                "format": "pdf",
            },
        },
    }


def load_config_payload(path: Path, fallback: TrainingConfig) -> dict[str, object]:
    payload = default_config_payload(fallback)
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            payload = deep_update(payload, json.load(handle))
    return payload


def run_settings_from_payload(payload: dict[str, object], fallback: TrainingConfig) -> RunSettings:
    physical = payload.get("physical", {})
    physical_parameters = physical.get("parameters", {}) if isinstance(physical, dict) else {}
    neural = payload.get("neural", {})
    neural_general = neural.get("general", {}) if isinstance(neural, dict) else {}
    training = payload.get("training", {})
    training_parameters = training.get("parameters", {}) if isinstance(training, dict) else {}
    training_loss = training.get("loss", {}) if isinstance(training, dict) else {}
    training_export = training.get("export", {}) if isinstance(training, dict) else {}

    model_name = neural.get("model", "FullPauliAGPPINN") if isinstance(neural, dict) else "FullPauliAGPPINN"
    if model_name != "FullPauliAGPPINN":
        raise ValueError(f"Unsupported model {model_name!r}; this runner expects FullPauliAGPPINN.")

    config = TrainingConfig(
        system=str(physical_parameters.get("system", fallback.system)),
        n_qubits=int(physical_parameters.get("num_qubits", physical_parameters.get("n_qubits", fallback.n_qubits))),
        distance=str(physical_parameters.get("distance", fallback.distance)),
        hamiltonian_source=str(physical_parameters.get("hamiltonian_source", fallback.hamiltonian_source)),
        t_initial=float(physical_parameters.get("t_initial", physical_parameters.get("t_min", fallback.t_initial))),
        physical_time=float(physical_parameters.get("T", physical_parameters.get("physical_time", fallback.physical_time))),
        schedule=str(physical_parameters.get("schedule", fallback.schedule)),
        hidden_layers=int(neural_general.get("n_hidden", neural_general.get("hidden_layers", fallback.hidden_layers))),
        hidden_width=int(neural_general.get("n_neurons", neural_general.get("hidden_width", fallback.hidden_width))),
        activation=str(neural_general.get("activation", fallback.activation)),
        layer_type=str(neural_general.get("layer_type", fallback.layer_type)),
    )
    if config.schedule != "sinusoidal_sin2":
        raise ValueError("Only the fixed sinusoidal_sin2 schedule is implemented.")
    if config.physical_time <= 0.0:
        raise ValueError("The physical time T must be positive.")

    return RunSettings(
        model=config,
        epochs=int(training_parameters.get("epochs", 10)),
        num_points=int(training_parameters.get("num_points", 32)),
        lr=float(training_parameters.get("lr", training_parameters.get("learning_rate", 1e-4))),
        optimizer=str(training.get("optimizer", training_parameters.get("optimizer", "SOAP"))) if isinstance(training, dict) else "SOAP",
        device=str(training.get("device", "auto")) if isinstance(training, dict) else "auto",
        seed=int(training_parameters.get("random_seed", training_parameters.get("seed", 7))),
        top_coefficients=int(training_export.get("top_coefficients", 12)),
        path_images=str(training_export.get("path_images", "Images/")),
        path_data=str(training_export.get("path_data", "Models_Data/")),
        residual_weight=float(training_loss.get("residual", 1.0)),
        agp_l2_weight=float(training_loss.get("agp_l2", 0.0)),
        soap_precondition_frequency=(
            int(training_parameters["soap_precondition_frequency"])
            if "soap_precondition_frequency" in training_parameters
            else None
        ),
        soap_max_precondition_dim=(
            int(training_parameters["soap_max_precondition_dim"])
            if "soap_max_precondition_dim" in training_parameters
            else None
        ),
        soap_weight_decay=(
            float(training_parameters["soap_weight_decay"])
            if "soap_weight_decay" in training_parameters
            else None
        ),
        soap_normalize_gradient=(
            bool(training_parameters["soap_normalize_gradient"])
            if "soap_normalize_gradient" in training_parameters
            else None
        ),
    )


def settings_to_payload(settings: RunSettings, *, device: torch.device | None = None) -> dict[str, object]:
    config = settings.model
    return {
        "physical": {
            "parameters": {
                "system": config.system,
                "num_qubits": config.n_qubits,
                "distance": config.distance,
                "hamiltonian_source": config.hamiltonian_source,
                "t_initial": config.t_initial,
                "T": config.physical_time,
                "t_final": config.t_final,
                "tau_range": [0.0, 1.0],
                "schedule": config.schedule,
            }
        },
        "neural": {
            "model": "FullPauliAGPPINN",
            "general": {
                "n_inputs": 1,
                "n_outputs": 4**config.n_qubits,
                "n_hidden": config.hidden_layers,
                "n_neurons": config.hidden_width,
                "activation": config.activation,
                "layer_type": config.layer_type,
            },
        },
        "training": {
            "device": str(device) if device is not None else settings.device,
            "optimizer": settings.optimizer,
            "parameters": {
                "epochs": settings.epochs,
                "num_points": settings.num_points,
                "lr": settings.lr,
                "random_seed": settings.seed,
                **soap_parameters_payload(settings),
            },
            "loss": {
                "residual": settings.residual_weight,
                "agp_l2": settings.agp_l2_weight,
            },
            "export": {
                "path_images": settings.path_images,
                "path_data": settings.path_data,
                "top_coefficients": settings.top_coefficients,
                "plot_quantity": "d_lambda_dt_times_C_P",
                "format": "pdf",
            },
        },
    }


def soap_parameters_payload(settings: RunSettings) -> dict[str, object]:
    payload: dict[str, object] = {}
    if settings.soap_precondition_frequency is not None:
        payload["soap_precondition_frequency"] = settings.soap_precondition_frequency
    if settings.soap_max_precondition_dim is not None:
        payload["soap_max_precondition_dim"] = settings.soap_max_precondition_dim
    if settings.soap_weight_decay is not None:
        payload["soap_weight_decay"] = settings.soap_weight_decay
    if settings.soap_normalize_gradient is not None:
        payload["soap_normalize_gradient"] = settings.soap_normalize_gradient
    return payload


def soap_kwargs(settings: RunSettings) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if settings.soap_precondition_frequency is not None:
        kwargs["precondition_frequency"] = settings.soap_precondition_frequency
    if settings.soap_max_precondition_dim is not None:
        kwargs["max_precondition_dim"] = settings.soap_max_precondition_dim
    if settings.soap_weight_decay is not None:
        kwargs["weight_decay"] = settings.soap_weight_decay
    if settings.soap_normalize_gradient is not None:
        kwargs["normalize_gradient"] = settings.soap_normalize_gradient
    return kwargs


def make_optimizer(model: torch.nn.Module, settings: RunSettings) -> tuple[torch.optim.Optimizer, dict[str, object]]:
    name = settings.optimizer
    lr = settings.lr
    optimizer = name.lower()
    if optimizer == "adam":
        instance = torch.optim.Adam(model.parameters(), lr=lr)
        return instance, {"requested": name, "actual": "Adam", "class": type(instance).__name__}
    if optimizer == "adamw":
        instance = torch.optim.AdamW(model.parameters(), lr=lr)
        return instance, {"requested": name, "actual": "AdamW", "class": type(instance).__name__}
    if optimizer in {"soap", "safe_mps_soap", "safempssoap"}:
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if pytorch_optimizer is None:
            print("SOAP is unavailable because pytorch-optimizer is not installed; falling back to AdamW.")
            instance = torch.optim.AdamW(params, lr=lr)
            return instance, {"requested": name, "actual": "AdamW", "class": type(instance).__name__, "fallback": True}
        optimizer_cls = SafeMPSSOAP if any(parameter.device.type == "mps" for parameter in params) else pytorch_optimizer.SOAP
        instance = optimizer_cls(params, lr=lr, **soap_kwargs(settings))
        return instance, {
            "requested": name,
            "actual": "SOAP",
            "class": optimizer_cls.__name__,
            "fallback": False,
            "soap_kwargs": soap_kwargs(settings),
        }
    raise ValueError(f"Unsupported optimizer {name!r}.")


def pauli_math_label(label: str) -> str:
    return rf"$\dot{{\lambda}}C_{{\mathrm{{{label}}}}}$"


def pauli_order(label: str) -> int:
    return sum(symbol != "I" for symbol in label)


def save_pdf(fig, images_dir: Path, stem: str) -> None:
    fig.savefig(images_dir / f"{stem}.pdf", format="pdf")


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


def rank_coefficients(coefficients: torch.Tensor, labels: list[str]) -> list[dict[str, object]]:
    values = coefficients.detach().cpu().numpy()
    importance = np.sqrt(np.mean(values * values, axis=0))
    mean_abs = np.mean(np.abs(values), axis=0)
    max_abs = np.max(np.abs(values), axis=0)
    rows = []
    for idx, label in enumerate(labels):
        rows.append(
            {
                "label": label,
                "index": idx,
                "order": pauli_order(label),
                "rms": float(importance[idx]),
                "mean_abs": float(mean_abs[idx]),
                "max_abs": float(max_abs[idx]),
            }
        )
    return sorted(rows, key=lambda row: (row["rms"], row["mean_abs"]), reverse=True)


def plot_loss_history(history: list[dict[str, float]], images_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_paper_style(plt)
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.semilogy([row["epoch"] for row in history], [row["total"] for row in history], label="total")
    ax.semilogy([row["epoch"] for row in history], [row["residual"] for row in history], label="residual")
    ax.set_xlabel("epoch", fontsize=LABEL_FS)
    ax.set_ylabel("loss", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS, length=TICK_LENGTH, width=TICK_WIDTH)
    ax.legend(fontsize=LEGEND_FS, frameon=False)
    fig.subplots_adjust(top=0.92, left=0.13, right=0.98, bottom=0.16)
    save_pdf(fig, images_dir, "losses")
    plt.close(fig)


def export_importance_table(
    coefficients: torch.Tensor,
    labels: list[str],
    data_dir: Path,
    *,
    top_k: int,
) -> list[dict[str, object]]:
    ranked = rank_coefficients(coefficients, labels)
    payload = {
        "coefficient_kind": "counterdiabatic_hamiltonian",
        "coefficient_definition": "d_lambda_dt * C_P(t)",
        "ranking_metric": "rms_over_time",
        "top_k": top_k,
        "all_terms": ranked,
        "top_nonidentity_terms": [row for row in ranked if row["order"] > 0][:top_k],
    }
    with (data_dir / "coefficient_importance.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return payload["top_nonidentity_terms"]


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
                "index": idx,
                "order": row["order"],
                "rms": row["rms"],
                "mean_abs": row["mean_abs"],
                "max_abs": row["max_abs"],
                "values": [float(value) for value in coeff_np[:, idx]],
            }
        )
    payload = {
        "coefficient_kind": "counterdiabatic_hamiltonian",
        "coefficient_definition": "d_lambda_dt * C_P(t)",
        "tau": tau_values,
        "physical_time": t_values,
        "d_lambda_dt": velocity_values,
        "ranking_metric": "rms_over_time",
        "top_coefficients": top_payload,
        "support_map_terms": top_payload[: min(len(top_payload), 16)],
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


def plot_top_coefficients(
    tau: torch.Tensor,
    coefficients: torch.Tensor,
    top_terms: list[dict[str, object]],
    images_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    from matplotlib.ticker import ScalarFormatter

    set_paper_style(plt)
    tau_np = tau.detach().cpu().view(-1).numpy()
    coeff_np = coefficients.detach().cpu().numpy()
    selected = top_terms[: min(len(top_terms), MAX_LABELED_TRACES)]
    fig, ax = plt.subplots(figsize=(6.7, 3.7))
    for plot_idx, row in enumerate(selected):
        color = OKABE_ITO[plot_idx % len(OKABE_ITO)]
        values = coeff_np[:, int(row["index"])]
        ax.plot(
            tau_np,
            values,
            color=color,
            linewidth=LINE_WIDTH,
        )
        label_idx = int(np.argmax(np.abs(values)))
        offset_points = (4, 7 if plot_idx % 2 == 0 else -9)
        text = ax.annotate(
            pauli_math_label(str(row["label"])),
            xy=(tau_np[label_idx], values[label_idx]),
            xytext=offset_points,
            textcoords="offset points",
            color=color,
            fontsize=LEGEND_FS,
            ha="left",
            va="center",
            annotation_clip=False,
        )
        text.set_path_effects([path_effects.withStroke(linewidth=2.4, foreground="white")])
    ax.axhline(0.0, color="0.3", linewidth=0.7)
    ax.margins(x=0.04, y=0.14)
    ax.set_xlabel(r"$\tau=t/T$", fontsize=LABEL_FS)
    ax.set_ylabel(r"$\dot{\lambda}(t) C_P(t)$", fontsize=LABEL_FS)
    ax.set_title("Largest counterdiabatic coefficients", fontsize=TITLE_FS)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.tick_params(axis="both", labelsize=TICK_FS, length=TICK_LENGTH, width=TICK_WIDTH)
    fig.subplots_adjust(top=0.88, left=0.11, right=0.98, bottom=0.17)
    save_pdf(fig, images_dir, "top_hcd_coefficients")
    plt.close(fig)


def plot_support_map(
    top_terms: list[dict[str, object]],
    n_qubits: int,
    images_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import ScalarFormatter

    set_paper_style(plt)
    selected = top_terms[: min(len(top_terms), 16)]
    if not selected:
        return
    code = {"I": 0, "X": 1, "Y": 2, "Z": 3}
    matrix = np.array([[code[symbol] for symbol in str(row["label"])] for row in selected], dtype=float)
    importance = np.array([float(row["rms"]) for row in selected])
    labels = [str(row["label"]) for row in selected]
    fig, (ax_map, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(7.8, max(3.2, 0.28 * len(selected) + 1.2)),
        gridspec_kw={"width_ratios": [max(1.7, 0.42 * n_qubits), 1.45], "wspace": 0.34},
    )
    cmap = ListedColormap(["#f7f7f7", "#0072B2", "#D55E00", "#009E73"])
    x_edges = np.arange(n_qubits + 1)
    y_edges = np.arange(len(selected) + 1)
    image = ax_map.pcolormesh(
        x_edges,
        y_edges,
        matrix,
        cmap=cmap,
        vmin=-0.5,
        vmax=3.5,
        edgecolors="white",
        linewidth=0.35,
        rasterized=False,
    )
    ax_map.invert_yaxis()
    ax_map.set_xticks(np.arange(n_qubits) + 0.5)
    ax_map.set_xticklabels([rf"$q_{{{idx}}}$" for idx in range(n_qubits)], fontsize=TICK_FS)
    ax_map.set_yticks(np.arange(len(selected)) + 0.5)
    ax_map.set_yticklabels([pauli_math_label(label) for label in labels], fontsize=TICK_FS)
    ax_map.set_xlabel("qubit", fontsize=LABEL_FS)
    ax_map.set_title("Top operator hyperedges", fontsize=TITLE_FS)
    ax_map.tick_params(axis="both", length=TICK_LENGTH, width=TICK_WIDTH)
    for row_idx, label in enumerate(labels):
        for qubit_idx, symbol in enumerate(label):
            ax_map.text(
                qubit_idx + 0.5,
                row_idx + 0.5,
                symbol,
                ha="center",
                va="center",
                fontsize=max(7, TICK_FS - 1),
                color="0.15",
            )

    y = np.arange(len(selected))
    ax_bar.barh(y, importance, color="#0072B2")
    ax_bar.invert_yaxis()
    ax_bar.set_yticks([])
    ax_bar.set_xlabel(r"$\mathrm{RMS}_\tau(\dot{\lambda}C_P)$", fontsize=LABEL_FS)
    ax_bar.set_title("importance", fontsize=TITLE_FS)
    ax_bar.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax_bar.tick_params(axis="x", labelsize=TICK_FS, length=TICK_LENGTH, width=TICK_WIDTH)
    fig.subplots_adjust(top=0.88, left=0.22, right=0.98, bottom=0.16, wspace=0.34)
    save_pdf(fig, images_dir, "hcd_coefficient_support_map")
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

    fig, (ax_heat, ax_order) = plt.subplots(
        1,
        2,
        figsize=(7.8, 3.4),
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.56},
    )
    cmap = mpl.colormaps["viridis"]
    heat = ax_heat.pcolormesh(
        np.arange(n_qubits + 1),
        np.arange(n_qubits + 1),
        pair_matrix,
        cmap=cmap,
        edgecolors="white",
        linewidth=0.35,
        rasterized=False,
    )
    ax_heat.invert_yaxis()
    ax_heat.set_xticks(np.arange(n_qubits) + 0.5)
    ax_heat.set_yticks(np.arange(n_qubits) + 0.5)
    ax_heat.set_xticklabels([rf"$q_{{{idx}}}$" for idx in range(n_qubits)], fontsize=TICK_FS)
    ax_heat.set_yticklabels([rf"$q_{{{idx}}}$" for idx in range(n_qubits)], fontsize=TICK_FS)
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
    fig.subplots_adjust(top=0.84, left=0.09, right=0.90, bottom=0.18, wspace=0.56)
    save_pdf(fig, images_dir, "hcd_connection_summary")
    plt.close(fig)


def export_coefficient_plots(
    tau: torch.Tensor,
    t: torch.Tensor,
    d_lambda_dt: torch.Tensor,
    coefficients: torch.Tensor,
    labels: list[str],
    images_dir: Path,
    data_dir: Path,
    *,
    top_k: int,
) -> None:
    top_terms = export_importance_table(coefficients, labels, data_dir, top_k=top_k)
    ranked_terms = rank_coefficients(coefficients, labels)
    n_qubits = len(labels[0])
    export_plot_numerics(tau, t, d_lambda_dt, coefficients, labels, top_terms, ranked_terms, data_dir)
    plot_top_coefficients(tau, coefficients, top_terms, images_dir)
    plot_support_map(top_terms, n_qubits, images_dir)
    plot_connection_summary(ranked_terms, n_qubits, images_dir)


def regenerate_plots_from_saved_run(settings: RunSettings, run_dir: Path) -> None:
    images_dir = run_dir / settings.path_images
    data_dir = run_dir / settings.path_data
    images_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    coefficient_path = data_dir / "final_agp_coefficients.pt"
    if not coefficient_path.is_file():
        raise FileNotFoundError(f"Cannot regenerate plots; missing {coefficient_path}.")

    payload = torch.load(coefficient_path, map_location="cpu")
    tau = payload.get("tau")
    t = payload.get("t")
    if tau is None and t is not None:
        t_flat = t.view(-1)
        tau = ((t_flat - t_flat[0]) / (t_flat[-1] - t_flat[0])).view(-1, 1)
    if t is None and tau is not None:
        t = settings.model.t_initial + settings.model.physical_time * tau
    if tau is None or t is None:
        raise KeyError("Saved coefficients must include at least tau or physical time t.")

    d_lambda_dt = payload["d_lambda_dt"]
    coefficients = payload.get("counterdiabatic_coefficients")
    if coefficients is None:
        coefficients = d_lambda_dt * payload["agp_coefficients"]
    labels = [str(label) for label in payload["pauli_labels"]]

    history_path = data_dir / "loss_history.json"
    if history_path.is_file():
        with history_path.open("r", encoding="utf-8") as handle:
            history = json.load(handle)
        plot_loss_history(history, images_dir)

    export_coefficient_plots(
        tau,
        t,
        d_lambda_dt,
        coefficients,
        labels,
        images_dir,
        data_dir,
        top_k=settings.top_coefficients,
    )


def run_training(settings: RunSettings, run_dir: Path) -> dict[str, float]:
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
    model = FullPauliAGPPINN(
        h0,
        h1,
        hidden_layers=config.hidden_layers,
        hidden_width=config.hidden_width,
        activation=config.activation,
        layer_type=config.layer_type,
        t_min=config.t_initial,
        t_max=config.t_final,
    ).to(device)
    optimizer, optimizer_info = make_optimizer(model, settings)
    loss_weights = FullPauliLossWeights(residual=settings.residual_weight, agp_l2=settings.agp_l2_weight)
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
                f"basis_size={int(row['basis_size'])} h_terms={int(row['hamiltonian_terms'])}"
            )

    images_dir = run_dir / settings.path_images
    data_dir = run_dir / settings.path_data
    images_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    with (data_dir / "loss_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
        handle.write("\n")
    run_metadata = settings_to_payload(settings, device=device)
    run_metadata["resolved"] = asdict(settings)
    run_metadata["derived"] = {
        "t_final": config.t_final,
        "output_terms": model.output_terms,
        "hamiltonian_terms": len(model.hamiltonian_labels),
        "plot_quantity": "d_lambda_dt_times_C_P",
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
            "pauli_labels": model.pauli_labels,
        },
        data_dir / "training_checkpoint.pt",
    )

    with torch.no_grad():
        prediction = model(t)
        final_coefficients = prediction["agp_coefficients"].detach().cpu()
        final_d_lambda_dt = prediction["d_lambda_dt"].detach().cpu()
        final_hcd_coefficients = final_d_lambda_dt * final_coefficients
        final_t = t.detach().cpu()
        final_tau = tau.detach().cpu()
        torch.save(
            {
                "t": final_t,
                "tau": final_tau,
                "physical_time_T": config.physical_time,
                "pauli_labels": model.pauli_labels,
                "agp_coefficients": final_coefficients,
                "counterdiabatic_coefficients": final_hcd_coefficients,
                "counterdiabatic_coefficient_definition": "d_lambda_dt * C_P(t)",
                "lambda": prediction["lambda"].detach().cpu(),
                "d_lambda_dt": final_d_lambda_dt,
            },
            data_dir / "final_agp_coefficients.pt",
        )

    try:
        plot_loss_history(history, images_dir)
        export_coefficient_plots(
            final_tau,
            final_t,
            final_d_lambda_dt,
            final_hcd_coefficients,
            model.pauli_labels,
            images_dir,
            data_dir,
            top_k=settings.top_coefficients,
        )
    except Exception as exc:  # pragma: no cover - plotting is noncritical for training.
        print(f"Skipping plot export: {exc}")

    return history[-1]


def main_for_config(config: TrainingConfig, run_dir: Path) -> None:
    parser = argparse.ArgumentParser(description="Train a fixed-schedule full-Pauli AGP PINN.")
    parser.add_argument("--config", type=Path, default=run_dir / "config.json")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--distance", default=None)
    parser.add_argument("--physical-time", type=float, default=None)
    parser.add_argument("--top-coefficients", type=int, default=None)
    parser.add_argument("--plots-only", action="store_true", help="Regenerate PDFs from Models_Data without training.")
    args = parser.parse_args()
    settings = run_settings_from_payload(load_config_payload(args.config, config), config)
    model_config = settings.model
    if args.distance is not None:
        model_config = replace(model_config, distance=args.distance)
    if args.physical_time is not None:
        model_config = replace(model_config, physical_time=args.physical_time)
    settings = replace(
        settings,
        model=model_config,
        epochs=args.epochs if args.epochs is not None else settings.epochs,
        num_points=args.num_points if args.num_points is not None else settings.num_points,
        lr=args.lr if args.lr is not None else settings.lr,
        optimizer=args.optimizer if args.optimizer is not None else settings.optimizer,
        device=args.device if args.device is not None else settings.device,
        seed=args.seed if args.seed is not None else settings.seed,
        top_coefficients=args.top_coefficients if args.top_coefficients is not None else settings.top_coefficients,
    )
    if args.plots_only:
        regenerate_plots_from_saved_run(settings, run_dir)
        print("regenerated_plots=true")
        return
    final = run_training(settings, run_dir)
    print(f"final_loss={final['total']:.6e} final_residual={final['residual']:.6e}")
