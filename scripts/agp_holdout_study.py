from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]

from projected_sparse_training_common import (  # noqa: E402
    LABEL_FS,
    LEGEND_FS,
    LINE_WIDTH,
    OKABE_ITO,
    ProjectedSparseLossWeights,
    TICK_FS,
    TICK_LENGTH,
    TICK_WIDTH,
    TITLE_FS,
    build_projected_support,
    compact_pauli_label,
    make_projected_model,
    pauli_weight,
    select_device,
    set_paper_style,
)
from agp_baseline_train import model_config_from_payload, parse_support_sizes  # noqa: E402
from utils import load_pauli_hamiltonian_pair, sort_pauli_labels  # noqa: E402


RUN_DIR = Path.cwd()
DEFAULT_CONFIG = Path("config.json")


def configure_run_dir(config_path: Path) -> None:
    global RUN_DIR
    RUN_DIR = config_path.resolve().parent


@dataclass(frozen=True)
class Thresholds:
    plateau: float
    holdout: float
    unseen: float
    top_stability: float
    top_fraction: float


def load_json(path: Path) -> dict[str, object] | list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def relpath(path: Path, base: Path | None = None) -> str:
    base = RUN_DIR if base is None else base
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def load_body_weights(model: torch.nn.Module, checkpoint: dict[str, object]) -> None:
    state = checkpoint["model_state_dict"]
    body_state = {
        key.removeprefix("body."): value
        for key, value in state.items()
        if key.startswith("body.")
    }
    model.body.load_state_dict(body_state)
    if "agp_log_gamma" in state and "agp_gate_logits" in state:
        support_metadata = checkpoint.get("config", {}).get("support", {})
        support_metadata = support_metadata if isinstance(support_metadata, dict) else {}
        calibration = support_metadata.get("agp_calibration", {})
        calibration = calibration if isinstance(calibration, dict) else {}
        device = next(model.parameters()).device
        model.agp_gate_temperature = float(calibration.get("gate_temperature", 1.0))
        model.agp_target_active_terms = int(calibration.get("target_active_terms", len(model.agp_labels)))
        model.agp_log_gamma = torch.nn.Parameter(state["agp_log_gamma"].detach().to(device).float().reshape(()))
        model.agp_gate_logits = torch.nn.Parameter(state["agp_gate_logits"].detach().to(device).float().flatten())


def load_checkpoint_labels(run_dir: Path) -> tuple[list[str], list[str]]:
    checkpoint = torch.load(run_dir / "Models_Data" / "training_checkpoint.pt", map_location="cpu")
    return [str(label) for label in checkpoint["agp_labels"]], [str(label) for label in checkpoint["residual_labels"]]


def norm_sq_subset(values: torch.Tensor, indices: list[int]) -> torch.Tensor:
    if not indices:
        return torch.zeros((), dtype=values.real.dtype, device=values.device)
    index = torch.tensor(indices, dtype=torch.long, device=values.device)
    subset = values.index_select(-1, index)
    return torch.mean(torch.sum(torch.abs(subset) ** 2, dim=-1).real)


def rms_per_label(values: torch.Tensor) -> np.ndarray:
    return torch.sqrt(torch.mean(torch.abs(values) ** 2, dim=0).real).detach().cpu().numpy()


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def relative_metric_with_reference_status(
    *,
    residual: torch.Tensor | float,
    reference: torch.Tensor | float,
    eps: float,
    term_count: int | None = None,
) -> tuple[float | None, dict[str, object]]:
    residual_value = scalar(residual) if isinstance(residual, torch.Tensor) else float(residual)
    reference_value = scalar(reference) if isinstance(reference, torch.Tensor) else float(reference)
    if term_count == 0:
        return None, {
            "valid": False,
            "reason": "empty_subset",
            "residual": residual_value,
            "reference_residual": reference_value,
            "eps": float(eps),
        }
    if reference_value <= float(eps):
        return None, {
            "valid": False,
            "reason": "zero_reference",
            "residual": residual_value,
            "reference_residual": reference_value,
            "eps": float(eps),
        }
    return residual_value / reference_value, {
        "valid": True,
        "reason": "finite_reference",
        "residual": residual_value,
        "reference_residual": reference_value,
        "eps": float(eps),
    }


def optional_float(value: object) -> float:
    return float("nan") if value is None else float(value)


def load_ranked_coefficient_labels(run_dir: Path) -> list[str]:
    payload = load_json(run_dir / "Models_Data" / "coefficient_importance.json")
    if not isinstance(payload, dict):
        raise TypeError("coefficient_importance.json must contain a JSON object.")
    all_terms = payload.get("all_terms", [])
    if not isinstance(all_terms, list):
        raise TypeError("coefficient_importance.json field 'all_terms' must be a list.")
    return [str(row["label"]) for row in all_terms if isinstance(row, dict) and "label" in row]


def top_fraction_set(run_dir: Path, fraction: float) -> set[str]:
    ranked = load_ranked_coefficient_labels(run_dir)
    if not ranked:
        return set()
    count = max(1, int(math.ceil(float(fraction) * len(ranked))))
    return set(ranked[:count])


def evaluate_one_run(
    *,
    run_dir: Path,
    config_payload: dict[str, object],
    residual_top_k: int,
    intermediate_top_k: int,
    device: torch.device,
    spectra_dir: Path,
    common_residual_labels: list[str] | None,
    holdout_basis_mode: str,
    holdout_basis_agp_terms: int | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    config = model_config_from_payload(config_payload)
    training = config_payload.get("training", {})
    parameters = training.get("parameters", {}) if isinstance(training, dict) else {}
    num_points = int(parameters.get("num_points", 16))

    checkpoint_path = run_dir / "Models_Data" / "training_checkpoint.pt"
    metadata_path = run_dir / "Models_Data" / "support_metadata.json"
    history_path = run_dir / "Models_Data" / "loss_history.json"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_metadata = load_json(metadata_path)
    train_history = load_json(history_path)
    if not isinstance(train_metadata, dict) or not isinstance(train_history, list):
        raise TypeError(f"Unexpected metadata/history shape in {run_dir}.")

    trained_agp_labels = [str(label) for label in checkpoint["agp_labels"]]
    trained_residual_labels = {str(label) for label in checkpoint["residual_labels"]}

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
        agp_top_k=len(trained_agp_labels),
        intermediate_top_k=intermediate_top_k,
        residual_top_k=residual_top_k,
        agp_labels=trained_agp_labels,
        residual_labels=common_residual_labels,
        stage=0,
    )
    if common_residual_labels is not None:
        support = dict(support)
        support["residual_labels"] = common_residual_labels
    model = make_projected_model(h0, h1, support, config, device)
    load_body_weights(model, checkpoint)
    model.eval()

    tau = torch.linspace(0.0, 1.0, num_points, device=device).view(-1, 1)
    t = config.t_initial + config.physical_time * tau
    weights = ProjectedSparseLossWeights(residual=1.0, agp_l2=0.0)
    with torch.no_grad():
        _, diagnostics = model.loss(t, weights=weights)
        residual = model.euler_lagrange_residual(t)
        reference = model.euler_lagrange_reference_residual(t)

    residual_labels = [str(label) for label in model.residual_labels]
    seen_indices = [idx for idx, label in enumerate(residual_labels) if label in trained_residual_labels]
    unseen_indices = [idx for idx, label in enumerate(residual_labels) if label not in trained_residual_labels]

    seen_residual = norm_sq_subset(residual, seen_indices)
    seen_reference = norm_sq_subset(reference, seen_indices)
    unseen_residual = norm_sq_subset(residual, unseen_indices)
    unseen_reference = norm_sq_subset(reference, unseen_indices)
    eps = float(torch.finfo(seen_residual.dtype).eps)
    seen_relative, seen_relative_status = relative_metric_with_reference_status(
        residual=seen_residual,
        reference=seen_reference,
        eps=eps,
        term_count=len(seen_indices),
    )
    unseen_relative, unseen_relative_status = relative_metric_with_reference_status(
        residual=unseen_residual,
        reference=unseen_reference,
        eps=eps,
        term_count=len(unseen_indices),
    )

    residual_rms = rms_per_label(residual)
    reference_rms = rms_per_label(reference)
    spectrum = sorted(
        [
            {
                "label": label,
                "index": idx,
                "order": pauli_weight(label),
                "seen_during_training": label in trained_residual_labels,
                "residual_rms": float(residual_rms[idx]),
                "reference_rms": float(reference_rms[idx]),
            }
            for idx, label in enumerate(residual_labels)
        ],
        key=lambda row: (float(row["residual_rms"]), int(row["order"])),
        reverse=True,
    )

    spectra_dir.mkdir(parents=True, exist_ok=True)
    spectrum_path = spectra_dir / f"holdout_residual_spectrum_agp_{len(model.agp_labels)}_residual_{len(model.residual_labels)}.json"
    with spectrum_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "trained_run": relpath(run_dir),
                "agp_terms": len(model.agp_labels),
                "holdout_residual_terms": len(model.residual_labels),
                "spectrum": spectrum,
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    full_basis = Decimal(4) ** int(config.n_qubits)
    best = min(train_history, key=lambda row: float(row["total"]))
    result = {
        "trained_run": relpath(run_dir),
        "n_qubits": int(config.n_qubits),
        "full_pauli_basis_size": str(full_basis),
        "agp_terms": len(model.agp_labels),
        "agp_fraction_of_full_basis": f"{Decimal(len(model.agp_labels)) / full_basis:.12E}",
        "train_residual_terms": int(train_metadata["final_residual_terms"]),
        "holdout_residual_terms": len(model.residual_labels),
        "holdout_basis_mode": holdout_basis_mode,
        "holdout_basis_agp_terms": holdout_basis_agp_terms,
        "seen_residual_terms": len(seen_indices),
        "unseen_residual_terms": len(unseen_indices),
        "intermediate_terms": len(model.intermediate_labels),
        "hamiltonian_terms": len(model.hamiltonian_labels),
        "first_commutator_nnz": model.first_commutator.nnz,
        "second_commutator_nnz": model.second_commutator.nnz,
        "training_final_relative_residual": float(train_history[-1]["relative_residual"]),
        "training_best_relative_residual": float(best["relative_residual"]),
        "holdout_total_residual": scalar(diagnostics["residual"]),
        "holdout_reference_residual": scalar(diagnostics["reference_residual"]),
        "holdout_relative_residual": scalar(diagnostics["relative_residual"]),
        "seen_residual": scalar(seen_residual),
        "seen_reference_residual": scalar(seen_reference),
        "seen_relative_residual": seen_relative,
        "seen_relative_residual_status": seen_relative_status,
        "seen_residual_per_term": scalar(seen_residual) / max(len(seen_indices), 1),
        "seen_reference_residual_per_term": scalar(seen_reference) / max(len(seen_indices), 1),
        "unseen_residual": scalar(unseen_residual),
        "unseen_reference_residual": scalar(unseen_reference),
        "unseen_relative_residual": unseen_relative,
        "unseen_relative_residual_status": unseen_relative_status,
        "unseen_residual_per_term": scalar(unseen_residual) / max(len(unseen_indices), 1),
        "unseen_reference_residual_per_term": scalar(unseen_reference) / max(len(unseen_indices), 1),
        "top_holdout_residual_terms": spectrum[:64],
        "spectrum_export": relpath(spectrum_path),
        "residual_basis_note": (
            "The trained network weights are unchanged. The residual basis is enlarged after training; "
            "unseen metrics use only holdout Pauli labels absent from the original training residual basis. "
            "By default the same holdout residual labels are used for every K in the sweep."
        ),
    }
    return result, spectrum


def build_common_holdout_residual_labels(
    *,
    run_dirs: list[Path],
    config_payload: dict[str, object],
    residual_top_k: int,
    intermediate_top_k: int,
) -> tuple[list[str], int]:
    config = model_config_from_payload(config_payload)
    hamiltonian_path = Path(config.hamiltonian_source)
    if not hamiltonian_path.is_absolute():
        hamiltonian_path = ROOT / hamiltonian_path
    h0, h1 = load_pauli_hamiltonian_pair(
        hamiltonian_path,
        system=config.system,
        n_qubits=config.n_qubits,
        distance=config.distance,
    )
    agp_labels: set[str] = set()
    for run_dir in run_dirs:
        run_agp_labels, _ = load_checkpoint_labels(run_dir)
        agp_labels.update(run_agp_labels)
    sorted_agp = sort_pauli_labels(agp_labels)
    support = build_projected_support(
        h0,
        h1,
        agp_top_k=len(sorted_agp),
        intermediate_top_k=intermediate_top_k,
        residual_top_k=residual_top_k,
        agp_labels=sorted_agp,
        stage=0,
    )
    return [str(label) for label in support["residual_labels"]], len(sorted_agp)


def add_stability_and_criteria(
    rows: list[dict[str, object]],
    run_dirs: list[Path],
    thresholds: Thresholds,
) -> dict[str, object]:
    top_sets = [top_fraction_set(run_dir, thresholds.top_fraction) for run_dir in run_dirs]
    largest_set = top_sets[-1] if top_sets else set()

    for idx, row in enumerate(rows):
        current_set = top_sets[idx]
        row["top_fraction"] = thresholds.top_fraction
        row["top_fraction_count"] = len(current_set)
        if largest_set and current_set:
            row["top_fraction_overlap_with_largest"] = len(current_set & largest_set) / min(len(current_set), len(largest_set))
        else:
            row["top_fraction_overlap_with_largest"] = 0.0
        if idx + 1 < len(top_sets) and current_set and top_sets[idx + 1]:
            row["top_fraction_overlap_with_next"] = len(current_set & top_sets[idx + 1]) / min(
                len(current_set),
                len(top_sets[idx + 1]),
            )
        else:
            row["top_fraction_overlap_with_next"] = None

    for idx, row in enumerate(rows):
        if idx + 1 < len(rows):
            current = float(row["training_final_relative_residual"])
            next_value = float(rows[idx + 1]["training_final_relative_residual"])
            improvement = (current - next_value) / current if current > 0.0 else 0.0
            plateau_pass = improvement <= thresholds.plateau
        else:
            improvement = None
            plateau_pass = None

        if idx + 1 < len(rows):
            stability_value = row["top_fraction_overlap_with_next"]
        elif idx > 0:
            stability_value = rows[idx - 1]["top_fraction_overlap_with_largest"]
        else:
            stability_value = row["top_fraction_overlap_with_largest"]
        stability_pass = (
            bool(stability_value is not None and float(stability_value) >= thresholds.top_stability)
        )

        unseen_relative = row.get("unseen_relative_residual")
        unseen_status = row.get("unseen_relative_residual_status", {})
        unseen_valid = bool(isinstance(unseen_status, dict) and unseen_status.get("valid", unseen_relative is not None))
        criteria = {
            "training_plateau": {
                "value": improvement,
                "threshold": thresholds.plateau,
                "pass": plateau_pass,
                "note": "Uses improvement in training relative residual when moving to the next larger K.",
            },
            "holdout_relative_residual": {
                "value": float(row["holdout_relative_residual"]),
                "threshold": thresholds.holdout,
                "pass": float(row["holdout_relative_residual"]) <= thresholds.holdout,
            },
            "unseen_relative_residual": {
                "value": unseen_relative,
                "threshold": thresholds.unseen,
                "pass": unseen_valid and unseen_relative is not None and float(unseen_relative) <= thresholds.unseen,
                "valid": unseen_valid,
                "note": "Invalid when the AGP=0 reference residual on the unseen subset is zero.",
            },
            "top_term_stability": {
                "value": stability_value,
                "threshold": thresholds.top_stability,
                "pass": stability_pass,
                "note": "Uses top-fraction overlap with the next K; for the largest K, uses previous-vs-largest overlap.",
            },
        }
        accepted = all(item["pass"] is True for item in criteria.values())
        criteria["accepted"] = {"pass": accepted}
        row["criteria"] = criteria

    accepted_rows = [row for row in rows if row["criteria"]["accepted"]["pass"] is True]
    if accepted_rows:
        k_min = int(accepted_rows[0]["agp_terms"])
        status = "found"
        conclusion = f"K_min={k_min} for the tested grid and thresholds."
    else:
        k_min = None
        status = "not_found_in_tested_grid"
        conclusion = (
            f"No tested support size passes all criteria; for this setup K_min is larger than "
            f"{max(int(row['agp_terms']) for row in rows)} or the support-generation rule must change."
        )
    return {
        "status": status,
        "k_min": k_min,
        "conclusion": conclusion,
        "thresholds": {
            "training_plateau_max_improvement": thresholds.plateau,
            "holdout_relative_residual_max": thresholds.holdout,
            "unseen_relative_residual_max": thresholds.unseen,
            "top_term_stability_min": thresholds.top_stability,
            "top_fraction": thresholds.top_fraction,
        },
    }


def plot_relative_residuals(rows: list[dict[str, object]], images_dir: Path, thresholds: Thresholds) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_paper_style(plt)
    x = np.asarray([int(row["agp_terms"]) for row in rows], dtype=float)
    series = [
        ("training", [float(row["training_final_relative_residual"]) for row in rows], OKABE_ITO[0], "o"),
        ("holdout", [float(row["holdout_relative_residual"]) for row in rows], OKABE_ITO[1], "s"),
        ("unseen", [optional_float(row["unseen_relative_residual"]) for row in rows], OKABE_ITO[2], "^"),
    ]
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    for label, values, color, marker in series:
        ax.semilogy(x, values, marker=marker, linewidth=LINE_WIDTH, color=color, label=label)
    ax.axhline(thresholds.holdout, color="0.35", linestyle="--", linewidth=0.8)
    ax.axhline(thresholds.unseen, color="0.55", linestyle=":", linewidth=0.8)
    ax.set_xlabel("AGP support size $K$", fontsize=LABEL_FS)
    ax.set_ylabel("relative residual", fontsize=LABEL_FS)
    n_qubits = rows[0]["n_qubits"] if rows else "?"
    ax.set_title(fr"$q={n_qubits}$ residual generalization", fontsize=TITLE_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS, length=TICK_LENGTH, width=TICK_WIDTH)
    fig.legend(loc="upper center", ncol=3, frameon=False, fontsize=LEGEND_FS, bbox_to_anchor=(0.53, 1.02))
    fig.subplots_adjust(top=0.80, left=0.13, right=0.98, bottom=0.16)
    fig.savefig(images_dir / "holdout_relative_residuals.pdf", format="pdf")
    plt.close(fig)


def plot_seen_unseen(rows: list[dict[str, object]], images_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_paper_style(plt)
    x = np.asarray([int(row["agp_terms"]) for row in rows], dtype=float)
    width = 58.0
    seen = np.asarray([float(row["seen_residual"]) for row in rows])
    unseen = np.asarray([float(row["unseen_residual"]) for row in rows])
    seen_rel = np.asarray([optional_float(row["seen_relative_residual"]) for row in rows])
    unseen_rel = np.asarray([optional_float(row["unseen_relative_residual"]) for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    axes[0].bar(x - width / 2.0, seen, width=width, color=OKABE_ITO[0], label="seen")
    axes[0].bar(x + width / 2.0, unseen, width=width, color=OKABE_ITO[1], label="unseen")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("AGP support size $K$", fontsize=LABEL_FS)
    axes[0].set_ylabel(r"$\|R(A)\|^2$", fontsize=LABEL_FS)
    axes[0].set_title("absolute residual", fontsize=TITLE_FS)

    axes[1].semilogy(x, seen_rel, marker="o", linewidth=LINE_WIDTH, color=OKABE_ITO[0], label="seen")
    axes[1].semilogy(x, unseen_rel, marker="s", linewidth=LINE_WIDTH, color=OKABE_ITO[1], label="unseen")
    axes[1].set_xlabel("AGP support size $K$", fontsize=LABEL_FS)
    axes[1].set_title("relative residual", fontsize=TITLE_FS)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=TICK_FS, length=TICK_LENGTH, width=TICK_WIDTH)
    fig.legend(loc="upper center", ncol=2, frameon=False, fontsize=LEGEND_FS, bbox_to_anchor=(0.53, 1.03))
    fig.subplots_adjust(top=0.78, left=0.10, right=0.98, bottom=0.18, wspace=0.32)
    fig.savefig(images_dir / "holdout_seen_unseen_residuals.pdf", format="pdf")
    plt.close(fig)


def plot_residual_spectrum(
    rows: list[dict[str, object]],
    spectra: dict[int, list[dict[str, object]]],
    images_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_paper_style(plt)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for idx, row in enumerate(rows):
        k = int(row["agp_terms"])
        values = np.asarray([float(item["residual_rms"]) for item in spectra[k]], dtype=float)
        ranks = np.arange(1, len(values) + 1)
        ax.loglog(ranks, values, linewidth=1.1, color=OKABE_ITO[idx % len(OKABE_ITO)], label=fr"$K={k}$")
    ax.set_xlabel("holdout residual rank", fontsize=LABEL_FS)
    ax.set_ylabel(r"RMS residual coefficient", fontsize=LABEL_FS)
    ax.set_title("holdout residual spectrum", fontsize=TITLE_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS, length=TICK_LENGTH, width=TICK_WIDTH)
    fig.legend(loc="upper center", ncol=5, frameon=False, fontsize=LEGEND_FS, bbox_to_anchor=(0.53, 1.03))
    fig.subplots_adjust(top=0.78, left=0.13, right=0.98, bottom=0.16)
    fig.savefig(images_dir / "holdout_residual_spectrum.pdf", format="pdf")
    plt.close(fig)


def plot_top_holdout_terms(row: dict[str, object], images_dir: Path, *, top_k: int = 20) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_paper_style(plt)
    terms = list(row["top_holdout_residual_terms"])[:top_k]
    terms = list(reversed(terms))
    labels = [compact_pauli_label(str(item["label"]), max_sites=5) for item in terms]
    values = np.asarray([float(item["residual_rms"]) for item in terms], dtype=float)
    colors = [OKABE_ITO[0] if bool(item["seen_during_training"]) else OKABE_ITO[1] for item in terms]
    y = np.arange(len(terms))

    fig, ax = plt.subplots(figsize=(6.3, 5.4))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=LEGEND_FS)
    ax.set_xscale("log")
    ax.set_xlabel(r"RMS residual coefficient", fontsize=LABEL_FS)
    ax.set_title(fr"largest holdout residual terms, $K={int(row['agp_terms'])}$", fontsize=TITLE_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS, length=TICK_LENGTH, width=TICK_WIDTH)
    fig.subplots_adjust(top=0.92, left=0.31, right=0.98, bottom=0.11)
    fig.savefig(images_dir / "holdout_top_residual_terms.pdf", format="pdf")
    plt.close(fig)


def plot_kmin_decision(rows: list[dict[str, object]], images_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    set_paper_style(plt)
    criteria_keys = [
        ("training_plateau", "plateau"),
        ("holdout_relative_residual", "holdout"),
        ("unseen_relative_residual", "unseen"),
        ("top_term_stability", "top terms"),
        ("accepted", "accepted"),
    ]
    matrix = np.zeros((len(criteria_keys), len(rows)), dtype=float)
    for col, row in enumerate(rows):
        criteria = row["criteria"]
        for row_idx, (key, _) in enumerate(criteria_keys):
            value = criteria[key]["pass"]
            matrix[row_idx, col] = 1.0 if value is True else 0.5 if value is None else 0.0

    cmap = ListedColormap(["#D55E00", "0.75", "#009E73"])
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([str(int(row["agp_terms"])) for row in rows], fontsize=TICK_FS)
    ax.set_yticks(np.arange(len(criteria_keys)))
    ax.set_yticklabels([label for _, label in criteria_keys], fontsize=TICK_FS)
    ax.set_xlabel("AGP support size $K$", fontsize=LABEL_FS)
    ax.set_title(r"$K_{\min}$ decision grid", fontsize=TITLE_FS)
    ax.tick_params(axis="both", length=0)
    for y_idx in range(len(criteria_keys)):
        for x_idx in range(len(rows)):
            text = "pass" if matrix[y_idx, x_idx] == 1.0 else "n/a" if matrix[y_idx, x_idx] == 0.5 else "fail"
            ax.text(x_idx, y_idx, text, ha="center", va="center", fontsize=LEGEND_FS, color="white" if text != "n/a" else "0.2")
    fig.subplots_adjust(top=0.86, left=0.18, right=0.98, bottom=0.20)
    fig.savefig(images_dir / "k_min_decision.pdf", format="pdf")
    plt.close(fig)


def write_outputs(
    *,
    rows: list[dict[str, object]],
    spectra: dict[int, list[dict[str, object]]],
    decision: dict[str, object],
    images_dir: Path,
    data_dir: Path,
    residual_top_k: int,
) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "Sparse AGP support-size holdout study. Each trained model is evaluated without retraining on an "
            "enlarged projected residual basis. By default this is a common holdout basis shared by all K."
        ),
        "holdout_residual_terms": residual_top_k,
        "decision": decision,
        "rows": rows,
    }
    summary_path = data_dir / f"holdout_study_residual_{residual_top_k}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    plot_relative_residuals(rows, images_dir, Thresholds(**{
        "plateau": float(decision["thresholds"]["training_plateau_max_improvement"]),
        "holdout": float(decision["thresholds"]["holdout_relative_residual_max"]),
        "unseen": float(decision["thresholds"]["unseen_relative_residual_max"]),
        "top_stability": float(decision["thresholds"]["top_term_stability_min"]),
        "top_fraction": float(decision["thresholds"]["top_fraction"]),
    }))
    plot_seen_unseen(rows, images_dir)
    plot_residual_spectrum(rows, spectra, images_dir)
    plot_top_holdout_terms(rows[-1], images_dir)
    plot_kmin_decision(rows, images_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configured sparse AGP holdout residual study.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--support-sizes", default=None, help="Comma or space separated AGP support sizes.")
    parser.add_argument("--residual-top-k", type=int, default=8192)
    parser.add_argument("--intermediate-top-k", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--plateau-threshold", type=float, default=0.10)
    parser.add_argument("--holdout-threshold", type=float, default=0.10)
    parser.add_argument("--unseen-threshold", type=float, default=1.0)
    parser.add_argument("--top-stability-threshold", type=float, default=0.85)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument(
        "--holdout-basis",
        choices=("union_agp", "per_run"),
        default="union_agp",
        help=(
            "union_agp builds one common holdout residual basis from the union of trained AGP supports; "
            "per_run rebuilds the holdout basis separately for each K."
        ),
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    configure_run_dir(config_path)
    payload = load_json(config_path)
    if not isinstance(payload, dict):
        raise TypeError("config.json must contain a JSON object.")
    support = payload.get("support_sweep", {})
    default_sizes = (
        [int(value) for value in support.get("agp_terms", [576, 768, 1024, 1536, 2048])]
        if isinstance(support, dict)
        else [576, 768, 1024, 1536, 2048]
    )
    support_sizes = parse_support_sizes(args.support_sizes, default_sizes)
    intermediate_top_k = (
        int(args.intermediate_top_k)
        if args.intermediate_top_k is not None
        else int(support.get("intermediate_top_k", 2048))
        if isinstance(support, dict)
        else 2048
    )
    summary = payload.get("summary", {})
    support_output_root = support.get("output_root") if isinstance(support, dict) else None
    runs_dir = RUN_DIR / str(
        support_output_root
        if support_output_root is not None
        else summary.get("runs_dir", "runs/")
        if isinstance(summary, dict)
        else "runs"
    )
    images_dir = RUN_DIR / str(summary.get("path_images", "Images/")) if isinstance(summary, dict) else RUN_DIR / "Images"
    data_dir = RUN_DIR / str(summary.get("path_data", "Models_Data/")) if isinstance(summary, dict) else RUN_DIR / "Models_Data"
    thresholds = Thresholds(
        plateau=float(args.plateau_threshold),
        holdout=float(args.holdout_threshold),
        unseen=float(args.unseen_threshold),
        top_stability=float(args.top_stability_threshold),
        top_fraction=float(args.top_fraction),
    )

    device = select_device(args.device)
    rows: list[dict[str, object]] = []
    spectra: dict[int, list[dict[str, object]]] = {}
    run_dirs = [runs_dir / f"agp_{support_size}" for support_size in support_sizes]
    if args.holdout_basis == "union_agp":
        common_residual_labels, holdout_basis_agp_terms = build_common_holdout_residual_labels(
            run_dirs=run_dirs,
            config_payload=payload,
            residual_top_k=int(args.residual_top_k),
            intermediate_top_k=intermediate_top_k,
        )
    else:
        common_residual_labels = None
        holdout_basis_agp_terms = None
    for run_dir, support_size in zip(run_dirs, support_sizes):
        print(
            f"evaluate_holdout agp_terms={support_size} residual_terms={args.residual_top_k} "
            f"basis={args.holdout_basis} device={device}"
        )
        row, spectrum = evaluate_one_run(
            run_dir=run_dir,
            config_payload=payload,
            residual_top_k=int(args.residual_top_k),
            intermediate_top_k=intermediate_top_k,
            device=device,
            spectra_dir=data_dir,
            common_residual_labels=common_residual_labels,
            holdout_basis_mode=str(args.holdout_basis),
            holdout_basis_agp_terms=holdout_basis_agp_terms,
        )
        rows.append(row)
        spectra[int(row["agp_terms"])] = spectrum
        print(
            f"done_holdout agp_terms={row['agp_terms']} "
            f"holdout_relative={row['holdout_relative_residual']:.6e} "
            f"unseen_relative={optional_float(row['unseen_relative_residual']):.6e}"
        )

    decision = add_stability_and_criteria(rows, run_dirs, thresholds)
    write_outputs(
        rows=rows,
        spectra=spectra,
        decision=decision,
        images_dir=images_dir,
        data_dir=data_dir,
        residual_top_k=int(args.residual_top_k),
    )
    print(json.dumps({"decision": decision, "summary": relpath(data_dir / f"holdout_study_residual_{args.residual_top_k}.json")}, indent=2))


if __name__ == "__main__":
    getcontext().prec = 80
    main()
