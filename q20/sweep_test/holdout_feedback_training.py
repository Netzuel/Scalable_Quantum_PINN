from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, replace
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from holdout_study import (  # noqa: E402
    Thresholds,
    build_common_holdout_residual_labels,
    evaluate_one_run,
    load_json,
)
from projected_sparse_training_common import (  # noqa: E402
    LABEL_FS,
    LEGEND_FS,
    LINE_WIDTH,
    OKABE_ITO,
    ProjectedRunSettings,
    ProjectedSparseLossWeights,
    TICK_FS,
    TICK_LENGTH,
    TICK_WIDTH,
    TITLE_FS,
    build_projected_support,
    export_results,
    make_optimizer,
    make_projected_model,
    select_device,
    set_paper_style,
    sort_pauli_labels,
    run_training,
    train_stage,
)
from training_script import DEFAULT_CONFIG, RUN_DIR, model_config_from_payload, settings_for_support  # noqa: E402
from utils import load_pauli_hamiltonian_pair  # noqa: E402


def load_body_state_from_checkpoint(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return {
        key.removeprefix("body."): value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith("body.")
    }


def load_checkpoint_labels(checkpoint_path: Path) -> tuple[list[str], list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return [str(label) for label in checkpoint["agp_labels"]], [str(label) for label in checkpoint["residual_labels"]]


def select_residual_additions(
    spectrum: list[dict[str, object]],
    current_residual_labels: set[str],
    *,
    add_terms: int,
    min_rms: float,
) -> list[dict[str, object]]:
    additions: list[dict[str, object]] = []
    for row in spectrum:
        label = str(row["label"])
        if label in current_residual_labels:
            continue
        if float(row["residual_rms"]) < min_rms:
            continue
        additions.append(row)
        if len(additions) >= add_terms:
            break
    return additions


def make_support_with_residual_labels(
    *,
    h0,
    h1,
    settings: ProjectedRunSettings,
    agp_labels: list[str],
    residual_labels: list[str],
    stage: int,
) -> dict[str, object]:
    support = build_projected_support(
        h0,
        h1,
        agp_top_k=len(agp_labels),
        intermediate_top_k=settings.intermediate_top_k,
        residual_top_k=max(settings.residual_top_k, len(residual_labels)),
        agp_labels=agp_labels,
        stage=stage,
    )
    support = dict(support)
    support["residual_labels"] = sort_pauli_labels(residual_labels)
    metadata = dict(support["metadata"])
    metadata["residual_selection_rule"] = "explicit_training_residual_labels"
    metadata["residual_terms_before_explicit_override"] = metadata["residual_terms"]
    metadata["residual_terms"] = len(support["residual_labels"])
    support["metadata"] = metadata
    return support


def train_feedback_round(
    *,
    run_dir: Path,
    payload: dict[str, object],
    settings: ProjectedRunSettings,
    agp_labels: list[str],
    residual_labels: list[str],
    body_state: dict[str, torch.Tensor],
    round_index: int,
    additions: list[dict[str, object]],
) -> tuple[dict[str, torch.Tensor], dict[str, float], dict[str, object]]:
    config = settings.model
    torch.manual_seed(settings.seed + round_index)
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
    support = make_support_with_residual_labels(
        h0=h0,
        h1=h1,
        settings=settings,
        agp_labels=agp_labels,
        residual_labels=residual_labels,
        stage=round_index,
    )
    model = make_projected_model(h0, h1, support, config, device)
    model.body.load_state_dict({key: value.to(device) for key, value in body_state.items()})

    optimizer, optimizer_info = make_optimizer(model, settings)
    loss_weights = ProjectedSparseLossWeights(residual=settings.residual_weight, agp_l2=settings.agp_l2_weight)
    tau = torch.linspace(0.0, 1.0, settings.num_points, device=device).view(-1, 1)
    t = config.t_initial + config.physical_time * tau
    history: list[dict[str, float]] = []
    train_stage(
        model,
        optimizer,
        loss_weights,
        t,
        stage=round_index,
        epochs=settings.epochs,
        global_epoch=0,
        history=history,
    )

    metadata = dict(support["metadata"])
    metadata["n_qubits"] = config.n_qubits
    metadata["device"] = str(device)
    metadata["full_pauli_basis_size"] = 4**config.n_qubits
    metadata["regime"] = "holdout_feedback_projected_sparse"
    metadata["feedback_round"] = round_index
    metadata["feedback_added_terms"] = additions
    metadata["feedback_added_term_count"] = len(additions)
    metadata["adaptive_enabled"] = False
    metadata["final_agp_terms"] = len(model.agp_labels)
    metadata["final_intermediate_terms"] = len(model.intermediate_labels)
    metadata["final_residual_terms"] = len(model.residual_labels)
    metadata["first_commutator_nnz"] = model.first_commutator.nnz
    metadata["second_commutator_nnz"] = model.second_commutator.nnz

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
    with (data_dir / "feedback_added_residual_terms.json").open("w", encoding="utf-8") as handle:
        json.dump(additions, handle, indent=2)
        handle.write("\n")

    run_metadata = {
        "physical": asdict(config),
        "training": asdict(settings),
        "support": metadata,
        "optimizer": optimizer_info,
        "source_config": payload,
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
    next_body_state = {key: value.detach().cpu() for key, value in model.body.state_dict().items()}
    return next_body_state, history[-1], metadata


def read_spectrum(path: Path) -> list[dict[str, object]]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    spectrum = payload.get("spectrum", [])
    if not isinstance(spectrum, list):
        raise TypeError(f"{path} field 'spectrum' must be a list.")
    return [row for row in spectrum if isinstance(row, dict)]


def resolve_holdout_residual_top_k(
    raw_value: object,
    *,
    initial_residual_terms: int,
    rounds: int,
    add_residual_terms: int,
    unseen_batches_after_final_iteration: int,
) -> tuple[int, dict[str, object]]:
    """Resolve the holdout residual budget.

    In automatic mode, keep at least one addition batch unseen after the final
    requested feedback round. This prevents an empty unseen set from appearing
    as a physically meaningful zero residual in the summary plots.
    """

    if rounds < 0:
        raise ValueError("Feedback iterations must be non-negative.")
    if add_residual_terms < 0:
        raise ValueError("Feedback residual additions must be non-negative.")
    unseen_batches = max(int(unseen_batches_after_final_iteration), 0)
    minimum_nonempty_unseen_budget = initial_residual_terms + rounds * add_residual_terms
    automatic_budget = initial_residual_terms + (rounds + unseen_batches) * add_residual_terms

    if raw_value is None:
        raw_value = "auto"
    if isinstance(raw_value, str) and raw_value.strip().lower() in {"auto", "automatic"}:
        resolved = automatic_budget
        mode = "auto"
    else:
        resolved = int(raw_value)
        mode = "explicit"

    if resolved < initial_residual_terms:
        raise ValueError(
            f"Resolved holdout residual budget Q={resolved} is smaller than the "
            f"initial training residual size {initial_residual_terms}."
        )

    return resolved, {
        "mode": mode,
        "resolved_holdout_residual_top_k": resolved,
        "initial_residual_terms": initial_residual_terms,
        "feedback_iterations": rounds,
        "add_residual_terms_per_iteration": add_residual_terms,
        "unseen_batches_after_final_iteration": unseen_batches,
        "minimum_budget_before_final_unseen_exhaustion": minimum_nonempty_unseen_budget,
        "automatic_budget_rule": (
            "Q = initial_residual_terms + "
            "(feedback_iterations + unseen_batches_after_final_iteration) * "
            "add_residual_terms_per_iteration"
        ),
        "final_round_expected_unseen_terms": max(resolved - minimum_nonempty_unseen_budget, 0),
    }


def plot_feedback_added_terms(rounds: list[dict[str, object]], images_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    set_paper_style(plt)
    x = np.asarray([int(row["round"]) for row in rounds], dtype=float)
    added = np.asarray([int(row["added_residual_terms"]) for row in rounds], dtype=float)
    residual_terms = np.asarray([int(row["train_residual_terms"]) for row in rounds], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    axes[0].bar(x, added, color=OKABE_ITO[0])
    axes[0].set_xlabel("feedback round", fontsize=LABEL_FS)
    axes[0].set_ylabel("added residual terms", fontsize=LABEL_FS)
    axes[0].set_title("holdout terms added", fontsize=TITLE_FS)
    axes[1].plot(x, residual_terms, marker="o", linewidth=LINE_WIDTH, color=OKABE_ITO[1])
    axes[1].set_xlabel("feedback round", fontsize=LABEL_FS)
    axes[1].set_ylabel("training residual terms", fontsize=LABEL_FS)
    axes[1].set_title("residual support growth", fontsize=TITLE_FS)
    axes[1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    for ax in axes:
        ax.tick_params(axis="both", labelsize=TICK_FS, length=TICK_LENGTH, width=TICK_WIDTH)
        ax.set_xticks(x)
    fig.subplots_adjust(top=0.84, left=0.11, right=0.98, bottom=0.19, wspace=0.34)
    fig.savefig(images_dir / "holdout_feedback_added_terms.pdf", format="pdf")
    plt.close(fig)


def plot_feedback_relative_residuals(rows: list[dict[str, object]], images_dir: Path, thresholds: Thresholds) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_paper_style(plt)
    x = np.asarray([int(row["feedback_round"]) for row in rows], dtype=float)
    unseen_values = [
        np.nan
        if int(row.get("unseen_residual_terms", 1)) == 0
        else float(row["unseen_relative_residual"])
        for row in rows
    ]
    series = [
        ("training", [float(row["training_final_relative_residual"]) for row in rows], OKABE_ITO[0], "o"),
        ("holdout", [float(row["holdout_relative_residual"]) for row in rows], OKABE_ITO[1], "s"),
        ("unseen", unseen_values, OKABE_ITO[2], "^"),
    ]
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    for label, values, color, marker in series:
        ax.semilogy(x, values, marker=marker, linewidth=LINE_WIDTH, color=color, label=label)
    ax.axhline(thresholds.holdout, color="0.35", linestyle="--", linewidth=0.8)
    ax.axhline(thresholds.unseen, color="0.55", linestyle=":", linewidth=0.8)
    ax.set_xlabel("feedback round", fontsize=LABEL_FS)
    ax.set_ylabel("relative residual", fontsize=LABEL_FS)
    ax.set_title(r"$q=20$ holdout-feedback residuals", fontsize=TITLE_FS)
    ax.set_xticks(x)
    ax.tick_params(axis="both", labelsize=TICK_FS, length=TICK_LENGTH, width=TICK_WIDTH)
    fig.legend(loc="upper center", ncol=3, frameon=False, fontsize=LEGEND_FS, bbox_to_anchor=(0.53, 1.02))
    fig.subplots_adjust(top=0.80, left=0.13, right=0.98, bottom=0.16)
    fig.savefig(images_dir / "holdout_feedback_relative_residuals.pdf", format="pdf")
    plt.close(fig)


def plot_feedback_seen_unseen(rows: list[dict[str, object]], images_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_paper_style(plt)
    x = np.asarray([int(row["feedback_round"]) for row in rows], dtype=float)
    width = 0.28
    seen = np.asarray([float(row["seen_residual"]) for row in rows])
    unseen = np.asarray(
        [
            np.nan if int(row.get("unseen_residual_terms", 1)) == 0 else float(row["unseen_residual"])
            for row in rows
        ]
    )
    seen_rel = np.asarray([float(row["seen_relative_residual"]) for row in rows])
    unseen_rel = np.asarray(
        [
            np.nan if int(row.get("unseen_residual_terms", 1)) == 0 else float(row["unseen_relative_residual"])
            for row in rows
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    axes[0].bar(x - width / 2.0, seen, width=width, color=OKABE_ITO[0], label="seen")
    axes[0].bar(x + width / 2.0, unseen, width=width, color=OKABE_ITO[1], label="unseen")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("feedback round", fontsize=LABEL_FS)
    axes[0].set_ylabel(r"$\|R(A)\|^2$", fontsize=LABEL_FS)
    axes[0].set_title("absolute residual", fontsize=TITLE_FS)

    axes[1].semilogy(x, seen_rel, marker="o", linewidth=LINE_WIDTH, color=OKABE_ITO[0], label="seen")
    axes[1].semilogy(x, unseen_rel, marker="s", linewidth=LINE_WIDTH, color=OKABE_ITO[1], label="unseen")
    axes[1].set_xlabel("feedback round", fontsize=LABEL_FS)
    axes[1].set_title("relative residual", fontsize=TITLE_FS)

    for ax in axes:
        ax.set_xticks(x)
        ax.tick_params(axis="both", labelsize=TICK_FS, length=TICK_LENGTH, width=TICK_WIDTH)
    fig.legend(loc="upper center", ncol=2, frameon=False, fontsize=LEGEND_FS, bbox_to_anchor=(0.53, 1.03))
    fig.subplots_adjust(top=0.78, left=0.10, right=0.98, bottom=0.18, wspace=0.32)
    fig.savefig(images_dir / "holdout_feedback_seen_unseen_residuals.pdf", format="pdf")
    plt.close(fig)


def plot_feedback_residual_spectrum(
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
        round_index = int(row["feedback_round"])
        values = np.asarray([float(item["residual_rms"]) for item in spectra[round_index]], dtype=float)
        ranks = np.arange(1, len(values) + 1)
        label = "baseline" if round_index == 0 else fr"round {round_index}"
        ax.loglog(ranks, values, linewidth=1.2, color=OKABE_ITO[idx % len(OKABE_ITO)], label=label)
    ax.set_xlabel("holdout residual rank", fontsize=LABEL_FS)
    ax.set_ylabel(r"RMS residual coefficient", fontsize=LABEL_FS)
    ax.set_title("holdout-feedback residual spectrum", fontsize=TITLE_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS, length=TICK_LENGTH, width=TICK_WIDTH)
    fig.legend(loc="upper center", ncol=min(len(rows), 4), frameon=False, fontsize=LEGEND_FS, bbox_to_anchor=(0.53, 1.03))
    fig.subplots_adjust(top=0.78, left=0.13, right=0.98, bottom=0.16)
    fig.savefig(images_dir / "holdout_feedback_residual_spectrum.pdf", format="pdf")
    plt.close(fig)


def write_feedback_spectrum(
    data_dir: Path,
    *,
    round_index: int,
    row: dict[str, object],
    spectrum: list[dict[str, object]],
) -> str:
    path = data_dir / f"holdout_feedback_spectrum_round_{round_index:02d}_agp_{row['agp_terms']}_residual_{row['holdout_residual_terms']}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "feedback_round": round_index,
                "agp_terms": row["agp_terms"],
                "holdout_residual_terms": row["holdout_residual_terms"],
                "spectrum": spectrum,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    return str(path)


def load_feedback_spectrum(data_dir: Path, *, round_index: int, residual_top_k: int) -> list[dict[str, object]]:
    matches = sorted(data_dir.glob(f"holdout_feedback_spectrum_round_{round_index:02d}_agp_*_residual_{residual_top_k}.json"))
    if not matches:
        raise FileNotFoundError(f"Missing feedback spectrum for round {round_index} in {data_dir}.")
    payload = load_json(matches[0])
    if not isinstance(payload, dict) or not isinstance(payload.get("spectrum"), list):
        raise TypeError(f"Unexpected feedback spectrum format in {matches[0]}.")
    return [row for row in payload["spectrum"] if isinstance(row, dict)]


def load_existing_feedback_state(
    *,
    output_dir: Path,
    data_dir: Path,
    residual_top_k: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[int, list[dict[str, object]]], int] | None:
    summary_path = data_dir / f"holdout_feedback_summary_residual_{residual_top_k}.json"
    if not summary_path.is_file():
        return None
    payload = load_json(summary_path)
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected feedback summary format in {summary_path}.")
    rows = [row for row in payload.get("rows", []) if isinstance(row, dict)]
    round_rows = [row for row in payload.get("rounds", []) if isinstance(row, dict)]
    if not rows:
        return None
    completed_round = max(int(row.get("feedback_round", 0)) for row in rows)
    spectra = {
        round_index: load_feedback_spectrum(data_dir, round_index=round_index, residual_top_k=residual_top_k)
        for round_index in range(completed_round + 1)
    }
    print(
        f"resume_feedback output={output_dir} completed_round={completed_round} "
        f"target_summary={summary_path.name}"
    )
    return rows, round_rows, spectra, completed_round


def write_feedback_summary(
    *,
    output_dir: Path,
    rows: list[dict[str, object]],
    spectra: dict[int, list[dict[str, object]]],
    round_rows: list[dict[str, object]],
    residual_top_k: int,
    thresholds: Thresholds,
    residual_budget: dict[str, object],
) -> None:
    images_dir = output_dir / "Images"
    data_dir = output_dir / "Models_Data"
    images_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    accepted = [
        row
        for row in rows
        if float(row["holdout_relative_residual"]) <= thresholds.holdout
        and float(row["unseen_relative_residual"]) <= thresholds.unseen
        and int(row.get("unseen_residual_terms", 0)) > 0
    ]
    if accepted:
        decision = {
            "status": "found_feedback_round",
            "round": int(accepted[0]["feedback_round"]),
            "conclusion": f"Feedback round {int(accepted[0]['feedback_round'])} passes holdout and unseen thresholds.",
            "thresholds": {
                "holdout_relative_residual_max": thresholds.holdout,
                "unseen_relative_residual_max": thresholds.unseen,
            },
        }
    else:
        decision = {
            "status": "not_found_in_feedback_run",
            "round": None,
            "conclusion": "No feedback round passes both holdout and unseen thresholds.",
            "thresholds": {
                "holdout_relative_residual_max": thresholds.holdout,
                "unseen_relative_residual_max": thresholds.unseen,
            },
        }
    payload = {
        "description": (
            "Holdout-feedback training: high-RMS unseen holdout residual strings are added to the "
            "training residual basis, while AGP support is kept fixed."
        ),
        "holdout_residual_terms": residual_top_k,
        "residual_budget": residual_budget,
        "decision": decision,
        "rounds": round_rows,
        "rows": rows,
    }
    with (data_dir / f"holdout_feedback_summary_residual_{residual_top_k}.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    plot_feedback_relative_residuals(rows, images_dir, thresholds)
    plot_feedback_seen_unseen(rows, images_dir)
    plot_feedback_residual_spectrum(rows, spectra, images_dir)
    plot_feedback_added_terms(round_rows, images_dir)
    if round_rows:
        final_round_dir = output_dir / str(round_rows[-1]["run_dir"])
        for filename in ("hcd_coefficient_support_map.pdf", "hcd_connection_summary.pdf"):
            source = final_round_dir / "Images" / filename
            if source.is_file():
                shutil.copy2(source, images_dir / filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train q=20 with holdout-residual feedback.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-agp-terms", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--add-residual-terms", type=int, default=None)
    parser.add_argument("--epochs-per-round", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--residual-top-k", type=int, default=None)
    parser.add_argument("--intermediate-top-k", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--min-rms", type=float, default=None)
    parser.add_argument("--unseen-residual-batches", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--holdout-threshold", type=float, default=None)
    parser.add_argument("--unseen-threshold", type=float, default=None)
    args = parser.parse_args()

    payload = load_json(args.config)
    if not isinstance(payload, dict):
        raise TypeError("config.json must contain a JSON object.")
    feedback = payload.get("holdout_feedback", {})
    feedback = feedback if isinstance(feedback, dict) else {}
    base_agp_terms = int(args.base_agp_terms if args.base_agp_terms is not None else feedback.get("base_agp_terms", 1024))
    rounds = int(args.rounds if args.rounds is not None else feedback.get("iterations", 1))
    add_residual_terms = int(
        args.add_residual_terms
        if args.add_residual_terms is not None
        else feedback.get("add_residual_terms_per_iteration", 1024)
    )
    epochs_per_round = int(
        args.epochs_per_round
        if args.epochs_per_round is not None
        else feedback.get("epochs_per_iteration", 1000)
    )
    residual_top_k_request = (
        args.residual_top_k
        if args.residual_top_k is not None
        else feedback.get("holdout_residual_top_k", "auto")
    )
    unseen_residual_batches = int(
        args.unseen_residual_batches
        if args.unseen_residual_batches is not None
        else feedback.get("unseen_residual_batches_after_final_iteration", 1)
    )
    lr = float(args.lr if args.lr is not None else feedback.get("lr", 1e-5))
    device_name = str(args.device if args.device is not None else feedback.get("device", "auto"))
    min_rms = float(args.min_rms if args.min_rms is not None else feedback.get("min_rms", 0.0))
    output_root_arg = args.output_root if args.output_root is not None else Path(str(feedback.get("output_root", "runs/holdout_feedback")))
    holdout_threshold = float(
        args.holdout_threshold if args.holdout_threshold is not None else feedback.get("holdout_threshold", 0.10)
    )
    unseen_threshold = float(
        args.unseen_threshold if args.unseen_threshold is not None else feedback.get("unseen_threshold", 1.0)
    )
    support = payload.get("support_sweep", {})
    intermediate_top_k = (
        int(args.intermediate_top_k)
        if args.intermediate_top_k is not None
        else int(support.get("intermediate_top_k", 2048))
        if isinstance(support, dict)
        else 2048
    )
    base_settings = settings_for_support(payload, base_agp_terms)
    feedback_settings = replace(
        base_settings,
        epochs=epochs_per_round,
        lr=lr,
        optimizer=str(args.optimizer) if args.optimizer is not None else base_settings.optimizer,
        intermediate_top_k=intermediate_top_k,
        device=device_name,
    )
    base_run = RUN_DIR / "runs" / f"agp_{base_agp_terms}"
    base_checkpoint = base_run / "Models_Data" / "training_checkpoint.pt"
    if not base_checkpoint.is_file():
        print(
            f"train_missing_baseline agp_terms={base_agp_terms} "
            f"epochs={base_settings.epochs} residual_terms={base_settings.residual_top_k}"
        )
        run_training(base_settings, base_run)
    agp_labels, residual_labels = load_checkpoint_labels(base_checkpoint)
    current_residual_labels = set(residual_labels)
    body_state = load_body_state_from_checkpoint(base_checkpoint)
    residual_top_k, residual_budget = resolve_holdout_residual_top_k(
        residual_top_k_request,
        initial_residual_terms=len(residual_labels),
        rounds=rounds,
        add_residual_terms=add_residual_terms,
        unseen_batches_after_final_iteration=unseen_residual_batches,
    )
    print(
        "resolved_feedback_residual_budget "
        f"mode={residual_budget['mode']} Q={residual_top_k} "
        f"initial={len(residual_labels)} rounds={rounds} "
        f"add={add_residual_terms} final_unseen_budget={residual_budget['final_round_expected_unseen_terms']}"
    )

    sweep_support = payload.get("support_sweep", {})
    support_sizes = (
        [int(value) for value in sweep_support.get("agp_terms", [base_agp_terms])]
        if isinstance(sweep_support, dict)
        else [base_agp_terms]
    )
    sweep_run_dirs = [RUN_DIR / "runs" / f"agp_{support_size}" for support_size in support_sizes]
    common_residual_labels, holdout_basis_agp_terms = build_common_holdout_residual_labels(
        run_dirs=sweep_run_dirs,
        config_payload=payload,
        residual_top_k=residual_top_k,
        intermediate_top_k=intermediate_top_k,
    )
    if len(common_residual_labels) < residual_top_k:
        print(
            "resolved_feedback_residual_budget_clipped "
            f"requested={residual_top_k} available={len(common_residual_labels)}"
        )
        residual_top_k = len(common_residual_labels)
        residual_budget = dict(residual_budget)
        residual_budget["resolved_holdout_residual_top_k"] = residual_top_k
        residual_budget["available_generated_residual_terms"] = len(common_residual_labels)
        residual_budget["final_round_expected_unseen_terms"] = max(
            residual_top_k - int(residual_budget["minimum_budget_before_final_unseen_exhaustion"]),
            0,
        )
    output_root = output_root_arg if output_root_arg.is_absolute() else RUN_DIR / output_root_arg
    output_dir = output_root / f"agp_{base_agp_terms}_residual_{residual_top_k}_add_{add_residual_terms}_rounds_{rounds}"
    data_dir = output_dir / "Models_Data"
    data_dir.mkdir(parents=True, exist_ok=True)
    existing_state = load_existing_feedback_state(
        output_dir=output_dir,
        data_dir=data_dir,
        residual_top_k=residual_top_k,
    )

    thresholds = Thresholds(
        plateau=1.0,
        holdout=holdout_threshold,
        unseen=unseen_threshold,
        top_stability=0.0,
        top_fraction=0.10,
    )
    if existing_state is None:
        rows: list[dict[str, object]] = []
        spectra: dict[int, list[dict[str, object]]] = {}
        round_rows: list[dict[str, object]] = []
        completed_round = 0

        print(f"evaluate_feedback_baseline agp_terms={base_agp_terms}")
        baseline_row, baseline_spectrum = evaluate_one_run(
            run_dir=base_run,
            config_payload=payload,
            residual_top_k=residual_top_k,
            intermediate_top_k=intermediate_top_k,
            device=select_device("cpu"),
            spectra_dir=data_dir,
            common_residual_labels=common_residual_labels,
            holdout_basis_mode="union_agp",
            holdout_basis_agp_terms=holdout_basis_agp_terms,
        )
        baseline_row["run_dir"] = str(base_run)
        baseline_row["feedback_round"] = 0
        rows.append(baseline_row)
        spectra[0] = baseline_spectrum
        baseline_row["spectrum_export"] = write_feedback_spectrum(
            data_dir,
            round_index=0,
            row=baseline_row,
            spectrum=baseline_spectrum,
        )
    else:
        rows, round_rows, spectra, completed_round = existing_state
        if completed_round >= rounds:
            write_feedback_summary(
                output_dir=output_dir,
                rows=rows,
                spectra=spectra,
                round_rows=round_rows,
                residual_top_k=residual_top_k,
                thresholds=thresholds,
                residual_budget=residual_budget,
            )
            print(f"feedback_already_complete rounds={rounds}")
            return
        last_checkpoint = output_dir / "runs" / f"round_{completed_round:02d}" / "Models_Data" / "training_checkpoint.pt"
        if completed_round > 0:
            agp_labels, residual_labels = load_checkpoint_labels(last_checkpoint)
            current_residual_labels = set(residual_labels)
            body_state = load_body_state_from_checkpoint(last_checkpoint)

    for round_index in range(completed_round + 1, rounds + 1):
        additions = select_residual_additions(
            spectra[round_index - 1],
            current_residual_labels,
            add_terms=add_residual_terms,
            min_rms=min_rms,
        )
        current_residual_labels.update(str(row["label"]) for row in additions)
        residual_labels = sort_pauli_labels(current_residual_labels)
        round_run = output_dir / "runs" / f"round_{round_index:02d}"
        print(
            f"train_feedback_round={round_index} agp_terms={len(agp_labels)} "
            f"residual_terms={len(residual_labels)} added={len(additions)} epochs={feedback_settings.epochs}"
        )
        body_state, final, metadata = train_feedback_round(
            run_dir=round_run,
            payload=payload,
            settings=feedback_settings,
            agp_labels=agp_labels,
            residual_labels=residual_labels,
            body_state=body_state,
            round_index=round_index,
            additions=additions,
        )
        row, spectrum = evaluate_one_run(
            run_dir=round_run,
            config_payload=payload,
            residual_top_k=residual_top_k,
            intermediate_top_k=intermediate_top_k,
            device=select_device("cpu"),
            spectra_dir=data_dir,
            common_residual_labels=common_residual_labels,
            holdout_basis_mode="union_agp",
            holdout_basis_agp_terms=holdout_basis_agp_terms,
        )
        row["run_dir"] = str(round_run.relative_to(output_dir))
        row["feedback_round"] = round_index
        rows.append(row)
        spectra[round_index] = spectrum
        row["spectrum_export"] = write_feedback_spectrum(
            data_dir,
            round_index=round_index,
            row=row,
            spectrum=spectrum,
        )
        round_rows.append(
            {
                "round": round_index,
                "run_dir": str(round_run.relative_to(output_dir)),
                "added_residual_terms": len(additions),
                "train_residual_terms": len(residual_labels),
                "training_final_relative_residual": float(final["relative_residual"]),
                "holdout_relative_residual": float(row["holdout_relative_residual"]),
                "unseen_relative_residual": float(row["unseen_relative_residual"]),
                "first_added_terms": additions[:32],
                "support_metadata": {
                    "first_commutator_nnz": metadata["first_commutator_nnz"],
                    "second_commutator_nnz": metadata["second_commutator_nnz"],
                    "final_intermediate_terms": metadata["final_intermediate_terms"],
                    "final_residual_terms": metadata["final_residual_terms"],
                },
            }
        )
        print(
            f"done_feedback_round={round_index} train_relative={final['relative_residual']:.6e} "
            f"holdout_relative={row['holdout_relative_residual']:.6e} "
            f"unseen_relative={row['unseen_relative_residual']:.6e}"
        )

    write_feedback_summary(
        output_dir=output_dir,
        rows=rows,
        spectra=spectra,
        round_rows=round_rows,
        residual_top_k=residual_top_k,
        thresholds=thresholds,
        residual_budget=residual_budget,
    )
    summary_path = output_dir / "Models_Data" / f"holdout_feedback_summary_residual_{residual_top_k}.json"
    try:
        summary_label = str(summary_path.relative_to(RUN_DIR))
    except ValueError:
        summary_label = str(summary_path)
    full_basis = Decimal(4) ** int(model_config_from_payload(payload).n_qubits)
    print(
        json.dumps(
            {
                "summary": summary_label,
                "base_agp_terms": base_agp_terms,
                "agp_fraction_of_full_basis": f"{Decimal(base_agp_terms) / full_basis:.12E}",
                "rounds": round_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    getcontext().prec = 80
    main()
