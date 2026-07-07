from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from decimal import Decimal, getcontext
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from projected_sparse_training_common import (  # noqa: E402
    ProjectedRunSettings,
    ProjectedTrainingConfig,
    run_training,
)


RUN_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = RUN_DIR / "config.json"


def load_payload(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def model_config_from_payload(payload: dict[str, object]) -> ProjectedTrainingConfig:
    physical = payload.get("physical", {})
    parameters = physical.get("parameters", {}) if isinstance(physical, dict) else {}
    neural = payload.get("neural", {})
    general = neural.get("general", {}) if isinstance(neural, dict) else {}
    return ProjectedTrainingConfig(
        system=str(parameters.get("system", "Hidrogen")),
        n_qubits=int(parameters.get("num_qubits", 20)),
        distance=str(parameters.get("distance", "1_0")),
        hamiltonian_source=str(parameters.get("hamiltonian_source", "Hamiltonians_to_use/pauli_decompositions/index.json")),
        t_initial=float(parameters.get("t_initial", 0.0)),
        physical_time=float(parameters.get("T", 1.0)),
        hidden_layers=int(general.get("n_hidden", 3)),
        hidden_width=int(general.get("n_neurons", 56)),
        activation=str(general.get("activation", "silu")),
        layer_type=str(general.get("layer_type", "quadratic")),
    )


def settings_for_support(payload: dict[str, object], agp_terms: int) -> ProjectedRunSettings:
    training = payload.get("training", {})
    parameters = training.get("parameters", {}) if isinstance(training, dict) else {}
    loss = training.get("loss", {}) if isinstance(training, dict) else {}
    export = training.get("export", {}) if isinstance(training, dict) else {}
    support = payload.get("support_sweep", {})
    adaptive = support.get("adaptive", {}) if isinstance(support, dict) else {}
    return ProjectedRunSettings(
        model=model_config_from_payload(payload),
        epochs=int(parameters.get("epochs", 5000)),
        num_points=int(parameters.get("num_points", 16)),
        lr=float(parameters.get("lr", 1e-4)),
        optimizer=str(training.get("optimizer", "AdamW")) if isinstance(training, dict) else "AdamW",
        device=str(training.get("device", "auto")) if isinstance(training, dict) else "auto",
        seed=int(parameters.get("random_seed", 11)),
        agp_top_k=int(agp_terms),
        intermediate_top_k=int(support.get("intermediate_top_k", 2048)) if isinstance(support, dict) else 2048,
        residual_top_k=int(support.get("residual_top_k", 2048)) if isinstance(support, dict) else 2048,
        allow_low_q_projected=False,
        adaptive_enabled=bool(adaptive.get("enabled", False)) if isinstance(adaptive, dict) else False,
        adaptive_stages=int(adaptive.get("stages", 1)) if isinstance(adaptive, dict) else 1,
        adaptive_growth_per_stage=int(adaptive.get("growth_terms_per_stage", 0)) if isinstance(adaptive, dict) else 0,
        top_coefficients=int(export.get("top_coefficients", 16)),
        residual_weight=float(loss.get("residual", 1.0)),
        agp_l2_weight=float(loss.get("agp_l2", 1e-8)),
        residual_block_normalization=str(loss.get("residual_block_normalization", "none")),
        agp_smoothness_weight=float(loss.get("agp_smoothness", 0.0)),
        agp_curvature_weight=float(loss.get("agp_curvature", 0.0)),
        path_images=str(export.get("path_images", "Images/")),
        path_data=str(export.get("path_data", "Models_Data/")),
    )


def read_run_summary(run_dir: Path, support_size: int, *, overlap_k: int) -> dict[str, object]:
    data_dir = run_dir / "Models_Data"
    metadata = json.loads((data_dir / "support_metadata.json").read_text())
    history = json.loads((data_dir / "loss_history.json").read_text())
    importance = json.loads((data_dir / "coefficient_importance.json").read_text())
    final = history[-1]
    best = min(history, key=lambda row: row["total"])
    top_terms = importance.get("top_terms", [])[:overlap_k]
    least_terms = importance.get("least_terms", importance.get("least_nonidentity_terms", []))[:overlap_k]
    full_basis = Decimal(4) ** int(metadata["n_qubits"])
    fraction = Decimal(int(metadata["final_agp_terms"])) / full_basis
    return {
        "support_size_requested": support_size,
        "run_dir": str(run_dir.relative_to(RUN_DIR)),
        "n_qubits": int(metadata["n_qubits"]),
        "full_pauli_basis_size": str(full_basis),
        "final_agp_terms": int(metadata["final_agp_terms"]),
        "agp_fraction_of_full_basis": f"{fraction:.12E}",
        "final_intermediate_terms": int(metadata["final_intermediate_terms"]),
        "final_residual_terms": int(metadata["final_residual_terms"]),
        "hamiltonian_terms": int(metadata["hamiltonian_terms"]),
        "final_total": float(final["total"]),
        "final_residual": float(final["residual"]),
        "final_relative_residual": float(final.get("relative_residual", 0.0)),
        "best_epoch": int(best["epoch"]),
        "best_total": float(best["total"]),
        "best_relative_residual": float(best.get("relative_residual", 0.0)),
        "top_labels": [str(row["label"]) for row in top_terms],
        "least_labels": [str(row["label"]) for row in least_terms],
    }


def add_overlap_metrics(rows: list[dict[str, object]], *, overlap_k: int) -> None:
    if not rows:
        return
    reference = rows[-1]
    ref_top = set(reference["top_labels"])
    ref_least = set(reference["least_labels"])
    for row in rows:
        top = set(row["top_labels"])
        least = set(row["least_labels"])
        row["top_overlap_with_largest"] = len(top & ref_top) / max(min(overlap_k, len(top), len(ref_top)), 1)
        row["least_overlap_with_largest"] = len(least & ref_least) / max(min(overlap_k, len(least), len(ref_least)), 1)


def save_summary_plot(rows: list[dict[str, object]], images_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    images_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )
    x = [int(row["final_agp_terms"]) for row in rows]
    y = [float(row["final_relative_residual"]) for row in rows]
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.plot(x, y, marker="o", linewidth=1.6, color="#0072B2")
    ax.set_xlabel("AGP terms", fontsize=12)
    ax.set_ylabel("relative residual", fontsize=12)
    ax.set_title("q=20 support-size sweep", fontsize=13)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.tick_params(axis="both", labelsize=10, length=4.0, width=0.8)
    fig.subplots_adjust(top=0.86, left=0.15, right=0.98, bottom=0.17)
    fig.savefig(images_dir / "support_sweep_relative_residual.pdf", format="pdf")
    plt.close(fig)


def write_sweep_summary(payload: dict[str, object], rows: list[dict[str, object]]) -> Path:
    summary = payload.get("summary", {})
    images_dir = RUN_DIR / str(summary.get("path_images", "Images/"))
    data_dir = RUN_DIR / str(summary.get("path_data", "Models_Data/"))
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "description": "q=20 fixed-support AGP sweep. Adaptive growth is disabled; support size is the controlled variable.",
        "rows": rows,
    }
    output_path = data_dir / "sweep_summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    save_summary_plot(rows, images_dir)
    return output_path


def parse_support_sizes(raw: str | None, default: list[int]) -> list[int]:
    if raw is None:
        return default
    return [int(item) for item in raw.replace(",", " ").split()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run q=20 fixed-support AGP sweep.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--support-sizes", default=None, help="Comma or space separated AGP support sizes.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip completed support-size runs.")
    parser.add_argument("--summary-only", action="store_true", help="Only rebuild sweep summary from existing runs.")
    args = parser.parse_args()

    payload = load_payload(args.config)
    support = payload.get("support_sweep", {})
    default_sizes = [int(value) for value in support.get("agp_terms", [576, 768, 1024, 1536, 2048])] if isinstance(support, dict) else [576, 768, 1024, 1536, 2048]
    support_sizes = parse_support_sizes(args.support_sizes, default_sizes)
    summary = payload.get("summary", {})
    support_output_root = support.get("output_root") if isinstance(support, dict) else None
    runs_dir = RUN_DIR / str(support_output_root if support_output_root is not None else summary.get("runs_dir", "runs/"))
    overlap_k = int(summary.get("top_k_overlap", 32))
    runs_dir.mkdir(parents=True, exist_ok=True)

    for support_size in support_sizes:
        run_dir = runs_dir / f"agp_{support_size}"
        complete = (run_dir / "Models_Data" / "final_agp_coefficients.pt").is_file()
        if args.summary_only or (args.resume and complete):
            print(f"skip_agp_terms={support_size} complete={complete}")
            continue
        settings = settings_for_support(payload, support_size)
        if args.epochs is not None:
            settings = replace(settings, epochs=args.epochs)
        print(f"start_agp_terms={support_size} epochs={settings.epochs} optimizer={settings.optimizer}")
        final = run_training(settings, run_dir)
        print(
            f"done_agp_terms={support_size} final_loss={final['total']:.6e} "
            f"relative_residual={final.get('relative_residual', 0.0):.6e}"
        )

    rows = [read_run_summary(runs_dir / f"agp_{support_size}", support_size, overlap_k=overlap_k) for support_size in support_sizes]
    add_overlap_metrics(rows, overlap_k=overlap_k)
    summary_path = write_sweep_summary(payload, rows)
    print(f"sweep_summary={summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    getcontext().prec = 80
    main()
