from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID_ROOT = Path("tests/diagonal_ising_grid")
DEFAULT_QUBITS = list(range(2, 21))
METHODS = ("no_cd", "kipu_dqfm_l1", "learned_sparse_agp")
METHOD_LABELS = {
    "no_cd": "no CD",
    "kipu_dqfm_l1": "Kipu/DQFM $l=1$",
    "learned_sparse_agp": "PINN sparse AGP",
}
PALETTE = {
    "no_cd": "#0072B2",
    "kipu_dqfm_l1": "#D55E00",
    "learned_sparse_agp": "#009E73",
    "exact": "#000000",
}


def parse_qubits(raw: str | None) -> list[int]:
    if raw is None:
        return list(DEFAULT_QUBITS)
    qubits: list[int] = []
    for chunk in raw.replace(",", " ").split():
        if "-" in chunk:
            start, end = chunk.split("-", maxsplit=1)
            qubits.extend(range(int(start), int(end) + 1))
        else:
            qubits.append(int(chunk))
    out = sorted(set(qubits))
    if any(q < 2 for q in out):
        raise ValueError("The grid benchmark expects q >= 2.")
    return out


def parse_phases(raw: str | None) -> list[str]:
    phases = ["prepare", "train", "validate", "aggregate", "plot"] if raw is None else [
        item.strip() for item in raw.split(",") if item.strip()
    ]
    allowed = {"prepare", "clean", "train", "validate", "aggregate", "plot"}
    invalid = sorted(set(phases) - allowed)
    if invalid:
        raise ValueError(f"Unknown phase(s): {invalid}. Allowed phases: {sorted(allowed)}")
    return phases


def non_identity_basis_size(q: int) -> int:
    return 4**int(q) - 1


def agp_terms_for_q(q: int, *, max_agp_terms: int) -> int:
    if int(q) <= 8:
        return 4 ** int(q)
    return min(int(max_agp_terms), 4 ** int(q))


def residual_seed_terms_for_q(q: int, *, max_residual_terms: int) -> int:
    if int(q) <= 8:
        return 4 ** int(q)
    return min(int(max_residual_terms), 4 ** int(q))


def learned_terms_for_k(k_terms: int, *, max_learned_terms: int) -> int:
    return min(int(max_learned_terms), int(k_terms))


def learned_term_sweep(k_terms: int, *, max_learned_terms: int) -> list[int]:
    default_terms = learned_terms_for_k(k_terms, max_learned_terms=max_learned_terms)
    if default_terms <= 1024:
        return [default_terms]
    return [1024, default_terms]


def q_dir(grid_root: Path, q: int) -> Path:
    return grid_root / f"q{int(q)}"


def config_path(grid_root: Path, q: int) -> Path:
    return q_dir(grid_root, q) / "config.json"


def grid_config(
    q: int,
    *,
    max_agp_terms: int,
    max_initial_residual_terms: int,
    holdout_residual_top_k: int,
    add_residual_terms: int,
    iterations: int,
    epochs: int,
    epochs_per_iteration: int,
    temporal_epochs: int,
    adaptive_temporal_epochs: int,
    evolution_steps: int,
    max_learned_terms: int,
    learned_action_cache_size: int,
) -> dict[str, object]:
    k_terms = agp_terms_for_q(q, max_agp_terms=max_agp_terms)
    q0_terms = residual_seed_terms_for_q(q, max_residual_terms=max_initial_residual_terms)
    intermediate_terms = max(k_terms, q0_terms)
    exact_full_basis = int(q) <= 8
    learned_terms = k_terms if exact_full_basis else learned_terms_for_k(k_terms, max_learned_terms=max_learned_terms)
    effective_holdout = 4 ** int(q) if exact_full_basis else int(holdout_residual_top_k)
    effective_additions = 0 if exact_full_basis else int(add_residual_terms)
    effective_iterations = 1 if exact_full_basis else int(iterations)
    target_active_terms = k_terms if exact_full_basis else min(max_learned_terms, k_terms)
    # Keep the baseline and feedback body architecture identical. Cross-activation
    # warm starts can evaluate SiLU-trained weights inside a PAU body and create
    # spurious large residuals at larger q.
    baseline_activation = "pau"
    return {
        "physical": {
            "parameters": {
                "system": "TransverseIsingDriverProblem",
                "num_qubits": int(q),
                "distance": "1_0",
                "hamiltonian_source": "Hamiltonians_to_use/pauli_decompositions/index.json",
                "t_initial": 0.0,
                "T": 1.0,
                "tau_range": [0.0, 1.0],
                "schedule": "trainable_bounded_envelope_sin2",
            }
        },
        "neural": {
            "model": "ProjectedSparseAGPPINN",
            "general": {
                "n_inputs": 1,
                "n_outputs": "agp_top_k",
                "n_hidden": 4,
                "n_neurons": 96,
                "activation": "pau",
                "layer_type": "quadratic",
            },
        },
        "default_pipeline": {
            "name": "diagonal_ising_fixed_k_support_swap_adaptive_temporal_refinement",
            "entrypoint": "scripts/agp_holdout_feedback.py",
            "agp_terms": k_terms,
            "description": (
                "Qubit-grid diagonal-Ising physical benchmark using the retained "
                "fixed-K support-swap curriculum, learned schedule, PAU network, "
                "and temporal/adaptive temporal refinement."
            ),
        },
        "support": {
            "allow_low_q_projected": True,
            "description": (
                "Allows q<=8 to use the projected training machinery with the "
                "complete 4**q Pauli basis for methodology parity with the "
                "large-q sparse curriculum."
            ),
        },
        "support_sweep": {
            "agp_terms": [k_terms],
            "output_root": "runs/baselines",
            "intermediate_top_k": intermediate_terms,
            "residual_top_k": q0_terms,
            "residual_projection": "largest_generated_commutator_terms",
            "agp_support_selection": {
                "strategy": "full_pauli_basis" if exact_full_basis else "nested_commutator_krylov_pool",
                "max_depth": 8,
                "max_frontier": 65536,
            },
            "adaptive": {"enabled": False, "stages": 1, "growth_terms_per_stage": 0},
        },
        "holdout_feedback": {
            "base_agp_terms": k_terms,
            "holdout_residual_top_k": effective_holdout,
            "iterations": effective_iterations,
            "add_residual_terms_per_iteration": effective_additions,
            "unseen_residual_batches_after_final_iteration": 0 if exact_full_basis else 1,
            "epochs_per_iteration": int(epochs_per_iteration),
            "lr": 1e-5,
            "device": "auto",
            "min_rms": 0.0,
            "holdout_threshold": 0.10,
            "unseen_threshold": 1.0,
            "baseline_root": "runs/baselines",
            "baseline_neural": {
                "n_hidden": 4,
                "n_neurons": 96,
                "activation": baseline_activation,
                "layer_type": "quadratic",
            },
            "support_swap": {
                "enabled": not exact_full_basis,
                "terms_per_iteration": 0 if exact_full_basis else min(256, max(k_terms // 64, 1)),
                "start_round": 2,
                "candidate_pool_multiplier": 16,
                "protect_top_fraction": 0.02,
                "new_gate_logit": 2.0,
            },
            "temporal_refinement": {
                "enabled": temporal_epochs > 0,
                "epochs": int(temporal_epochs),
                "num_points": 64,
                "lr": 3e-6,
                "optimizer": "AdamW",
                "run_dir": "temporal_refinement",
            },
            "adaptive_temporal_refinement": {
                "enabled": adaptive_temporal_epochs > 0,
                "epochs": int(adaptive_temporal_epochs),
                "dense_points": 257,
                "num_points": 64,
                "lr": 1.5e-6,
                "optimizer": "AdamW",
                "run_dir": "adaptive_temporal_refinement",
                "weight_power": 0.5,
                "min_weight": 0.25,
                "max_weight": 4.0,
                "difficulty": "residual_x_cd_norm",
            },
            "keep_round_images": False,
            "output_root": "runs/fixed_k_holdout_feedback_trainable_schedule_w96_l4_pau_support_swap_adaptive_temporal_refinement_v1",
        },
        "agp_calibration": {
            "enabled": True,
            "gamma_lr": 0.01,
            "gate_lr": 0.01,
            "initial_gamma": 1.0,
            "target_active_terms": target_active_terms,
            "gate_temperature": 1.0,
            "active_logit": 4.0,
            "inactive_logit": -8.0,
            "budget_weight": 1.0,
            "binary_weight": 0.001,
            "scale_l2_weight": 0.0001,
        },
        "training": {
            "device": "auto",
            "optimizer": "AdamW",
            "parameters": {
                "epochs": int(epochs),
                "num_points": 16,
                "lr": 1e-4,
                "random_seed": 11,
            },
            "loss": {
                "residual": 1.0,
                "agp_l2": 1e-8,
                "schedule_monotonic": 10.0,
                "schedule_correction_l2": 0.0001,
            },
            "export": {
                "path_images": "Images/",
                "path_data": "Models_Data/",
                "top_coefficients": 32,
                "plot_quantity": "d_lambda_dt_times_C_P",
                "format": "pdf",
            },
        },
        "schedule_optimization": {
            "enabled": True,
            "base": "sinusoidal_sin2",
            "correction_amplitude": 2.4,
            "hidden_width": 32,
            "hidden_layers": 2,
            "activation": "tanh",
            "lr": 0.001,
        },
        "physical_validation": {
            "entrypoint": "scripts/agp_physical_validation.py",
            "protocols": ["no_cd", "kipu_dqfm_l1", "learned_sparse_agp"],
            "schedule": "sinusoidal_sin2",
            "statevector_qubits": int(q),
            "evolution_steps": int(evolution_steps),
            "prefer_residual_calibrated": False,
            "learned_top_terms": learned_terms,
            "learned_top_terms_sweep": [k_terms]
            if exact_full_basis
            else learned_term_sweep(k_terms, max_learned_terms=max_learned_terms),
            "learned_scale_sweep": [1.0],
            "learned_action_cache_size": int(learned_action_cache_size),
            "selection_metric": "energy_error",
            "trained_run_selection": "best_holdout_residual",
            "description": (
                "Statevector diagnostic for the diagonal-Ising grid benchmark. "
                "The final Hamiltonian is diagonal, so exact final ground-space "
                "energy and Z/ZZ observables are available analytically."
            ),
        },
        "summary": {
            "top_k_overlap": 32,
            "path_images": "runs/support_sweep_summary/Images/",
            "path_data": "runs/support_sweep_summary/Models_Data/",
            "runs_dir": "runs/",
        },
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def run_command(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def prepare_one(q: int, args: argparse.Namespace) -> Path:
    grid_root = ROOT / args.grid_root
    destination = config_path(grid_root, q)
    config = grid_config(
        q,
        max_agp_terms=args.max_agp_terms,
        max_initial_residual_terms=args.initial_residual_terms,
        holdout_residual_top_k=args.holdout_residual_terms,
        add_residual_terms=args.add_residual_terms,
        iterations=args.iterations,
        epochs=args.epochs,
        epochs_per_iteration=args.epochs_per_iteration,
        temporal_epochs=args.temporal_epochs,
        adaptive_temporal_epochs=args.adaptive_temporal_epochs,
        evolution_steps=args.evolution_steps,
        max_learned_terms=args.max_learned_terms,
        learned_action_cache_size=args.learned_action_cache_size,
    )
    write_json(destination, config)
    run_command(
        [
            sys.executable,
            "scripts/build_driver_problem_hamiltonian.py",
            "--num-qubits",
            str(q),
            "--distance",
            "1.0",
            "--update-index",
        ]
    )
    return destination


def clean_one(config: Path) -> None:
    run_command([sys.executable, "scripts/agp_restart.py", "--config", str(config)])


def training_summary_exists(config: Path) -> bool:
    q_root = config.parent
    return any(q_root.glob("runs/**/holdout_feedback_summary_residual_*.json"))


def physical_summary_paths(q_root: Path) -> list[Path]:
    return sorted(q_root.glob("runs/**/physical_validation_summary.json"), key=lambda path: path.stat().st_mtime)


def physical_summary_exists(config: Path) -> bool:
    return bool(physical_summary_paths(config.parent))


def train_one(config: Path, *, force: bool) -> None:
    if training_summary_exists(config) and not force:
        print(f"skip_train config={config.relative_to(ROOT)} reason=holdout_feedback_summary_exists")
        return
    run_command([sys.executable, "-u", "scripts/agp_holdout_feedback.py", "--config", str(config)])


def validate_one(config: Path, *, force: bool) -> None:
    if physical_summary_exists(config) and not force:
        print(f"skip_validation config={config.relative_to(ROOT)} reason=physical_summary_exists")
        return
    run_command([sys.executable, "-u", "scripts/agp_physical_validation.py", "--config", str(config)])


def flatten_summary(q: int, summary_path: Path) -> list[dict[str, object]]:
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows: list[dict[str, object]] = []
    results = payload.get("results", {})
    if not isinstance(results, dict):
        return rows
    for method in METHODS:
        row = results.get(method)
        if not isinstance(row, dict):
            continue
        out = {
            "q": int(q),
            "method": method,
            "method_label": METHOD_LABELS[method].replace("$", ""),
            "trained_run": payload.get("trained_run"),
            "hilbert_dimension": payload.get("hilbert_dimension"),
            "ground_state_degeneracy": payload.get("ground_state_degeneracy"),
        }
        for key in (
            "final_energy",
            "ground_energy",
            "energy_error",
            "ground_state_fidelity",
            "excitation_probability",
            "z_rmse",
            "nearest_neighbor_zz_rmse",
            "energy_error_quotient_vs_no_cd",
            "excitation_probability_quotient_vs_no_cd",
            "z_rmse_quotient_vs_no_cd",
            "nearest_neighbor_zz_rmse_quotient_vs_no_cd",
            "learned_terms",
            "learned_scale",
            "retained_rms_norm_fraction",
        ):
            if key in row:
                out[key] = row[key]
        rows.append(out)
    return rows


def aggregate(grid_root: Path, qubits: Iterable[int]) -> list[dict[str, object]]:
    all_rows: list[dict[str, object]] = []
    missing: list[int] = []
    for q in qubits:
        summaries = physical_summary_paths(q_dir(grid_root, q))
        if not summaries:
            missing.append(int(q))
            continue
        all_rows.extend(flatten_summary(q, summaries[-1]))
    data_dir = grid_root / "Models_Data"
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / "grid_physical_validation_summary.json"
    write_json(json_path, {"rows": all_rows, "missing_qubits": missing})
    csv_path = data_dir / "grid_physical_validation_summary.csv"
    fieldnames = sorted({key for row in all_rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    write_markdown_table(data_dir / "grid_physical_validation_summary.md", all_rows, missing)
    print(f"grid_summary_json={json_path.relative_to(ROOT)}")
    print(f"grid_summary_csv={csv_path.relative_to(ROOT)}")
    return all_rows


def format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown_table(path: Path, rows: list[dict[str, object]], missing: list[int]) -> None:
    columns = [
        "q",
        "method",
        "final_energy",
        "ground_energy",
        "energy_error",
        "ground_state_fidelity",
        "z_rmse",
        "nearest_neighbor_zz_rmse",
    ]
    lines = [
        "# Qubit-Grid Physical Validation Summary",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in sorted(rows, key=lambda item: (int(item["q"]), str(item["method"]))):
        lines.append("| " + " | ".join(format_cell(row.get(column)) for column in columns) + " |")
    if missing:
        lines.extend(["", f"Missing physical-validation summaries for q={missing}."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_paper_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "mathtext.rm": "stix",
            "mathtext.it": "stix:italic",
            "mathtext.bf": "stix:bold",
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
        }
    )


def rows_by_method(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {method: [] for method in METHODS}
    for row in rows:
        method = str(row.get("method"))
        if method in grouped:
            grouped[method].append(row)
    for method_rows in grouped.values():
        method_rows.sort(key=lambda item: int(item["q"]))
    return grouped


def numeric_series(rows: list[dict[str, object]], key: str) -> tuple[list[int], list[float]]:
    x: list[int] = []
    y: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        x.append(int(row["q"]))
        y.append(float(value))
    return x, y


def save_all_formats(fig, path_without_suffix: Path) -> None:
    for suffix in ("pdf", "eps", "png"):
        fig.savefig(path_without_suffix.with_suffix(f".{suffix}"), format=suffix, dpi=300)


def plot_grid(grid_root: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        print("skip_plot reason=no_rows")
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_paper_style()
    images_dir = grid_root / "Images"
    images_dir.mkdir(parents=True, exist_ok=True)
    grouped = rows_by_method(rows)

    metric_specs = [
        ("energy_error", r"Energy error", True),
        ("excitation_probability", r"Ground infidelity $1-F_0$", True),
        ("z_rmse", r"$\langle Z_i\rangle$ RMSE", True),
        ("nearest_neighbor_zz_rmse", r"$\langle Z_i Z_{i+1}\rangle$ RMSE", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
    for ax, (key, ylabel, log_scale) in zip(axes.ravel(), metric_specs, strict=True):
        for method in METHODS:
            x, y = numeric_series(grouped[method], key)
            if not x:
                continue
            ax.plot(
                x,
                [max(value, 1e-15) for value in y],
                marker="o",
                linewidth=1.4,
                markersize=4.2,
                color=PALETTE[method],
                label=METHOD_LABELS[method],
            )
        if log_scale:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(labelsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Number of qubits $q$", fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.86, hspace=0.34, wspace=0.34)
    save_all_formats(fig, images_dir / "grid_physical_metrics_vs_qubits")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    exact_by_q: dict[int, float] = {}
    for row in rows:
        if "ground_energy" in row:
            exact_by_q[int(row["q"])] = float(row["ground_energy"])
    x_exact = sorted(exact_by_q)
    ax.plot(
        x_exact,
        [exact_by_q[q] for q in x_exact],
        color=PALETTE["exact"],
        linewidth=1.6,
        marker="s",
        markersize=4.0,
        label="exact ground energy",
    )
    for method in METHODS:
        x, y = numeric_series(grouped[method], "final_energy")
        if not x:
            continue
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.3,
            markersize=4.0,
            color=PALETTE[method],
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel(r"Number of qubits $q$", fontsize=10)
    ax.set_ylabel(r"Final energy expectation", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.96)
    save_all_formats(fig, images_dir / "grid_final_energy_vs_qubits")
    plt.close(fig)
    print(f"grid_figures_dir={images_dir.relative_to(ROOT)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and aggregate the diagonal-Ising AGP qubit grid benchmark.")
    parser.add_argument("--grid-root", type=Path, default=DEFAULT_GRID_ROOT)
    parser.add_argument("--qubits", default=None, help="Qubit list like '2-20' or '2,3,4'.")
    parser.add_argument("--phases", default=None, help="Comma-separated phases: prepare,clean,train,validate,aggregate,plot.")
    parser.add_argument("--force", action="store_true", help="Rerun training/validation even if completed artifacts exist.")
    parser.add_argument("--max-agp-terms", type=int, default=32768)
    parser.add_argument("--initial-residual-terms", type=int, default=4096)
    parser.add_argument("--holdout-residual-terms", type=int, default=65536)
    parser.add_argument("--add-residual-terms", type=int, default=3072)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--epochs-per-iteration", type=int, default=1000)
    parser.add_argument("--temporal-epochs", type=int, default=2500)
    parser.add_argument("--adaptive-temporal-epochs", type=int, default=1500)
    parser.add_argument("--evolution-steps", type=int, default=96)
    parser.add_argument("--max-learned-terms", type=int, default=2048)
    parser.add_argument("--learned-action-cache-size", type=int, default=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    qubits = parse_qubits(args.qubits)
    phases = parse_phases(args.phases)
    grid_root = ROOT / args.grid_root
    configs = [config_path(grid_root, q) for q in qubits]

    if "prepare" in phases:
        configs = [prepare_one(q, args) for q in qubits]
    if "clean" in phases:
        for config in configs:
            clean_one(config)
    if "train" in phases:
        for config in configs:
            train_one(config, force=bool(args.force))
    if "validate" in phases:
        for config in configs:
            validate_one(config, force=bool(args.force))
    rows: list[dict[str, object]] = []
    if "aggregate" in phases or "plot" in phases:
        rows = aggregate(grid_root, qubits)
    if "plot" in phases:
        plot_grid(grid_root, rows)


if __name__ == "__main__":
    main()
