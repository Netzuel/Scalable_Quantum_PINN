"""Run and aggregate the fixed q15/q20/q25 size-intensive PINN study."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = (
    ROOT
    / "tests"
    / "sparse_agp_curriculum"
    / "transverse_field_diagonal_ising"
)
MANIFEST_PATH = SCENARIO_DIR / "size_intensive_pinn_study.json"


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def config_path_for_q(manifest: Mapping[str, object], q: int) -> Path:
    configs = manifest.get("configs", {})
    if not isinstance(configs, Mapping) or str(q) not in configs:
        raise ValueError(f"q={q} is not declared in {MANIFEST_PATH}")
    return (SCENARIO_DIR / str(configs[str(q)])).resolve()


def trained_run_path(config_path: Path, config: Mapping[str, object]) -> Path:
    validation = config.get("tensor_network_validation", {})
    if not isinstance(validation, Mapping):
        raise TypeError("tensor_network_validation must be a JSON object.")
    return config_path.parent / str(validation["trained_run"])


def tn_summary_path(config_path: Path, config: Mapping[str, object]) -> Path:
    validation = config["tensor_network_validation"]
    if not isinstance(validation, Mapping):
        raise TypeError("tensor_network_validation must be a JSON object.")
    return (
        trained_run_path(config_path, config)
        / str(validation.get("output_dir", "mpo_validation"))
        / "Models_Data"
        / "mps_physical_validation_summary.json"
    )


def exact_summary_path(config_path: Path, config: Mapping[str, object]) -> Path:
    validation = config.get("physical_validation", {})
    if not isinstance(validation, Mapping):
        raise TypeError("physical_validation must be a JSON object.")
    return (
        config_path.parent
        / str(validation["trained_run"])
        / "Models_Data"
        / "physical_validation_summary.json"
    )


def validate_config(config_path: Path) -> None:
    config = load_json(config_path)
    physical_parameters = config["physical"]["parameters"]  # type: ignore[index]
    q = int(physical_parameters["num_qubits"])
    duration = float(physical_parameters["T"])
    if duration != 1.0:
        raise ValueError(
            f"q={q} must use the canonical fixed physical duration T=1; got T={duration}."
        )
    feedback = config["holdout_feedback"]
    neural = config["neural"]
    validation = config["tensor_network_validation"]
    if not isinstance(feedback, Mapping) or not isinstance(neural, Mapping) or not isinstance(validation, Mapping):
        raise TypeError(f"Malformed size-intensive configuration: {config_path}")
    general = neural.get("general", {})
    if not isinstance(general, Mapping) or general.get("coefficient_architecture") != "independent_outputs":
        raise ValueError(f"q={q} does not use the conventional independent-output PINN.")
    if bool(feedback.get("allow_legacy_baseline_reuse", True)):
        raise ValueError(f"q={q} permits legacy or cross-system baseline reuse.")
    if list(validation.get("protocols", [])) != ["learned_sparse_agp"]:
        raise ValueError(f"q={q} must evaluate only learned_sparse_agp in this study.")
    scaling = config.get("size_intensive_scaling", {})
    training = config.get("training", {})
    loss = training.get("loss", {}) if isinstance(training, Mapping) else {}
    if (
        isinstance(scaling, Mapping)
        and str(scaling.get("version", "")).endswith("_action")
        and (
            not isinstance(loss, Mapping)
            or float(loss.get("variational_action", 0.0)) <= 0.0
        )
    ):
        raise ValueError(f"q={q} action candidate must use a positive variational-action weight.")
    learned_terms = int(feedback["base_agp_terms"])
    resolutions = validation.get("resolutions", [])
    if not isinstance(resolutions, Sequence) or not resolutions:
        raise ValueError(f"q={q} has no tensor-network resolutions.")
    if any(
        not isinstance(row, Mapping) or int(row.get("learned_terms", -1)) != learned_terms
        for row in resolutions
    ):
        raise ValueError(f"q={q} tensor-network validation does not deploy full K={learned_terms}.")


def run_command(command: Sequence[str], *, cwd: Path) -> None:
    print(f"run cwd={cwd} command={' '.join(command)}", flush=True)
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    subprocess.run(list(command), cwd=cwd, check=True, env=environment)


def execution_flags_for_q(
    manifest: Mapping[str, object],
    q: int,
    *,
    clean: bool,
    train: bool,
) -> dict[str, bool]:
    """Protect a declared retained anchor from destructive study actions."""

    retained_anchor = manifest.get("retained_anchor_q")
    is_anchor = retained_anchor is not None and int(retained_anchor) == int(q)
    return {
        "clean": bool(clean) and not is_anchor,
        "train": bool(train) and not is_anchor,
    }


def run_one(config_path: Path, *, clean: bool, train: bool, validate: bool) -> None:
    validate_config(config_path)
    config = load_json(config_path)
    q = int(config["physical"]["parameters"]["num_qubits"])  # type: ignore[index]
    python = sys.executable
    if clean:
        run_command(
            [python, str(ROOT / "scripts" / "agp_restart.py"), "--config", str(config_path)],
            cwd=config_path.parent,
        )
    if train:
        run_command(
            [python, "-u", str(ROOT / "scripts" / "agp_holdout_feedback.py"), "--config", str(config_path)],
            cwd=config_path.parent,
        )
    if validate:
        if q == 15:
            run_command(
                [
                    python,
                    "-u",
                    str(ROOT / "tests" / "sparse_agp_curriculum" / "scripts" / "agp_physical_validation.py"),
                    "--config",
                    str(config_path),
                ],
                cwd=config_path.parent,
            )
        run_command(
            [
                python,
                "-u",
                str(ROOT / "tests" / "sparse_agp_curriculum" / "scripts" / "agp_mps_validation.py"),
                "--config",
                str(config_path),
            ],
            cwd=config_path.parent,
        )


def result_row(config_path: Path) -> dict[str, object]:
    config = load_json(config_path)
    q = int(config["physical"]["parameters"]["num_qubits"])  # type: ignore[index]
    expected_terms = int(config["holdout_feedback"]["base_agp_terms"])  # type: ignore[index]
    if q <= 15:
        summary_path = exact_summary_path(config_path, config)
        if not summary_path.is_file():
            return {"q": q, "status": "not_tested", "summary": str(summary_path)}
        payload = load_json(summary_path)
        results = payload.get("results", {})
        row = results.get("learned_sparse_agp", {}) if isinstance(results, Mapping) else {}
        row = row if isinstance(row, Mapping) else {}
        evaluated_terms = int(row.get("learned_terms", -1) or -1)
        full_support = evaluated_terms == expected_terms
        fidelity = row.get("ground_state_fidelity") if full_support else None
        energy = row.get("final_energy") if full_support else None
        return {
            "q": q,
            "status": "ok" if fidelity is not None else "not_tested",
            "certification": "exact_statevector" if fidelity is not None else "not_tested",
            "expected_terms": expected_terms,
            "evaluated_terms": evaluated_terms,
            "full_support": full_support,
            "final_energy": energy,
            "ground_energy": row.get("ground_energy"),
            "energy_error": row.get("energy_error") if energy is not None else None,
            "ground_state_fidelity": fidelity,
            "summary": str(summary_path),
        }
    summary_path = tn_summary_path(config_path, config)
    if not summary_path.is_file():
        return {"q": q, "status": "not_tested", "summary": str(summary_path)}
    payload = load_json(summary_path)
    results = payload.get("results", {})
    row = results.get("learned_sparse_agp", {}) if isinstance(results, Mapping) else {}
    row = row if isinstance(row, Mapping) else {}
    diagnostics = row.get("mps_diagnostics", {})
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    completed = int(diagnostics.get("completed_steps", 0) or 0)
    planned = int(diagnostics.get("steps", 0) or 0)
    evaluated_terms = int(diagnostics.get("evaluated_cd_terms", -1) or -1)
    complete = diagnostics.get("status") == "ok" and planned > 0 and completed == planned
    full_support = evaluated_terms == expected_terms
    fidelity = row.get("ground_state_fidelity") if complete and full_support else None
    energy = row.get("final_energy") if complete and full_support else None
    ground_energy = row.get("ground_energy", payload.get("ground_energy"))
    return {
        "q": q,
        "status": "ok" if fidelity is not None else "not_tested",
        "certification": payload.get("certification", {}).get("status", "not_tested")
        if isinstance(payload.get("certification"), Mapping)
        else "not_tested",
        "expected_terms": expected_terms,
        "evaluated_terms": evaluated_terms,
        "full_support": full_support,
        "final_energy": energy,
        "ground_energy": ground_energy,
        "energy_error": row.get("energy_error") if energy is not None else None,
        "ground_state_fidelity": fidelity,
        "summary": str(summary_path),
    }


def assess_acceptance(
    rows: Sequence[Mapping[str, object]],
    *,
    minimum_fidelity: float,
    maximum_adjacent_drop: float,
) -> dict[str, object]:
    ordered = sorted(rows, key=lambda row: int(row["q"]))
    per_size: dict[str, object] = {}
    adjacent: list[dict[str, object]] = []
    for row in ordered:
        fidelity = row.get("ground_state_fidelity")
        passed = fidelity is not None and float(fidelity) >= minimum_fidelity
        per_size[str(row["q"])] = {
            "fidelity": fidelity,
            "minimum": minimum_fidelity,
            "status": "pass" if passed else ("not_tested" if fidelity is None else "fail"),
        }
    for left, right in zip(ordered, ordered[1:]):
        left_fidelity = left.get("ground_state_fidelity")
        right_fidelity = right.get("ground_state_fidelity")
        drop = (
            float(left_fidelity) - float(right_fidelity)
            if left_fidelity is not None and right_fidelity is not None
            else None
        )
        adjacent.append(
            {
                "from_q": int(left["q"]),
                "to_q": int(right["q"]),
                "fidelity_drop": drop,
                "maximum": maximum_adjacent_drop,
                "status": "not_tested" if drop is None else ("pass" if drop <= maximum_adjacent_drop else "fail"),
            }
        )
    statuses = [entry["status"] for entry in per_size.values()] + [entry["status"] for entry in adjacent]
    status = "pass" if statuses and all(value == "pass" for value in statuses) else (
        "not_tested" if any(value == "not_tested" for value in statuses) else "fail"
    )
    return {"status": status, "per_size": per_size, "adjacent": adjacent}


def plot_results(rows: Sequence[Mapping[str, object]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    complete = [row for row in sorted(rows, key=lambda item: int(item["q"])) if row.get("ground_state_fidelity") is not None]
    if not complete:
        return
    q_values = [int(row["q"]) for row in complete]
    fidelities = [float(row["ground_state_fidelity"]) for row in complete]
    energy_errors = [max(float(row["energy_error"]), 1e-15) for row in complete]
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "mathtext.rm": "stix",
            "mathtext.it": "stix:italic",
            "mathtext.bf": "stix:bold",
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.25))
    axes[0].plot(q_values, fidelities, color="#0072B2", marker="o", linewidth=1.8)
    axes[0].axhline(0.95, color="#D55E00", linestyle="--", linewidth=1.2, label="acceptance")
    axes[0].set_xlabel(r"number of qubits $q$")
    axes[0].set_ylabel(r"ground-state fidelity $F_0(T)$")
    axes[0].set_ylim(min(0.9, min(fidelities) - 0.01), 1.005)
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].plot(q_values, energy_errors, color="#009E73", marker="s", linewidth=1.8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"number of qubits $q$")
    axes[1].set_ylabel(r"energy error $|E(T)-E_0|$")
    for axis in axes:
        axis.set_xticks(q_values)
        axis.grid(True, alpha=0.22, linewidth=0.6)
        axis.tick_params(direction="in", width=0.8)
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.18, top=0.94, wspace=0.34)
    fig.savefig(output_dir / "size_intensive_pinn_scaling.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(output_dir / "size_intensive_pinn_scaling.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def aggregate(manifest: Mapping[str, object], qubits: Sequence[int]) -> dict[str, object]:
    rows = [result_row(config_path_for_q(manifest, q)) for q in qubits]
    acceptance_config = manifest.get("acceptance", {})
    if not isinstance(acceptance_config, Mapping):
        raise TypeError("Study acceptance must be a JSON object.")
    acceptance = assess_acceptance(
        rows,
        minimum_fidelity=float(acceptance_config["minimum_ground_fidelity"]),
        maximum_adjacent_drop=float(acceptance_config["maximum_adjacent_fidelity_drop"]),
    )
    acceptance["role"] = acceptance_config.get(
        "role",
        "promotion_gate",
    )
    output_dir = SCENARIO_DIR / str(manifest.get("results_dir", "size_intensive_pinn_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "methodology": manifest.get("methodology"),
        "benchmark_status": manifest.get("benchmark_status"),
        "promotion_decision": manifest.get("promotion_decision"),
        "rows": rows,
        "acceptance": acceptance,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    columns = (
        "q",
        "status",
        "certification",
        "expected_terms",
        "evaluated_terms",
        "full_support",
        "final_energy",
        "ground_energy",
        "energy_error",
        "ground_state_fidelity",
    )
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    plot_results(rows, output_dir)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, nargs="+", default=[15, 20, 25])
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_json(MANIFEST_PATH)
    for q in args.qubits:
        flags = execution_flags_for_q(
            manifest,
            q,
            clean=bool(args.clean),
            train=bool(args.train),
        )
        run_one(
            config_path_for_q(manifest, q),
            clean=flags["clean"],
            train=flags["train"],
            validate=bool(args.validate),
        )
    if not args.skip_aggregate:
        print(json.dumps(aggregate(manifest, args.qubits), indent=2), flush=True)


if __name__ == "__main__":
    main()
