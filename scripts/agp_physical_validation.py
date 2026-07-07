from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import SparsePauliOperator, _commutator_pauli_labels_unchecked, load_pauli_hamiltonian_pair  # noqa: E402


RUN_DIR = Path.cwd()
DEFAULT_CONFIG = Path("config.json")


def configure_run_dir(config_path: Path) -> None:
    global RUN_DIR
    RUN_DIR = config_path.resolve().parent


@dataclass(frozen=True)
class PauliAction:
    label: str
    flipped_indices: np.ndarray
    phase: np.ndarray


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return payload


def schedule_sin2(t: float, total_time: float) -> tuple[float, float]:
    tau = float(t) / float(total_time)
    lam = float(np.sin(0.5 * np.pi * tau) ** 2)
    dlam_dt = float(0.5 * np.pi / float(total_time) * np.sin(np.pi * tau))
    return lam, dlam_dt


def build_pauli_action(label: str) -> PauliAction:
    n_qubits = len(label)
    dim = 1 << n_qubits
    indices = np.arange(dim, dtype=np.int64)
    flip_mask = np.int64(0)
    for site, symbol in enumerate(label):
        if symbol in {"X", "Y"}:
            bit = n_qubits - 1 - site
            flip_mask ^= np.int64(1 << bit)
    flipped = indices ^ flip_mask
    phase = np.ones(dim, dtype=np.complex128)
    for site, symbol in enumerate(label):
        bit = n_qubits - 1 - site
        bit_values = (flipped >> bit) & 1
        z_eigen = 1.0 - 2.0 * bit_values.astype(np.float64)
        if symbol == "I":
            continue
        if symbol == "X":
            continue
        elif symbol == "Y":
            phase *= 1j * z_eigen
        elif symbol == "Z":
            phase *= z_eigen
        else:
            raise ValueError(f"Invalid Pauli symbol {symbol!r} in {label!r}.")
    return PauliAction(label=label, flipped_indices=flipped, phase=phase)


def build_action_cache(labels: list[str]) -> dict[str, PauliAction]:
    return {label: build_pauli_action(label) for label in sorted(set(labels))}


def apply_pauli_sum(
    psi: np.ndarray,
    terms: Mapping[str, complex],
    actions: Mapping[str, PauliAction],
) -> np.ndarray:
    out = np.zeros_like(psi, dtype=np.complex128)
    for label, coeff in terms.items():
        coeff = complex(coeff)
        if abs(coeff) == 0.0:
            continue
        action = actions[label]
        out += coeff * action.phase * psi[action.flipped_indices]
    return out


def diagonal_energies(terms: Mapping[str, complex], n_qubits: int) -> np.ndarray:
    dim = 1 << n_qubits
    indices = np.arange(dim, dtype=np.int64)
    energies = np.zeros(dim, dtype=np.float64)
    for label, coeff in terms.items():
        if any(symbol not in {"I", "Z"} for symbol in label):
            raise ValueError(f"Expected a diagonal I/Z label, got {label!r}.")
        value = np.ones(dim, dtype=np.float64)
        for site, symbol in enumerate(label):
            if symbol == "I":
                continue
            bit = n_qubits - 1 - site
            bit_values = (indices >> bit) & 1
            value *= 1.0 - 2.0 * bit_values.astype(np.float64)
        energies += float(np.real(coeff)) * value
    return energies


def z_expectations(probabilities: np.ndarray, n_qubits: int) -> np.ndarray:
    indices = np.arange(1 << n_qubits, dtype=np.int64)
    values = np.zeros(n_qubits, dtype=np.float64)
    for site in range(n_qubits):
        bit = n_qubits - 1 - site
        z_values = 1.0 - 2.0 * (((indices >> bit) & 1).astype(np.float64))
        values[site] = float(np.dot(probabilities, z_values))
    return values


def zz_expectations(probabilities: np.ndarray, n_qubits: int) -> np.ndarray:
    z_values = z_expectations_by_site(n_qubits)
    return np.asarray(
        [float(np.dot(probabilities, z_values[site] * z_values[site + 1])) for site in range(n_qubits - 1)],
        dtype=np.float64,
    )


def z_expectations_by_site(n_qubits: int) -> np.ndarray:
    indices = np.arange(1 << n_qubits, dtype=np.int64)
    values = np.empty((n_qubits, 1 << n_qubits), dtype=np.float64)
    for site in range(n_qubits):
        bit = n_qubits - 1 - site
        values[site] = 1.0 - 2.0 * (((indices >> bit) & 1).astype(np.float64))
    return values


def operator_inner(left: SparsePauliOperator, right: SparsePauliOperator) -> complex:
    labels = set(left.labels) | set(right.labels)
    return sum(np.conjugate(left.coefficient(label)) * right.coefficient(label) for label in labels)


def variational_l1_agp(h0: SparsePauliOperator, h1: SparsePauliOperator, lam: float) -> SparsePauliOperator:
    h_ad = h0.scale(1.0 - lam).add(h1, scale=lam)
    d_h = h1 - h0
    candidate = h_ad.commutator(d_h).scale(1.0j)
    if not candidate.terms:
        return SparsePauliOperator.zero(h0.n_qubits)
    direction = candidate.commutator(h_ad)
    denominator = direction.l2_norm_sq()
    if denominator <= 1e-24:
        return SparsePauliOperator.zero(h0.n_qubits)
    alpha = float(np.real(operator_inner(direction, d_h.scale(1.0j))) / denominator)
    return candidate.scale(alpha)


def commutator_terms(
    left_terms: Mapping[str, complex],
    right_terms: Mapping[str, complex],
    *,
    n_qubits: int,
) -> dict[str, complex]:
    out: defaultdict[str, complex] = defaultdict(complex)
    for left_label, left_coeff in left_terms.items():
        for right_label, right_coeff in right_terms.items():
            item = _commutator_pauli_labels_unchecked(left_label, right_label)
            if item is None:
                continue
            phase, out_label = item
            out[out_label] += complex(left_coeff) * complex(right_coeff) * phase
    return SparsePauliOperator(dict(out), n_qubits=n_qubits).terms


def learned_term_selection(coefficient_path: Path, max_terms: int) -> dict[str, object]:
    payload = torch.load(coefficient_path, map_location="cpu")
    labels = [str(label) for label in payload["pauli_labels"]]
    coefficients = np.asarray(payload["counterdiabatic_coefficients"], dtype=np.float64)
    tau_grid = np.asarray(payload["tau"], dtype=np.float64).reshape(-1)
    rms = np.sqrt(np.mean(coefficients * coefficients, axis=0))
    ranking = np.argsort(-rms)
    selected_idx = ranking[: min(int(max_terms), len(ranking))]
    total_norm_sq = float(np.sum(rms * rms))
    retained_norm_sq = float(np.sum(rms[selected_idx] * rms[selected_idx]))
    return {
        "tau": tau_grid,
        "labels": [labels[idx] for idx in selected_idx],
        "coefficients": coefficients[:, selected_idx],
        "selected_indices": [int(idx) for idx in selected_idx],
        "selected_rms": [float(rms[idx]) for idx in selected_idx],
        "selected_terms": int(len(selected_idx)),
        "available_terms": int(len(labels)),
        "retained_rms_norm_fraction": retained_norm_sq / total_norm_sq if total_norm_sq > 0.0 else 0.0,
    }


def interpolate_coefficients(tau_grid: np.ndarray, coefficients: np.ndarray, tau: float) -> np.ndarray:
    tau = float(np.clip(tau, tau_grid[0], tau_grid[-1]))
    if tau <= tau_grid[0]:
        return coefficients[0]
    if tau >= tau_grid[-1]:
        return coefficients[-1]
    upper = int(np.searchsorted(tau_grid, tau, side="right"))
    lower = upper - 1
    width = tau_grid[upper] - tau_grid[lower]
    if width <= 0.0:
        return coefficients[lower]
    weight = (tau - tau_grid[lower]) / width
    return (1.0 - weight) * coefficients[lower] + weight * coefficients[upper]


def evolve_state(
    *,
    protocol: str,
    h0: SparsePauliOperator,
    final_energies: np.ndarray,
    h0_actions: Mapping[str, PauliAction],
    learned: dict[str, object] | None,
    learned_actions: Mapping[str, PauliAction],
    total_time: float,
    steps: int,
) -> np.ndarray:
    n_qubits = h0.n_qubits
    dim = 1 << n_qubits
    psi = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
    dt = float(total_time) / int(steps)

    def h_apply(t: float, state: np.ndarray) -> np.ndarray:
        lam, dlam_dt = schedule_sin2(t, total_time)
        out = (1.0 - lam) * apply_pauli_sum(state, h0.terms, h0_actions)
        out += lam * final_energies * state
        if protocol in {"nested_l1", "kipu_dqfm_l1"}:
            agp = variational_l1_agp(h0, h1_global, lam)
            if agp.terms and abs(dlam_dt) > 0.0:
                l1_actions = build_action_cache(list(agp.terms))
                out += dlam_dt * apply_pauli_sum(state, agp.terms, l1_actions)
        elif protocol == "learned_sparse_agp":
            if learned is None:
                raise ValueError("learned payload is required for learned_sparse_agp.")
            tau = t / total_time
            coeffs = interpolate_coefficients(
                learned["tau"],
                learned["coefficients"],
                tau,
            )
            cd_terms = {
                label: float(coeff)
                for label, coeff in zip(learned["labels"], coeffs, strict=True)
                if abs(float(coeff)) > 0.0
            }
            out += apply_pauli_sum(state, cd_terms, learned_actions)
        return out

    for step in range(int(steps)):
        t = step * dt
        k1 = -1j * h_apply(t, psi)
        k2 = -1j * h_apply(t + 0.5 * dt, psi + 0.5 * dt * k1)
        k3 = -1j * h_apply(t + 0.5 * dt, psi + 0.5 * dt * k2)
        k4 = -1j * h_apply(t + dt, psi + dt * k3)
        psi = psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        psi = psi / np.linalg.norm(psi)
    return psi


def protocol_metrics(
    psi: np.ndarray,
    *,
    final_energies: np.ndarray,
    ground_indices: np.ndarray,
    target_z: np.ndarray,
    target_zz: np.ndarray,
) -> dict[str, float]:
    probabilities = np.abs(psi) ** 2
    probabilities = probabilities / float(np.sum(probabilities))
    n_qubits = int(np.log2(probabilities.size))
    energy = float(np.dot(probabilities, final_energies))
    ground_energy = float(final_energies[ground_indices[0]])
    ground_fidelity = float(np.sum(probabilities[ground_indices]))
    z_values = z_expectations(probabilities, n_qubits)
    zz_values = zz_expectations(probabilities, n_qubits)
    return {
        "final_energy": energy,
        "ground_energy": ground_energy,
        "energy_error": energy - ground_energy,
        "ground_state_fidelity": ground_fidelity,
        "excitation_probability": 1.0 - ground_fidelity,
        "z_rmse": float(np.sqrt(np.mean((z_values - target_z) ** 2))),
        "nearest_neighbor_zz_rmse": float(np.sqrt(np.mean((zz_values - target_zz) ** 2))),
    }


def add_no_cd_quotients(results: dict[str, dict[str, float]]) -> None:
    baseline = results["no_cd"]
    for name, row in results.items():
        row["energy_error_quotient_vs_no_cd"] = row["energy_error"] / max(baseline["energy_error"], 1e-15)
        row["excitation_probability_quotient_vs_no_cd"] = row["excitation_probability"] / max(
            baseline["excitation_probability"],
            1e-15,
        )
        row["z_rmse_quotient_vs_no_cd"] = row["z_rmse"] / max(baseline["z_rmse"], 1e-15)
        row["nearest_neighbor_zz_rmse_quotient_vs_no_cd"] = row["nearest_neighbor_zz_rmse"] / max(
            baseline["nearest_neighbor_zz_rmse"],
            1e-15,
        )


def save_plot(results: dict[str, dict[str, float]], images_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images_dir.mkdir(parents=True, exist_ok=True)
    protocols = list(results)
    metrics = [
        ("energy_error", "energy error"),
        ("excitation_probability", "excitation prob."),
        ("z_rmse", r"$\langle Z_i\rangle$ RMSE"),
        ("nearest_neighbor_zz_rmse", r"$\langle Z_iZ_{i+1}\rangle$ RMSE"),
    ]
    colors = ["#0072B2", "#D55E00", "#009E73"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(10.8, 3.2))
    for ax, (key, title) in zip(axes, metrics, strict=True):
        values = [max(float(results[name][key]), 1e-15) for name in protocols]
        ax.bar(np.arange(len(protocols)), values, color=colors[: len(protocols)])
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(len(protocols)))
        ax.set_xticklabels(protocols, rotation=30, ha="right", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.31, top=0.84, wspace=0.42)
    fig.savefig(images_dir / "physical_validation_observables.pdf", format="pdf")
    plt.close(fig)


def final_run_from_summary(config: dict[str, object]) -> Path:
    feedback = config.get("holdout_feedback", {})
    feedback = feedback if isinstance(feedback, dict) else {}
    base_agp_terms = int(feedback.get("base_agp_terms", 16384))
    rounds = int(feedback.get("iterations", 10))
    add_terms = int(feedback.get("add_residual_terms_per_iteration", 1024))
    initial_residual = int(config.get("support_sweep", {}).get("residual_top_k", 2048))  # type: ignore[union-attr]
    unseen_batches = int(feedback.get("unseen_residual_batches_after_final_iteration", 1))
    residual_request = feedback.get("holdout_residual_top_k", "auto")
    if str(residual_request).lower() == "auto":
        residual_top_k = initial_residual + (rounds + unseen_batches) * add_terms
    else:
        residual_top_k = int(residual_request)
    output_root = RUN_DIR / str(feedback.get("output_root", "runs/fixed_k_holdout_feedback_v1"))
    expected = (
        output_root
        / f"agp_{base_agp_terms}_residual_{residual_top_k}_add_{add_terms}_rounds_{rounds}"
        / "rounds"
        / f"round_{rounds:02d}"
    )
    if expected.is_dir():
        return expected
    matches = sorted(output_root.glob(f"agp_{base_agp_terms}_residual_*_add_*_rounds_{rounds}/rounds/round_{rounds:02d}"))
    if matches:
        return matches[-1]
    return expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare physical observables after sparse AGP training.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trained-run", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max-learned-terms", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    configure_run_dir(config_path)
    config = load_json(config_path)
    physical = config.get("physical", {})
    parameters = physical.get("parameters", {}) if isinstance(physical, dict) else {}
    validation = config.get("physical_validation", {})
    validation = validation if isinstance(validation, dict) else {}
    n_qubits = int(parameters.get("num_qubits", 15))
    total_time = float(parameters.get("T", 1.0))
    steps = int(args.steps if args.steps is not None else validation.get("evolution_steps", 96))
    max_learned_terms = int(
        args.max_learned_terms if args.max_learned_terms is not None else validation.get("learned_top_terms", 256)
    )
    trained_run = args.trained_run or final_run_from_summary(config)
    if not trained_run.is_absolute():
        trained_run = RUN_DIR / trained_run
    output_dir = args.output_dir or trained_run
    if not output_dir.is_absolute():
        output_dir = RUN_DIR / output_dir
    images_dir = output_dir / "Images"
    data_dir = output_dir / "Models_Data"
    data_dir.mkdir(parents=True, exist_ok=True)

    hamiltonian_source = Path(str(parameters.get("hamiltonian_source", "Hamiltonians_to_use/pauli_decompositions/index.json")))
    if not hamiltonian_source.is_absolute():
        hamiltonian_source = ROOT / hamiltonian_source
    h0, h1 = load_pauli_hamiltonian_pair(
        hamiltonian_source,
        system=str(parameters.get("system", "TransverseIsingDriverProblem")),
        n_qubits=n_qubits,
        distance=str(parameters.get("distance", "1_0")),
    )
    global h1_global
    h1_global = h1
    final_energies = diagonal_energies(h1.terms, n_qubits)
    ground_energy = float(np.min(final_energies))
    ground_indices = np.where(np.isclose(final_energies, ground_energy, rtol=0.0, atol=1e-10))[0]
    ground_probs = np.zeros_like(final_energies)
    ground_probs[ground_indices] = 1.0 / len(ground_indices)
    target_z = z_expectations(ground_probs, n_qubits)
    target_zz = zz_expectations(ground_probs, n_qubits)

    coefficient_path = trained_run / "Models_Data" / "final_agp_coefficients.pt"
    if not coefficient_path.is_file():
        raise FileNotFoundError(f"Missing trained AGP coefficients: {coefficient_path}")
    learned = learned_term_selection(coefficient_path, max_learned_terms)

    h0_actions = build_action_cache(list(h0.terms))
    learned_actions = build_action_cache(list(learned["labels"]))

    results: dict[str, dict[str, float]] = {}
    for protocol in ("no_cd", "kipu_dqfm_l1", "learned_sparse_agp"):
        print(f"evolve_protocol={protocol} steps={steps}")
        psi = evolve_state(
            protocol=protocol,
            h0=h0,
            final_energies=final_energies,
            h0_actions=h0_actions,
            learned=learned,
            learned_actions=learned_actions,
            total_time=total_time,
            steps=steps,
        )
        results[protocol] = protocol_metrics(
            psi,
            final_energies=final_energies,
            ground_indices=ground_indices,
            target_z=target_z,
            target_zz=target_zz,
        )
    add_no_cd_quotients(results)

    payload = {
        "description": (
            "Statevector physical diagnostic comparing no-CD, the Kipu/DQFM-style first-order "
            "nested-commutator CD approximator, and the learned sparse AGP. This is intentionally "
            "not a scalable large-q path."
        ),
        "trained_run": str(trained_run.relative_to(RUN_DIR) if trained_run.is_relative_to(RUN_DIR) else trained_run),
        "n_qubits": n_qubits,
        "hilbert_dimension": int(1 << n_qubits),
        "schedule": "sinusoidal_sin2",
        "total_time": total_time,
        "steps": steps,
        "ground_energy": ground_energy,
        "ground_state_degeneracy": int(len(ground_indices)),
        "learned_agp_truncation": {
            "selected_terms": learned["selected_terms"],
            "available_terms": learned["available_terms"],
            "retained_rms_norm_fraction": learned["retained_rms_norm_fraction"],
        },
        "results": results,
    }
    with (data_dir / "physical_validation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    save_plot(results, images_dir)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    h1_global: SparsePauliOperator
    main()
