from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import SparsePauliOperator, _commutator_pauli_labels_unchecked, load_pauli_hamiltonian_pair  # noqa: E402
from scripts.agp_plot_annotations import plot_physical_comparison_table  # noqa: E402


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


@dataclass(frozen=True)
class LearnedVariantSpec:
    name: str
    max_terms: int
    scale: float
    is_default: bool = False


class LazyPauliActionCache(Mapping[str, PauliAction]):
    """Bounded Pauli-action cache for larger statevector diagnostics.

    A dense action for one q20 Pauli string stores two arrays of length 2**20.
    Keeping thousands of them resident is unnecessary and can exhaust memory.
    This mapping builds actions on demand and evicts old entries while keeping
    the existing ``actions[label]`` call sites unchanged.
    """

    def __init__(self, labels: Sequence[str], *, max_items: int) -> None:
        if max_items < 1:
            raise ValueError("max_items must be positive for LazyPauliActionCache.")
        self._labels = set(labels)
        self._max_items = int(max_items)
        self._cache: OrderedDict[str, PauliAction] = OrderedDict()

    def __getitem__(self, label: str) -> PauliAction:
        if label not in self._labels:
            raise KeyError(label)
        action = self._cache.get(label)
        if action is not None:
            self._cache.move_to_end(label)
            return action
        action = build_pauli_action(label)
        self._cache[label] = action
        if len(self._cache) > self._max_items:
            self._cache.popitem(last=False)
        return action

    def __iter__(self) -> Iterator[str]:
        return iter(self._labels)

    def __len__(self) -> int:
        return len(self._labels)


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


def build_action_cache(labels: list[str], *, max_items: int | None = None) -> Mapping[str, PauliAction]:
    unique_labels = sorted(set(labels))
    if max_items is not None and int(max_items) < len(unique_labels):
        return LazyPauliActionCache(unique_labels, max_items=int(max_items))
    return {label: build_pauli_action(label) for label in unique_labels}


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


def parse_int_sweep(value: object, *, default: Sequence[int]) -> list[int]:
    if value is None:
        raw_values: Sequence[object] = list(default)
    elif isinstance(value, str):
        raw_values = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, int):
        raw_values = [value]
    elif isinstance(value, Sequence):
        raw_values = value
    else:
        raise TypeError(f"Cannot parse integer sweep from {value!r}.")

    out: list[int] = []
    seen: set[int] = set()
    for item in raw_values:
        parsed = int(item)
        if parsed <= 0:
            raise ValueError("learned term counts must be positive.")
        if parsed not in seen:
            out.append(parsed)
            seen.add(parsed)
    return out


def parse_float_sweep(value: object, *, default: Sequence[float]) -> list[float]:
    if value is None:
        raw_values: Sequence[object] = list(default)
    elif isinstance(value, str):
        raw_values = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (int, float)):
        raw_values = [value]
    elif isinstance(value, Sequence):
        raw_values = value
    else:
        raise TypeError(f"Cannot parse float sweep from {value!r}.")

    out: list[float] = []
    seen: set[str] = set()
    for item in raw_values:
        parsed = float(item)
        key = f"{parsed:.12g}"
        if key not in seen:
            out.append(parsed)
            seen.add(key)
    return out


def scale_token(scale: float) -> str:
    return f"{float(scale):g}".replace("-", "m").replace(".", "p")


def build_learned_variant_specs(
    validation: Mapping[str, object],
    *,
    max_terms_override: int | None,
    term_sweep_override: object | None,
    scale_sweep_override: object | None,
) -> list[LearnedVariantSpec]:
    default_terms = int(max_terms_override if max_terms_override is not None else validation.get("learned_top_terms", 256))
    term_source = term_sweep_override if term_sweep_override is not None else validation.get("learned_top_terms_sweep")
    scale_source = scale_sweep_override if scale_sweep_override is not None else validation.get("learned_scale_sweep")

    term_counts = parse_int_sweep(term_source, default=[default_terms])
    scales = parse_float_sweep(scale_source, default=[1.0])
    if default_terms not in term_counts:
        term_counts.append(default_terms)
    if not any(np.isclose(scale, 1.0, rtol=0.0, atol=1e-12) for scale in scales):
        scales.append(1.0)

    specs: dict[tuple[int, str], LearnedVariantSpec] = {}
    for terms in term_counts:
        for scale in scales:
            is_default = terms == default_terms and np.isclose(scale, 1.0, rtol=0.0, atol=1e-12)
            name = "learned_sparse_agp" if is_default else f"learned_sparse_agp_terms_{terms}_scale_{scale_token(scale)}"
            specs[(terms, f"{scale:.12g}")] = LearnedVariantSpec(
                name=name,
                max_terms=terms,
                scale=float(scale),
                is_default=bool(is_default),
            )
    return sorted(specs.values(), key=lambda spec: (spec.max_terms, 0 if spec.is_default else 1, spec.scale))


def select_best_learned_variant(
    variant_results: Mapping[str, Mapping[str, float]],
    *,
    metric: str = "energy_error",
) -> dict[str, object]:
    if not variant_results:
        raise ValueError("Cannot select a learned variant from an empty result set.")
    if metric == "ground_state_fidelity":
        name, row = max(
            variant_results.items(),
            key=lambda item: (float(item[1]["ground_state_fidelity"]), -float(item[1]["energy_error"])),
        )
    else:
        name, row = min(
            variant_results.items(),
            key=lambda item: (float(item[1][metric]), -float(item[1]["ground_state_fidelity"])),
        )
    return {
        "name": name,
        "selection_metric": metric,
        "energy_error": float(row["energy_error"]),
        "ground_state_fidelity": float(row["ground_state_fidelity"]),
        "excitation_probability": float(row["excitation_probability"]),
        "learned_terms": int(row["learned_terms"]),
        "learned_scale": float(row["learned_scale"]),
        "retained_rms_norm_fraction": float(row["retained_rms_norm_fraction"]),
    }


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
    lambda_grid = np.asarray(payload.get("lambda", []), dtype=np.float64).reshape(-1)
    d_lambda_dt_grid = np.asarray(payload.get("d_lambda_dt", []), dtype=np.float64).reshape(-1)
    rms = np.sqrt(np.mean(coefficients * coefficients, axis=0))
    ranking = np.argsort(-rms)
    selected_idx = ranking[: min(int(max_terms), len(ranking))]
    total_norm_sq = float(np.sum(rms * rms))
    retained_norm_sq = float(np.sum(rms[selected_idx] * rms[selected_idx]))
    return {
        "tau": tau_grid,
        "lambda": lambda_grid if lambda_grid.shape == tau_grid.shape else None,
        "d_lambda_dt": d_lambda_dt_grid if d_lambda_dt_grid.shape == tau_grid.shape else None,
        "labels": [labels[idx] for idx in selected_idx],
        "coefficients": coefficients[:, selected_idx],
        "selected_indices": [int(idx) for idx in selected_idx],
        "selected_rms": [float(rms[idx]) for idx in selected_idx],
        "selected_terms": int(len(selected_idx)),
        "available_terms": int(len(labels)),
        "retained_rms_norm_sq": retained_norm_sq,
        "total_rms_norm_sq": total_norm_sq,
        "retained_rms_norm_fraction": retained_norm_sq / total_norm_sq if total_norm_sq > 0.0 else 0.0,
        "schedule_source": payload.get("schedule", "exported_agp_coefficients"),
    }


def subset_learned_terms(learned: Mapping[str, object], max_terms: int) -> dict[str, object]:
    labels = list(learned["labels"])[:max_terms]  # type: ignore[index]
    coefficients = np.asarray(learned["coefficients"], dtype=np.float64)[:, :max_terms]  # type: ignore[index]
    selected_indices = list(learned["selected_indices"])[:max_terms]  # type: ignore[index]
    selected_rms = [float(value) for value in list(learned["selected_rms"])[:max_terms]]  # type: ignore[index]
    total_norm_sq = float(learned.get("total_rms_norm_sq", np.sum(np.asarray(selected_rms) ** 2)))
    retained_norm_sq = float(np.sum(np.asarray(selected_rms, dtype=np.float64) ** 2))
    return {
        "tau": learned["tau"],
        "lambda": learned.get("lambda"),
        "d_lambda_dt": learned.get("d_lambda_dt"),
        "labels": labels,
        "coefficients": coefficients,
        "selected_indices": selected_indices,
        "selected_rms": selected_rms,
        "selected_terms": int(len(labels)),
        "available_terms": int(learned["available_terms"]),  # type: ignore[arg-type]
        "retained_rms_norm_sq": retained_norm_sq,
        "total_rms_norm_sq": total_norm_sq,
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


def interpolate_scalar(tau_grid: np.ndarray, values: np.ndarray, tau: float) -> float:
    return float(interpolate_coefficients(tau_grid, values[:, None], tau)[0])


def learned_schedule(learned: Mapping[str, object], t: float, total_time: float) -> tuple[float, float]:
    tau_grid = np.asarray(learned["tau"], dtype=np.float64)
    lambda_grid = learned.get("lambda")
    d_lambda_dt_grid = learned.get("d_lambda_dt")
    if lambda_grid is None or d_lambda_dt_grid is None:
        return schedule_sin2(t, total_time)
    tau = float(t) / float(total_time)
    lam = interpolate_scalar(tau_grid, np.asarray(lambda_grid, dtype=np.float64), tau)
    dlam_dt = interpolate_scalar(tau_grid, np.asarray(d_lambda_dt_grid, dtype=np.float64), tau)
    return lam, dlam_dt


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
    learned_scale: float = 1.0,
    schedule: Mapping[str, object] | None = None,
) -> np.ndarray:
    n_qubits = h0.n_qubits
    dim = 1 << n_qubits
    psi = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
    dt = float(total_time) / int(steps)

    def h_apply(t: float, state: np.ndarray) -> np.ndarray:
        if schedule is None:
            lam, dlam_dt = schedule_sin2(t, total_time)
        else:
            lam, dlam_dt = learned_schedule(schedule, t, total_time)
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
                label: float(learned_scale) * float(coeff)
                for label, coeff in zip(learned["labels"], coeffs, strict=True)
                if abs(float(learned_scale) * float(coeff)) > 0.0
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


def add_quotients_vs_no_cd(results: dict[str, dict[str, float]], baseline: Mapping[str, float]) -> None:
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


def add_no_cd_quotients(results: dict[str, dict[str, float]]) -> None:
    add_quotients_vs_no_cd(results, results["no_cd"])


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
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442"]
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


def refresh_hcd_connection_summary(trained_run: Path, output_dir: Path) -> None:
    coefficient_path = trained_run / "Models_Data" / "final_agp_coefficients.pt"
    if not coefficient_path.is_file():
        print(f"skip_hcd_connection_summary_refresh missing_coefficients={coefficient_path}")
        return

    from projected_sparse_training_common import plot_connection_summary, rank_coefficients

    payload = torch.load(coefficient_path, map_location="cpu")
    coefficients = payload.get("counterdiabatic_coefficients")
    if coefficients is None:
        coefficients = payload["d_lambda_dt"] * payload["agp_coefficients"]
    labels = [str(label) for label in payload["pauli_labels"]]
    ranked = rank_coefficients(coefficients, labels)
    images_dir = output_dir / "Images"
    images_dir.mkdir(parents=True, exist_ok=True)
    plot_connection_summary(ranked, len(labels[0]), images_dir)


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
    output_dir = output_root / f"agp_{base_agp_terms}_residual_{residual_top_k}_add_{add_terms}_rounds_{rounds}"
    validation = config.get("physical_validation", {})
    validation = validation if isinstance(validation, dict) else {}
    if str(validation.get("trained_run_selection", "")).lower() == "best_holdout_residual":
        best = best_residual_run_from_summary(output_dir=output_dir, residual_top_k=residual_top_k)
        if best is not None:
            return best
    refined = preferred_refinement_run_from_summary(output_dir=output_dir, residual_top_k=residual_top_k)
    if refined is not None:
        return refined
    expected = output_dir / "rounds" / f"round_{rounds:02d}"
    if expected.is_dir():
        return expected
    matches = sorted(output_root.glob(f"agp_{base_agp_terms}_residual_*_add_*_rounds_{rounds}"))
    if matches:
        refined = preferred_refinement_run_from_summary(output_dir=matches[-1], residual_top_k=residual_top_k)
        if refined is not None:
            return refined
        return matches[-1] / "rounds" / f"round_{rounds:02d}"
    return expected


def metric_value(row: Mapping[str, object], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(parsed):
            return parsed
    return None


def best_residual_run_from_summary(*, output_dir: Path, residual_top_k: int) -> Path | None:
    summary_path = output_dir / "Models_Data" / f"holdout_feedback_summary_residual_{residual_top_k}.json"
    if not summary_path.is_file():
        return None
    payload = load_json(summary_path)
    candidates: list[tuple[float, Path]] = []

    rows = payload.get("rows", [])
    if isinstance(rows, list) and rows:
        baseline = rows[0]
        if isinstance(baseline, dict):
            score = metric_value(baseline, "holdout_relative_residual", "training_final_relative_residual")
            run_dir = baseline.get("run_dir")
            if score is not None and run_dir:
                path = Path(str(run_dir))
                if not path.is_absolute():
                    path = RUN_DIR / path
                candidates.append((score, path))

    for key, default_dir in (
        ("temporal_refinement", "temporal_refinement"),
        ("adaptive_temporal_refinement", "adaptive_temporal_refinement"),
    ):
        refinement = payload.get(key, {})
        if not isinstance(refinement, dict) or not bool(refinement.get("enabled", False)):
            continue
        score = metric_value(refinement, "holdout_relative_residual", "training_final_relative_residual")
        if score is None:
            continue
        run_dir = output_dir / str(refinement.get("run_dir", default_dir))
        candidates.append((score, run_dir))

    valid = [
        (score, path)
        for score, path in candidates
        if (path / "Models_Data" / "final_agp_coefficients.pt").is_file()
    ]
    if not valid:
        return None
    return min(valid, key=lambda item: item[0])[1]


def preferred_refinement_run_from_summary(*, output_dir: Path, residual_top_k: int) -> Path | None:
    for key, default_dir in (
        ("adaptive_temporal_refinement", "adaptive_temporal_refinement"),
        ("temporal_refinement", "temporal_refinement"),
    ):
        refined = refinement_run_from_summary(
            output_dir=output_dir,
            residual_top_k=residual_top_k,
            summary_key=key,
            default_dir=default_dir,
        )
        if refined is not None:
            return refined
    return None


def temporal_refinement_run_from_summary(*, output_dir: Path, residual_top_k: int) -> Path | None:
    return refinement_run_from_summary(
        output_dir=output_dir,
        residual_top_k=residual_top_k,
        summary_key="temporal_refinement",
        default_dir="temporal_refinement",
    )


def refinement_run_from_summary(
    *,
    output_dir: Path,
    residual_top_k: int,
    summary_key: str,
    default_dir: str,
) -> Path | None:
    summary_path = output_dir / "Models_Data" / f"holdout_feedback_summary_residual_{residual_top_k}.json"
    if not summary_path.is_file():
        return None
    payload = load_json(summary_path)
    refinement = payload.get(summary_key, {})
    if not isinstance(refinement, dict) or not bool(refinement.get("enabled", False)):
        return None
    run_dir = output_dir / str(refinement.get("run_dir", default_dir))
    return run_dir if run_dir.is_dir() else None


def residual_calibrated_run_from_summary(config: dict[str, object]) -> Path:
    calibration = config.get("residual_calibration", {})
    calibration = calibration if isinstance(calibration, dict) else {}
    suffix = str(calibration.get("output_suffix", "residual_calibrated"))
    source = final_run_from_summary(config)
    return source.with_name(f"{source.name}_{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare physical observables after sparse AGP training.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trained-run", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max-learned-terms", type=int, default=None)
    parser.add_argument(
        "--learned-term-sweep",
        type=str,
        default=None,
        help="Comma-separated learned AGP term counts to deploy in the physical benchmark.",
    )
    parser.add_argument(
        "--learned-scale-sweep",
        type=str,
        default=None,
        help="Comma-separated global scale factors for the learned counterdiabatic Hamiltonian.",
    )
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
    variant_specs = build_learned_variant_specs(
        validation,
        max_terms_override=args.max_learned_terms,
        term_sweep_override=args.learned_term_sweep,
        scale_sweep_override=args.learned_scale_sweep,
    )
    selection_metric = str(validation.get("selection_metric", "energy_error"))
    configured_run = validation.get("trained_run")
    if args.trained_run is not None:
        trained_run = args.trained_run
    elif configured_run:
        trained_run = Path(str(configured_run))
    elif bool(validation.get("prefer_residual_calibrated", False)):
        trained_run = residual_calibrated_run_from_summary(config)
    else:
        trained_run = final_run_from_summary(config)
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
    learned_full = learned_term_selection(coefficient_path, max(spec.max_terms for spec in variant_specs))

    h0_actions = build_action_cache(list(h0.terms))
    learned_cache_size = validation.get("learned_action_cache_size")
    learned_actions = build_action_cache(
        list(learned_full["labels"]),
        max_items=int(learned_cache_size) if learned_cache_size is not None else None,
    )

    results: dict[str, dict[str, float]] = {}
    for protocol in ("no_cd", "kipu_dqfm_l1"):
        print(f"evolve_protocol={protocol} steps={steps}")
        psi = evolve_state(
            protocol=protocol,
            h0=h0,
            final_energies=final_energies,
            h0_actions=h0_actions,
            learned=None,
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

    learned_variant_results: dict[str, dict[str, float]] = {}
    default_learned: dict[str, object] | None = None
    for spec in variant_specs:
        learned = subset_learned_terms(learned_full, spec.max_terms)
        print(
            f"evolve_protocol={spec.name} steps={steps} "
            f"learned_terms={learned['selected_terms']} learned_scale={spec.scale:g}"
        )
        psi = evolve_state(
            protocol="learned_sparse_agp",
            h0=h0,
            final_energies=final_energies,
            h0_actions=h0_actions,
            learned=learned,
            learned_actions=learned_actions,
            total_time=total_time,
            steps=steps,
            learned_scale=spec.scale,
            schedule=learned,
        )
        row = protocol_metrics(
            psi,
            final_energies=final_energies,
            ground_indices=ground_indices,
            target_z=target_z,
            target_zz=target_zz,
        )
        row["learned_terms"] = int(learned["selected_terms"])
        row["learned_scale"] = float(spec.scale)
        row["retained_rms_norm_fraction"] = float(learned["retained_rms_norm_fraction"])
        learned_variant_results[spec.name] = row
        if spec.is_default:
            default_learned = learned
            results["learned_sparse_agp"] = dict(row)

    if "learned_sparse_agp" not in results:
        default_spec = variant_specs[0]
        default_learned = subset_learned_terms(learned_full, default_spec.max_terms)
        results["learned_sparse_agp"] = dict(learned_variant_results[default_spec.name])

    best_learned_variant = select_best_learned_variant(learned_variant_results, metric=selection_metric)
    if best_learned_variant["name"] != "learned_sparse_agp":
        results["learned_sparse_agp_best"] = dict(learned_variant_results[str(best_learned_variant["name"])])
    add_no_cd_quotients(results)
    add_quotients_vs_no_cd(learned_variant_results, results["no_cd"])

    assert default_learned is not None

    payload = {
        "description": (
            "Statevector physical diagnostic comparing no-CD, the Kipu/DQFM-style first-order "
            "nested-commutator CD approximator, and learned sparse AGP deployment variants. This is intentionally "
            "not a scalable large-q path."
        ),
        "trained_run": str(trained_run.relative_to(RUN_DIR) if trained_run.is_relative_to(RUN_DIR) else trained_run),
        "n_qubits": n_qubits,
        "hilbert_dimension": int(1 << n_qubits),
        "schedule": "sinusoidal_sin2",
        "learned_protocol_schedule": (
            "exported_lambda_grid" if default_learned.get("lambda") is not None else "sinusoidal_sin2"
        ),
        "total_time": total_time,
        "steps": steps,
        "ground_energy": ground_energy,
        "ground_state_degeneracy": int(len(ground_indices)),
        "learned_agp_truncation": {
            "selected_terms": default_learned["selected_terms"],
            "available_terms": default_learned["available_terms"],
            "retained_rms_norm_fraction": default_learned["retained_rms_norm_fraction"],
            "action_cache": (
                {
                    "mode": "lazy_lru",
                    "max_items": int(learned_cache_size),
                }
                if learned_cache_size is not None
                else {"mode": "eager"}
            ),
        },
        "learned_variant_specs": [asdict(spec) for spec in variant_specs],
        "learned_variant_results": learned_variant_results,
        "best_learned_variant": best_learned_variant,
        "results": results,
    }
    with (data_dir / "physical_validation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    save_plot(results, images_dir)
    plot_physical_comparison_table(images_dir, payload)
    refresh_hcd_connection_summary(trained_run, output_dir)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    h1_global: SparsePauliOperator
    main()
