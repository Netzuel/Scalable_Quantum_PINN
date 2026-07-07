from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict, deque
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from agp_coupled_curriculum import select_agp_additions_from_residual  # noqa: E402
from agp_holdout_study import load_body_weights, load_json, norm_sq_subset, rms_per_label, scalar  # noqa: E402
from agp_oracle_tools import score_candidates_with_omp  # noqa: E402
from projected_sparse_training_common import (  # noqa: E402
    LABEL_FS,
    LEGEND_FS,
    LINE_WIDTH,
    OKABE_ITO,
    TICK_FS,
    TICK_LENGTH,
    TICK_WIDTH,
    TITLE_FS,
    build_projected_support,
    commutator_generated_scores,
    hamiltonian_importance,
    make_projected_model,
    merge_scores,
    operator_importance,
    pauli_weight,
    ranked_label_scores,
    select_device,
    set_paper_style,
)
from agp_baseline_train import model_config_from_payload  # noqa: E402
from utils import load_pauli_hamiltonian_pair, sort_pauli_labels  # noqa: E402


DEFAULT_CONFIG = ROOT / "tests" / "q20" / "sweep_test" / "config.json"
RUN_DIR = DEFAULT_CONFIG.parent
CERTIFICATION_FILENAME = "sparse_agp_support_certification.json"


def configure_run_dir(config_path: Path) -> None:
    global RUN_DIR
    RUN_DIR = config_path.resolve().parent


def gate(value: float | None, threshold: float, *, lower_is_better: bool = True) -> dict[str, object]:
    if value is None:
        return {"status": "not tested", "value": None, "threshold": float(threshold)}
    passed = value <= threshold if lower_is_better else value >= threshold
    return {"status": "pass" if passed else "fail", "value": float(value), "threshold": float(threshold)}


def classify_claim_level(checks: dict[str, dict[str, object]]) -> str:
    statuses = [str(row.get("status", "not tested")) for row in checks.values()]
    if any(status == "fail" for status in statuses):
        return "projected_sparse_agp_experiment"
    if statuses and all(status == "pass" for status in statuses):
        return "certified_sparse_agp_for_this_path_and_tolerance"

    core_keys = ("training_residual", "holdout_residual", "unseen_residual", "multi_holdout")
    if all(str(checks.get(key, {}).get("status")) == "pass" for key in core_keys):
        return "candidate_robust_sparse_agp"
    return "projected_sparse_agp_experiment"


def select_top_labels(
    scored_labels: Iterable[tuple[str, float]],
    *,
    count: int,
    excluded: set[str] | None = None,
) -> list[str]:
    excluded = excluded or set()
    selected: list[str] = []
    seen: set[str] = set()
    for label, _ in scored_labels:
        label = str(label)
        if label in excluded or label in seen:
            continue
        selected.append(label)
        seen.add(label)
        if len(selected) >= int(count):
            break
    return sort_pauli_labels(selected)


def select_order_stratified_labels(
    scored_labels: Iterable[tuple[str, float]],
    *,
    count: int,
    excluded: set[str] | None = None,
) -> list[str]:
    """Select high-score residual labels while forcing Pauli-order diversity."""

    excluded = excluded or set()
    groups: dict[int, deque[tuple[str, float]]] = defaultdict(deque)
    seen: set[str] = set()
    for label, score in scored_labels:
        label = str(label)
        if label in excluded or label in seen:
            continue
        groups[pauli_weight(label)].append((label, float(score)))
        seen.add(label)

    ordered_weights = sorted(
        groups,
        key=lambda weight: (groups[weight][0][1], -weight),
        reverse=True,
    )
    selected: list[str] = []
    selected_set: set[str] = set()
    while len(selected) < int(count) and any(groups[weight] for weight in ordered_weights):
        for weight in ordered_weights:
            if len(selected) >= int(count):
                break
            if not groups[weight]:
                continue
            label, _ = groups[weight].popleft()
            selected.append(label)
            selected_set.add(label)

    if len(selected) < int(count):
        for label, _ in scored_labels:
            label = str(label)
            if label in excluded or label in selected_set:
                continue
            selected.append(label)
            selected_set.add(label)
            if len(selected) >= int(count):
                break
    return sort_pauli_labels(selected)


def select_seeded_labels(
    scored_labels: list[tuple[str, float]],
    *,
    count: int,
    excluded: set[str] | None = None,
    seed: int,
    pool_multiplier: int,
) -> list[str]:
    excluded = excluded or set()
    pool: list[str] = []
    seen: set[str] = set()
    pool_size = max(int(count) * max(int(pool_multiplier), 1), int(count))
    for label, _ in scored_labels:
        label = str(label)
        if label in excluded or label in seen:
            continue
        pool.append(label)
        seen.add(label)
        if len(pool) >= pool_size:
            break
    if len(pool) <= int(count):
        return sort_pauli_labels(pool)
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(pool), size=int(count), replace=False)
    return sort_pauli_labels([pool[int(idx)] for idx in indices])


def fixed_feedback_output_roots(config_path: Path = DEFAULT_CONFIG) -> list[Path]:
    roots: list[Path] = []
    if config_path.is_file():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        feedback = payload.get("holdout_feedback", {})
        if isinstance(feedback, dict) and feedback.get("output_root"):
            root = Path(str(feedback["output_root"]))
            roots.append(root if root.is_absolute() else RUN_DIR / root)
    roots.append(RUN_DIR / "runs" / "fixed_k_holdout_feedback_v2")

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            unique.append(root)
            seen.add(resolved)
    return unique


def latest_holdout_feedback_summary(config_path: Path = DEFAULT_CONFIG) -> Path:
    candidates = sorted(
        (
            path
            for root in fixed_feedback_output_roots(config_path)
            for path in root.glob("*/Models_Data/holdout_feedback_summary_residual_*.json")
        ),
        key=lambda path: (path.stat().st_mtime, str(path)),
    )
    if not candidates:
        roots = ", ".join(str(root) for root in fixed_feedback_output_roots(config_path))
        raise FileNotFoundError(f"No holdout-feedback summary found under configured roots: {roots}.")
    return candidates[-1]


def resolve_trained_run_dir(summary_path: Path, row: dict[str, object]) -> Path:
    raw = row.get("run_dir", row.get("trained_run"))
    if raw is None:
        raise KeyError("Final summary row has neither 'run_dir' nor 'trained_run'.")
    path = Path(str(raw))
    if path.is_absolute():
        return path
    output_dir = summary_path.parents[1]
    run_dir = output_dir / path
    if run_dir.is_dir():
        return run_dir
    return RUN_DIR / path


def load_checkpoint_labels(checkpoint_path: Path) -> tuple[list[str], list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return [str(label) for label in checkpoint["agp_labels"]], [str(label) for label in checkpoint["residual_labels"]]


def load_hamiltonian_context(config_payload: dict[str, object]):
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
    return config, h0, h1


def generated_residual_score_pairs(
    h0,
    h1,
    *,
    agp_labels: list[str],
    intermediate_top_k: int,
) -> list[tuple[str, float]]:
    h_score = hamiltonian_importance(h0, h1)
    delta_score = operator_importance(h1 - h0)
    commutator = h0.commutator(h1)
    endpoint_score = {label: abs(coeff) for label, coeff in commutator.terms.items()}
    agp_score = {label: endpoint_score.get(label, 1.0) for label in agp_labels}
    intermediate_scores = commutator_generated_scores(agp_score, h_score)
    bounded_intermediate = dict(ranked_label_scores(intermediate_scores)[: int(intermediate_top_k)])
    generator_scores = merge_scores(delta_score, bounded_intermediate)
    residual_scores = merge_scores(endpoint_score, commutator_generated_scores(generator_scores, h_score))
    return ranked_label_scores(residual_scores)


def generated_support_cache(
    h0,
    h1,
    *,
    agp_labels: list[str],
    intermediate_top_k: int,
) -> dict[str, object]:
    h_labels = sort_pauli_labels(set(h0.labels) | set(h1.labels))
    h_score = hamiltonian_importance(h0, h1)
    delta_score = operator_importance(h1 - h0)
    commutator = h0.commutator(h1)
    endpoint_score = {label: abs(coeff) for label, coeff in commutator.terms.items()}
    agp_score = {label: endpoint_score.get(label, 1.0) for label in agp_labels}
    intermediate_scores = commutator_generated_scores(agp_score, h_score)
    bounded_intermediate_pairs = ranked_label_scores(intermediate_scores)[: int(intermediate_top_k)]
    bounded_intermediate_scores = {label: score for label, score in bounded_intermediate_pairs}
    intermediate_labels = sort_pauli_labels(
        set(h_labels) | set(agp_labels) | {label for label, _ in bounded_intermediate_pairs}
    )
    generator_scores = merge_scores(delta_score, bounded_intermediate_scores)
    residual_scores = merge_scores(endpoint_score, commutator_generated_scores(generator_scores, h_score))
    return {
        "residual_score_pairs": ranked_label_scores(residual_scores),
        "intermediate_labels": intermediate_labels,
        "metadata": {
            "strategy": "cached_generated_commutator_projected_residual",
            "endpoint_commutator_terms": len(commutator.terms),
            "generated_intermediate_candidate_terms": len(intermediate_scores),
            "generated_residual_candidate_terms": len(residual_scores),
            "intermediate_top_k": int(intermediate_top_k),
            "intermediate_terms": len(intermediate_labels),
        },
    }


def relpath(path: Path, base: Path | None = None) -> str:
    base = RUN_DIR if base is None else base
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def build_cached_generated_probe(
    *,
    scored_labels: list[tuple[str, float]],
    feedback_residual_labels: list[str],
    extra_excluded_labels: list[str] | None,
    probe_residual_terms: int,
    probe_source_agp_terms: int,
    probe_name: str,
) -> tuple[list[str], dict[str, object]]:
    excluded = set(feedback_residual_labels)
    if extra_excluded_labels is not None:
        excluded.update(str(label) for label in extra_excluded_labels)
    labels = select_top_labels(scored_labels, count=int(probe_residual_terms), excluded=excluded)
    return labels, {
        "probe_name": probe_name,
        "probe_residual_terms_requested": int(probe_residual_terms),
        "probe_residual_terms": len(labels),
        "probe_source_agp_terms": int(probe_source_agp_terms),
        "excluded_feedback_residual_terms": len(set(feedback_residual_labels)),
        "excluded_total_terms": len(excluded),
        "generated_residual_candidate_terms": len(scored_labels),
        "selection_rule": (
            "Fixed disjoint residual probe selected from one cached generated residual-score list. "
            "Labels already present in the training residual basis and prior probes are excluded; "
            "the trained model is replayed on this basis without retraining."
        ),
    }


def evaluate_basis(
    *,
    name: str,
    labels: list[str],
    run_dir: Path,
    config_payload: dict[str, object],
    config,
    h0,
    h1,
    intermediate_labels: list[str],
    support_cache_metadata: dict[str, object],
    data_dir: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
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
    residual_labels = sort_pauli_labels(labels)
    support = {
        "agp_labels": trained_agp_labels,
        "intermediate_labels": intermediate_labels,
        "residual_labels": residual_labels,
        "metadata": {
            **support_cache_metadata,
            "residual_selection_rule": "explicit_certification_residual_labels",
            "residual_terms": len(residual_labels),
            "basis_name": name,
        },
    }

    device = select_device("cpu")
    model = make_projected_model(h0, h1, support, config, device)
    load_body_weights(model, checkpoint)
    model.eval()

    tau = torch.linspace(0.0, 1.0, num_points, device=device).view(-1, 1)
    t = config.t_initial + config.physical_time * tau
    with torch.no_grad():
        residual = model.euler_lagrange_residual(t)
        reference = model.euler_lagrange_reference_residual(t)
    seen_indices = [idx for idx, label in enumerate(residual_labels) if label in trained_residual_labels]
    unseen_indices = [idx for idx, label in enumerate(residual_labels) if label not in trained_residual_labels]
    seen_residual = norm_sq_subset(residual, seen_indices)
    seen_reference = norm_sq_subset(reference, seen_indices)
    unseen_residual = norm_sq_subset(residual, unseen_indices)
    unseen_reference = norm_sq_subset(reference, unseen_indices)
    eps = torch.finfo(seen_residual.dtype).eps
    seen_relative = seen_residual / torch.clamp(seen_reference, min=eps)
    unseen_relative = unseen_residual / torch.clamp(unseen_reference, min=eps)

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

    spectra_dir = data_dir / f"{name}_projection"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    spectrum_path = spectra_dir / f"certification_spectrum_{name}_agp_{len(trained_agp_labels)}_residual_{len(residual_labels)}.json"
    spectrum_path.write_text(
        json.dumps(
            {
                "trained_run": relpath(run_dir),
                "basis_name": name,
                "agp_terms": len(trained_agp_labels),
                "residual_terms": len(residual_labels),
                "spectrum": spectrum,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    full_basis = Decimal(4) ** int(config.n_qubits)
    best = min(train_history, key=lambda row: float(row["total"]))
    row = {
        "trained_run": relpath(run_dir),
        "basis_name": name,
        "n_qubits": int(config.n_qubits),
        "full_pauli_basis_size": str(full_basis),
        "agp_terms": len(trained_agp_labels),
        "agp_fraction_of_full_basis": f"{Decimal(len(trained_agp_labels)) / full_basis:.12E}",
        "train_residual_terms": int(train_metadata["final_residual_terms"]),
        "holdout_residual_terms": len(residual_labels),
        "holdout_basis_mode": name,
        "holdout_basis_agp_terms": None,
        "seen_residual_terms": len(seen_indices),
        "unseen_residual_terms": len(unseen_indices),
        "intermediate_terms": len(intermediate_labels),
        "hamiltonian_terms": len(model.hamiltonian_labels),
        "first_commutator_nnz": model.first_commutator.nnz,
        "second_commutator_nnz": model.second_commutator.nnz,
        "training_final_relative_residual": float(train_history[-1]["relative_residual"]),
        "training_best_relative_residual": float(best["relative_residual"]),
        "holdout_total_residual": scalar(torch.mean(torch.sum(torch.abs(residual) ** 2, dim=-1).real)),
        "holdout_reference_residual": scalar(torch.mean(torch.sum(torch.abs(reference) ** 2, dim=-1).real)),
        "holdout_relative_residual": scalar(
            torch.mean(torch.sum(torch.abs(residual) ** 2, dim=-1).real)
            / torch.clamp(torch.mean(torch.sum(torch.abs(reference) ** 2, dim=-1).real), min=eps)
        ),
        "seen_residual": scalar(seen_residual),
        "seen_reference_residual": scalar(seen_reference),
        "seen_relative_residual": scalar(seen_relative),
        "unseen_residual": scalar(unseen_residual),
        "unseen_reference_residual": scalar(unseen_reference),
        "unseen_relative_residual": scalar(unseen_relative),
        "top_holdout_residual_terms": spectrum[:64],
        "spectrum_export": relpath(spectrum_path),
        "selection_rule": "explicit_certification_residual_labels_with_cached_intermediate_support",
    }
    return row, spectrum


def _basis_metrics_from_union(
    *,
    name: str,
    labels: list[str],
    union_labels: list[str],
    residual_union: torch.Tensor,
    reference_union: torch.Tensor,
    trained_residual_labels: set[str],
    run_dir: Path,
    config,
    train_metadata: dict[str, object],
    train_history: list[dict[str, object]],
    agp_labels: list[str],
    intermediate_labels: list[str],
    hamiltonian_terms: int,
    first_commutator_nnz: int,
    second_commutator_nnz: int,
    data_dir: Path,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray, np.ndarray]:
    union_index = {label: idx for idx, label in enumerate(union_labels)}
    indices = [union_index[label] for label in labels if label in union_index]
    if len(indices) != len(labels):
        missing = len(labels) - len(indices)
        raise RuntimeError(f"Basis {name!r} has {missing} labels absent from the union residual matrix.")
    index_tensor = torch.tensor(indices, dtype=torch.long, device=residual_union.device)
    residual = residual_union.index_select(-1, index_tensor)
    reference = reference_union.index_select(-1, index_tensor)
    seen_indices = [idx for idx, label in enumerate(labels) if label in trained_residual_labels]
    unseen_indices = [idx for idx, label in enumerate(labels) if label not in trained_residual_labels]
    seen_residual = norm_sq_subset(residual, seen_indices)
    seen_reference = norm_sq_subset(reference, seen_indices)
    unseen_residual = norm_sq_subset(residual, unseen_indices)
    unseen_reference = norm_sq_subset(reference, unseen_indices)
    eps = torch.finfo(seen_residual.dtype).eps
    seen_relative = seen_residual / torch.clamp(seen_reference, min=eps)
    unseen_relative = unseen_residual / torch.clamp(unseen_reference, min=eps)
    total_residual = torch.mean(torch.sum(torch.abs(residual) ** 2, dim=-1).real)
    total_reference = torch.mean(torch.sum(torch.abs(reference) ** 2, dim=-1).real)
    total_relative = total_residual / torch.clamp(total_reference, min=eps)

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
            for idx, label in enumerate(labels)
        ],
        key=lambda row: (float(row["residual_rms"]), int(row["order"])),
        reverse=True,
    )
    spectra_dir = data_dir / f"{name}_projection"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    spectrum_path = spectra_dir / f"certification_spectrum_{name}_agp_{len(agp_labels)}_residual_{len(labels)}.json"
    spectrum_path.write_text(
        json.dumps(
            {
                "trained_run": relpath(run_dir),
                "basis_name": name,
                "agp_terms": len(agp_labels),
                "residual_terms": len(labels),
                "spectrum": spectrum,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    full_basis = Decimal(4) ** int(config.n_qubits)
    best = min(train_history, key=lambda row: float(row["total"]))
    row = {
        "trained_run": relpath(run_dir),
        "basis_name": name,
        "n_qubits": int(config.n_qubits),
        "full_pauli_basis_size": str(full_basis),
        "agp_terms": len(agp_labels),
        "agp_fraction_of_full_basis": f"{Decimal(len(agp_labels)) / full_basis:.12E}",
        "train_residual_terms": int(train_metadata["final_residual_terms"]),
        "holdout_residual_terms": len(labels),
        "holdout_basis_mode": name,
        "holdout_basis_agp_terms": None,
        "seen_residual_terms": len(seen_indices),
        "unseen_residual_terms": len(unseen_indices),
        "intermediate_terms": len(intermediate_labels),
        "hamiltonian_terms": int(hamiltonian_terms),
        "first_commutator_nnz": int(first_commutator_nnz),
        "second_commutator_nnz": int(second_commutator_nnz),
        "training_final_relative_residual": float(train_history[-1]["relative_residual"]),
        "training_best_relative_residual": float(best["relative_residual"]),
        "holdout_total_residual": scalar(total_residual),
        "holdout_reference_residual": scalar(total_reference),
        "holdout_relative_residual": scalar(total_relative),
        "seen_residual": scalar(seen_residual),
        "seen_reference_residual": scalar(seen_reference),
        "seen_relative_residual": scalar(seen_relative),
        "unseen_residual": scalar(unseen_residual),
        "unseen_reference_residual": scalar(unseen_reference),
        "unseen_relative_residual": scalar(unseen_relative),
        "top_holdout_residual_terms": spectrum[:64],
        "spectrum_export": relpath(spectrum_path),
        "selection_rule": "explicit_certification_residual_labels_sliced_from_union_replay",
    }
    return (
        row,
        spectrum,
        residual.detach().cpu().numpy().astype(np.complex128, copy=False),
        reference.detach().cpu().numpy().astype(np.complex128, copy=False),
    )


def evaluate_bases_with_union(
    *,
    named_bases: dict[str, list[str]],
    run_dir: Path,
    config,
    h0,
    h1,
    intermediate_labels: list[str],
    support_cache_metadata: dict[str, object],
    data_dir: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, list[dict[str, object]]], dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray]:
    checkpoint_path = run_dir / "Models_Data" / "training_checkpoint.pt"
    metadata_path = run_dir / "Models_Data" / "support_metadata.json"
    history_path = run_dir / "Models_Data" / "loss_history.json"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_metadata = load_json(metadata_path)
    train_history = load_json(history_path)
    if not isinstance(train_metadata, dict) or not isinstance(train_history, list):
        raise TypeError(f"Unexpected metadata/history shape in {run_dir}.")
    agp_labels = [str(label) for label in checkpoint["agp_labels"]]
    trained_residual_labels = {str(label) for label in checkpoint["residual_labels"]}
    union_labels = sort_pauli_labels({label for labels in named_bases.values() for label in labels})
    support = {
        "agp_labels": agp_labels,
        "intermediate_labels": intermediate_labels,
        "residual_labels": union_labels,
        "metadata": {
            **support_cache_metadata,
            "residual_selection_rule": "union_of_certification_residual_labels",
            "residual_terms": len(union_labels),
        },
    }
    print(f"build_union_certification_model residual_terms={len(union_labels)}", flush=True)
    device = select_device("cpu")
    model = make_projected_model(h0, h1, support, config, device)
    load_body_weights(model, checkpoint)
    model.eval()

    num_points = 16
    tau = torch.linspace(0.0, 1.0, num_points, device=device).view(-1, 1)
    t = config.t_initial + config.physical_time * tau
    with torch.no_grad():
        residual_union = model.euler_lagrange_residual(t)
        reference_union = model.euler_lagrange_reference_residual(t)

    rows: dict[str, dict[str, object]] = {}
    spectra: dict[str, list[dict[str, object]]] = {}
    matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, labels in named_bases.items():
        print(f"slice_certification_basis name={name} terms={len(labels)}", flush=True)
        row, spectrum, residual_by_tau, reference_by_tau = _basis_metrics_from_union(
            name=name,
            labels=labels,
            union_labels=union_labels,
            residual_union=residual_union,
            reference_union=reference_union,
            trained_residual_labels=trained_residual_labels,
            run_dir=run_dir,
            config=config,
            train_metadata=train_metadata,
            train_history=train_history,
            agp_labels=agp_labels,
            intermediate_labels=intermediate_labels,
            hamiltonian_terms=len(model.hamiltonian_labels),
            first_commutator_nnz=model.first_commutator.nnz,
            second_commutator_nnz=model.second_commutator.nnz,
            data_dir=data_dir,
        )
        rows[name] = row
        spectra[name] = spectrum
        matrices[name] = (residual_by_tau, reference_by_tau)
    return rows, spectra, matrices, tau.squeeze(-1).detach().cpu().numpy()


def residual_matrices_for_basis(
    *,
    run_dir: Path,
    config_payload: dict[str, object],
    config,
    h0,
    h1,
    residual_labels: list[str],
    intermediate_labels: list[str],
    support_cache_metadata: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    checkpoint_path = run_dir / "Models_Data" / "training_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    agp_labels = [str(label) for label in checkpoint["agp_labels"]]
    support = {
        "agp_labels": agp_labels,
        "intermediate_labels": intermediate_labels,
        "residual_labels": sort_pauli_labels(residual_labels),
        "metadata": {
            **support_cache_metadata,
            "residual_selection_rule": "explicit_certification_residual_labels",
            "residual_terms": len(residual_labels),
        },
    }
    device = select_device("cpu")
    model = make_projected_model(h0, h1, support, config, device)
    load_body_weights(model, checkpoint)
    model.eval()

    training = config_payload.get("training", {})
    parameters = training.get("parameters", {}) if isinstance(training, dict) else {}
    num_points = int(parameters.get("num_points", 16))
    tau = torch.linspace(0.0, 1.0, num_points, device=device).view(-1, 1)
    t = config.t_initial + config.physical_time * tau
    with torch.no_grad():
        residual = model.euler_lagrange_residual(t)
        reference = model.euler_lagrange_reference_residual(t)
    return (
        tau.squeeze(-1).detach().cpu().numpy(),
        residual.detach().cpu().numpy().astype(np.complex128, copy=False),
        reference.detach().cpu().numpy().astype(np.complex128, copy=False),
    )


def relative_energy(residual_by_tau: np.ndarray, reference_by_tau: np.ndarray) -> tuple[float, float, float]:
    residual = float(np.mean(np.sum(np.abs(residual_by_tau) ** 2, axis=1).real))
    reference = float(np.mean(np.sum(np.abs(reference_by_tau) ** 2, axis=1).real))
    return residual, reference, residual / max(reference, np.finfo(float).eps)


def evaluate_q_sweep(rows: list[dict[str, object]], *, plateau_threshold: float) -> dict[str, object]:
    if len(rows) < 2:
        return {
            "status": "not tested",
            "rows": rows,
            "threshold": float(plateau_threshold),
            "note": "At least two residual-basis sizes are required.",
        }
    changes: list[dict[str, object]] = []
    for previous, current in zip(rows, rows[1:]):
        previous_value = float(previous["holdout_relative_residual"])
        current_value = float(current["holdout_relative_residual"])
        change = abs(current_value - previous_value) / max(previous_value, np.finfo(float).eps)
        changes.append(
            {
                "from_terms": int(previous["holdout_residual_terms"]),
                "to_terms": int(current["holdout_residual_terms"]),
                "previous_relative_residual": previous_value,
                "current_relative_residual": current_value,
                "absolute_fractional_change": float(change),
            }
        )
    max_change = max(float(row["absolute_fractional_change"]) for row in changes)
    return {
        "status": "pass" if max_change <= float(plateau_threshold) else "fail",
        "rows": rows,
        "changes": changes,
        "max_absolute_fractional_change": max_change,
        "threshold": float(plateau_threshold),
        "selection_rule": "Nested generated residual prefixes from the current AGP support.",
    }


def summarize_multi_holdout(rows: list[dict[str, object]], *, threshold: float) -> dict[str, object]:
    worst = max((float(row["holdout_relative_residual"]) for row in rows), default=None)
    return {
        "status": "not tested"
        if worst is None
        else "pass"
        if worst <= float(threshold)
        else "fail",
        "max_relative_residual": worst,
        "threshold": float(threshold),
        "rows": rows,
        "selection_rule": (
            "Final trained weights are replayed on several generated residual bases: the original "
            "feedback holdout, disjoint fixed probes, an order-stratified generated probe, and a "
            "seeded generated probe."
        ),
    }


def omitted_term_pressure(
    *,
    spectrum: list[dict[str, object]],
    residual_labels: list[str],
    residual_by_tau: np.ndarray,
    reference_by_tau: np.ndarray,
    tau_values: np.ndarray,
    agp_labels: list[str],
    h0,
    h1,
    max_candidates: int,
    residual_candidate_terms: int,
    hamiltonian_candidate_terms: int,
    omp_hamiltonian_terms: int,
    omp_max_candidates: int,
    report_terms: int,
    gain_threshold: float,
) -> dict[str, object]:
    raw_candidates = select_agp_additions_from_residual(
        spectrum,
        set(agp_labels),
        h0=h0,
        h1=h1,
        add_terms=int(max_candidates),
        max_agp_terms=None,
        residual_candidate_terms=int(residual_candidate_terms),
        hamiltonian_candidate_terms=int(hamiltonian_candidate_terms),
        min_residual_rms=0.0,
        min_score=0.0,
        unseen_only=True,
    )
    scored = score_candidates_with_omp(
        raw_candidates,
        residual_labels=residual_labels,
        residual_by_tau=residual_by_tau,
        tau_values=tau_values,
        h0=h0,
        h1=h1,
        hamiltonian_top_k=int(omp_hamiltonian_terms),
        max_candidates=int(omp_max_candidates),
    )
    residual_energy_value, reference_energy_value, relative_value = relative_energy(residual_by_tau, reference_by_tau)
    top_score = float(scored[0].get("omp_score", 0.0)) if scored else 0.0
    best_fractional_gain = top_score / max(residual_energy_value, np.finfo(float).eps)
    best_reference_gain = top_score / max(reference_energy_value, np.finfo(float).eps)
    return {
        "status": "pass" if best_fractional_gain <= float(gain_threshold) else "fail",
        "candidate_count": len(raw_candidates),
        "scored_candidate_count": len(scored),
        "residual_terms": len(residual_labels),
        "residual_energy": residual_energy_value,
        "reference_energy": reference_energy_value,
        "relative_residual": relative_value,
        "best_single_candidate_omp_score": top_score,
        "best_single_candidate_fractional_residual_gain": float(best_fractional_gain),
        "best_single_candidate_reference_gain": float(best_reference_gain),
        "gain_threshold": float(gain_threshold),
        "top_omitted_candidates": scored[: int(report_terms)],
        "selection_rule": (
            "Candidates are inverse-commutator AGP labels generated from high-RMS residual strings "
            "outside the active AGP support, then re-ranked by a projected symbolic matching-pursuit "
            "double-commutator score."
        ),
        "caveat": (
            "This is an adversarial search inside a generated candidate pool. A pass means no "
            "large omitted direction was found by this oracle, not that all 4**q Pauli strings "
            "were exhaustively checked."
        ),
    }


def plot_certification_report(report: dict[str, object], images_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style(plt)

    multi_holdout = report["multi_holdout"]
    holdout_rows = list(multi_holdout["rows"]) if isinstance(multi_holdout, dict) else []
    labels = [str(row["basis_name"]).replace("_", "\n") for row in holdout_rows]
    values = [float(row["holdout_relative_residual"]) for row in holdout_rows]

    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.3))
    x = np.arange(len(values), dtype=float)
    axes[0].bar(x, values, color=OKABE_ITO[0])
    axes[0].axhline(float(multi_holdout["threshold"]), color="0.35", linestyle="--", linewidth=0.8)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right", fontsize=LEGEND_FS)
    axes[0].set_ylabel("relative residual", fontsize=LABEL_FS)
    axes[0].set_title("multi-holdout replay", fontsize=TITLE_FS)

    q_sweep = report["q_sweep"]
    q_rows = list(q_sweep["rows"]) if isinstance(q_sweep, dict) else []
    if q_rows:
        qx = np.asarray([int(row["holdout_residual_terms"]) for row in q_rows], dtype=float)
        qy = np.asarray([float(row["holdout_relative_residual"]) for row in q_rows], dtype=float)
        axes[1].semilogy(qx, qy, marker="o", linewidth=LINE_WIDTH, color=OKABE_ITO[1])
    axes[1].set_xlabel("residual basis size $Q$", fontsize=LABEL_FS)
    axes[1].set_title("$Q$-sweep plateau", fontsize=TITLE_FS)
    axes[1].set_ylabel("relative residual", fontsize=LABEL_FS)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=TICK_FS, length=TICK_LENGTH, width=TICK_WIDTH)
    fig.subplots_adjust(top=0.86, left=0.10, right=0.98, bottom=0.28, wspace=0.36)
    fig.savefig(images_dir / "support_robustness_certification.pdf", format="pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Certify sparse AGP support robustness without retraining.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--intermediate-top-k", type=int, default=None)
    parser.add_argument("--probe-residual-terms", type=int, default=4096)
    parser.add_argument("--probe-test-residual-terms", type=int, default=None)
    parser.add_argument("--probe-source-agp-terms", type=int, default=None)
    parser.add_argument("--q-sweep-terms", default="4096,8192,13312")
    parser.add_argument("--random-seed", type=int, default=20260707)
    parser.add_argument("--random-pool-multiplier", type=int, default=8)
    parser.add_argument("--target-train", type=float, default=0.10)
    parser.add_argument("--target-holdout", type=float, default=0.10)
    parser.add_argument("--target-unseen", type=float, default=1.0)
    parser.add_argument("--q-plateau-threshold", type=float, default=0.10)
    parser.add_argument("--omitted-gain-threshold", type=float, default=0.05)
    parser.add_argument("--omitted-candidates", type=int, default=512)
    parser.add_argument("--candidate-residual-terms", type=int, default=2048)
    parser.add_argument("--candidate-hamiltonian-terms", type=int, default=64)
    parser.add_argument("--omp-hamiltonian-terms", type=int, default=48)
    parser.add_argument("--omp-max-candidates", type=int, default=512)
    parser.add_argument("--omitted-report-terms", type=int, default=32)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    configure_run_dir(args.config)
    getcontext().prec = 80
    config_payload = load_json(args.config)
    if not isinstance(config_payload, dict):
        raise TypeError("config.json must contain a JSON object.")
    support_payload = config_payload.get("support_sweep", {})
    intermediate_top_k = (
        int(args.intermediate_top_k)
        if args.intermediate_top_k is not None
        else int(support_payload.get("intermediate_top_k", 2048))
        if isinstance(support_payload, dict)
        else 2048
    )
    summary_path = args.summary or latest_holdout_feedback_summary(args.config)
    summary_payload = load_json(summary_path)
    if not isinstance(summary_payload, dict):
        raise TypeError(f"{summary_path} must contain a JSON object.")
    rows = [row for row in summary_payload.get("rows", []) if isinstance(row, dict)]
    if not rows:
        raise RuntimeError(f"No rows found in {summary_path}.")
    final_row = dict(rows[-1])
    run_dir = resolve_trained_run_dir(summary_path, final_row)
    output_dir = summary_path.parents[1]
    data_dir = output_dir / "Models_Data"
    images_dir = output_dir / "Images"
    checkpoint_path = run_dir / "Models_Data" / "training_checkpoint.pt"
    agp_labels, training_residual_labels = load_checkpoint_labels(checkpoint_path)
    config, h0, h1 = load_hamiltonian_context(config_payload)
    print(
        f"build_generated_residual_scores agp_terms={len(agp_labels)} "
        f"intermediate_top_k={intermediate_top_k}",
        flush=True,
    )
    support_cache = generated_support_cache(
        h0,
        h1,
        agp_labels=agp_labels,
        intermediate_top_k=intermediate_top_k,
    )
    generated_scores = list(support_cache["residual_score_pairs"])
    intermediate_labels = [str(label) for label in support_cache["intermediate_labels"]]
    support_cache_metadata = dict(support_cache["metadata"])
    print(f"generated_residual_scores count={len(generated_scores)}", flush=True)

    probe_terms = int(args.probe_residual_terms)
    probe_test_terms = int(args.probe_test_residual_terms or probe_terms)
    probe_source_agp_terms = int(args.probe_source_agp_terms or len(agp_labels))
    probe_gate_labels, probe_gate_metadata = build_cached_generated_probe(
        scored_labels=generated_scores,
        feedback_residual_labels=training_residual_labels,
        extra_excluded_labels=None,
        probe_source_agp_terms=probe_source_agp_terms,
        probe_residual_terms=probe_terms,
        probe_name="support_certification_probe_gate",
    )
    probe_watch_labels, probe_watch_metadata = build_cached_generated_probe(
        scored_labels=generated_scores,
        feedback_residual_labels=training_residual_labels,
        extra_excluded_labels=probe_gate_labels,
        probe_source_agp_terms=probe_source_agp_terms,
        probe_residual_terms=probe_terms,
        probe_name="support_certification_probe_watch",
    )
    probe_test_labels, probe_test_metadata = build_cached_generated_probe(
        scored_labels=generated_scores,
        feedback_residual_labels=training_residual_labels,
        extra_excluded_labels=probe_gate_labels + probe_watch_labels,
        probe_source_agp_terms=probe_source_agp_terms,
        probe_residual_terms=probe_test_terms,
        probe_name="support_certification_probe_test",
    )
    stratified_excluded = set(training_residual_labels) | set(probe_gate_labels) | set(probe_watch_labels) | set(probe_test_labels)
    stratified_labels = select_order_stratified_labels(
        generated_scores,
        count=probe_terms,
        excluded=stratified_excluded,
    )
    seeded_labels = select_seeded_labels(
        generated_scores,
        count=probe_terms,
        excluded=stratified_excluded | set(stratified_labels),
        seed=int(args.random_seed),
        pool_multiplier=int(args.random_pool_multiplier),
    )

    multi_rows: list[dict[str, object]] = []
    spectra_by_basis: dict[str, list[dict[str, object]]] = {}
    cached = dict(final_row)
    cached["basis_name"] = "feedback_holdout_existing"
    cached["selection_rule"] = "cached_final_feedback_holdout_from_training_summary"
    multi_rows.append(cached)

    q_sweep_terms = [
        int(item)
        for chunk in str(args.q_sweep_terms).replace(",", " ").split()
        for item in [chunk.strip()]
        if item
    ]
    named_bases: dict[str, list[str]] = {
        "probe_gate_disjoint": probe_gate_labels,
        "probe_watch_disjoint": probe_watch_labels,
        "probe_test_disjoint": probe_test_labels,
        "order_stratified_generated": stratified_labels,
        "seeded_generated": seeded_labels,
    }
    q_seen: set[int] = set()
    for terms in sorted(q_sweep_terms):
        if terms in q_seen:
            continue
        q_seen.add(terms)
        q_labels = select_top_labels(generated_scores, count=terms)
        named_bases[f"q_sweep_top_generated_{len(q_labels)}"] = q_labels

    union_rows, spectra_by_basis, matrices_by_basis, tau_values = evaluate_bases_with_union(
        named_bases=named_bases,
        run_dir=run_dir,
        config=config,
        h0=h0,
        h1=h1,
        intermediate_labels=intermediate_labels,
        support_cache_metadata=support_cache_metadata,
        data_dir=data_dir,
    )
    for name in (
        "probe_gate_disjoint",
        "probe_watch_disjoint",
        "probe_test_disjoint",
        "order_stratified_generated",
        "seeded_generated",
    ):
        if name in union_rows:
            multi_rows.append(union_rows[name])
    q_sweep_rows = [
        union_rows[f"q_sweep_top_generated_{terms}"]
        for terms in sorted(q_seen)
        if f"q_sweep_top_generated_{terms}" in union_rows
    ]

    adversarial_labels = probe_test_labels or stratified_labels or probe_gate_labels
    adversarial_spectrum = spectra_by_basis.get("probe_test_disjoint", [])
    residual_by_tau, reference_by_tau = matrices_by_basis["probe_test_disjoint"]
    pressure = omitted_term_pressure(
        spectrum=adversarial_spectrum or [],
        residual_labels=adversarial_labels,
        residual_by_tau=residual_by_tau,
        reference_by_tau=reference_by_tau,
        tau_values=tau_values,
        agp_labels=agp_labels,
        h0=h0,
        h1=h1,
        max_candidates=int(args.omitted_candidates),
        residual_candidate_terms=int(args.candidate_residual_terms),
        hamiltonian_candidate_terms=int(args.candidate_hamiltonian_terms),
        omp_hamiltonian_terms=int(args.omp_hamiltonian_terms),
        omp_max_candidates=int(args.omp_max_candidates),
        report_terms=int(args.omitted_report_terms),
        gain_threshold=float(args.omitted_gain_threshold),
    )

    multi_holdout = summarize_multi_holdout(multi_rows, threshold=float(args.target_holdout))
    q_sweep = evaluate_q_sweep(q_sweep_rows, plateau_threshold=float(args.q_plateau_threshold))
    checks = {
        "training_residual": gate(float(final_row.get("training_final_relative_residual")), float(args.target_train)),
        "holdout_residual": gate(float(final_row.get("holdout_relative_residual")), float(args.target_holdout)),
        "unseen_residual": gate(float(final_row.get("unseen_relative_residual")), float(args.target_unseen)),
        "multi_holdout": {
            "status": multi_holdout["status"],
            "value": multi_holdout["max_relative_residual"],
            "threshold": multi_holdout["threshold"],
        },
        "q_sweep_plateau": {
            "status": q_sweep["status"],
            "value": q_sweep.get("max_absolute_fractional_change"),
            "threshold": q_sweep["threshold"],
        },
        "omitted_term_pressure": {
            "status": pressure["status"],
            "value": pressure["best_single_candidate_fractional_residual_gain"],
            "threshold": pressure["gain_threshold"],
        },
        "k_sweep_plateau": {
            "status": "not tested",
            "note": "Only one K=4**7 fixed-support run is present in this q20 feedback artifact.",
        },
        "seed_stability": {
            "status": "not tested",
            "note": "Requires independently trained seeds under the same support-selection rule.",
        },
        "prune_and_retest": {
            "status": "not tested",
            "note": "Requires retraining or replaying pruned supports on the fixed probes.",
        },
        "physical_validation": {
            "status": "not tested",
            "note": "No state-evolution or observable validation is performed by this residual-only script.",
        },
    }
    claim_level = classify_claim_level(checks)
    full_basis = Decimal(4) ** int(config.n_qubits)
    report = {
        "schema_version": 1,
        "methodology": "sparse_agp_support_robustness_certification_v1",
        "criteria_file": "AGP_CERTIFICATION_CRITERIA.md",
        "summary": str(summary_path),
        "trained_run": str(run_dir),
        "n_qubits": int(config.n_qubits),
        "full_pauli_basis_size": str(full_basis),
        "agp_terms": len(agp_labels),
        "agp_fraction_of_full_basis": f"{Decimal(len(agp_labels)) / full_basis:.12E}",
        "training_residual_terms": len(training_residual_labels),
        "generated_residual_candidate_terms": len(generated_scores),
        "limits": {
            "full_basis_enumerated": False,
            "reason": (
                "For q>8 the full 4**q Pauli basis is not enumerated. Certification is based on "
                "generated sparse residual/proposal pools and must be interpreted as a sparse "
                "certificate, not an unrestricted full-basis proof."
            ),
        },
        "probe_metadata": {
            "probe_gate": probe_gate_metadata,
            "probe_watch": probe_watch_metadata,
            "probe_test": probe_test_metadata,
            "order_stratified_terms": len(stratified_labels),
            "seeded_generated_terms": len(seeded_labels),
            "random_seed": int(args.random_seed),
        },
        "multi_holdout": multi_holdout,
        "q_sweep": q_sweep,
        "omitted_term_pressure": pressure,
        "checks": checks,
        "claim_level": claim_level,
        "decision": {
            "certified": claim_level == "certified_sparse_agp_for_this_path_and_tolerance",
            "reason": "Any failed strict gate or untested required gate downgrades the claim level.",
        },
    }
    output_path = args.output or (data_dir / CERTIFICATION_FILENAME)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_certification_report(report, images_dir)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "plot": str(images_dir / "support_robustness_certification.pdf"),
                "claim_level": claim_level,
                "certified": report["decision"]["certified"],
                "multi_holdout_status": multi_holdout["status"],
                "q_sweep_status": q_sweep["status"],
                "omitted_term_pressure_status": pressure["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
