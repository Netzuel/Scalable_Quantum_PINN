from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict, replace
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from holdout_feedback_training import (  # noqa: E402
    load_body_state_from_checkpoint,
    load_checkpoint_labels,
    make_support_with_residual_labels,
    plot_feedback_added_terms,
    plot_feedback_relative_residuals,
    plot_feedback_residual_spectrum,
    plot_feedback_seen_unseen,
    resolve_holdout_residual_top_k,
    select_residual_additions,
    write_feedback_spectrum,
)
from holdout_study import Thresholds, build_common_holdout_residual_labels, evaluate_one_run, load_json  # noqa: E402
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
)
from training_script import DEFAULT_CONFIG, RUN_DIR, model_config_from_payload, settings_for_support  # noqa: E402
from utils import _commutator_pauli_labels_unchecked, load_pauli_hamiltonian_pair, pauli_weight  # noqa: E402


PAULI_PRODUCT = {
    ("I", "I"): "I",
    ("I", "X"): "X",
    ("I", "Y"): "Y",
    ("I", "Z"): "Z",
    ("X", "I"): "X",
    ("X", "X"): "I",
    ("X", "Y"): "Z",
    ("X", "Z"): "Y",
    ("Y", "I"): "Y",
    ("Y", "X"): "Z",
    ("Y", "Y"): "I",
    ("Y", "Z"): "X",
    ("Z", "I"): "Z",
    ("Z", "X"): "Y",
    ("Z", "Y"): "X",
    ("Z", "Z"): "I",
}


def pauli_product_label(left: str, right: str) -> str:
    if len(left) != len(right):
        raise ValueError("Pauli labels must have the same length.")
    return "".join(PAULI_PRODUCT[(a, b)] for a, b in zip(left, right))


def inverse_commutator_source(output_label: str, right_label: str) -> str | None:
    """Return a Pauli string P such that [P, right_label] can produce output_label."""

    candidate = pauli_product_label(output_label, right_label)
    item = _commutator_pauli_labels_unchecked(candidate, right_label)
    if item is None:
        return None
    _, produced = item
    return candidate if produced == output_label else None


def hamiltonian_scores(h0, h1) -> dict[str, float]:
    labels = set(h0.labels) | set(h1.labels)
    return {label: max(abs(h0.coefficient(label)), abs(h1.coefficient(label))) for label in labels}


def select_agp_additions_from_residual(
    spectrum: list[dict[str, object]],
    current_agp_labels: set[str],
    *,
    h0,
    h1,
    add_terms: int,
    max_agp_terms: int | None,
    residual_candidate_terms: int,
    hamiltonian_candidate_terms: int,
    min_residual_rms: float,
    min_score: float,
    unseen_only: bool,
) -> list[dict[str, object]]:
    if add_terms <= 0:
        return []
    if max_agp_terms is not None:
        add_terms = min(add_terms, max(max_agp_terms - len(current_agp_labels), 0))
    if add_terms <= 0:
        return []

    h_scores = hamiltonian_scores(h0, h1)
    h_ranked = sorted(h_scores.items(), key=lambda item: item[1], reverse=True)[:hamiltonian_candidate_terms]
    identity = "I" * h0.n_qubits
    candidates: dict[str, dict[str, object]] = {}

    for row in spectrum[:residual_candidate_terms]:
        if unseen_only and bool(row.get("seen_during_training", False)):
            continue
        residual_rms = float(row["residual_rms"])
        if residual_rms < min_residual_rms:
            continue
        residual_label = str(row["label"])
        for h2_label, h2_score in h_ranked:
            intermediate_label = inverse_commutator_source(residual_label, h2_label)
            if intermediate_label is None:
                continue
            for h1_label, h1_score in h_ranked:
                candidate_label = inverse_commutator_source(intermediate_label, h1_label)
                if candidate_label is None:
                    continue
                if candidate_label == identity or candidate_label in current_agp_labels:
                    continue
                order_penalty = max(pauli_weight(candidate_label), 1) ** 0.5
                score = residual_rms * float(h1_score) * float(h2_score) / order_penalty
                if score < min_score:
                    continue
                entry = candidates.get(candidate_label)
                if entry is None:
                    candidates[candidate_label] = {
                        "label": candidate_label,
                        "score": float(score),
                        "order": pauli_weight(candidate_label),
                        "source_residual_label": residual_label,
                        "source_residual_rms": residual_rms,
                        "intermediate_label": intermediate_label,
                        "left_hamiltonian_label": h1_label,
                        "right_hamiltonian_label": h2_label,
                        "left_hamiltonian_score": float(h1_score),
                        "right_hamiltonian_score": float(h2_score),
                    }
                else:
                    entry["score"] = float(entry["score"]) + float(score)
                    if residual_rms > float(entry["source_residual_rms"]):
                        entry.update(
                            {
                                "source_residual_label": residual_label,
                                "source_residual_rms": residual_rms,
                                "intermediate_label": intermediate_label,
                                "left_hamiltonian_label": h1_label,
                                "right_hamiltonian_label": h2_label,
                                "left_hamiltonian_score": float(h1_score),
                                "right_hamiltonian_score": float(h2_score),
                            }
                        )

    ranked = sorted(
        candidates.values(),
        key=lambda row: (float(row["score"]), float(row["source_residual_rms"]), -int(row["order"]), str(row["label"])),
        reverse=True,
    )
    return ranked[:add_terms]


def merge_agp_candidate_additions(
    *,
    feedback_candidates: list[dict[str, object]],
    probe_candidates: list[dict[str, object]],
    current_agp_labels: set[str],
    add_terms: int,
    probe_score_weight: float,
) -> list[dict[str, object]]:
    """Merge feedback-driven and fixed-probe-driven AGP proposals."""

    merged: dict[str, dict[str, object]] = {}
    for source_name, weight, candidates in (
        ("feedback", 1.0, feedback_candidates),
        ("probe_gate", float(probe_score_weight), probe_candidates),
    ):
        for rank, candidate in enumerate(candidates, start=1):
            label = str(candidate["label"])
            if label in current_agp_labels:
                continue
            score = float(candidate.get("score", 0.0)) * weight
            entry = merged.get(label)
            if entry is None:
                entry = {
                    "label": label,
                    "score": 0.0,
                    "order": int(candidate.get("order", pauli_weight(label))),
                    "sources": [],
                    "feedback_score": 0.0,
                    "probe_gate_score": 0.0,
                    "best_source": source_name,
                    "best_source_rank": rank,
                    "best_source_score": float(candidate.get("score", 0.0)),
                }
                merged[label] = entry
            entry["score"] = float(entry["score"]) + score
            source_payload = dict(candidate)
            source_payload["source"] = source_name
            source_payload["source_rank"] = rank
            source_payload["weighted_score"] = score
            entry["sources"].append(source_payload)
            score_key = "feedback_score" if source_name == "feedback" else "probe_gate_score"
            entry[score_key] = float(entry[score_key]) + float(candidate.get("score", 0.0))
            if float(candidate.get("score", 0.0)) > float(entry["best_source_score"]):
                entry["best_source"] = source_name
                entry["best_source_rank"] = rank
                entry["best_source_score"] = float(candidate.get("score", 0.0))

    ranked = sorted(
        merged.values(),
        key=lambda row: (
            float(row["score"]),
            float(row["probe_gate_score"]),
            float(row["feedback_score"]),
            -int(row["order"]),
            str(row["label"]),
        ),
        reverse=True,
    )
    return ranked[: max(int(add_terms), 0)]


def build_fixed_probe_residual_labels(
    *,
    h0,
    h1,
    feedback_residual_labels: list[str],
    extra_excluded_labels: list[str] | None = None,
    probe_agp_terms: int,
    probe_residual_terms: int,
    intermediate_top_k: int,
    probe_name: str,
) -> tuple[list[str], dict[str, object]]:
    """Build a fixed residual probe disjoint from the feedback/training pool."""

    excluded = set(feedback_residual_labels)
    if extra_excluded_labels is not None:
        excluded.update(str(label) for label in extra_excluded_labels)
    requested = max(int(probe_residual_terms) + len(excluded), int(probe_residual_terms))
    support: dict[str, object] | None = None
    filtered: list[str] = []
    for _ in range(6):
        support = build_projected_support(
            h0,
            h1,
            agp_top_k=int(probe_agp_terms),
            intermediate_top_k=int(intermediate_top_k),
            residual_top_k=requested,
            stage=0,
        )
        filtered = [str(label) for label in support["residual_labels"] if str(label) not in excluded]
        if len(filtered) >= int(probe_residual_terms):
            break
        requested *= 2
    if support is None:
        raise RuntimeError("Could not build a frozen probe residual support.")
    labels = sort_pauli_labels(filtered[: int(probe_residual_terms)])
    metadata = dict(support["metadata"])
    metadata.update(
        {
            "probe_name": probe_name,
            "probe_residual_terms_requested": int(probe_residual_terms),
            "probe_residual_terms": len(labels),
            "probe_agp_terms": int(probe_agp_terms),
            "excluded_feedback_residual_terms": len(excluded),
            "selection_rule": (
                "Fixed disjoint residual probe generated once from an enlarged endpoint-commutator "
                "AGP support. Probe labels are never added to the training residual basis. "
                "The gate probe may accept or reject curriculum steps; the test probe is only reported."
            ),
        }
    )
    return labels, metadata


def _final_output_modules(model: torch.nn.Module) -> list[torch.nn.Linear]:
    if hasattr(model.body, "layers"):
        final_layer = model.body.layers[-1]
        modules = [
            getattr(final_layer, branch)
            for branch in ("linear", "quad_left", "quad_right")
            if hasattr(final_layer, branch)
        ]
        return [module for module in modules if isinstance(module, torch.nn.Linear)]
    if hasattr(model.body, "network"):
        return [module for module in model.body.network if isinstance(module, torch.nn.Linear)][-1:]
    return []


def _install_new_row_training_hooks(
    model: torch.nn.Module,
    *,
    new_agp_labels: list[str],
) -> tuple[list[torch.utils.hooks.RemovableHandle], dict[str, bool]]:
    """Freeze hidden parameters and allow warm-up gradients only on new output rows."""

    new_rows = [idx for idx, label in enumerate(model.agp_labels) if label in set(new_agp_labels)]
    if not new_rows:
        return [], {}
    previous_requires_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    handles: list[torch.utils.hooks.RemovableHandle] = []
    for module in _final_output_modules(model):
        for parameter in (module.weight, module.bias):
            parameter.requires_grad_(True)
            mask = torch.zeros(parameter.shape[0], dtype=parameter.dtype, device=parameter.device)
            mask[new_rows] = 1.0
            view_shape = (parameter.shape[0],) + (1,) * (parameter.ndim - 1)
            handles.append(parameter.register_hook(lambda grad, m=mask.view(view_shape): grad * m))
    return handles, previous_requires_grad


def _restore_requires_grad(
    model: torch.nn.Module,
    previous_requires_grad: dict[str, bool],
    handles: list[torch.utils.hooks.RemovableHandle],
) -> None:
    for handle in handles:
        handle.remove()
    if previous_requires_grad:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(previous_requires_grad.get(name, True))
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)


def write_probe_spectrum(
    data_dir: Path,
    *,
    round_index: int,
    row: dict[str, object],
    spectrum: list[dict[str, object]],
    probe_name: str,
) -> str:
    path = data_dir / (
        f"{probe_name}_spectrum_round_{round_index:02d}_"
        f"agp_{row['agp_terms']}_residual_{row['holdout_residual_terms']}.json"
    )
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "feedback_round": round_index,
                "probe_name": probe_name,
                "agp_terms": row["agp_terms"],
                "probe_residual_terms": row["holdout_residual_terms"],
                "spectrum": spectrum,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    return str(path)


def floored_relative_residual(*, total: float, reference: float, floor: float) -> float:
    return float(total) / max(float(reference), float(floor))


def probe_row_for_export(row: dict[str, object], *, prefix: str) -> dict[str, object]:
    return {
        "agp_terms": row["agp_terms"],
        "holdout_residual_terms": row[f"{prefix}_residual_terms"],
    }


def attach_probe_metrics(
    row: dict[str, object],
    *,
    probe_gate_row: dict[str, object],
    probe_test_row: dict[str, object],
    probe_gate_terms: int,
    probe_test_terms: int,
    reference_floor: float,
) -> None:
    for prefix, probe_row, terms in (
        ("probe_gate", probe_gate_row, probe_gate_terms),
        ("probe_test", probe_test_row, probe_test_terms),
    ):
        row[f"{prefix}_residual_terms"] = int(terms)
        row[f"{prefix}_total_residual"] = probe_row["holdout_total_residual"]
        row[f"{prefix}_reference_residual"] = probe_row["holdout_reference_residual"]
        row[f"{prefix}_relative_residual"] = probe_row["holdout_relative_residual"]
        row[f"{prefix}_relative_residual_floored"] = floored_relative_residual(
            total=float(probe_row["holdout_total_residual"]),
            reference=float(probe_row["holdout_reference_residual"]),
            floor=reference_floor,
        )
        row[f"{prefix}_unseen_relative_residual"] = probe_row["unseen_relative_residual"]

    # Backward-compatible aliases for older summary readers and plots.
    row["frozen_probe_residual_terms"] = row["probe_gate_residual_terms"]
    row["frozen_probe_total_residual"] = row["probe_gate_total_residual"]
    row["frozen_probe_reference_residual"] = row["probe_gate_reference_residual"]
    row["frozen_probe_relative_residual"] = row["probe_gate_relative_residual"]
    row["frozen_probe_unseen_relative_residual"] = row["probe_gate_unseen_relative_residual"]


def step_gate_decision(
    *,
    previous_feedback_row: dict[str, object],
    candidate_feedback_row: dict[str, object],
    residual_candidate_count: int,
    agp_candidate_count: int,
    attempt_kind: str,
    probe_max_worsening_factor: float,
    probe_max_worsening_delta: float,
    probe_absolute_max_worsening_factor: float,
    probe_absolute_max_worsening_delta: float,
    feedback_max_worsening_factor: float,
    reference_floor: float,
) -> dict[str, object]:
    candidate_count = int(residual_candidate_count) + int(agp_candidate_count)
    if candidate_count <= 0:
        return {
            "accepted": True,
            "status": "no_support_candidates",
            "candidate_count": 0,
            "residual_candidate_count": 0,
            "agp_candidate_count": 0,
            "attempt_kind": attempt_kind,
            "reason": "No support changes were proposed.",
        }

    previous_probe = floored_relative_residual(
        total=float(previous_feedback_row["probe_gate_total_residual"]),
        reference=float(previous_feedback_row["probe_gate_reference_residual"]),
        floor=reference_floor,
    )
    candidate_probe = floored_relative_residual(
        total=float(candidate_feedback_row["probe_gate_total_residual"]),
        reference=float(candidate_feedback_row["probe_gate_reference_residual"]),
        floor=reference_floor,
    )
    previous_probe_abs = float(previous_feedback_row["probe_gate_total_residual"])
    candidate_probe_abs = float(candidate_feedback_row["probe_gate_total_residual"])
    previous_feedback = float(previous_feedback_row["holdout_relative_residual"])
    candidate_feedback = float(candidate_feedback_row["holdout_relative_residual"])
    probe_limit = previous_probe * float(probe_max_worsening_factor) + float(probe_max_worsening_delta)
    probe_abs_limit = previous_probe_abs * float(probe_absolute_max_worsening_factor) + float(
        probe_absolute_max_worsening_delta
    )
    feedback_limit = previous_feedback * float(feedback_max_worsening_factor)
    probe_pass = candidate_probe <= probe_limit
    probe_abs_pass = candidate_probe_abs <= probe_abs_limit
    feedback_pass = candidate_feedback <= feedback_limit
    return {
        "accepted": bool(probe_pass and probe_abs_pass and feedback_pass),
        "status": (
            "accepted_step_gate"
            if probe_pass and probe_abs_pass and feedback_pass
            else "rejected_step_gate"
        ),
        "candidate_count": int(candidate_count),
        "residual_candidate_count": int(residual_candidate_count),
        "agp_candidate_count": int(agp_candidate_count),
        "attempt_kind": attempt_kind,
        "previous_probe_relative_residual": previous_probe,
        "candidate_probe_relative_residual": candidate_probe,
        "probe_acceptance_limit": probe_limit,
        "probe_pass": bool(probe_pass),
        "previous_probe_total_residual": previous_probe_abs,
        "candidate_probe_total_residual": candidate_probe_abs,
        "probe_total_acceptance_limit": probe_abs_limit,
        "probe_total_pass": bool(probe_abs_pass),
        "probe_reference_floor": float(reference_floor),
        "previous_feedback_relative_residual": previous_feedback,
        "candidate_feedback_relative_residual": candidate_feedback,
        "feedback_acceptance_limit": feedback_limit,
        "feedback_pass": bool(feedback_pass),
        "reason": (
            "Candidate support step accepted because feedback and probe-gate residuals "
            "remain within configured worsening tolerances."
            if probe_pass and probe_abs_pass and feedback_pass
            else "Candidate support step rejected because it worsens feedback or probe-gate residuals."
        ),
    }


def transfer_body_state_to_model(
    model: torch.nn.Module,
    body_state: dict[str, torch.Tensor],
    *,
    old_agp_labels: list[str],
    new_agp_labels: list[str],
) -> None:
    """Load compatible hidden weights and map old output rows into an expanded AGP head."""

    state = model.body.state_dict()
    for key, value in body_state.items():
        if key in state and state[key].shape == value.shape:
            state[key].copy_(value.to(state[key].device))
    model.body.load_state_dict(state)

    old_index = {label: idx for idx, label in enumerate(old_agp_labels)}
    row_pairs = [(new_idx, old_index[label]) for new_idx, label in enumerate(new_agp_labels) if label in old_index]
    new_only = [new_idx for new_idx, label in enumerate(new_agp_labels) if label not in old_index]
    if not row_pairs and not new_only:
        return

    with torch.no_grad():
        if hasattr(model.body, "layers"):
            final_prefix = f"layers.{len(model.body.layers) - 1}"
            branches = ("linear", "quad_left", "quad_right")
            for branch in branches:
                weight_key = f"{final_prefix}.{branch}.weight"
                bias_key = f"{final_prefix}.{branch}.bias"
                if weight_key not in state or weight_key not in body_state:
                    continue
                weight = state[weight_key]
                bias = state[bias_key]
                old_weight = body_state[weight_key].to(weight.device)
                old_bias = body_state[bias_key].to(bias.device)
                for new_idx, old_idx in row_pairs:
                    weight[new_idx].copy_(old_weight[old_idx])
                    bias[new_idx].copy_(old_bias[old_idx])
                if branch == "linear":
                    for new_idx in new_only:
                        weight[new_idx].zero_()
                        bias[new_idx].zero_()
            model.body.load_state_dict(state)
            return

        if hasattr(model.body, "network"):
            linear_indices = [
                idx
                for idx, module in enumerate(model.body.network)
                if isinstance(module, torch.nn.Linear)
            ]
            if not linear_indices:
                return
            final_idx = linear_indices[-1]
            weight_key = f"network.{final_idx}.weight"
            bias_key = f"network.{final_idx}.bias"
            if weight_key not in state or weight_key not in body_state:
                return
            weight = state[weight_key]
            bias = state[bias_key]
            old_weight = body_state[weight_key].to(weight.device)
            old_bias = body_state[bias_key].to(bias.device)
            for new_idx, old_idx in row_pairs:
                weight[new_idx].copy_(old_weight[old_idx])
                bias[new_idx].copy_(old_bias[old_idx])
            for new_idx in new_only:
                weight[new_idx].zero_()
                bias[new_idx].zero_()
            model.body.load_state_dict(state)


def agp_index_pairs(current_labels: list[str], reference_labels: list[str]) -> tuple[list[int], list[int]]:
    current_index = {label: idx for idx, label in enumerate(current_labels)}
    reference_index = {label: idx for idx, label in enumerate(reference_labels)}
    common = [label for label in reference_labels if label in current_index]
    return [current_index[label] for label in common], [reference_index[label] for label in common]


def make_trust_region_reference(
    *,
    h0,
    h1,
    settings: ProjectedRunSettings,
    config,
    previous_agp_labels: list[str],
    residual_labels: list[str],
    body_state: dict[str, torch.Tensor],
    device: torch.device,
    round_index: int,
) -> torch.nn.Module:
    support = make_support_with_residual_labels(
        h0=h0,
        h1=h1,
        settings=settings,
        agp_labels=previous_agp_labels,
        residual_labels=residual_labels,
        stage=round_index,
    )
    reference_model = make_projected_model(h0, h1, support, config, device)
    reference_model.body.load_state_dict({key: value.to(device) for key, value in body_state.items()})
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    return reference_model


def train_stage_with_trust_region(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_weights: ProjectedSparseLossWeights,
    t: torch.Tensor,
    *,
    stage: int,
    epochs: int,
    global_epoch: int,
    history: list[dict[str, float]],
    reference_model: torch.nn.Module | None,
    current_reference_indices: list[int],
    previous_reference_indices: list[int],
    trust_region_weight: float,
) -> int:
    current_index_tensor: torch.Tensor | None = None
    previous_index_tensor: torch.Tensor | None = None
    if (
        reference_model is not None
        and trust_region_weight > 0.0
        and current_reference_indices
        and previous_reference_indices
    ):
        current_index_tensor = torch.tensor(current_reference_indices, dtype=torch.long, device=t.device)
        previous_index_tensor = torch.tensor(previous_reference_indices, dtype=torch.long, device=t.device)

    for local_epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = model.loss(t, weights=loss_weights)
        diagnostics = dict(diagnostics)
        if current_index_tensor is not None and previous_index_tensor is not None and reference_model is not None:
            current_agp = model(t)["agp_coefficients"].index_select(-1, current_index_tensor)
            with torch.no_grad():
                reference_agp = reference_model(t)["agp_coefficients"].index_select(-1, previous_index_tensor)
            trust_loss = torch.mean((current_agp - reference_agp.to(current_agp.device)) ** 2)
            loss = loss + float(trust_region_weight) * trust_loss
            diagnostics["trust_region"] = trust_loss
            diagnostics["trust_region_weight"] = torch.as_tensor(float(trust_region_weight), device=t.device)
            diagnostics["total"] = loss
        loss.backward()
        optimizer.step()
        row = {"epoch": float(global_epoch), "stage": float(stage), "stage_epoch": float(local_epoch)}
        row.update({key: float(value.detach().cpu().item()) for key, value in diagnostics.items()})
        history.append(row)
        if global_epoch == 0 or local_epoch == epochs - 1:
            extra = f" trust={row['trust_region']:.6e}" if "trust_region" in row else ""
            print(
                f"stage={stage:02d} epoch={global_epoch:04d} loss={row['total']:.6e} "
                f"residual={row['residual']:.6e} agp_terms={int(row['agp_terms'])} "
                f"residual_terms={int(row['residual_terms'])}{extra}"
            )
        global_epoch += 1
    return global_epoch


def train_coupled_round(
    *,
    run_dir: Path,
    payload: dict[str, object],
    settings: ProjectedRunSettings,
    agp_labels: list[str],
    previous_agp_labels: list[str],
    residual_labels: list[str],
    body_state: dict[str, torch.Tensor],
    round_index: int,
    residual_additions: list[dict[str, object]],
    agp_additions: list[dict[str, object]],
    warmup_epochs: int,
    trust_region_weight: float,
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
    transfer_body_state_to_model(
        model,
        {key: value.to(device) for key, value in body_state.items()},
        old_agp_labels=previous_agp_labels,
        new_agp_labels=model.agp_labels,
    )
    reference_model: torch.nn.Module | None = None
    current_reference_indices: list[int] = []
    previous_reference_indices: list[int] = []
    if float(trust_region_weight) > 0.0 and previous_agp_labels:
        reference_model = make_trust_region_reference(
            h0=h0,
            h1=h1,
            settings=settings,
            config=config,
            previous_agp_labels=previous_agp_labels,
            residual_labels=residual_labels,
            body_state=body_state,
            device=device,
            round_index=round_index,
        )
        current_reference_indices, previous_reference_indices = agp_index_pairs(
            model.agp_labels,
            reference_model.agp_labels,
        )

    loss_weights = ProjectedSparseLossWeights(residual=settings.residual_weight, agp_l2=settings.agp_l2_weight)
    tau = torch.linspace(0.0, 1.0, settings.num_points, device=device).view(-1, 1)
    t = config.t_initial + config.physical_time * tau
    history: list[dict[str, float]] = []
    optimizer_info: dict[str, object]
    optimizer_stages: list[dict[str, object]] = []
    global_epoch = 0
    new_agp_labels = [str(row["label"]) for row in agp_additions]
    warmup_epochs = min(max(int(warmup_epochs), 0), max(int(settings.epochs) - 1, 0))
    if warmup_epochs > 0 and new_agp_labels:
        handles, previous_requires_grad = _install_new_row_training_hooks(model, new_agp_labels=new_agp_labels)
        warmup_optimizer, warmup_optimizer_info = make_optimizer(model, settings)
        warmup_optimizer_info = dict(warmup_optimizer_info)
        warmup_optimizer_info["stage"] = "new_agp_row_warmup"
        warmup_optimizer_info["epochs"] = warmup_epochs
        warmup_optimizer_info["trainable_new_agp_terms"] = len(new_agp_labels)
        optimizer_stages.append(warmup_optimizer_info)
        try:
            global_epoch = train_stage_with_trust_region(
                model,
                warmup_optimizer,
                loss_weights,
                t,
                stage=round_index,
                epochs=warmup_epochs,
                global_epoch=global_epoch,
                history=history,
                reference_model=reference_model,
                current_reference_indices=current_reference_indices,
                previous_reference_indices=previous_reference_indices,
                trust_region_weight=trust_region_weight,
            )
        finally:
            _restore_requires_grad(model, previous_requires_grad, handles)

    optimizer, optimizer_info = make_optimizer(model, settings)
    optimizer_info = dict(optimizer_info)
    optimizer_info["stage"] = "full_model_finetune"
    optimizer_info["epochs"] = int(settings.epochs) - global_epoch
    optimizer_info["trust_region_weight"] = float(trust_region_weight)
    optimizer_stages.append(optimizer_info)
    train_stage_with_trust_region(
        model,
        optimizer,
        loss_weights,
        t,
        stage=round_index,
        epochs=int(settings.epochs) - global_epoch,
        global_epoch=global_epoch,
        history=history,
        reference_model=reference_model,
        current_reference_indices=current_reference_indices,
        previous_reference_indices=previous_reference_indices,
        trust_region_weight=trust_region_weight,
    )

    metadata = dict(support["metadata"])
    metadata["n_qubits"] = config.n_qubits
    metadata["device"] = str(device)
    metadata["full_pauli_basis_size"] = 4**config.n_qubits
    metadata["regime"] = "coupled_residual_agp_curriculum"
    metadata["curriculum_round"] = round_index
    metadata["residual_added_terms"] = residual_additions
    metadata["residual_added_term_count"] = len(residual_additions)
    metadata["agp_added_terms"] = agp_additions
    metadata["agp_added_term_count"] = len(agp_additions)
    metadata["agp_terms_before_growth"] = len(previous_agp_labels)
    metadata["trust_region_weight"] = float(trust_region_weight)
    metadata["trust_region_common_agp_terms"] = len(current_reference_indices)
    metadata["adaptive_enabled"] = True
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
    with (data_dir / "coupled_added_residual_terms.json").open("w", encoding="utf-8") as handle:
        json.dump(residual_additions, handle, indent=2)
        handle.write("\n")
    with (data_dir / "coupled_added_agp_terms.json").open("w", encoding="utf-8") as handle:
        json.dump(agp_additions, handle, indent=2)
        handle.write("\n")

    run_metadata = {
        "physical": asdict(config),
        "training": asdict(settings),
        "support": metadata,
        "optimizer": optimizer_info,
        "optimizer_stages": optimizer_stages,
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
    export_results(model, tau, t, images_dir, data_dir, metadata, history, top_k=settings.top_coefficients)
    next_body_state = {key: value.detach().cpu() for key, value in model.body.state_dict().items()}
    return next_body_state, history[-1], metadata


def plot_coupled_support_growth(rows: list[dict[str, object]], round_rows: list[dict[str, object]], images_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    set_paper_style(plt)
    x = np.asarray([int(row["feedback_round"]) for row in rows], dtype=float)
    agp_terms = np.asarray([int(row["agp_terms"]) for row in rows], dtype=float)
    residual_terms = np.asarray([int(row["train_residual_terms"]) for row in rows], dtype=float)
    round_x = np.asarray([int(row["round"]) for row in round_rows], dtype=float)
    added_agp = np.asarray([int(row["added_agp_terms"]) for row in round_rows], dtype=float)
    added_residual = np.asarray([int(row["added_residual_terms"]) for row in round_rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    axes[0].plot(x, agp_terms, marker="o", linewidth=LINE_WIDTH, color=OKABE_ITO[0], label=r"$K$")
    axes[0].plot(x, residual_terms, marker="s", linewidth=LINE_WIDTH, color=OKABE_ITO[1], label=r"$R_{\mathrm{train}}$")
    axes[0].set_xlabel("curriculum round", fontsize=LABEL_FS)
    axes[0].set_ylabel("Pauli strings", fontsize=LABEL_FS)
    axes[0].set_title("support growth", fontsize=TITLE_FS)
    axes[0].legend(frameon=False, fontsize=LEGEND_FS)
    axes[0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    width = 0.32
    axes[1].bar(round_x - width / 2.0, added_agp, width=width, color=OKABE_ITO[0], label="AGP")
    axes[1].bar(round_x + width / 2.0, added_residual, width=width, color=OKABE_ITO[1], label="residual")
    axes[1].set_xlabel("curriculum round", fontsize=LABEL_FS)
    axes[1].set_title("terms added per round", fontsize=TITLE_FS)
    axes[1].legend(frameon=False, fontsize=LEGEND_FS)
    axes[1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    for ax in axes:
        ax.set_xticks(x)
        ax.tick_params(axis="both", labelsize=TICK_FS, length=TICK_LENGTH, width=TICK_WIDTH)
    fig.subplots_adjust(top=0.84, left=0.10, right=0.98, bottom=0.18, wspace=0.34)
    fig.savefig(images_dir / "coupled_curriculum_support_growth.pdf", format="pdf")
    plt.close(fig)


def plot_residuals_vs_agp_terms(rows: list[dict[str, object]], images_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_paper_style(plt)
    k_values = np.asarray([int(row["agp_terms"]) for row in rows], dtype=float)
    series = [
        ("training", [float(row["training_final_relative_residual"]) for row in rows], OKABE_ITO[0], "o"),
        ("holdout", [float(row["holdout_relative_residual"]) for row in rows], OKABE_ITO[1], "s"),
        (
            "unseen",
            [
                np.nan
                if int(row.get("unseen_residual_terms", 1)) == 0
                else float(row["unseen_relative_residual"])
                for row in rows
            ],
            OKABE_ITO[2],
            "^",
        ),
    ]
    if rows and "probe_gate_relative_residual" in rows[0]:
        series.append(
            (
                "probe gate",
                [float(row["probe_gate_relative_residual"]) for row in rows],
                OKABE_ITO[3],
                "D",
            )
        )
    if rows and "probe_test_relative_residual" in rows[0]:
        series.append(
            (
                "probe test",
                [float(row["probe_test_relative_residual"]) for row in rows],
                OKABE_ITO[4],
                "v",
            )
        )
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    for label, values, color, marker in series:
        ax.semilogy(k_values, values, marker=marker, linewidth=LINE_WIDTH, color=color, label=label)
    ax.set_xlabel(r"AGP support size $K$", fontsize=LABEL_FS)
    ax.set_ylabel("relative residual", fontsize=LABEL_FS)
    ax.set_title("residual decrease with AGP growth", fontsize=TITLE_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS, length=TICK_LENGTH, width=TICK_WIDTH)
    fig.legend(loc="upper center", ncol=min(len(series), 5), frameon=False, fontsize=LEGEND_FS, bbox_to_anchor=(0.53, 1.02))
    fig.subplots_adjust(top=0.80, left=0.13, right=0.98, bottom=0.16)
    fig.savefig(images_dir / "coupled_curriculum_residuals_vs_agp_terms.pdf", format="pdf")
    plt.close(fig)


def plot_probe_gate(rows: list[dict[str, object]], round_rows: list[dict[str, object]], images_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_paper_style(plt)
    x = np.asarray([int(row["feedback_round"]) for row in rows], dtype=float)
    probe_gate = np.asarray([float(row["probe_gate_relative_residual"]) for row in rows], dtype=float)
    probe_test = np.asarray([float(row["probe_test_relative_residual"]) for row in rows], dtype=float)
    feedback = np.asarray([float(row["holdout_relative_residual"]) for row in rows], dtype=float)
    round_x = np.asarray([int(row["round"]) for row in round_rows], dtype=float)
    accepted = np.asarray([1.0 if bool(row.get("agp_growth_accepted", False)) else 0.0 for row in round_rows])
    added_agp = np.asarray([int(row["added_agp_terms"]) for row in round_rows], dtype=float)
    added_residual = np.asarray([int(row["added_residual_terms"]) for row in round_rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.3))
    axes[0].semilogy(x, feedback, marker="s", linewidth=LINE_WIDTH, color=OKABE_ITO[1], label="feedback")
    axes[0].semilogy(x, probe_gate, marker="D", linewidth=LINE_WIDTH, color=OKABE_ITO[3], label="probe gate")
    axes[0].semilogy(x, probe_test, marker="v", linewidth=LINE_WIDTH, color=OKABE_ITO[4], label="probe test")
    axes[0].set_xlabel("curriculum round", fontsize=LABEL_FS)
    axes[0].set_ylabel("relative residual", fontsize=LABEL_FS)
    axes[0].set_title("probe-gated residuals", fontsize=TITLE_FS)
    axes[0].legend(frameon=False, fontsize=LEGEND_FS)

    colors = [OKABE_ITO[2] if value > 0.5 else OKABE_ITO[1] for value in accepted]
    width = 0.34
    axes[1].bar(round_x - width / 2.0, added_agp, width=width, color=colors, label="AGP")
    axes[1].bar(round_x + width / 2.0, added_residual, width=width, color=OKABE_ITO[0], label="residual")
    axes[1].set_xlabel("curriculum round", fontsize=LABEL_FS)
    axes[1].set_ylabel("accepted terms", fontsize=LABEL_FS)
    axes[1].set_title("step gate decisions", fontsize=TITLE_FS)
    axes[1].set_xticks(round_x)
    axes[1].legend(frameon=False, fontsize=LEGEND_FS)
    for ax in axes:
        ax.tick_params(axis="both", labelsize=TICK_FS, length=TICK_LENGTH, width=TICK_WIDTH)
    fig.subplots_adjust(top=0.84, left=0.10, right=0.98, bottom=0.18, wspace=0.35)
    fig.savefig(images_dir / "coupled_curriculum_probe_gate.pdf", format="pdf")
    plt.close(fig)


def write_coupled_summary(
    *,
    output_dir: Path,
    rows: list[dict[str, object]],
    spectra: dict[int, list[dict[str, object]]],
    probe_gate_spectra: dict[int, list[dict[str, object]]],
    probe_test_spectra: dict[int, list[dict[str, object]]],
    round_rows: list[dict[str, object]],
    residual_top_k: int,
    thresholds: Thresholds,
    residual_budget: dict[str, object],
    agp_growth_config: dict[str, object],
    probe_config: dict[str, object],
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
        and float(row["probe_gate_relative_residual"]) <= thresholds.unseen
        and float(row["probe_test_relative_residual"]) <= thresholds.unseen
        and int(row.get("unseen_residual_terms", 0)) > 0
    ]
    decision = {
        "status": "found_coupled_round" if accepted else "not_found_in_coupled_run",
        "round": int(accepted[0]["feedback_round"]) if accepted else None,
        "thresholds": {
            "holdout_relative_residual_max": thresholds.holdout,
            "unseen_relative_residual_max": thresholds.unseen,
            "probe_gate_relative_residual_max": thresholds.unseen,
            "probe_test_relative_residual_max": thresholds.unseen,
        },
    }
    if accepted:
        decision["conclusion"] = (
            f"Coupled round {int(accepted[0]['feedback_round'])} passes feedback holdout, "
            "feedback unseen, probe-gate, and probe-test thresholds."
        )
    else:
        decision["conclusion"] = "No coupled round passes feedback holdout, feedback unseen, probe-gate, and probe-test thresholds."

    payload = {
        "description": (
            "Coupled curriculum: high-RMS holdout residual strings expand the training residual basis, "
            "and inverse-commutator scoring proposes new AGP Pauli strings. Step-level gates "
            "may reject or backtrack support growth when a fixed probe-gate basis worsens."
        ),
        "holdout_residual_terms": residual_top_k,
        "residual_budget": residual_budget,
        "agp_growth": agp_growth_config,
        "probe": probe_config,
        "decision": decision,
        "rounds": round_rows,
        "rows": rows,
        "probe_gate_spectrum_rounds": sorted(probe_gate_spectra),
        "probe_test_spectrum_rounds": sorted(probe_test_spectra),
    }
    with (data_dir / f"coupled_curriculum_summary_residual_{residual_top_k}.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    plot_feedback_relative_residuals(rows, images_dir, thresholds)
    plot_feedback_seen_unseen(rows, images_dir)
    plot_feedback_residual_spectrum(rows, spectra, images_dir)
    plot_feedback_added_terms(round_rows, images_dir)
    plot_coupled_support_growth(rows, round_rows, images_dir)
    plot_residuals_vs_agp_terms(rows, images_dir)
    plot_probe_gate(rows, round_rows, images_dir)
    if round_rows:
        final_round_dir = output_dir / str(round_rows[-1]["run_dir"])
        for filename in ("hcd_coefficient_support_map.pdf", "hcd_connection_summary.pdf"):
            source = final_round_dir / "Images" / filename
            if source.is_file():
                shutil.copy2(source, images_dir / filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train q=20 with coupled residual and AGP-support curricula.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-agp-terms", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--add-residual-terms", type=int, default=None)
    parser.add_argument("--add-agp-terms", type=int, default=None)
    parser.add_argument("--max-agp-terms", type=int, default=None)
    parser.add_argument("--epochs-per-round", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--residual-top-k", default=None)
    parser.add_argument("--intermediate-top-k", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--min-residual-rms", type=float, default=None)
    parser.add_argument("--min-agp-score", type=float, default=None)
    parser.add_argument("--candidate-residual-terms", type=int, default=None)
    parser.add_argument("--candidate-hamiltonian-terms", type=int, default=None)
    parser.add_argument("--include-seen-residuals-for-agp", action="store_true")
    parser.add_argument("--probe-residual-terms", type=int, default=None)
    parser.add_argument("--probe-test-residual-terms", type=int, default=None)
    parser.add_argument("--probe-source-agp-terms", type=int, default=None)
    parser.add_argument("--proposal-multiplier", type=int, default=None)
    parser.add_argument("--probe-score-weight", type=float, default=None)
    parser.add_argument("--probe-max-worsening-factor", type=float, default=None)
    parser.add_argument("--probe-max-worsening-delta", type=float, default=None)
    parser.add_argument("--probe-absolute-max-worsening-factor", type=float, default=None)
    parser.add_argument("--probe-absolute-max-worsening-delta", type=float, default=None)
    parser.add_argument("--probe-reference-floor", type=float, default=None)
    parser.add_argument("--feedback-max-worsening-factor", type=float, default=None)
    parser.add_argument("--new-agp-warmup-epochs", type=int, default=None)
    parser.add_argument("--trust-region-weight", type=float, default=None)
    parser.add_argument("--unseen-residual-batches", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--holdout-threshold", type=float, default=None)
    parser.add_argument("--unseen-threshold", type=float, default=None)
    args = parser.parse_args()

    payload = load_json(args.config)
    if not isinstance(payload, dict):
        raise TypeError("config.json must contain a JSON object.")
    coupled = payload.get("coupled_curriculum", {})
    coupled = coupled if isinstance(coupled, dict) else {}
    base_agp_terms = int(args.base_agp_terms if args.base_agp_terms is not None else coupled.get("base_agp_terms", 1024))
    rounds = int(args.rounds if args.rounds is not None else coupled.get("iterations", 10))
    add_residual_terms = int(
        args.add_residual_terms
        if args.add_residual_terms is not None
        else coupled.get("add_residual_terms_per_iteration", 1024)
    )
    add_agp_terms = int(
        args.add_agp_terms
        if args.add_agp_terms is not None
        else coupled.get("add_agp_terms_per_iteration", 128)
    )
    max_agp_raw = args.max_agp_terms if args.max_agp_terms is not None else coupled.get("max_agp_terms", None)
    max_agp_terms = int(max_agp_raw) if max_agp_raw is not None else None
    epochs_per_round = int(
        args.epochs_per_round
        if args.epochs_per_round is not None
        else coupled.get("epochs_per_iteration", 1000)
    )
    residual_top_k_request = args.residual_top_k if args.residual_top_k is not None else coupled.get("holdout_residual_top_k", "auto")
    unseen_residual_batches = int(
        args.unseen_residual_batches
        if args.unseen_residual_batches is not None
        else coupled.get("unseen_residual_batches_after_final_iteration", 1)
    )
    lr = float(args.lr if args.lr is not None else coupled.get("lr", 1e-5))
    device_name = str(args.device if args.device is not None else coupled.get("device", "auto"))
    min_residual_rms = float(
        args.min_residual_rms if args.min_residual_rms is not None else coupled.get("min_residual_rms", 0.0)
    )
    min_agp_score = float(args.min_agp_score if args.min_agp_score is not None else coupled.get("min_agp_score", 0.0))
    candidate_residual_terms = int(
        args.candidate_residual_terms
        if args.candidate_residual_terms is not None
        else coupled.get("candidate_residual_terms", 512)
    )
    candidate_hamiltonian_terms = int(
        args.candidate_hamiltonian_terms
        if args.candidate_hamiltonian_terms is not None
        else coupled.get("candidate_hamiltonian_terms", 64)
    )
    probe_gate_residual_terms = int(
        args.probe_residual_terms
        if args.probe_residual_terms is not None
        else coupled.get("probe_gate_residual_terms", coupled.get("probe_residual_terms", 4096))
    )
    probe_test_residual_terms = int(
        args.probe_test_residual_terms
        if args.probe_test_residual_terms is not None
        else coupled.get("probe_test_residual_terms", probe_gate_residual_terms)
    )
    probe_source_agp_raw = (
        args.probe_source_agp_terms
        if args.probe_source_agp_terms is not None
        else coupled.get("probe_source_agp_terms", None)
    )
    proposal_multiplier = int(
        args.proposal_multiplier
        if args.proposal_multiplier is not None
        else coupled.get("proposal_multiplier", 4)
    )
    probe_score_weight = float(
        args.probe_score_weight
        if args.probe_score_weight is not None
        else coupled.get("probe_score_weight", 1.0)
    )
    probe_max_worsening_factor = float(
        args.probe_max_worsening_factor
        if args.probe_max_worsening_factor is not None
        else coupled.get("probe_max_worsening_factor", 1.05)
    )
    probe_max_worsening_delta = float(
        args.probe_max_worsening_delta
        if args.probe_max_worsening_delta is not None
        else coupled.get("probe_max_worsening_delta", 0.0)
    )
    probe_absolute_max_worsening_factor = float(
        args.probe_absolute_max_worsening_factor
        if args.probe_absolute_max_worsening_factor is not None
        else coupled.get("probe_absolute_max_worsening_factor", 1.10)
    )
    probe_absolute_max_worsening_delta = float(
        args.probe_absolute_max_worsening_delta
        if args.probe_absolute_max_worsening_delta is not None
        else coupled.get("probe_absolute_max_worsening_delta", 0.0)
    )
    probe_reference_floor = float(
        args.probe_reference_floor if args.probe_reference_floor is not None else coupled.get("probe_reference_floor", 1e-12)
    )
    feedback_max_worsening_factor = float(
        args.feedback_max_worsening_factor
        if args.feedback_max_worsening_factor is not None
        else coupled.get("feedback_max_worsening_factor", 1.10)
    )
    new_agp_warmup_epochs = int(
        args.new_agp_warmup_epochs
        if args.new_agp_warmup_epochs is not None
        else coupled.get("new_agp_warmup_epochs", 100)
    )
    trust_region_weight = float(
        args.trust_region_weight if args.trust_region_weight is not None else coupled.get("trust_region_weight", 0.0)
    )
    residual_backtracking_factors = coupled.get("residual_backtracking_factors", [1.0, 0.5, 0.25, 0.0])
    if not isinstance(residual_backtracking_factors, list):
        raise TypeError("coupled_curriculum.residual_backtracking_factors must be a list.")
    resolved_residual_backtracking_counts: list[int] = []
    for raw_factor in residual_backtracking_factors:
        factor = float(raw_factor)
        if factor <= 1.0:
            count = int(round(add_residual_terms * factor))
        else:
            count = int(round(factor))
        count = min(max(count, 0), add_residual_terms)
        if count not in resolved_residual_backtracking_counts:
            resolved_residual_backtracking_counts.append(count)
    if add_residual_terms not in resolved_residual_backtracking_counts:
        resolved_residual_backtracking_counts.insert(0, add_residual_terms)
    if 0 not in resolved_residual_backtracking_counts:
        resolved_residual_backtracking_counts.append(0)
    unseen_only = bool(coupled.get("use_unseen_residuals_for_agp", True)) and not args.include_seen_residuals_for_agp
    output_root_arg = args.output_root if args.output_root is not None else Path(str(coupled.get("output_root", "runs/coupled_curriculum")))
    holdout_threshold = float(
        args.holdout_threshold if args.holdout_threshold is not None else coupled.get("holdout_threshold", 0.10)
    )
    unseen_threshold = float(args.unseen_threshold if args.unseen_threshold is not None else coupled.get("unseen_threshold", 1.0))

    support = payload.get("support_sweep", {})
    intermediate_top_k = (
        int(args.intermediate_top_k)
        if args.intermediate_top_k is not None
        else int(support.get("intermediate_top_k", 2048))
        if isinstance(support, dict)
        else 2048
    )
    base_settings = settings_for_support(payload, base_agp_terms)
    coupled_settings = replace(
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
    previous_agp_labels = list(agp_labels)
    current_agp_labels = set(agp_labels)
    current_residual_labels = set(residual_labels)
    body_state = load_body_state_from_checkpoint(base_checkpoint)
    residual_top_k, residual_budget = resolve_holdout_residual_top_k(
        residual_top_k_request,
        initial_residual_terms=len(residual_labels),
        rounds=rounds,
        add_residual_terms=add_residual_terms,
        unseen_batches_after_final_iteration=unseen_residual_batches,
    )

    config = model_config_from_payload(payload)
    hamiltonian_path = Path(config.hamiltonian_source)
    if not hamiltonian_path.is_absolute():
        hamiltonian_path = ROOT / hamiltonian_path
    h0, h1 = load_pauli_hamiltonian_pair(
        hamiltonian_path,
        system=config.system,
        n_qubits=config.n_qubits,
        distance=config.distance,
    )

    if residual_top_k <= len(residual_labels):
        feedback_residual_labels = sort_pauli_labels(residual_labels)[:residual_top_k]
        holdout_basis_agp_terms = len(agp_labels)
    else:
        feedback_residual_labels, holdout_basis_agp_terms = build_common_holdout_residual_labels(
            run_dirs=[base_run],
            config_payload=payload,
            residual_top_k=residual_top_k,
            intermediate_top_k=intermediate_top_k,
        )
    if len(feedback_residual_labels) < residual_top_k:
        print(
            "resolved_coupled_residual_budget_clipped "
            f"requested={residual_top_k} available={len(feedback_residual_labels)}"
        )
        residual_top_k = len(feedback_residual_labels)
        residual_budget = dict(residual_budget)
        residual_budget["resolved_holdout_residual_top_k"] = residual_top_k
        residual_budget["available_generated_residual_terms"] = len(feedback_residual_labels)
        residual_budget["final_round_expected_unseen_terms"] = max(
            residual_top_k - int(residual_budget["minimum_budget_before_final_unseen_exhaustion"]),
            0,
        )
    probe_source_agp_terms = (
        int(probe_source_agp_raw)
        if probe_source_agp_raw is not None
        else max(max_agp_terms or base_agp_terms + rounds * add_agp_terms, base_agp_terms)
    )
    probe_gate_residual_labels, probe_gate_metadata = build_fixed_probe_residual_labels(
        h0=h0,
        h1=h1,
        feedback_residual_labels=feedback_residual_labels,
        probe_agp_terms=probe_source_agp_terms,
        probe_residual_terms=probe_gate_residual_terms,
        intermediate_top_k=intermediate_top_k,
        probe_name="probe_gate",
    )
    probe_test_residual_labels, probe_test_metadata = build_fixed_probe_residual_labels(
        h0=h0,
        h1=h1,
        feedback_residual_labels=feedback_residual_labels,
        extra_excluded_labels=probe_gate_residual_labels,
        probe_agp_terms=probe_source_agp_terms,
        probe_residual_terms=probe_test_residual_terms,
        intermediate_top_k=intermediate_top_k,
        probe_name="probe_test",
    )

    output_root = output_root_arg if output_root_arg.is_absolute() else RUN_DIR / output_root_arg
    output_dir = output_root / (
        f"base_agp_{base_agp_terms}_residual_{residual_top_k}_probeG_{len(probe_gate_residual_labels)}_"
        f"probeT_{len(probe_test_residual_labels)}_addR_{add_residual_terms}_addA_{add_agp_terms}_gated_rounds_{rounds}"
    )
    data_dir = output_dir / "Models_Data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "feedback_residual_labels.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "labels": feedback_residual_labels,
                "count": len(feedback_residual_labels),
                "holdout_basis_agp_terms": holdout_basis_agp_terms,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    with (data_dir / "probe_gate_residual_labels.json").open("w", encoding="utf-8") as handle:
        json.dump({"labels": probe_gate_residual_labels, "metadata": probe_gate_metadata}, handle, indent=2)
        handle.write("\n")
    with (data_dir / "probe_test_residual_labels.json").open("w", encoding="utf-8") as handle:
        json.dump({"labels": probe_test_residual_labels, "metadata": probe_test_metadata}, handle, indent=2)
        handle.write("\n")
    thresholds = Thresholds(plateau=1.0, holdout=holdout_threshold, unseen=unseen_threshold, top_stability=0.0, top_fraction=0.10)

    agp_growth_config = {
        "base_agp_terms": base_agp_terms,
        "add_agp_terms_per_iteration": add_agp_terms,
        "max_agp_terms": max_agp_terms,
        "proposal_multiplier": proposal_multiplier,
        "probe_score_weight": probe_score_weight,
        "probe_max_worsening_factor": probe_max_worsening_factor,
        "probe_max_worsening_delta": probe_max_worsening_delta,
        "probe_absolute_max_worsening_factor": probe_absolute_max_worsening_factor,
        "probe_absolute_max_worsening_delta": probe_absolute_max_worsening_delta,
        "probe_reference_floor": probe_reference_floor,
        "feedback_max_worsening_factor": feedback_max_worsening_factor,
        "new_agp_warmup_epochs": new_agp_warmup_epochs,
        "trust_region_weight": trust_region_weight,
        "residual_backtracking_counts": resolved_residual_backtracking_counts,
        "candidate_residual_terms": candidate_residual_terms,
        "candidate_hamiltonian_terms": candidate_hamiltonian_terms,
        "min_residual_rms": min_residual_rms,
        "min_agp_score": min_agp_score,
        "use_unseen_residuals_for_agp": unseen_only,
        "selection_rule": (
            "New AGP strings are scored by symbolic inverse commutator paths "
            "P -> [P, H] -> [[P, H], H] that can produce high-RMS holdout residual strings."
        ),
    }
    probe_config = {
        "probe_gate_residual_terms": len(probe_gate_residual_labels),
        "probe_gate_residual_terms_requested": probe_gate_residual_terms,
        "probe_test_residual_terms": len(probe_test_residual_labels),
        "probe_test_residual_terms_requested": probe_test_residual_terms,
        "probe_source_agp_terms": probe_source_agp_terms,
        "probe_disjoint_from_feedback": True,
        "probe_gate_labels_export": str((data_dir / "probe_gate_residual_labels.json").relative_to(output_dir)),
        "probe_test_labels_export": str((data_dir / "probe_test_residual_labels.json").relative_to(output_dir)),
        "feedback_labels_export": str((data_dir / "feedback_residual_labels.json").relative_to(output_dir)),
        "probe_gate_metadata": probe_gate_metadata,
        "probe_test_metadata": probe_test_metadata,
    }

    rows: list[dict[str, object]] = []
    spectra: dict[int, list[dict[str, object]]] = {}
    probe_gate_spectra: dict[int, list[dict[str, object]]] = {}
    probe_test_spectra: dict[int, list[dict[str, object]]] = {}
    round_rows: list[dict[str, object]] = []

    print(
        "resolved_coupled_curriculum "
        f"Q={residual_top_k} base_K={base_agp_terms} rounds={rounds} "
        f"addR={add_residual_terms} addA={add_agp_terms} maxK={max_agp_terms} "
        f"probe_gate={len(probe_gate_residual_labels)} probe_test={len(probe_test_residual_labels)}"
    )
    print(f"evaluate_coupled_baseline agp_terms={base_agp_terms}")
    baseline_row, baseline_spectrum = evaluate_one_run(
        run_dir=base_run,
        config_payload=payload,
        residual_top_k=residual_top_k,
        intermediate_top_k=intermediate_top_k,
        device=select_device("cpu"),
        spectra_dir=data_dir / "feedback_projection",
        common_residual_labels=feedback_residual_labels,
        holdout_basis_mode="fixed_feedback",
        holdout_basis_agp_terms=holdout_basis_agp_terms,
    )
    baseline_probe_gate_row, baseline_probe_gate_spectrum = evaluate_one_run(
        run_dir=base_run,
        config_payload=payload,
        residual_top_k=len(probe_gate_residual_labels),
        intermediate_top_k=intermediate_top_k,
        device=select_device("cpu"),
        spectra_dir=data_dir / "probe_gate_projection",
        common_residual_labels=probe_gate_residual_labels,
        holdout_basis_mode="probe_gate",
        holdout_basis_agp_terms=probe_source_agp_terms,
    )
    baseline_probe_test_row, baseline_probe_test_spectrum = evaluate_one_run(
        run_dir=base_run,
        config_payload=payload,
        residual_top_k=len(probe_test_residual_labels),
        intermediate_top_k=intermediate_top_k,
        device=select_device("cpu"),
        spectra_dir=data_dir / "probe_test_projection",
        common_residual_labels=probe_test_residual_labels,
        holdout_basis_mode="probe_test",
        holdout_basis_agp_terms=probe_source_agp_terms,
    )
    baseline_row["run_dir"] = str(base_run)
    baseline_row["feedback_round"] = 0
    attach_probe_metrics(
        baseline_row,
        probe_gate_row=baseline_probe_gate_row,
        probe_test_row=baseline_probe_test_row,
        probe_gate_terms=len(probe_gate_residual_labels),
        probe_test_terms=len(probe_test_residual_labels),
        reference_floor=probe_reference_floor,
    )
    rows.append(baseline_row)
    spectra[0] = baseline_spectrum
    probe_gate_spectra[0] = baseline_probe_gate_spectrum
    probe_test_spectra[0] = baseline_probe_test_spectrum
    baseline_row["spectrum_export"] = write_feedback_spectrum(
        data_dir,
        round_index=0,
        row=baseline_row,
        spectrum=baseline_spectrum,
    )
    baseline_row["probe_gate_spectrum_export"] = write_probe_spectrum(
        data_dir,
        round_index=0,
        row=baseline_probe_gate_row,
        spectrum=baseline_probe_gate_spectrum,
        probe_name="probe_gate",
    )
    baseline_row["probe_test_spectrum_export"] = write_probe_spectrum(
        data_dir,
        round_index=0,
        row=baseline_probe_test_row,
        spectrum=baseline_probe_test_spectrum,
        probe_name="probe_test",
    )

    for round_index in range(1, rounds + 1):
        previous_body_state = {key: value.clone() for key, value in body_state.items()}
        previous_agp_labels_for_round = list(previous_agp_labels)
        previous_feedback_row = rows[-1]
        selected_residual_additions = select_residual_additions(
            spectra[round_index - 1],
            current_residual_labels,
            add_terms=add_residual_terms,
            min_rms=min_residual_rms,
        )
        candidate_pool_size = max(add_agp_terms * max(proposal_multiplier, 1), add_agp_terms)
        feedback_agp_candidates = select_agp_additions_from_residual(
            spectra[round_index - 1],
            current_agp_labels,
            h0=h0,
            h1=h1,
            add_terms=candidate_pool_size,
            max_agp_terms=max_agp_terms,
            residual_candidate_terms=candidate_residual_terms,
            hamiltonian_candidate_terms=candidate_hamiltonian_terms,
            min_residual_rms=min_residual_rms,
            min_score=min_agp_score,
            unseen_only=unseen_only,
        )
        probe_agp_candidates = select_agp_additions_from_residual(
            probe_gate_spectra[round_index - 1],
            current_agp_labels,
            h0=h0,
            h1=h1,
            add_terms=candidate_pool_size,
            max_agp_terms=max_agp_terms,
            residual_candidate_terms=candidate_residual_terms,
            hamiltonian_candidate_terms=candidate_hamiltonian_terms,
            min_residual_rms=min_residual_rms,
            min_score=min_agp_score,
            unseen_only=False,
        )
        proposed_agp_additions = merge_agp_candidate_additions(
            feedback_candidates=feedback_agp_candidates,
            probe_candidates=probe_agp_candidates,
            current_agp_labels=current_agp_labels,
            add_terms=add_agp_terms,
            probe_score_weight=probe_score_weight,
        )
        round_run = output_dir / "runs" / f"round_{round_index:02d}"
        attempt_gates: list[dict[str, object]] = []
        accepted_payload: dict[str, object] | None = None
        attempt_index = 0
        for residual_count in resolved_residual_backtracking_counts:
            residual_additions = selected_residual_additions[:residual_count]
            attempt_specs: list[tuple[str, list[dict[str, object]], int]] = []
            if proposed_agp_additions:
                label = "residual_agp" if residual_additions else "agp_only"
                attempt_specs.append((label, proposed_agp_additions, new_agp_warmup_epochs))
            if residual_additions:
                attempt_specs.append(("residual_only", [], 0))
            for attempt_kind, agp_additions, warmup_epochs in attempt_specs:
                attempt_index += 1
                residual_labels = sort_pauli_labels(
                    current_residual_labels | {str(item["label"]) for item in residual_additions}
                )
                proposed_agp_labels = sort_pauli_labels(
                    current_agp_labels | {str(item["label"]) for item in agp_additions}
                )
                if round_run.exists():
                    shutil.rmtree(round_run)
                print(
                    f"train_coupled_round={round_index} attempt={attempt_index} kind={attempt_kind} "
                    f"agp_terms={len(proposed_agp_labels)} added_agp={len(agp_additions)} "
                    f"residual_terms={len(residual_labels)} added_residual={len(residual_additions)} "
                    f"epochs={coupled_settings.epochs} warmup={warmup_epochs} trust={trust_region_weight:.3e}"
                )
                candidate_body_state, candidate_final, candidate_metadata = train_coupled_round(
                    run_dir=round_run,
                    payload=payload,
                    settings=coupled_settings,
                    agp_labels=proposed_agp_labels,
                    previous_agp_labels=previous_agp_labels_for_round,
                    residual_labels=residual_labels,
                    body_state=previous_body_state,
                    round_index=round_index,
                    residual_additions=residual_additions,
                    agp_additions=agp_additions,
                    warmup_epochs=warmup_epochs,
                    trust_region_weight=trust_region_weight,
                )
                candidate_row, candidate_spectrum = evaluate_one_run(
                    run_dir=round_run,
                    config_payload=payload,
                    residual_top_k=residual_top_k,
                    intermediate_top_k=intermediate_top_k,
                    device=select_device("cpu"),
                    spectra_dir=data_dir / "feedback_projection",
                    common_residual_labels=feedback_residual_labels,
                    holdout_basis_mode="fixed_feedback",
                    holdout_basis_agp_terms=holdout_basis_agp_terms,
                )
                candidate_probe_gate_row, candidate_probe_gate_spectrum = evaluate_one_run(
                    run_dir=round_run,
                    config_payload=payload,
                    residual_top_k=len(probe_gate_residual_labels),
                    intermediate_top_k=intermediate_top_k,
                    device=select_device("cpu"),
                    spectra_dir=data_dir / "probe_gate_projection",
                    common_residual_labels=probe_gate_residual_labels,
                    holdout_basis_mode="probe_gate",
                    holdout_basis_agp_terms=probe_source_agp_terms,
                )
                candidate_probe_test_row, candidate_probe_test_spectrum = evaluate_one_run(
                    run_dir=round_run,
                    config_payload=payload,
                    residual_top_k=len(probe_test_residual_labels),
                    intermediate_top_k=intermediate_top_k,
                    device=select_device("cpu"),
                    spectra_dir=data_dir / "probe_test_projection",
                    common_residual_labels=probe_test_residual_labels,
                    holdout_basis_mode="probe_test",
                    holdout_basis_agp_terms=probe_source_agp_terms,
                )
                attach_probe_metrics(
                    candidate_row,
                    probe_gate_row=candidate_probe_gate_row,
                    probe_test_row=candidate_probe_test_row,
                    probe_gate_terms=len(probe_gate_residual_labels),
                    probe_test_terms=len(probe_test_residual_labels),
                    reference_floor=probe_reference_floor,
                )
                gate = step_gate_decision(
                    previous_feedback_row=previous_feedback_row,
                    candidate_feedback_row=candidate_row,
                    residual_candidate_count=len(residual_additions),
                    agp_candidate_count=len(agp_additions),
                    attempt_kind=attempt_kind,
                    probe_max_worsening_factor=probe_max_worsening_factor,
                    probe_max_worsening_delta=probe_max_worsening_delta,
                    probe_absolute_max_worsening_factor=probe_absolute_max_worsening_factor,
                    probe_absolute_max_worsening_delta=probe_absolute_max_worsening_delta,
                    feedback_max_worsening_factor=feedback_max_worsening_factor,
                    reference_floor=probe_reference_floor,
                )
                gate["attempt_index"] = attempt_index
                gate["residual_backtracking_count"] = int(residual_count)
                gate["probe_test_relative_residual"] = float(candidate_row["probe_test_relative_residual"])
                gate["probe_test_relative_residual_floored"] = float(candidate_row["probe_test_relative_residual_floored"])
                if bool(gate["accepted"]):
                    accepted_payload = {
                        "body_state": candidate_body_state,
                        "final": candidate_final,
                        "metadata": candidate_metadata,
                        "row": candidate_row,
                        "spectrum": candidate_spectrum,
                        "probe_gate_spectrum": candidate_probe_gate_spectrum,
                        "probe_test_spectrum": candidate_probe_test_spectrum,
                        "residual_labels": residual_labels,
                        "agp_labels": proposed_agp_labels,
                        "residual_additions": residual_additions,
                        "agp_additions": agp_additions,
                        "gate": gate,
                    }
                    attempt_gates.append(dict(gate))
                    break
                rejected_run = output_dir / "runs" / (
                    f"round_{round_index:02d}_rejected_{attempt_index:02d}_"
                    f"{attempt_kind}_R{len(residual_additions)}_A{len(agp_additions)}"
                )
                if rejected_run.exists():
                    shutil.rmtree(rejected_run)
                shutil.move(str(round_run), str(rejected_run))
                gate["rejected_run_dir"] = str(rejected_run.relative_to(output_dir))
                gate["rejected_agp_terms"] = agp_additions[:64]
                attempt_gates.append(dict(gate))
                print(
                    f"reject_coupled_round={round_index} attempt={attempt_index} kind={attempt_kind} "
                    f"feedback={gate['candidate_feedback_relative_residual']:.6e}/"
                    f"{gate['feedback_acceptance_limit']:.6e} "
                    f"probe_gate={gate['candidate_probe_relative_residual']:.6e}/"
                    f"{gate['probe_acceptance_limit']:.6e} "
                    f"probe_gate_abs={gate['candidate_probe_total_residual']:.6e}/"
                    f"{gate['probe_total_acceptance_limit']:.6e}"
                )
            if accepted_payload is not None:
                break

        if accepted_payload is None:
            gate = {
                "accepted": False,
                "status": "rejected_no_safe_step",
                "candidate_count": len(selected_residual_additions) + len(proposed_agp_additions),
                "residual_candidate_count": len(selected_residual_additions),
                "agp_candidate_count": len(proposed_agp_additions),
                "attempts": attempt_gates,
                "reason": "All residual/AGP backtracking attempts violated the feedback or probe-gate tolerances.",
            }
            row = dict(previous_feedback_row)
            row["feedback_round"] = round_index
            row["agp_growth_gate"] = gate
            body_state = previous_body_state
            final = {"relative_residual": float(row["training_final_relative_residual"])}
            metadata = {
                "first_commutator_nnz": row["first_commutator_nnz"],
                "second_commutator_nnz": row["second_commutator_nnz"],
                "final_intermediate_terms": row["intermediate_terms"],
                "final_residual_terms": row["train_residual_terms"],
                "final_agp_terms": row["agp_terms"],
            }
            residual_additions = []
            agp_additions = []
            residual_labels = sort_pauli_labels(current_residual_labels)
            agp_labels = sort_pauli_labels(current_agp_labels)
            spectrum = spectra[round_index - 1]
            probe_gate_spectrum = probe_gate_spectra[round_index - 1]
            probe_test_spectrum = probe_test_spectra[round_index - 1]
        else:
            body_state = accepted_payload["body_state"]
            final = accepted_payload["final"]
            metadata = accepted_payload["metadata"]
            row = accepted_payload["row"]
            spectrum = accepted_payload["spectrum"]
            probe_gate_spectrum = accepted_payload["probe_gate_spectrum"]
            probe_test_spectrum = accepted_payload["probe_test_spectrum"]
            residual_labels = accepted_payload["residual_labels"]
            agp_labels = accepted_payload["agp_labels"]
            residual_additions = accepted_payload["residual_additions"]
            agp_additions = accepted_payload["agp_additions"]
            gate = accepted_payload["gate"]
            gate = dict(gate)
            gate["attempts"] = attempt_gates
            row["agp_growth_gate"] = gate
            current_residual_labels = set(residual_labels)
            current_agp_labels = set(agp_labels)
            previous_agp_labels = list(agp_labels)
            row["run_dir"] = str(round_run.relative_to(output_dir))
            row["feedback_round"] = round_index

        rows.append(row)
        spectra[round_index] = spectrum
        probe_gate_spectra[round_index] = probe_gate_spectrum
        probe_test_spectra[round_index] = probe_test_spectrum
        row["spectrum_export"] = write_feedback_spectrum(
            data_dir,
            round_index=round_index,
            row=row,
            spectrum=spectrum,
        )
        row["probe_gate_spectrum_export"] = write_probe_spectrum(
            data_dir,
            round_index=round_index,
            row=probe_row_for_export(row, prefix="probe_gate"),
            spectrum=probe_gate_spectrum,
            probe_name="probe_gate",
        )
        row["probe_test_spectrum_export"] = write_probe_spectrum(
            data_dir,
            round_index=round_index,
            row=probe_row_for_export(row, prefix="probe_test"),
            spectrum=probe_test_spectrum,
            probe_name="probe_test",
        )
        round_rows.append(
            {
                "round": round_index,
                "run_dir": str(row["run_dir"]),
                "added_residual_terms": len(residual_additions),
                "added_agp_terms": len(agp_additions),
                "proposed_agp_terms": len(proposed_agp_additions),
                "proposed_residual_terms": len(selected_residual_additions),
                "agp_growth_accepted": bool(gate["accepted"]),
                "agp_growth_gate": gate,
                "attempts": attempt_gates,
                "train_residual_terms": len(residual_labels),
                "agp_terms": len(agp_labels),
                "training_final_relative_residual": float(final["relative_residual"]),
                "holdout_relative_residual": float(row["holdout_relative_residual"]),
                "unseen_relative_residual": float(row["unseen_relative_residual"]),
                "probe_gate_relative_residual": float(row["probe_gate_relative_residual"]),
                "probe_gate_relative_residual_floored": float(row["probe_gate_relative_residual_floored"]),
                "probe_test_relative_residual": float(row["probe_test_relative_residual"]),
                "probe_test_relative_residual_floored": float(row["probe_test_relative_residual_floored"]),
                "frozen_probe_relative_residual": float(row["probe_gate_relative_residual"]),
                "first_added_residual_terms": residual_additions[:32],
                "first_added_agp_terms": agp_additions[:32],
                "support_metadata": {
                    "first_commutator_nnz": metadata["first_commutator_nnz"],
                    "second_commutator_nnz": metadata["second_commutator_nnz"],
                    "final_intermediate_terms": metadata["final_intermediate_terms"],
                    "final_residual_terms": metadata["final_residual_terms"],
                    "final_agp_terms": metadata["final_agp_terms"],
                },
            }
        )
        print(
            f"done_coupled_round={round_index} K={len(agp_labels)} "
            f"train_relative={final['relative_residual']:.6e} "
            f"holdout_relative={row['holdout_relative_residual']:.6e} "
            f"unseen_relative={row['unseen_relative_residual']:.6e} "
            f"probe_gate={row['probe_gate_relative_residual']:.6e} "
            f"probe_test={row['probe_test_relative_residual']:.6e} "
            f"step_gate={gate['status']}"
        )

    write_coupled_summary(
        output_dir=output_dir,
        rows=rows,
        spectra=spectra,
        probe_gate_spectra=probe_gate_spectra,
        probe_test_spectra=probe_test_spectra,
        round_rows=round_rows,
        residual_top_k=residual_top_k,
        thresholds=thresholds,
        residual_budget=residual_budget,
        agp_growth_config=agp_growth_config,
        probe_config=probe_config,
    )
    summary_path = output_dir / "Models_Data" / f"coupled_curriculum_summary_residual_{residual_top_k}.json"
    full_basis = Decimal(4) ** int(config.n_qubits)
    print(
        json.dumps(
            {
                "summary": str(summary_path.relative_to(RUN_DIR)),
                "base_agp_terms": base_agp_terms,
                "final_agp_terms": len(agp_labels),
                "final_agp_fraction_of_full_basis": f"{Decimal(len(agp_labels)) / full_basis:.12E}",
                "rounds": round_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    getcontext().prec = 80
    main()
