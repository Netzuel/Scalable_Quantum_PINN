from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Callable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]

from agp_holdout_study import (  # noqa: E402
    Thresholds,
    build_common_holdout_residual_labels,
    evaluate_one_run,
    feedback_threshold_decision,
    fixed_unseen_plot_series,
    load_json,
    moving_unseen_diagnostics,
    optional_float,
)
from agp_residual_probes import (  # noqa: E402
    FixedUnseenProbeConfig,
    fixed_unseen_metrics,
    select_fixed_unseen_probes,
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
    plan_fixed_k_support_swap,
    plot_connection_summary,
    plot_support_map,
    preferred_calibration_labels_from_support,
    projected_trainable_state,
    projected_trainable_state_from_checkpoint,
    rank_coefficients,
    remap_trainable_state_for_agp_labels,
    restore_projected_trainable_state,
    select_device,
    set_paper_style,
    sort_pauli_labels,
    train_stage,
)
from agp_baseline_train import (  # noqa: E402
    configure_run_dir as configure_baseline_run_dir,
    model_config_from_payload,
    run_training,
    settings_for_support,
)
from models import PadeActivation  # noqa: E402
from utils import load_pauli_hamiltonian_pair  # noqa: E402


RUN_DIR = Path.cwd()
DEFAULT_CONFIG = Path("config.json")
ROUND_RUNS_DIRNAME = "rounds"
LEGACY_ROUND_RUNS_DIRNAME = "runs"
FIXED_UNSEEN_ROW_FIELDS = (
    "fixed_unseen_active_terms",
    "fixed_unseen_active_residual",
    "fixed_unseen_active_reference_residual",
    "fixed_unseen_active_relative",
    "fixed_unseen_active_status",
    "fixed_unseen_null_terms",
    "fixed_unseen_null_absolute_per_term",
    "fixed_unseen_null_scaled",
)
CERTIFICATION_PROBE_MANIFEST_NAMES = (
    "probe_gate_residual_labels.json",
    "probe_watch_residual_labels.json",
    "probe_test_residual_labels.json",
)


@dataclass(frozen=True)
class SupportSwapSettings:
    enabled: bool = False
    terms_per_iteration: int = 0
    start_round: int = 2
    candidate_pool_multiplier: int = 16
    protect_top_fraction: float = 0.02
    new_gate_logit: float = 2.0


@dataclass(frozen=True)
class TemporalRefinementSettings:
    enabled: bool = False
    epochs: int = 0
    num_points: int = 0
    lr: float = 0.0
    optimizer: str = ""
    run_dir: str = "temporal_refinement"


@dataclass(frozen=True)
class AdaptiveTemporalRefinementSettings:
    enabled: bool = False
    epochs: int = 0
    dense_points: int = 0
    num_points: int = 0
    lr: float = 0.0
    optimizer: str = ""
    run_dir: str = "adaptive_temporal_refinement"
    weight_power: float = 0.5
    min_weight: float = 0.25
    max_weight: float = 4.0
    difficulty: str = "residual"


@dataclass(frozen=True)
class PauTransferStabilitySettings:
    enabled: bool = True
    max_initial_relative_residual: float = 1.0e8
    fallback: str = "silu_rational_fit"


def fixed_unseen_probe_settings_from_feedback(
    feedback: Mapping[str, object],
) -> FixedUnseenProbeConfig:
    raw = feedback.get("fixed_unseen_probes", {})
    if not isinstance(raw, Mapping):
        return FixedUnseenProbeConfig()
    return FixedUnseenProbeConfig(
        enabled=bool(raw.get("enabled", False)),
        active_terms=max(0, int(raw.get("active_terms", 0))),
        null_terms=max(0, int(raw.get("null_terms", 0))),
        reference_rms_threshold=max(0.0, float(raw.get("reference_rms_threshold", 1.0e-12))),
        seed=int(raw.get("seed", 0)),
        candidate_multiplier=max(1, int(raw.get("candidate_multiplier", 4))),
    )


def save_fixed_unseen_probe(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2)
        handle.write("\n")


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_fixed_unseen_probe_manifest(
    payload: Mapping[str, object],
    *,
    certification_eligible: bool,
    provenance: str,
) -> dict[str, object]:
    """Attach immutable provenance and a content hash to a fixed-probe manifest."""

    manifest = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    manifest["certification_eligible"] = bool(certification_eligible)
    manifest["provenance"] = str(provenance)
    manifest["certification_reason"] = (
        "established_before_training"
        if certification_eligible
        else "historical_diagnostic_backfill"
    )
    manifest["manifest_sha256"] = _stable_hash(manifest)
    return manifest


def _label_identity(labels: Collection[str]) -> dict[str, object]:
    ordered = [str(label) for label in labels]
    return {
        "labels": ordered,
        "count": len(ordered),
        "sha256": _stable_hash(ordered),
    }


def fixed_unseen_probe_manifest_identity(
    *,
    settings: FixedUnseenProbeConfig,
    candidate_universe_labels: list[str],
    excluded_labels: Collection[str],
    requested_candidate_terms: int,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "requested_active_terms": int(settings.active_terms),
        "requested_null_terms": int(settings.null_terms),
        "reference_rms_threshold": float(settings.reference_rms_threshold),
        "seed": int(settings.seed),
        "candidate_multiplier": int(settings.candidate_multiplier),
        "requested_candidate_terms": int(requested_candidate_terms),
        "candidate_universe": _label_identity(candidate_universe_labels),
        "excluded_labels": _label_identity(sorted(str(label) for label in excluded_labels)),
    }


def _reference_rms_metadata(
    *,
    candidate_labels: list[str],
    reference_rms: np.ndarray,
    probe: Mapping[str, object],
) -> dict[str, object]:
    values = [float(value) for value in np.asarray(reference_rms, dtype=float)]
    by_label = {label: value for label, value in zip(candidate_labels, values, strict=True)}
    selected = {
        "active": [
            {"label": label, "reference_rms": float(by_label[label])}
            for label in probe.get("active_labels", [])
        ],
        "null": [
            {"label": label, "reference_rms": float(by_label[label])}
            for label in probe.get("null_labels", [])
        ],
    }
    return {
        "candidate_sha256": _stable_hash(
            [{"label": label, "reference_rms": value} for label, value in zip(candidate_labels, values, strict=True)]
        ),
        "selected": selected,
        "selected_sha256": _stable_hash(selected),
    }


def load_or_validate_fixed_unseen_probe(
    path: Path,
    *,
    expected_excluded_labels: Collection[str],
    expected_identity: Mapping[str, object] | None = None,
    expected_reference_rms_metadata: Mapping[str, object] | None = None,
    expected_selected_labels: Mapping[str, Collection[str]] | None = None,
) -> dict[str, object]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    stored_hash = payload.get("manifest_sha256")
    if stored_hash is not None:
        hashed_payload = {key: value for key, value in payload.items() if key != "manifest_sha256"}
        if str(stored_hash) != _stable_hash(hashed_payload):
            raise ValueError("immutable fixed unseen probe manifest hash does not match its contents")
    active_labels = [str(label) for label in payload.get("active_labels", [])]
    null_labels = [str(label) for label in payload.get("null_labels", [])]
    if set(active_labels) & set(null_labels):
        raise ValueError("immutable fixed unseen probe partitions overlap")
    overlap = (set(active_labels) | set(null_labels)) & {str(label) for label in expected_excluded_labels}
    if overlap:
        raise ValueError(
            "immutable fixed unseen probe intersects the current excluded residual labels: "
            f"{sorted(overlap)[:8]}"
        )
    if expected_identity is not None:
        for key in (
            "schema_version",
            "requested_active_terms",
            "requested_null_terms",
            "reference_rms_threshold",
            "seed",
            "candidate_multiplier",
            "requested_candidate_terms",
            "excluded_labels",
        ):
            if payload.get(key) != expected_identity.get(key):
                raise ValueError(f"immutable fixed unseen probe {key} identity does not match the current request")
        if payload.get("candidate_universe") != expected_identity.get("candidate_universe"):
            raise ValueError("immutable fixed unseen probe candidate universe identity does not match the current request")
    if expected_selected_labels is not None:
        for key, labels in (("active_labels", expected_selected_labels.get("active", [])), ("null_labels", expected_selected_labels.get("null", []))):
            if payload.get(key, []) != [str(label) for label in labels]:
                raise ValueError(f"immutable fixed unseen probe selected {key} do not match the current request")
    metadata = payload.get("reference_rms_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("immutable fixed unseen probe is missing reference RMS metadata")
    if expected_reference_rms_metadata is not None and dict(metadata) != dict(expected_reference_rms_metadata):
        raise ValueError("immutable fixed unseen probe reference RMS metadata does not match the current request")
    payload["active_labels"] = active_labels
    payload["null_labels"] = null_labels
    return payload


def build_fixed_unseen_probe(
    *,
    candidate_labels: list[str],
    excluded_labels: set[str],
    reference_rms: np.ndarray,
    settings: FixedUnseenProbeConfig,
) -> dict[str, object]:
    if not settings.enabled:
        return {
            "enabled": False,
            "active_labels": [],
            "null_labels": [],
            "active_reference_rms": [],
            "null_reference_rms": [],
            "requested_active_terms": settings.active_terms,
            "requested_null_terms": settings.null_terms,
            "status": "disabled",
            "reference_rms_threshold": settings.reference_rms_threshold,
        }
    probe = select_fixed_unseen_probes(
        candidate_labels,
        reference_rms,
        excluded_labels=excluded_labels,
        config=settings,
    )
    probe.update(
        {
            "enabled": True,
            "reference_rms_threshold": settings.reference_rms_threshold,
            "seed": settings.seed,
            "candidate_multiplier": settings.candidate_multiplier,
            "candidate_terms": len(candidate_labels),
            "excluded_terms": len(excluded_labels),
        }
    )
    return probe


def build_expanding_fixed_unseen_probe(
    *,
    generate_candidates: Callable[[int], list[str]],
    reference_rms_for_labels: Callable[[list[str]], np.ndarray],
    settings: FixedUnseenProbeConfig,
    moving_holdout_terms: int,
    excluded_labels: Collection[str],
    initial_request: int,
    resource_cap: int | None,
) -> tuple[dict[str, object], list[str]]:
    moving_holdout_terms = int(moving_holdout_terms)
    request = max(int(initial_request), moving_holdout_terms)
    cap = int(resource_cap) if resource_cap is not None else None
    if cap is not None:
        if cap < moving_holdout_terms:
            raise ValueError(
                "fixed unseen candidate cap must be at least the mandatory moving holdout "
                f"request ({moving_holdout_terms}); got {cap}"
            )
        request = min(request, cap)
    history: list[dict[str, object]] = []
    seen_hashes: set[str] = set()
    final_probe: dict[str, object] | None = None
    final_universe: list[str] = []
    final_reference = np.empty(0, dtype=float)
    final_excluded: set[str] = set()
    insufficiency_reason: str | None = None

    while True:
        universe = [str(label) for label in generate_candidates(request)]
        moving_labels = universe[: int(moving_holdout_terms)]
        candidate_tail = universe[int(moving_holdout_terms) :]
        effective_excluded = {str(label) for label in excluded_labels} | set(moving_labels)
        reference_rms = reference_rms_for_labels(candidate_tail)
        probe = build_fixed_unseen_probe(
            candidate_labels=candidate_tail,
            excluded_labels=effective_excluded,
            reference_rms=reference_rms,
            settings=settings,
        )
        history.append(
            {
                "requested_candidate_terms": request,
                "realized_candidate_terms": len(universe),
                "realized_tail_terms": len(candidate_tail),
                "realized_active_terms": len(probe["active_labels"]),
                "realized_null_terms": len(probe["null_labels"]),
            }
        )
        final_probe = probe
        final_universe = universe
        final_reference = reference_rms
        final_excluded = effective_excluded
        universe_hash = _stable_hash(universe)
        if not settings.enabled or probe.get("status") == "complete":
            break
        if len(universe) < request or universe_hash in seen_hashes:
            insufficiency_reason = "generator_saturated"
            break
        if cap is not None and request >= cap:
            insufficiency_reason = "resource_cap_reached"
            break
        seen_hashes.add(universe_hash)
        request = request * 2 if cap is None else min(request * 2, cap)

    if final_probe is None:
        raise RuntimeError("fixed unseen probe expansion produced no candidate universe")
    final_probe.update(
        fixed_unseen_probe_manifest_identity(
            settings=settings,
            candidate_universe_labels=final_universe,
            excluded_labels=final_excluded,
            requested_candidate_terms=history[-1]["requested_candidate_terms"],
        )
    )
    final_probe.update(
        {
            "moving_holdout_terms": int(moving_holdout_terms),
            "requested_candidate_terms_initial": int(initial_request),
            "requested_candidate_terms_final": int(history[-1]["requested_candidate_terms"]),
            "resource_cap": cap,
            "realized_candidate_terms": len(final_universe),
            "realized_tail_terms": max(len(final_universe) - int(moving_holdout_terms), 0),
            "expansion_history": history,
            "insufficiency_reason": (
                "disabled" if not settings.enabled else insufficiency_reason
            ),
            "reference_rms_metadata": _reference_rms_metadata(
                candidate_labels=final_universe[int(moving_holdout_terms) :],
                reference_rms=final_reference,
                probe=final_probe,
            ),
        }
    )
    return final_probe, final_universe


def _fixed_unseen_metrics_from_totals(
    *,
    active_terms: int,
    active_residual: float,
    active_reference: float,
    null_terms: int,
    null_residual: float,
    null_reference: float,
    reference_floor: float,
) -> dict[str, object]:
    values = torch.zeros((1, active_terms + null_terms), dtype=torch.float64)
    reference = torch.zeros_like(values)
    if active_terms:
        values[0, :active_terms] = float(np.sqrt(max(active_residual, 0.0) / active_terms))
        reference[0, :active_terms] = float(np.sqrt(max(active_reference, 0.0) / active_terms))
    if null_terms:
        values[0, active_terms:] = float(np.sqrt(max(null_residual, 0.0) / null_terms))
        reference[0, active_terms:] = float(np.sqrt(max(null_reference, 0.0) / null_terms))
    return fixed_unseen_metrics(
        residual=values,
        reference=reference,
        active_indices=list(range(active_terms)),
        null_indices=list(range(active_terms, active_terms + null_terms)),
        reference_floor=reference_floor,
    )


def fixed_unseen_probe_row_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    return {
        "fixed_unseen_active_terms": metrics["active_terms"],
        "fixed_unseen_active_residual": metrics["active_residual"],
        "fixed_unseen_active_reference_residual": metrics["active_reference_residual"],
        "fixed_unseen_active_relative": metrics["active_relative"],
        "fixed_unseen_active_status": metrics["active_status"],
        "fixed_unseen_null_terms": metrics["null_terms"],
        "fixed_unseen_null_absolute_per_term": metrics["null_absolute_per_term"],
        "fixed_unseen_null_scaled": metrics["null_scaled"],
    }


def merge_fixed_unseen_probe_metrics(
    row: Mapping[str, object],
    metrics: Mapping[str, object],
) -> dict[str, object]:
    merged = dict(row)
    merged.update(
        {
            key: metrics[key]
            for key in FIXED_UNSEEN_ROW_FIELDS
            if key in metrics
        }
    )
    return merged


def evaluate_fixed_unseen_probe(
    *,
    run_dir: Path,
    config_payload: dict[str, object],
    probe_metadata: Mapping[str, object],
    intermediate_top_k: int,
    device: torch.device,
) -> dict[str, object]:
    active_labels = [str(label) for label in probe_metadata.get("active_labels", [])]
    null_labels = [str(label) for label in probe_metadata.get("null_labels", [])]
    checkpoint_labels = load_checkpoint_labels(run_dir / "Models_Data" / "training_checkpoint.pt")[1]
    overlap = (set(active_labels) | set(null_labels)) & set(checkpoint_labels)
    if overlap:
        raise AssertionError(
            "fixed unseen probes must not intersect checkpoint training residual labels: "
            f"{sorted(overlap)[:8]}"
        )

    empty = {
        "holdout_total_residual": 0.0,
        "holdout_reference_residual": 0.0,
    }

    def evaluate_partition(labels: list[str], name: str) -> dict[str, object]:
        if not labels:
            return empty
        row, _ = evaluate_one_run(
            run_dir=run_dir,
            config_payload=config_payload,
            residual_top_k=len(labels),
            intermediate_top_k=intermediate_top_k,
            device=device,
            spectra_dir=run_dir / "Models_Data" / "fixed_unseen_probe_spectra" / name,
            common_residual_labels=labels,
            holdout_basis_mode=f"fixed_unseen_{name}",
            holdout_basis_agp_terms=None,
        )
        return row

    active_row = evaluate_partition(active_labels, "active")
    null_row = evaluate_partition(null_labels, "null")
    reference_floor = float(probe_metadata.get("reference_rms_threshold", 1.0e-12)) ** 2
    metrics = fixed_unseen_probe_row_metrics(_fixed_unseen_metrics_from_totals(
        active_terms=len(active_labels),
        active_residual=float(active_row["holdout_total_residual"]),
        active_reference=float(active_row["holdout_reference_residual"]),
        null_terms=len(null_labels),
        null_residual=float(null_row["holdout_total_residual"]),
        null_reference=float(null_row["holdout_reference_residual"]),
        reference_floor=reference_floor,
    ))
    return metrics


def configured_certification_probe_labels(
    payload: Mapping[str, object],
    feedback: Mapping[str, object],
) -> set[str]:
    """Collect explicitly configured gate/watch/test probe labels for exclusion."""

    labels: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, str):
            labels.add(value)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                collect(item)
        elif isinstance(value, Mapping):
            for key in ("labels", "residual_labels", "probe_labels"):
                if key in value:
                    collect(value[key])

    for name in ("probe_gate", "probe_watch", "probe_test"):
        collect(feedback.get(name))
        collect(payload.get(name))
    certification = payload.get("certification_probes")
    if isinstance(certification, Mapping):
        for name in ("probe_gate", "probe_watch", "probe_test"):
            collect(certification.get(name))
    return labels


def configured_generated_run_roots(
    payload: Mapping[str, object],
    *,
    scenario_root: Path,
    extra_roots: Collection[Path] = (),
) -> list[Path]:
    """Return canonical scenario run roots declared by the active config."""

    roots = {scenario_root / "runs", *(Path(root) for root in extra_roots)}

    def collect(value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in {"output_root", "baseline_root", "run_root"} and isinstance(item, (str, Path)):
                    root = Path(item)
                    roots.add(root if root.is_absolute() else scenario_root / root)
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(payload)
    return sorted({root.resolve() for root in roots}, key=str)


def load_existing_certification_probe_labels(
    roots: Collection[Path],
) -> tuple[set[str], list[Path]]:
    """Load persisted gate/watch/test labels from configured scenario run roots."""

    manifest_paths: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for filename in CERTIFICATION_PROBE_MANIFEST_NAMES:
            manifest_paths.update(path.resolve() for path in root.rglob(filename))

    labels: set[str] = set()
    paths = sorted(manifest_paths, key=str)
    for path in paths:
        payload = load_json(path)
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object.")
        manifest_labels = payload.get("labels", [])
        if not isinstance(manifest_labels, list):
            raise TypeError(f"{path} labels must be a JSON list.")
        labels.update(str(label) for label in manifest_labels)
    return labels, paths


def fixed_unseen_probe_candidate_cap(
    feedback: Mapping[str, object],
    *,
    moving_holdout_terms: int | None = None,
    initial_request: int | None = None,
) -> int | None:
    if moving_holdout_terms is None:
        if initial_request is None:
            raise TypeError("moving_holdout_terms is required")
        moving_holdout_terms = int(initial_request)
    raw = feedback.get("fixed_unseen_probes", {})
    raw = raw if isinstance(raw, Mapping) else {}
    for key in ("max_candidate_terms", "candidate_request_cap", "generator_cap", "resource_cap"):
        if key in raw:
            cap = int(raw[key])
            if cap < int(moving_holdout_terms):
                raise ValueError(
                    "holdout_feedback.fixed_unseen_probes candidate cap must be at least the "
                    f"mandatory moving holdout request ({int(moving_holdout_terms)}); got {cap}"
                )
            return cap
    return None


def fixed_unseen_reference_rms(
    *,
    h0,
    h1,
    settings: ProjectedRunSettings,
    agp_labels: list[str],
    candidate_labels: list[str],
) -> np.ndarray:
    if not candidate_labels:
        return np.empty(0, dtype=float)
    support = make_support_with_residual_labels(
        h0=h0,
        h1=h1,
        settings=settings,
        agp_labels=agp_labels,
        residual_labels=candidate_labels,
        stage=0,
    )
    device = select_device("cpu")
    model = make_projected_model(h0, h1, support, settings.model, device)
    tau = torch.linspace(0.0, 1.0, settings.num_points, device=device).view(-1, 1)
    t = settings.model.t_initial + settings.model.physical_time * tau
    with torch.no_grad():
        reference = model.euler_lagrange_reference_residual(t)
    return torch.sqrt(torch.mean(torch.abs(reference) ** 2, dim=0).real).detach().cpu().numpy()


def feedback_refinements_complete(
    summary: dict[str, object],
    output_dir: Path,
    temporal: TemporalRefinementSettings,
    adaptive: AdaptiveTemporalRefinementSettings,
) -> bool:
    required = (
        (temporal.enabled, "temporal_refinement", temporal.run_dir),
        (adaptive.enabled, "adaptive_temporal_refinement", adaptive.run_dir),
    )
    for enabled, summary_key, default_run_dir in required:
        if not enabled:
            continue
        entry = summary.get(summary_key, {})
        if not isinstance(entry, dict) or not bool(entry.get("enabled", False)):
            return False
        run_dir = output_dir / str(entry.get("run_dir", default_run_dir))
        if not (run_dir / "Models_Data" / "training_checkpoint.pt").is_file():
            return False
    return True


def configure_run_dir(config_path: Path) -> None:
    global RUN_DIR
    RUN_DIR = config_path.resolve().parent
    configure_baseline_run_dir(config_path)


def round_run_dir(output_dir: Path, round_index: int) -> Path:
    return output_dir / ROUND_RUNS_DIRNAME / f"round_{round_index:02d}"


def normalize_round_run_label(label: object) -> str:
    raw = str(label)
    legacy_prefix = f"{LEGACY_ROUND_RUNS_DIRNAME}/round_"
    if raw.startswith(legacy_prefix):
        return f"{ROUND_RUNS_DIRNAME}/{raw[len(LEGACY_ROUND_RUNS_DIRNAME) + 1:]}"
    return raw


def load_checkpoint_labels(checkpoint_path: Path) -> tuple[list[str], list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return [str(label) for label in checkpoint["agp_labels"]], [str(label) for label in checkpoint["residual_labels"]]


def support_swap_settings_from_feedback(feedback: dict[str, object]) -> SupportSwapSettings:
    raw = feedback.get("support_swap", {})
    if not isinstance(raw, dict):
        return SupportSwapSettings()
    return SupportSwapSettings(
        enabled=bool(raw.get("enabled", False)),
        terms_per_iteration=max(0, int(raw.get("terms_per_iteration", 0))),
        start_round=max(1, int(raw.get("start_round", 2))),
        candidate_pool_multiplier=max(1, int(raw.get("candidate_pool_multiplier", 16))),
        protect_top_fraction=max(0.0, float(raw.get("protect_top_fraction", 0.02))),
        new_gate_logit=float(raw.get("new_gate_logit", 2.0)),
    )


def pau_transfer_stability_settings_from_feedback(
    feedback: dict[str, object],
) -> PauTransferStabilitySettings:
    raw = feedback.get("pau_transfer_stability", {})
    if not isinstance(raw, dict):
        return PauTransferStabilitySettings()
    return PauTransferStabilitySettings(
        enabled=bool(raw.get("enabled", True)),
        max_initial_relative_residual=max(0.0, float(raw.get("max_initial_relative_residual", 1.0e8))),
        fallback=str(raw.get("fallback", "silu_rational_fit")),
    )


def temporal_refinement_settings_from_feedback(feedback: dict[str, object]) -> TemporalRefinementSettings:
    raw = feedback.get("temporal_refinement", {})
    if not isinstance(raw, dict):
        return TemporalRefinementSettings()
    return TemporalRefinementSettings(
        enabled=bool(raw.get("enabled", False)),
        epochs=max(0, int(raw.get("epochs", 0))),
        num_points=max(0, int(raw.get("num_points", 0))),
        lr=max(0.0, float(raw.get("lr", 0.0))),
        optimizer=str(raw.get("optimizer", "")),
        run_dir=str(raw.get("run_dir", "temporal_refinement")),
    )


def adaptive_temporal_refinement_settings_from_feedback(
    feedback: dict[str, object],
) -> AdaptiveTemporalRefinementSettings:
    raw = feedback.get("adaptive_temporal_refinement", {})
    if not isinstance(raw, dict):
        return AdaptiveTemporalRefinementSettings()
    return AdaptiveTemporalRefinementSettings(
        enabled=bool(raw.get("enabled", False)),
        epochs=max(0, int(raw.get("epochs", 0))),
        dense_points=max(0, int(raw.get("dense_points", 0))),
        num_points=max(0, int(raw.get("num_points", 0))),
        lr=max(0.0, float(raw.get("lr", 0.0))),
        optimizer=str(raw.get("optimizer", "")),
        run_dir=str(raw.get("run_dir", "adaptive_temporal_refinement")),
        weight_power=max(0.0, float(raw.get("weight_power", 0.5))),
        min_weight=max(0.0, float(raw.get("min_weight", 0.25))),
        max_weight=max(0.0, float(raw.get("max_weight", 4.0))),
        difficulty=str(raw.get("difficulty", "residual")),
    )


def make_adaptive_tau_grid(
    dense_tau: torch.Tensor,
    difficulty: torch.Tensor,
    *,
    num_points: int,
    weight_power: float,
    min_weight: float,
    max_weight: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Build a monotone collocation grid concentrated near hard residual times."""

    if num_points < 2:
        raise ValueError("adaptive temporal refinement requires at least two time points.")
    dense = dense_tau.detach().flatten().to(dtype=torch.float32)
    score = difficulty.detach().flatten().to(dtype=torch.float32, device=dense.device)
    if dense.numel() != score.numel():
        raise ValueError("dense_tau and difficulty must have the same length.")
    if dense.numel() < 2:
        raise ValueError("adaptive temporal refinement requires at least two dense time points.")
    if max_weight < min_weight:
        raise ValueError("adaptive temporal refinement max_weight must be >= min_weight.")

    order = torch.argsort(dense)
    dense = dense.index_select(0, order)
    score = score.index_select(0, order)
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    eps = torch.finfo(score.dtype).eps
    mean_score = torch.mean(score).clamp_min(eps)
    relative_score = score / mean_score
    if weight_power == 0.0:
        weights = torch.ones_like(relative_score)
    else:
        weights = relative_score.clamp_min(eps).pow(weight_power)
    weights = weights.clamp(min=min_weight, max=max_weight)
    delta = torch.diff(dense).clamp_min(eps)
    segment_mass = 0.5 * (weights[:-1] + weights[1:]) * delta
    total_mass = torch.sum(segment_mass)
    if not torch.isfinite(total_mass) or float(total_mass.item()) <= 0.0:
        tau = torch.linspace(float(dense[0]), float(dense[-1]), num_points, device=dense.device)
        metadata = {
            "num_points": int(num_points),
            "dense_points": int(dense.numel()),
            "min_weight": float(weights.min().item()),
            "max_weight": float(weights.max().item()),
            "mean_weight": float(weights.mean().item()),
            "fallback": "uniform_zero_mass",
        }
        return tau.view(-1, 1), metadata

    cdf = torch.cat([torch.zeros(1, device=dense.device, dtype=dense.dtype), torch.cumsum(segment_mass, dim=0)])
    cdf = cdf / total_mass
    targets = torch.linspace(0.0, 1.0, num_points, device=dense.device, dtype=dense.dtype)
    right = torch.searchsorted(cdf, targets, right=False).clamp(min=1, max=cdf.numel() - 1)
    left = right - 1
    width = (cdf[right] - cdf[left]).clamp_min(eps)
    frac = (targets - cdf[left]) / width
    tau = dense[left] + frac * (dense[right] - dense[left])
    tau[0] = dense[0]
    tau[-1] = dense[-1]
    metadata = {
        "num_points": int(num_points),
        "dense_points": int(dense.numel()),
        "min_weight": float(weights.min().item()),
        "max_weight": float(weights.max().item()),
        "mean_weight": float(weights.mean().item()),
        "max_difficulty": float(score.max().item()),
        "mean_difficulty": float(score.mean().item()),
        "weight_power": float(weight_power),
    }
    return tau.view(-1, 1), metadata


def compact_support_swap_plan(plan: dict[str, object] | None, *, preview_terms: int = 32) -> dict[str, object]:
    if not plan:
        return {"enabled": False, "swap_count": 0}
    preview = max(0, int(preview_terms))
    compact: dict[str, object] = {
        "enabled": bool(plan.get("enabled", False)),
        "swap_count": int(plan.get("swap_count", 0)),
        "reason": str(plan.get("reason", "unknown")),
    }
    for key in ("removed_labels", "added_labels", "candidate_rows"):
        values = plan.get(key, [])
        if isinstance(values, list):
            compact[key] = values[:preview]
    for key in ("protected_label_count", "protect_top_fraction"):
        if key in plan:
            compact[key] = plan[key]
    return compact


def payload_with_feedback_baseline_neural(payload: dict[str, object]) -> dict[str, object]:
    feedback = payload.get("holdout_feedback", {})
    feedback = feedback if isinstance(feedback, dict) else {}
    baseline_neural = feedback.get("baseline_neural", {})
    if not isinstance(baseline_neural, dict) or not baseline_neural:
        return payload

    out = copy.deepcopy(payload)
    neural = out.setdefault("neural", {})
    if not isinstance(neural, dict):
        neural = {}
        out["neural"] = neural
    general = neural.setdefault("general", {})
    if not isinstance(general, dict):
        general = {}
        neural["general"] = general
    general.update(baseline_neural)
    return out


def load_body_state_from_checkpoint(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    state = projected_trainable_state_from_checkpoint(checkpoint_path)
    body_state = state.get("body", {})
    if not isinstance(body_state, dict):
        raise TypeError(f"Missing body state in {checkpoint_path}.")
    return {str(key): value for key, value in body_state.items() if isinstance(value, torch.Tensor)}


def load_coefficient_importance_rows(run_dir: Path) -> list[dict[str, object]]:
    path = run_dir / "Models_Data" / "coefficient_importance.json"
    if not path.is_file():
        return []
    payload = load_json(path)
    if not isinstance(payload, dict):
        return []
    rows = payload.get("all_terms", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def select_residual_additions(
    spectrum: list[dict[str, object]],
    current_residual_labels: set[str],
    *,
    excluded_labels: Collection[str] = (),
    add_terms: int,
    min_rms: float,
) -> list[dict[str, object]]:
    additions: list[dict[str, object]] = []
    excluded = {str(label) for label in excluded_labels}
    for row in spectrum:
        label = str(row["label"])
        if label in current_residual_labels or label in excluded:
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
        residual_labels=residual_labels,
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


def load_feedback_hamiltonian_pair(config):
    hamiltonian_path = Path(config.hamiltonian_source)
    if not hamiltonian_path.is_absolute():
        hamiltonian_path = ROOT / hamiltonian_path
    return load_pauli_hamiltonian_pair(
        hamiltonian_path,
        system=config.system,
        n_qubits=config.n_qubits,
        distance=config.distance,
    )


def make_feedback_model_from_state(
    *,
    h0,
    h1,
    settings: ProjectedRunSettings,
    agp_labels: list[str],
    residual_labels: list[str],
    trainable_state: dict[str, object],
    stage: int,
    device: torch.device,
):
    support = make_support_with_residual_labels(
        h0=h0,
        h1=h1,
        settings=settings,
        agp_labels=agp_labels,
        residual_labels=residual_labels,
        stage=stage,
    )
    model = make_projected_model(h0, h1, support, settings.model, device)
    restore_projected_trainable_state(
        model,
        trainable_state,
        settings=settings,
        preferred_active_labels=preferred_calibration_labels_from_support(support),
    )
    return model, support


def adaptive_temporal_difficulty(
    *,
    payload: dict[str, object],
    settings: ProjectedRunSettings,
    agp_labels: list[str],
    residual_labels: list[str],
    trainable_state: dict[str, object],
    stage: int,
    dense_points: int,
    difficulty: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    config = settings.model
    device = select_device(settings.device)
    h0, h1 = load_feedback_hamiltonian_pair(config)
    model, _ = make_feedback_model_from_state(
        h0=h0,
        h1=h1,
        settings=settings,
        agp_labels=agp_labels,
        residual_labels=residual_labels,
        trainable_state=trainable_state,
        stage=stage,
        device=device,
    )
    dense_tau = torch.linspace(0.0, 1.0, dense_points, device=device).view(-1, 1)
    t = config.t_initial + config.physical_time * dense_tau
    with torch.no_grad():
        residual = model.euler_lagrange_residual(t)
        residual_score = torch.sum(torch.abs(residual) ** 2, dim=-1).real
        if difficulty == "residual_x_cd_norm":
            prediction = model(t)
            cd_coefficients = prediction["d_lambda_dt"] * prediction["agp_coefficients"]
            cd_norm = torch.sqrt(torch.mean(torch.abs(cd_coefficients) ** 2, dim=-1).real)
            normalized_cd_norm = cd_norm / torch.mean(cd_norm).clamp_min(torch.finfo(cd_norm.dtype).eps)
            residual_score = residual_score * normalized_cd_norm
    return dense_tau.detach().cpu(), residual_score.detach().cpu()


def train_feedback_round(
    *,
    run_dir: Path,
    payload: dict[str, object],
    settings: ProjectedRunSettings,
    agp_labels: list[str],
    residual_labels: list[str],
    trainable_state: dict[str, object],
    round_index: int,
    additions: list[dict[str, object]],
    support_swap_plan: dict[str, object] | None = None,
    tau_override: torch.Tensor | None = None,
    temporal_sampling_metadata: dict[str, object] | None = None,
    pau_transfer_stability: PauTransferStabilitySettings | None = None,
) -> tuple[dict[str, object], dict[str, float], dict[str, object]]:
    config = settings.model
    torch.manual_seed(settings.seed + round_index)
    device = select_device(settings.device)
    h0, h1 = load_feedback_hamiltonian_pair(config)
    model, support = make_feedback_model_from_state(
        h0=h0,
        h1=h1,
        settings=settings,
        agp_labels=agp_labels,
        residual_labels=residual_labels,
        trainable_state=trainable_state,
        stage=round_index,
        device=device,
    )

    loss_weights = ProjectedSparseLossWeights(
        residual=settings.residual_weight,
        agp_l2=settings.agp_l2_weight,
        residual_block_normalization=settings.residual_block_normalization,
        agp_smoothness=settings.agp_smoothness_weight,
        agp_curvature=settings.agp_curvature_weight,
        schedule_monotonic=settings.schedule_monotonic_weight,
        schedule_correction_l2=settings.schedule_correction_l2_weight,
        calibration_budget=settings.calibration_budget_weight,
        calibration_binary=settings.calibration_binary_weight,
        calibration_scale_l2=settings.calibration_scale_l2_weight,
    )
    if tau_override is None:
        tau = torch.linspace(0.0, 1.0, settings.num_points, device=device).view(-1, 1)
    else:
        tau = tau_override.detach().to(device=device, dtype=torch.float32).view(-1, 1)
    t = config.t_initial + config.physical_time * tau
    transfer_metadata: dict[str, object] = {"enabled": False, "triggered": False}
    if pau_transfer_stability is not None and pau_transfer_stability.enabled and round_index == 1:
        initial_loss, initial_diagnostics = model.loss(t, weights=loss_weights)
        initial_relative = float(initial_diagnostics["relative_residual"].detach().cpu())
        transfer_metadata = {
            "enabled": True,
            "triggered": False,
            "initial_relative_residual": initial_relative,
            "max_initial_relative_residual": pau_transfer_stability.max_initial_relative_residual,
            "fallback": pau_transfer_stability.fallback,
        }
        if not np.isfinite(initial_relative) or initial_relative > pau_transfer_stability.max_initial_relative_residual:
            if pau_transfer_stability.fallback != "silu_rational_fit":
                raise ValueError(f"Unsupported PAU transfer fallback {pau_transfer_stability.fallback!r}.")
            reset_count = 0
            for module in model.body.modules():
                if isinstance(module, PadeActivation):
                    module.reset_to_silu_rational_fit()
                    reset_count += 1
            fallback_loss, fallback_diagnostics = model.loss(t, weights=loss_weights)
            fallback_relative = float(fallback_diagnostics["relative_residual"].detach().cpu())
            transfer_metadata.update(
                {
                    "triggered": True,
                    "reset_activation_count": reset_count,
                    "fallback_relative_residual": fallback_relative,
                }
            )
            print(
                "pau_transfer_stability_fallback "
                f"initial_relative={initial_relative:.6e} fallback_relative={fallback_relative:.6e} "
                f"activations={reset_count}"
            )
            del fallback_loss, fallback_diagnostics
        del initial_loss, initial_diagnostics
        model.zero_grad(set_to_none=True)
    optimizer, optimizer_info = make_optimizer(model, settings)
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
    metadata["support_swap"] = compact_support_swap_plan(support_swap_plan)
    metadata["pau_transfer_stability"] = transfer_metadata
    metadata["adaptive_enabled"] = False
    if temporal_sampling_metadata is not None:
        metadata["temporal_sampling"] = temporal_sampling_metadata
    metadata["final_agp_terms"] = len(model.agp_labels)
    metadata["final_intermediate_terms"] = len(model.intermediate_labels)
    metadata["final_residual_terms"] = len(model.residual_labels)
    metadata["first_commutator_nnz"] = model.first_commutator.nnz
    metadata["second_commutator_nnz"] = model.second_commutator.nnz
    if model.has_agp_calibration():
        metadata["agp_calibration"] = {
            "enabled": True,
            "training_mode": "joint_in_curriculum",
            "target_active_terms": int(getattr(model, "agp_target_active_terms", len(model.agp_labels))),
            "gate_temperature": float(getattr(model, "agp_gate_temperature", 1.0)),
            "uses_ground_truth_observables": False,
            "objective": "projected_euler_lagrange_residual_with_trainable_scale_and_gates",
        }

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
    next_trainable_state = projected_trainable_state(model)
    return next_trainable_state, history[-1], metadata


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


def fit_residual_budget_to_available(
    *,
    residual_top_k: int,
    add_residual_terms: int,
    residual_budget: dict[str, object],
    available_residual_terms: int,
    initial_residual_terms: int,
    rounds: int,
    unseen_batches_after_final_iteration: int,
) -> tuple[int, int, dict[str, object]]:
    if available_residual_terms < initial_residual_terms:
        raise ValueError(
            f"Available residual labels ({available_residual_terms}) are fewer than the initial "
            f"training residual labels ({initial_residual_terms})."
        )
    fitted = dict(residual_budget)
    requested_residual_top_k = int(residual_top_k)
    requested_add = int(add_residual_terms)
    unseen_batches = max(int(unseen_batches_after_final_iteration), 0)
    effective_residual_top_k = min(requested_residual_top_k, int(available_residual_terms))
    effective_add = requested_add
    status = "unchanged"

    denominator = int(rounds) + unseen_batches
    if denominator > 0:
        max_add_preserving_rounds = max(
            (effective_residual_top_k - int(initial_residual_terms)) // denominator,
            0,
        )
        if effective_add > max_add_preserving_rounds:
            effective_add = max_add_preserving_rounds
            status = "auto_reduced_additions_to_preserve_rounds"

    minimum_budget_before_final_unseen_exhaustion = int(initial_residual_terms) + int(rounds) * effective_add
    final_unseen_terms = max(effective_residual_top_k - minimum_budget_before_final_unseen_exhaustion, 0)
    fitted.update(
        {
            "residual_budget_fit_status": status,
            "requested_holdout_residual_top_k": requested_residual_top_k,
            "requested_add_residual_terms_per_iteration": requested_add,
            "available_generated_residual_terms": int(available_residual_terms),
            "resolved_holdout_residual_top_k": effective_residual_top_k,
            "effective_add_residual_terms_per_iteration": effective_add,
            "add_residual_terms_per_iteration": effective_add,
            "minimum_budget_before_final_unseen_exhaustion": minimum_budget_before_final_unseen_exhaustion,
            "final_round_expected_unseen_terms": final_unseen_terms,
            "automatic_fit_rule": (
                "After generated residual labels are known, use all available holdout labels but reduce "
                "the per-round addition size when needed so feedback_iterations rounds and the requested "
                "post-final unseen batches remain nonempty."
            ),
        }
    )
    return effective_residual_top_k, effective_add, fitted


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
        if int(row.get("unseen_residual_terms", 1)) == 0 or row.get("unseen_relative_residual") is None
        else optional_float(row["unseen_relative_residual"])
        for row in rows
    ]
    series = [
        ("training", [float(row["training_final_relative_residual"]) for row in rows], OKABE_ITO[0], "o"),
        ("holdout", [float(row["holdout_relative_residual"]) for row in rows], OKABE_ITO[1], "s"),
        ("moving unseen quotient", unseen_values, OKABE_ITO[2], "^"),
    ]
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    for label, values, color, marker in series:
        ax.semilogy(x, values, marker=marker, linewidth=LINE_WIDTH, color=color, label=label)
    ax.axhline(thresholds.holdout, color="0.35", linestyle="--", linewidth=0.8)
    ax.axhline(thresholds.unseen, color="0.55", linestyle=":", linewidth=0.8)
    ax.set_xlabel("feedback round", fontsize=LABEL_FS)
    ax.set_ylabel("relative residual", fontsize=LABEL_FS)
    n_qubits = int(rows[0].get("n_qubits", 15)) if rows else 15
    ax.set_title(fr"$q={n_qubits}$ holdout-feedback residuals", fontsize=TITLE_FS)
    ax.set_xticks(x)
    ax.tick_params(axis="both", labelsize=TICK_FS, length=TICK_LENGTH, width=TICK_WIDTH)
    fig.legend(loc="upper center", ncol=3, frameon=False, fontsize=LEGEND_FS, bbox_to_anchor=(0.53, 1.02))
    fig.subplots_adjust(top=0.80, left=0.13, right=0.98, bottom=0.16)
    fig.savefig(images_dir / "holdout_feedback_relative_residuals.pdf", format="pdf")
    plt.close(fig)


def plot_fixed_unseen_probes(
    rows: list[dict[str, object]],
    output_path: Path,
    *,
    unseen_threshold: float,
) -> None:
    """Plot the stable active quotient separately from null leakage diagnostics."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_paper_style(plt)
    series = fixed_unseen_plot_series(rows)
    rounds = series["rounds"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    axes[0].semilogy(
        rounds,
        series["active_relative"],
        marker="o",
        linewidth=LINE_WIDTH,
        color=OKABE_ITO[0],
        label=str(series["labels"]["active_relative"]),
    )
    axes[0].axhline(unseen_threshold, linestyle=":", color="0.5", linewidth=0.8)
    axes[0].set_ylabel("relative residual", fontsize=LABEL_FS)
    axes[0].set_title("fixed active unseen gate", fontsize=TITLE_FS)
    axes[1].semilogy(
        rounds,
        series["null_absolute_per_term"],
        marker="s",
        linewidth=LINE_WIDTH,
        color=OKABE_ITO[1],
        label=str(series["labels"]["null_absolute_per_term"]),
    )
    axes[1].semilogy(
        rounds,
        series["null_scaled"],
        marker="^",
        linewidth=LINE_WIDTH,
        color=OKABE_ITO[2],
        label=str(series["labels"]["null_scaled"]),
    )
    axes[1].set_title("fixed null leakage", fontsize=TITLE_FS)
    for axis in axes:
        axis.set_xlabel("feedback round", fontsize=LABEL_FS)
        axis.set_xticks(rounds)
        axis.tick_params(axis="both", labelsize=TICK_FS, length=TICK_LENGTH, width=TICK_WIDTH)
        axis.legend(frameon=False, fontsize=LEGEND_FS)
    fig.subplots_adjust(top=0.84, left=0.10, right=0.98, bottom=0.18, wspace=0.34)
    fig.savefig(output_path, format="pdf")
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
    seen_rel = np.asarray([optional_float(row["seen_relative_residual"]) for row in rows])
    unseen_rel = np.asarray(
        [
            np.nan
            if int(row.get("unseen_residual_terms", 1)) == 0 or row.get("unseen_relative_residual") is None
            else optional_float(row["unseen_relative_residual"])
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
    for row in rows:
        if "run_dir" in row:
            row["run_dir"] = normalize_round_run_label(row["run_dir"])
    for row in round_rows:
        if "run_dir" in row:
            row["run_dir"] = normalize_round_run_label(row["run_dir"])
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


def assert_fixed_unseen_manifest_lifecycle(
    *,
    output_dir: Path,
    data_dir: Path,
    residual_top_k: int,
    enabled: bool = True,
) -> None:
    """Prevent normal training from upgrading historical runs into fixed-probe runs."""
    if not enabled:
        return
    manifest_path = data_dir / "fixed_unseen_probe_labels.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        if not isinstance(manifest, Mapping) or not bool(manifest.get("certification_eligible", False)):
            raise RuntimeError(
                "Cannot resume normal training from a diagnostics-only fixed unseen manifest. "
                "Start a new run root; historical diagnostic backfill remains certification-ineligible."
            )
        return

    markers: list[Path] = []
    summary_path = data_dir / f"holdout_feedback_summary_residual_{residual_top_k}.json"
    if summary_path.is_file():
        markers.append(summary_path)
    markers.extend(output_dir.glob("rounds/round_*"))
    markers.extend(output_dir.glob("runs/round_*"))
    markers.extend(output_dir.glob("temporal_refinement"))
    markers.extend(output_dir.glob("adaptive_temporal_refinement"))
    markers.extend(output_dir.rglob("training_checkpoint.pt"))
    markers.extend(data_dir.glob("holdout_feedback_spectrum_round_*.json"))
    markers = sorted({path.resolve() for path in markers}, key=str)
    if not markers:
        return

    preview = ", ".join(str(path) for path in markers[:3])
    if len(markers) > 3:
        preview += ", ..."
    raise RuntimeError(
        "Cannot create a certification-eligible fixed unseen probe manifest in "
        f"existing run root {output_dir}: prior feedback state was found ({preview}) "
        "but fixed_unseen_probe_labels.json is missing. Start a new run root for "
        "normal training. An explicit diagnostics-only refresh may backfill "
        "historical metrics, but remains certification-ineligible."
    )


def write_feedback_summary(
    *,
    output_dir: Path,
    rows: list[dict[str, object]],
    spectra: dict[int, list[dict[str, object]]],
    round_rows: list[dict[str, object]],
    residual_top_k: int,
    thresholds: Thresholds,
    residual_budget: dict[str, object],
    fixed_unseen_probe: Mapping[str, object] | None = None,
    temporal_refinement: dict[str, object] | None = None,
    adaptive_temporal_refinement: dict[str, object] | None = None,
    keep_round_images: bool = True,
) -> None:
    images_dir = output_dir / "Images"
    data_dir = output_dir / "Models_Data"
    images_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    decision = feedback_threshold_decision(
        rows,
        holdout_threshold=thresholds.holdout,
        unseen_threshold=thresholds.unseen,
        fixed_unseen_probe=fixed_unseen_probe,
    )
    payload = {
        "description": (
            "Holdout-feedback training: high-RMS unseen holdout residual strings are added to the "
            "training residual basis, while AGP support is kept fixed."
        ),
        "holdout_residual_terms": residual_top_k,
        "residual_budget": residual_budget,
        "fixed_unseen_probe": dict(fixed_unseen_probe) if fixed_unseen_probe is not None else None,
        "decision": decision,
        "moving_unseen_diagnostic": moving_unseen_diagnostics(rows),
        "rounds": round_rows,
        "rows": rows,
    }
    if temporal_refinement is not None:
        payload["temporal_refinement"] = temporal_refinement
    if adaptive_temporal_refinement is not None:
        payload["adaptive_temporal_refinement"] = adaptive_temporal_refinement
    with (data_dir / f"holdout_feedback_summary_residual_{residual_top_k}.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    plot_feedback_relative_residuals(rows, images_dir, thresholds)
    plot_fixed_unseen_probes(
        rows,
        images_dir / "holdout_feedback_fixed_unseen_probes.pdf",
        unseen_threshold=thresholds.unseen,
    )
    plot_feedback_seen_unseen(rows, images_dir)
    plot_feedback_residual_spectrum(rows, spectra, images_dir)
    plot_feedback_added_terms(round_rows, images_dir)
    if round_rows:
        final_round_dir = output_dir / str(round_rows[-1]["run_dir"])
        if temporal_refinement and temporal_refinement.get("enabled", False):
            candidate_dir = output_dir / str(temporal_refinement.get("run_dir", ""))
            if candidate_dir.is_dir():
                final_round_dir = candidate_dir
        if adaptive_temporal_refinement and adaptive_temporal_refinement.get("enabled", False):
            candidate_dir = output_dir / str(adaptive_temporal_refinement.get("run_dir", ""))
            if candidate_dir.is_dir():
                final_round_dir = candidate_dir
        coefficient_path = final_round_dir / "Models_Data" / "final_agp_coefficients.pt"
        if coefficient_path.is_file():
            coefficient_payload = torch.load(coefficient_path, map_location="cpu")
            labels = [str(label) for label in coefficient_payload["pauli_labels"]]
            ranked = rank_coefficients(coefficient_payload["counterdiabatic_coefficients"], labels)
            plot_support_map(ranked[:16], len(labels[0]), images_dir)
            plot_connection_summary(ranked, len(labels[0]), images_dir)
        else:
            for filename in ("hcd_coefficient_support_map.pdf", "hcd_connection_summary.pdf"):
                source = final_round_dir / "Images" / filename
                if source.is_file():
                    shutil.copy2(source, images_dir / filename)
    if not keep_round_images:
        for round_images in sorted((output_dir / ROUND_RUNS_DIRNAME).glob("round_*/Images")):
            shutil.rmtree(round_images, ignore_errors=True)


def build_holdout_feedback_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a configured sparse AGP with holdout-residual feedback.")
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
    parser.add_argument("--baseline-root", type=Path, default=None)
    parser.add_argument("--holdout-threshold", type=float, default=None)
    parser.add_argument("--unseen-threshold", type=float, default=None)
    parser.add_argument("--keep-round-images", action="store_true")
    parser.add_argument(
        "--refresh-fixed-unseen-only",
        action="store_true",
        help=(
            "Diagnostics-only historical backfill. Loads existing checkpoints and updates fixed-unseen "
            "summaries without training or modifying checkpoints; the resulting manifest is ineligible for certification."
        ),
    )
    return parser


def parse_holdout_feedback_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_holdout_feedback_argument_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_holdout_feedback_args(argv)

    config_path = args.config.resolve()
    configure_run_dir(config_path)
    payload = load_json(config_path)
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
    support_swap_settings = support_swap_settings_from_feedback(feedback)
    fixed_unseen_probe_settings = fixed_unseen_probe_settings_from_feedback(feedback)
    pau_transfer_stability_settings = pau_transfer_stability_settings_from_feedback(feedback)
    temporal_refinement_settings = temporal_refinement_settings_from_feedback(feedback)
    adaptive_temporal_settings = adaptive_temporal_refinement_settings_from_feedback(feedback)
    output_root_arg = args.output_root if args.output_root is not None else Path(str(feedback.get("output_root", "runs/holdout_feedback")))
    baseline_root_arg = args.baseline_root if args.baseline_root is not None else Path(str(feedback.get("baseline_root", "runs/baselines")))
    keep_round_images = bool(args.keep_round_images or feedback.get("keep_round_images", False))
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
    baseline_payload = payload_with_feedback_baseline_neural(payload)
    base_settings = settings_for_support(baseline_payload, base_agp_terms)
    feedback_base_settings = settings_for_support(payload, base_agp_terms)
    feedback_settings = replace(
        feedback_base_settings,
        epochs=epochs_per_round,
        lr=lr,
        optimizer=str(args.optimizer) if args.optimizer is not None else base_settings.optimizer,
        intermediate_top_k=intermediate_top_k,
        device=device_name,
    )
    baseline_root = baseline_root_arg if baseline_root_arg.is_absolute() else RUN_DIR / baseline_root_arg
    base_run = baseline_root / f"agp_{base_agp_terms}"
    legacy_base_run = RUN_DIR / "runs" / f"agp_{base_agp_terms}"
    base_checkpoint = base_run / "Models_Data" / "training_checkpoint.pt"
    if not base_checkpoint.is_file() and legacy_base_run != base_run:
        legacy_checkpoint = legacy_base_run / "Models_Data" / "training_checkpoint.pt"
        if legacy_checkpoint.is_file():
            print(f"use_legacy_baseline source={legacy_base_run.relative_to(RUN_DIR)}")
            base_run = legacy_base_run
            base_checkpoint = legacy_checkpoint
    if not base_checkpoint.is_file():
        if args.refresh_fixed_unseen_only:
            raise FileNotFoundError(
                "Diagnostics-only fixed-unseen refresh requires an existing baseline checkpoint; "
                "it will not train a missing baseline."
            )
        print(
            f"train_missing_baseline agp_terms={base_agp_terms} "
            f"epochs={base_settings.epochs} residual_terms={base_settings.residual_top_k}"
        )
        run_training(base_settings, base_run, baseline_payload)
    agp_labels, residual_labels = load_checkpoint_labels(base_checkpoint)
    current_residual_labels = set(residual_labels)
    trainable_state = projected_trainable_state_from_checkpoint(base_checkpoint)
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
    sweep_run_dirs = [
        base_run
        if support_size == base_agp_terms
        else baseline_root / f"agp_{support_size}"
        for support_size in support_sizes
    ]
    initial_fixed_probe_request = residual_top_k + fixed_unseen_probe_settings.candidate_multiplier * (
        fixed_unseen_probe_settings.active_terms + fixed_unseen_probe_settings.null_terms
    )
    fixed_probe_candidate_cap = fixed_unseen_probe_candidate_cap(
        feedback,
        moving_holdout_terms=residual_top_k,
    )
    initial_candidate_request = (
        initial_fixed_probe_request
        if fixed_probe_candidate_cap is None
        else min(initial_fixed_probe_request, fixed_probe_candidate_cap)
    )
    candidate_residual_labels, holdout_basis_agp_terms = build_common_holdout_residual_labels(
        run_dirs=sweep_run_dirs,
        config_payload=payload,
        residual_top_k=initial_candidate_request,
        intermediate_top_k=intermediate_top_k,
    )
    if len(candidate_residual_labels) < residual_top_k:
        print(
            "resolved_feedback_residual_budget_available "
            f"requested={residual_top_k} available={len(candidate_residual_labels)}"
        )
    residual_top_k, add_residual_terms, residual_budget = fit_residual_budget_to_available(
        residual_top_k=residual_top_k,
        add_residual_terms=add_residual_terms,
        residual_budget=residual_budget,
        available_residual_terms=len(candidate_residual_labels),
        initial_residual_terms=len(residual_labels),
        rounds=rounds,
        unseen_batches_after_final_iteration=unseen_residual_batches,
    )
    fixed_probe_request = max(
        residual_top_k,
        residual_top_k
        + fixed_unseen_probe_settings.candidate_multiplier
        * (fixed_unseen_probe_settings.active_terms + fixed_unseen_probe_settings.null_terms),
    )
    if len(candidate_residual_labels) < fixed_probe_request:
        bounded_fixed_probe_request = (
            fixed_probe_request
            if fixed_probe_candidate_cap is None
            else min(fixed_probe_request, fixed_probe_candidate_cap)
        )
        candidate_residual_labels, holdout_basis_agp_terms = build_common_holdout_residual_labels(
            run_dirs=sweep_run_dirs,
            config_payload=payload,
            residual_top_k=bounded_fixed_probe_request,
            intermediate_top_k=intermediate_top_k,
        )
    common_residual_labels = candidate_residual_labels[:residual_top_k]
    print(
        "fitted_feedback_residual_budget "
        f"Q={residual_top_k} add={add_residual_terms} "
        f"status={residual_budget['residual_budget_fit_status']} "
        f"final_unseen_budget={residual_budget['final_round_expected_unseen_terms']}"
    )
    output_root = output_root_arg if output_root_arg.is_absolute() else RUN_DIR / output_root_arg
    output_dir = output_root / f"agp_{base_agp_terms}_residual_{residual_top_k}_add_{add_residual_terms}_rounds_{rounds}"
    data_dir = output_dir / "Models_Data"
    assert_fixed_unseen_manifest_lifecycle(
        output_dir=output_dir,
        data_dir=data_dir,
        residual_top_k=residual_top_k,
        enabled=fixed_unseen_probe_settings.enabled and not args.refresh_fixed_unseen_only,
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    hamiltonian_path = Path(feedback_settings.model.hamiltonian_source)
    if not hamiltonian_path.is_absolute():
        hamiltonian_path = ROOT / hamiltonian_path
    h0_swap, h1_swap = load_pauli_hamiltonian_pair(
        hamiltonian_path,
        system=feedback_settings.model.system,
        n_qubits=feedback_settings.model.n_qubits,
        distance=feedback_settings.model.distance,
    )
    certification_run_roots = configured_generated_run_roots(
        payload,
        scenario_root=RUN_DIR,
        extra_roots=(baseline_root, output_root),
    )
    persisted_certification_labels, certification_manifest_paths = (
        load_existing_certification_probe_labels(certification_run_roots)
    )
    certification_probe_labels = (
        configured_certification_probe_labels(payload, feedback)
        | persisted_certification_labels
    )
    fixed_probe_base_excluded_labels = set(residual_labels) | certification_probe_labels
    historical_training_residual_labels: set[str] = set()
    if args.refresh_fixed_unseen_only:
        historical_checkpoints = sorted(
            {
                base_checkpoint.resolve(),
                *(
                    checkpoint.resolve()
                    for checkpoint in output_dir.glob("**/Models_Data/training_checkpoint.pt")
                ),
            },
            key=str,
        )
        if not historical_checkpoints:
            raise FileNotFoundError(
                "Diagnostics-only fixed-unseen refresh requires at least one retained checkpoint."
            )
        for checkpoint in historical_checkpoints:
            _, checkpoint_residual_labels = load_checkpoint_labels(checkpoint)
            historical_training_residual_labels.update(checkpoint_residual_labels)
        fixed_probe_base_excluded_labels.update(historical_training_residual_labels)

    def generate_fixed_probe_candidates(request: int) -> list[str]:
        labels, _ = build_common_holdout_residual_labels(
            run_dirs=sweep_run_dirs,
            config_payload=payload,
            residual_top_k=request,
            intermediate_top_k=intermediate_top_k,
        )
        return labels

    expected_fixed_unseen_probe, candidate_residual_labels = build_expanding_fixed_unseen_probe(
        generate_candidates=generate_fixed_probe_candidates,
        reference_rms_for_labels=lambda labels: fixed_unseen_reference_rms(
            h0=h0_swap,
            h1=h1_swap,
            settings=feedback_settings,
            agp_labels=agp_labels,
            candidate_labels=labels,
        ),
        settings=fixed_unseen_probe_settings,
        moving_holdout_terms=residual_top_k,
        excluded_labels=fixed_probe_base_excluded_labels,
        initial_request=fixed_probe_request,
        resource_cap=fixed_probe_candidate_cap,
    )
    common_residual_labels = candidate_residual_labels[:residual_top_k]
    fixed_probe_excluded_labels = set(residual_labels) | set(common_residual_labels) | certification_probe_labels
    fixed_probe_path = data_dir / "fixed_unseen_probe_labels.json"
    if fixed_probe_path.is_file():
        fixed_unseen_probe = load_or_validate_fixed_unseen_probe(
            fixed_probe_path,
            expected_excluded_labels=fixed_probe_excluded_labels,
            expected_identity=expected_fixed_unseen_probe,
            expected_reference_rms_metadata=expected_fixed_unseen_probe["reference_rms_metadata"],
            expected_selected_labels={
                "active": expected_fixed_unseen_probe["active_labels"],
                "null": expected_fixed_unseen_probe["null_labels"],
            },
        )
        if args.refresh_fixed_unseen_only and bool(fixed_unseen_probe.get("certification_eligible", False)):
            raise RuntimeError(
                "Diagnostics-only fixed-unseen refresh cannot overwrite or reclassify a "
                "certification-eligible manifest. Use a normal valid resume instead."
            )
    else:
        fixed_unseen_probe = build_fixed_unseen_probe_manifest(
            expected_fixed_unseen_probe,
            certification_eligible=not args.refresh_fixed_unseen_only,
            provenance=(
                "diagnostic_backfill"
                if args.refresh_fixed_unseen_only
                else "established_before_training"
            ),
        )
        fixed_unseen_probe["certification_probe_excluded_terms"] = len(certification_probe_labels)
        fixed_unseen_probe["historical_training_residual_excluded_terms"] = len(
            historical_training_residual_labels
        )
        fixed_unseen_probe["certification_probe_manifest_paths"] = [
            str(path) for path in certification_manifest_paths
        ]
        fixed_unseen_probe = build_fixed_unseen_probe_manifest(
            fixed_unseen_probe,
            certification_eligible=not args.refresh_fixed_unseen_only,
            provenance=(
                "diagnostic_backfill"
                if args.refresh_fixed_unseen_only
                else "established_before_training"
            ),
        )
        save_fixed_unseen_probe(fixed_probe_path, fixed_unseen_probe)
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
        baseline_row = merge_fixed_unseen_probe_metrics(
            baseline_row,
            evaluate_fixed_unseen_probe(
                run_dir=base_run,
                config_payload=payload,
                probe_metadata=fixed_unseen_probe,
                intermediate_top_k=intermediate_top_k,
                device=select_device("cpu"),
            ),
        )
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
        for row in rows:
            feedback_round = int(row.get("feedback_round", 0))
            checkpoint_run = base_run if feedback_round == 0 else output_dir / str(row["run_dir"])
            row.update(merge_fixed_unseen_probe_metrics(
                row,
                evaluate_fixed_unseen_probe(
                    run_dir=checkpoint_run,
                    config_payload=payload,
                    probe_metadata=fixed_unseen_probe,
                    intermediate_top_k=intermediate_top_k,
                    device=select_device("cpu"),
                ),
            ))
        rows_by_round = {int(row.get("feedback_round", 0)): row for row in rows}
        for round_row in round_rows:
            source = rows_by_round.get(int(round_row.get("round", 0)))
            if source is not None:
                round_row.update(
                    {key: value for key, value in source.items() if key.startswith("fixed_unseen_")}
                )
        if args.refresh_fixed_unseen_only:
            if completed_round < rounds:
                raise RuntimeError(
                    "Diagnostics-only fixed-unseen refresh found an incomplete feedback run. "
                    "It will not train missing rounds."
                )
            summary_path = data_dir / f"holdout_feedback_summary_residual_{residual_top_k}.json"
            summary_payload = load_json(summary_path)
            if not isinstance(summary_payload, dict):
                raise TypeError(f"Unexpected feedback summary format in {summary_path}.")
            temporal_summary = summary_payload.get("temporal_refinement")
            adaptive_summary = summary_payload.get("adaptive_temporal_refinement")
            checkpoint_runs = {
                base_run.resolve(),
                *(
                    checkpoint.parent.parent.resolve()
                    for checkpoint in output_dir.glob("**/Models_Data/training_checkpoint.pt")
                ),
            }
            evaluated_runs = {base_run.resolve()}
            for row in rows:
                feedback_round = int(row.get("feedback_round", 0))
                if feedback_round > 0:
                    evaluated_runs.add((output_dir / str(row["run_dir"])).resolve())
            for summary in (temporal_summary, adaptive_summary):
                if not isinstance(summary, dict) or not bool(summary.get("enabled", False)):
                    continue
                refinement_run = output_dir / str(summary["run_dir"])
                if not (refinement_run / "Models_Data" / "training_checkpoint.pt").is_file():
                    raise FileNotFoundError(
                        "Diagnostics-only fixed-unseen refresh requires every listed refinement checkpoint: "
                        f"{refinement_run}"
                    )
                summary.update(
                    merge_fixed_unseen_probe_metrics(
                        summary,
                        evaluate_fixed_unseen_probe(
                            run_dir=refinement_run,
                            config_payload=payload,
                            probe_metadata=fixed_unseen_probe,
                            intermediate_top_k=intermediate_top_k,
                            device=select_device("cpu"),
                        ),
                    )
                )
                evaluated_runs.add(refinement_run.resolve())
            if checkpoint_runs != evaluated_runs:
                missing = sorted(str(path) for path in checkpoint_runs - evaluated_runs)
                extra = sorted(str(path) for path in evaluated_runs - checkpoint_runs)
                raise RuntimeError(
                    "Diagnostics-only fixed-unseen refresh must evaluate every available checkpoint; "
                    f"missing={missing} extra={extra}"
                )
            write_feedback_summary(
                output_dir=output_dir,
                rows=rows,
                spectra=spectra,
                round_rows=round_rows,
                residual_top_k=residual_top_k,
                thresholds=thresholds,
                residual_budget=residual_budget,
                fixed_unseen_probe=fixed_unseen_probe,
                temporal_refinement=temporal_summary if isinstance(temporal_summary, dict) else None,
                adaptive_temporal_refinement=adaptive_summary if isinstance(adaptive_summary, dict) else None,
                keep_round_images=keep_round_images,
            )
            print(
                "fixed_unseen_diagnostic_backfill "
                f"checkpoints={len(checkpoint_runs)} certification_eligible=false"
            )
            return
        last_checkpoint = round_run_dir(output_dir, completed_round) / "Models_Data" / "training_checkpoint.pt"
        if completed_round > 0:
            agp_labels, residual_labels = load_checkpoint_labels(last_checkpoint)
            current_residual_labels = set(residual_labels)
            trainable_state = projected_trainable_state_from_checkpoint(last_checkpoint)
        if completed_round >= rounds:
            summary_path = data_dir / f"holdout_feedback_summary_residual_{residual_top_k}.json"
            summary_payload = load_json(summary_path)
            if isinstance(summary_payload, dict) and feedback_refinements_complete(
                summary_payload,
                output_dir,
                temporal_refinement_settings,
                adaptive_temporal_settings,
            ):
                temporal_summary = summary_payload.get("temporal_refinement")
                adaptive_summary = summary_payload.get("adaptive_temporal_refinement")
                for summary in (temporal_summary, adaptive_summary):
                    if not isinstance(summary, dict) or not bool(summary.get("enabled", False)):
                        continue
                    refinement_run = output_dir / str(summary["run_dir"])
                    summary.update(merge_fixed_unseen_probe_metrics(
                        summary,
                        evaluate_fixed_unseen_probe(
                            run_dir=refinement_run,
                            config_payload=payload,
                            probe_metadata=fixed_unseen_probe,
                            intermediate_top_k=intermediate_top_k,
                            device=select_device("cpu"),
                        ),
                    ))
                write_feedback_summary(
                    output_dir=output_dir,
                    rows=rows,
                    spectra=spectra,
                    round_rows=round_rows,
                    residual_top_k=residual_top_k,
                    thresholds=thresholds,
                    residual_budget=residual_budget,
                    fixed_unseen_probe=fixed_unseen_probe,
                    temporal_refinement=temporal_summary if isinstance(temporal_summary, dict) else None,
                    adaptive_temporal_refinement=adaptive_summary if isinstance(adaptive_summary, dict) else None,
                    keep_round_images=keep_round_images,
                )
                print(f"feedback_already_complete rounds={rounds}")
                return
            print(f"resume_feedback_refinements completed_round={completed_round}")

    for round_index in range(completed_round + 1, rounds + 1):
        previous_run = base_run if round_index == 1 else round_run_dir(output_dir, round_index - 1)
        support_swap_plan: dict[str, object] = {"enabled": False, "swap_count": 0}
        if (
            support_swap_settings.enabled
            and support_swap_settings.terms_per_iteration > 0
            and round_index >= support_swap_settings.start_round
        ):
            old_agp_labels = list(agp_labels)
            support_swap_plan = plan_fixed_k_support_swap(
                current_agp_labels=old_agp_labels,
                coefficient_importance=load_coefficient_importance_rows(previous_run),
                residual_spectrum=spectra[round_index - 1],
                h0=h0_swap,
                h1=h1_swap,
                max_swaps=support_swap_settings.terms_per_iteration,
                candidate_pool_size=(
                    support_swap_settings.terms_per_iteration
                    * support_swap_settings.candidate_pool_multiplier
                ),
                protect_top_fraction=support_swap_settings.protect_top_fraction,
            )
            if int(support_swap_plan.get("swap_count", 0)) > 0:
                agp_labels = [str(label) for label in support_swap_plan["new_agp_labels"]]
                trainable_state = remap_trainable_state_for_agp_labels(
                    trainable_state,
                    old_labels=old_agp_labels,
                    new_labels=agp_labels,
                    removed_labels=[str(label) for label in support_swap_plan.get("removed_labels", [])],
                    added_labels=[str(label) for label in support_swap_plan.get("added_labels", [])],
                    new_gate_logit=support_swap_settings.new_gate_logit,
                )
                print(
                    f"support_swap_round={round_index} swapped={support_swap_plan['swap_count']} "
                    f"removed={support_swap_plan.get('removed_labels', [])[:5]} "
                    f"added={support_swap_plan.get('added_labels', [])[:5]}"
                )
        additions = select_residual_additions(
            spectra[round_index - 1],
            current_residual_labels,
            excluded_labels=(
                set(fixed_unseen_probe.get("active_labels", []))
                | set(fixed_unseen_probe.get("null_labels", []))
            ),
            add_terms=add_residual_terms,
            min_rms=min_rms,
        )
        current_residual_labels.update(str(row["label"]) for row in additions)
        residual_labels = sort_pauli_labels(current_residual_labels)
        round_run = round_run_dir(output_dir, round_index)
        print(
            f"train_feedback_round={round_index} agp_terms={len(agp_labels)} "
            f"residual_terms={len(residual_labels)} added={len(additions)} epochs={feedback_settings.epochs}"
        )
        trainable_state, final, metadata = train_feedback_round(
            run_dir=round_run,
            payload=payload,
            settings=feedback_settings,
            agp_labels=agp_labels,
            residual_labels=residual_labels,
            trainable_state=trainable_state,
            round_index=round_index,
            additions=additions,
            support_swap_plan=support_swap_plan,
            pau_transfer_stability=pau_transfer_stability_settings,
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
        row = merge_fixed_unseen_probe_metrics(
            row,
            evaluate_fixed_unseen_probe(
                run_dir=round_run,
                config_payload=payload,
                probe_metadata=fixed_unseen_probe,
                intermediate_top_k=intermediate_top_k,
                device=select_device("cpu"),
            ),
        )
        rows.append(row)
        spectra[round_index] = spectrum
        row["spectrum_export"] = write_feedback_spectrum(
            data_dir,
            round_index=round_index,
            row=row,
            spectrum=spectrum,
        )
        round_summary = {
                "round": round_index,
                "run_dir": str(round_run.relative_to(output_dir)),
                "added_residual_terms": len(additions),
                "train_residual_terms": len(residual_labels),
                "training_final_relative_residual": float(final["relative_residual"]),
                "holdout_relative_residual": float(row["holdout_relative_residual"]),
                "unseen_relative_residual": row["unseen_relative_residual"],
                "unseen_relative_residual_status": row.get("unseen_relative_residual_status"),
                "unseen_residual": row.get("unseen_residual"),
                "unseen_reference_residual": row.get("unseen_reference_residual"),
                "unseen_residual_per_term": row.get("unseen_residual_per_term"),
                "first_added_terms": additions[:32],
                "support_metadata": {
                    "first_commutator_nnz": metadata["first_commutator_nnz"],
                    "second_commutator_nnz": metadata["second_commutator_nnz"],
                    "final_intermediate_terms": metadata["final_intermediate_terms"],
                    "final_residual_terms": metadata["final_residual_terms"],
                },
                "support_swap": metadata.get("support_swap", {"enabled": False, "swap_count": 0}),
            }
        round_summary.update(
            {
                key: row[key]
                for key in (
                    "fixed_unseen_active_terms",
                    "fixed_unseen_active_residual",
                    "fixed_unseen_active_reference_residual",
                    "fixed_unseen_active_relative",
                    "fixed_unseen_active_status",
                    "fixed_unseen_null_terms",
                    "fixed_unseen_null_absolute_per_term",
                    "fixed_unseen_null_scaled",
                )
                if key in row
            }
        )
        round_rows.append(round_summary)
        print(
            f"done_feedback_round={round_index} train_relative={final['relative_residual']:.6e} "
            f"holdout_relative={row['holdout_relative_residual']:.6e} "
            f"unseen_relative={optional_float(row['unseen_relative_residual']):.6e} "
            f"unseen_status={row.get('unseen_relative_residual_status', {}).get('reason', 'unknown')}"
        )
        write_feedback_summary(
            output_dir=output_dir,
            rows=rows,
            spectra=spectra,
            round_rows=round_rows,
            residual_top_k=residual_top_k,
            thresholds=thresholds,
            residual_budget=residual_budget,
            fixed_unseen_probe=fixed_unseen_probe,
            keep_round_images=keep_round_images,
        )

    temporal_refinement_summary: dict[str, object] | None = None
    if temporal_refinement_settings.enabled:
        if temporal_refinement_settings.epochs <= 0:
            raise ValueError("holdout_feedback.temporal_refinement.epochs must be positive when enabled.")
        if temporal_refinement_settings.num_points <= 0:
            raise ValueError("holdout_feedback.temporal_refinement.num_points must be positive when enabled.")
        if temporal_refinement_settings.lr <= 0.0:
            raise ValueError("holdout_feedback.temporal_refinement.lr must be positive when enabled.")
        refinement_settings = replace(
            feedback_settings,
            epochs=temporal_refinement_settings.epochs,
            num_points=temporal_refinement_settings.num_points,
            lr=temporal_refinement_settings.lr,
            optimizer=(
                temporal_refinement_settings.optimizer
                if temporal_refinement_settings.optimizer
                else feedback_settings.optimizer
            ),
        )
        refinement_run = output_dir / temporal_refinement_settings.run_dir
        print(
            "train_temporal_refinement "
            f"run_dir={temporal_refinement_settings.run_dir} "
            f"agp_terms={len(agp_labels)} residual_terms={len(residual_labels)} "
            f"epochs={refinement_settings.epochs} num_points={refinement_settings.num_points} "
            f"lr={refinement_settings.lr:g}"
        )
        trainable_state, refined_final, refined_metadata = train_feedback_round(
            run_dir=refinement_run,
            payload=payload,
            settings=refinement_settings,
            agp_labels=agp_labels,
            residual_labels=residual_labels,
            trainable_state=trainable_state,
            round_index=rounds + 1,
            additions=[],
            support_swap_plan={"enabled": False, "swap_count": 0, "reason": "temporal_refinement"},
        )
        refined_row, _ = evaluate_one_run(
            run_dir=refinement_run,
            config_payload=payload,
            residual_top_k=residual_top_k,
            intermediate_top_k=intermediate_top_k,
            device=select_device("cpu"),
            spectra_dir=data_dir,
            common_residual_labels=common_residual_labels,
            holdout_basis_mode="union_agp",
            holdout_basis_agp_terms=holdout_basis_agp_terms,
        )
        refined_row = merge_fixed_unseen_probe_metrics(
            refined_row,
            evaluate_fixed_unseen_probe(
                run_dir=refinement_run,
                config_payload=payload,
                probe_metadata=fixed_unseen_probe,
                intermediate_top_k=intermediate_top_k,
                device=select_device("cpu"),
            ),
        )
        temporal_refinement_summary = {
            "enabled": True,
            "run_dir": str(refinement_run.relative_to(output_dir)),
            "source_round": rounds,
            "epochs": refinement_settings.epochs,
            "num_points": refinement_settings.num_points,
            "lr": refinement_settings.lr,
            "optimizer": refinement_settings.optimizer,
            "training_final_relative_residual": float(refined_final["relative_residual"]),
            "holdout_relative_residual": float(refined_row["holdout_relative_residual"]),
            "unseen_relative_residual": refined_row["unseen_relative_residual"],
            "unseen_relative_residual_status": refined_row.get("unseen_relative_residual_status"),
            "support_metadata": {
                "first_commutator_nnz": refined_metadata["first_commutator_nnz"],
                "second_commutator_nnz": refined_metadata["second_commutator_nnz"],
                "final_intermediate_terms": refined_metadata["final_intermediate_terms"],
                "final_residual_terms": refined_metadata["final_residual_terms"],
            },
        }
        temporal_refinement_summary.update(
            {
                key: refined_row[key]
                for key in refined_row
                if key.startswith("fixed_unseen_")
            }
        )
        print(
            "done_temporal_refinement "
            f"train_relative={refined_final['relative_residual']:.6e} "
            f"holdout_relative={refined_row['holdout_relative_residual']:.6e} "
            f"unseen_relative={optional_float(refined_row['unseen_relative_residual']):.6e}"
        )

    adaptive_temporal_summary: dict[str, object] | None = None
    if adaptive_temporal_settings.enabled:
        if adaptive_temporal_settings.epochs <= 0:
            raise ValueError("holdout_feedback.adaptive_temporal_refinement.epochs must be positive when enabled.")
        if adaptive_temporal_settings.dense_points <= 1:
            raise ValueError("holdout_feedback.adaptive_temporal_refinement.dense_points must be greater than one when enabled.")
        if adaptive_temporal_settings.num_points <= 1:
            raise ValueError("holdout_feedback.adaptive_temporal_refinement.num_points must be greater than one when enabled.")
        if adaptive_temporal_settings.lr <= 0.0:
            raise ValueError("holdout_feedback.adaptive_temporal_refinement.lr must be positive when enabled.")
        if adaptive_temporal_settings.max_weight < adaptive_temporal_settings.min_weight:
            raise ValueError("holdout_feedback.adaptive_temporal_refinement.max_weight must be >= min_weight.")
        adaptive_settings = replace(
            feedback_settings,
            epochs=adaptive_temporal_settings.epochs,
            num_points=adaptive_temporal_settings.num_points,
            lr=adaptive_temporal_settings.lr,
            optimizer=(
                adaptive_temporal_settings.optimizer
                if adaptive_temporal_settings.optimizer
                else feedback_settings.optimizer
            ),
        )
        dense_tau, difficulty = adaptive_temporal_difficulty(
            payload=payload,
            settings=adaptive_settings,
            agp_labels=agp_labels,
            residual_labels=residual_labels,
            trainable_state=trainable_state,
            stage=rounds + 2,
            dense_points=adaptive_temporal_settings.dense_points,
            difficulty=adaptive_temporal_settings.difficulty,
        )
        adaptive_tau, temporal_sampling_metadata = make_adaptive_tau_grid(
            dense_tau,
            difficulty,
            num_points=adaptive_temporal_settings.num_points,
            weight_power=adaptive_temporal_settings.weight_power,
            min_weight=adaptive_temporal_settings.min_weight,
            max_weight=adaptive_temporal_settings.max_weight,
        )
        temporal_sampling_metadata.update(
            {
                "enabled": True,
                "difficulty": adaptive_temporal_settings.difficulty,
                "uses_ground_truth_observables": False,
                "source": "projected_euler_lagrange_residual_on_dense_time_grid",
            }
        )
        adaptive_run = output_dir / adaptive_temporal_settings.run_dir
        print(
            "train_adaptive_temporal_refinement "
            f"run_dir={adaptive_temporal_settings.run_dir} "
            f"agp_terms={len(agp_labels)} residual_terms={len(residual_labels)} "
            f"epochs={adaptive_settings.epochs} dense_points={adaptive_temporal_settings.dense_points} "
            f"num_points={adaptive_settings.num_points} lr={adaptive_settings.lr:g}"
        )
        trainable_state, adaptive_final, adaptive_metadata = train_feedback_round(
            run_dir=adaptive_run,
            payload=payload,
            settings=adaptive_settings,
            agp_labels=agp_labels,
            residual_labels=residual_labels,
            trainable_state=trainable_state,
            round_index=rounds + 2,
            additions=[],
            support_swap_plan={"enabled": False, "swap_count": 0, "reason": "adaptive_temporal_refinement"},
            tau_override=adaptive_tau,
            temporal_sampling_metadata=temporal_sampling_metadata,
        )
        adaptive_row, _ = evaluate_one_run(
            run_dir=adaptive_run,
            config_payload=payload,
            residual_top_k=residual_top_k,
            intermediate_top_k=intermediate_top_k,
            device=select_device("cpu"),
            spectra_dir=data_dir,
            common_residual_labels=common_residual_labels,
            holdout_basis_mode="union_agp",
            holdout_basis_agp_terms=holdout_basis_agp_terms,
        )
        adaptive_row = merge_fixed_unseen_probe_metrics(
            adaptive_row,
            evaluate_fixed_unseen_probe(
                run_dir=adaptive_run,
                config_payload=payload,
                probe_metadata=fixed_unseen_probe,
                intermediate_top_k=intermediate_top_k,
                device=select_device("cpu"),
            ),
        )
        adaptive_temporal_summary = {
            "enabled": True,
            "run_dir": str(adaptive_run.relative_to(output_dir)),
            "source": (
                temporal_refinement_summary["run_dir"]
                if temporal_refinement_summary is not None
                else str(round_run_dir(output_dir, rounds).relative_to(output_dir))
            ),
            "epochs": adaptive_settings.epochs,
            "dense_points": adaptive_temporal_settings.dense_points,
            "num_points": adaptive_settings.num_points,
            "lr": adaptive_settings.lr,
            "optimizer": adaptive_settings.optimizer,
            "difficulty": adaptive_temporal_settings.difficulty,
            "temporal_sampling": temporal_sampling_metadata,
            "training_final_relative_residual": float(adaptive_final["relative_residual"]),
            "holdout_relative_residual": float(adaptive_row["holdout_relative_residual"]),
            "unseen_relative_residual": adaptive_row["unseen_relative_residual"],
            "unseen_relative_residual_status": adaptive_row.get("unseen_relative_residual_status"),
            "support_metadata": {
                "first_commutator_nnz": adaptive_metadata["first_commutator_nnz"],
                "second_commutator_nnz": adaptive_metadata["second_commutator_nnz"],
                "final_intermediate_terms": adaptive_metadata["final_intermediate_terms"],
                "final_residual_terms": adaptive_metadata["final_residual_terms"],
            },
        }
        adaptive_temporal_summary.update(
            {
                key: adaptive_row[key]
                for key in adaptive_row
                if key.startswith("fixed_unseen_")
            }
        )
        print(
            "done_adaptive_temporal_refinement "
            f"train_relative={adaptive_final['relative_residual']:.6e} "
            f"holdout_relative={adaptive_row['holdout_relative_residual']:.6e} "
            f"unseen_relative={optional_float(adaptive_row['unseen_relative_residual']):.6e}"
        )

    write_feedback_summary(
        output_dir=output_dir,
        rows=rows,
        spectra=spectra,
        round_rows=round_rows,
        residual_top_k=residual_top_k,
        thresholds=thresholds,
        residual_budget=residual_budget,
        fixed_unseen_probe=fixed_unseen_probe,
        temporal_refinement=temporal_refinement_summary,
        adaptive_temporal_refinement=adaptive_temporal_summary,
        keep_round_images=keep_round_images,
    )
    summary_path = output_dir / "Models_Data" / f"holdout_feedback_summary_residual_{residual_top_k}.json"
    try:
        summary_label = str(summary_path.relative_to(RUN_DIR))
    except ValueError:
        summary_label = str(summary_path)
    full_basis = Decimal(4) ** int(model_config_from_payload(payload).n_qubits)
    final_round = round_rows[-1] if round_rows else {}
    compact_summary = {
        "summary": summary_label,
        "base_agp_terms": base_agp_terms,
        "agp_fraction_of_full_basis": f"{Decimal(base_agp_terms) / full_basis:.12E}",
        "rounds": len(round_rows),
        "final_round_holdout_relative_residual": final_round.get("holdout_relative_residual"),
        "temporal_holdout_relative_residual": (
            temporal_refinement_summary or {}
        ).get("holdout_relative_residual"),
        "adaptive_temporal_holdout_relative_residual": (
            adaptive_temporal_summary or {}
        ).get("holdout_relative_residual"),
    }
    print("holdout_feedback_summary " + json.dumps(compact_summary, sort_keys=True))


if __name__ == "__main__":
    getcontext().prec = 80
    main()
