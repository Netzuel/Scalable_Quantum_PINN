"""Resample a trained projected AGP checkpoint without further optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
for path in (SCRIPTS_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agp_holdout_study import model_config_from_checkpoint_or_payload  # noqa: E402
from projected_sparse_training_common import (  # noqa: E402
    make_projected_export_model,
    projected_trainable_state_from_checkpoint,
    restore_projected_trainable_state,
    sample_projected_export_payload,
    select_device,
    settings_from_payload,
)


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def physical_export_sha256(payload: Mapping[str, object]) -> str:
    """Hash every ordered label and tensor that defines the deployed CD operator."""

    labels = payload.get("pauli_labels")
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        raise TypeError("A physical AGP export requires an ordered pauli_labels sequence.")
    digest = hashlib.sha256(b"projected_sparse_agp_physical_export_v2\0")
    digest.update(
        json.dumps(
            [str(label) for label in labels],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )
    tensor_fields = (
        "tau",
        "t",
        "lambda",
        "d_lambda_d_tau",
        "d_lambda_dt",
        "raw_agp_coefficients",
        "agp_coefficients",
        "counterdiabatic_coefficients",
        "calibration_gates",
    )
    for field in tensor_fields:
        tensor = payload.get(field)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Physical AGP export field {field!r} must be a tensor.")
        value = tensor.detach().cpu().contiguous()
        digest.update(field.encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    gamma = payload.get("calibration_gamma")
    if gamma is not None:
        digest.update(b"calibration_gamma\0")
        digest.update(float(gamma).hex().encode("ascii"))
    return digest.hexdigest()


def resample_checkpoint_export(
    *,
    config_path: Path,
    trained_run: Path,
    output_dir: Path,
    num_points: int,
    device: str = "cpu",
) -> Path:
    """Reconstruct and densely sample one immutable checkpoint artifact."""

    if int(num_points) < 2:
        raise ValueError("num_points must be at least two.")
    config_path = Path(config_path).resolve()
    trained_run = Path(trained_run).resolve()
    output_dir = Path(output_dir).resolve()
    checkpoint_path = trained_run / "Models_Data" / "training_checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing training checkpoint: {checkpoint_path}")
    final_path = output_dir / "Models_Data" / "final_agp_coefficients.pt"
    if final_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable resampled export: {final_path}")

    outer_payload = _load_json(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{checkpoint_path} must contain a checkpoint dictionary.")
    checkpoint_config = checkpoint.get("config", {})
    checkpoint_config = checkpoint_config if isinstance(checkpoint_config, dict) else {}
    embedded_source = checkpoint_config.get("source_config")
    source_payload = embedded_source if isinstance(embedded_source, dict) else outer_payload
    source_config_origin = "embedded_checkpoint" if isinstance(embedded_source, dict) else "external_config"
    if not isinstance(source_payload, dict):
        source_payload = outer_payload
    model_config = model_config_from_checkpoint_or_payload(checkpoint, source_payload)
    settings = settings_from_payload(source_payload, model_config)
    trainable_state = projected_trainable_state_from_checkpoint(checkpoint_path)
    agp_labels = [str(label) for label in checkpoint.get("agp_labels", [])]
    if not agp_labels:
        raise ValueError("Checkpoint must contain a non-empty AGP label list.")

    torch_device = select_device(device)
    model = make_projected_export_model(
        model_config,
        agp_labels,
        torch_device,
    )
    restore_projected_trainable_state(model, trainable_state, settings=settings)
    model.eval()
    tau = torch.linspace(0.0, 1.0, int(num_points), device=torch_device).view(-1, 1)
    t = model_config.t_initial + model_config.physical_time * tau
    source_hash = _sha256(checkpoint_path)
    provenance = {
        "kind": "checkpoint_temporal_resampling",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": source_hash,
        "source_config": str(config_path),
        "source_config_origin": source_config_origin,
        "source_config_sha256": _canonical_json_sha256(source_payload),
        "num_points": int(num_points),
        "device": str(torch_device),
        "optimization_steps": 0,
        "uses_ground_truth_observables": False,
    }
    support_metadata_path = trained_run / "Models_Data" / "support_metadata.json"
    if support_metadata_path.is_file():
        support_metadata = _load_json(support_metadata_path)
    else:
        raw_support = checkpoint_config.get("support", {})
        support_metadata = dict(raw_support) if isinstance(raw_support, dict) else {}
    export_payload = sample_projected_export_payload(model, tau, t, support_metadata)
    provenance["physical_export_sha256"] = physical_export_sha256(export_payload)
    export_payload["resampling_provenance"] = provenance

    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export_payload, final_path)
    with (final_path.parent / "resolved_source_config.json").open("w", encoding="utf-8") as handle:
        json.dump(source_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    manifest_path = final_path.parent / "resampling_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")
    return final_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trained-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-points", type=int, default=257)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = resample_checkpoint_export(
        config_path=args.config,
        trained_run=args.trained_run,
        output_dir=args.output_dir,
        num_points=args.num_points,
        device=args.device,
    )
    print(output)


if __name__ == "__main__":
    main()
