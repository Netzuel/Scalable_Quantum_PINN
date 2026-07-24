"""Canonical physical-validation identity shared by statevector and MPO CLIs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


VALIDATION_IDENTITY_KEYS = (
    "n_qubits",
    "hamiltonian_identity",
    "schedule_identity",
    "schedule_parameters_identity",
    "total_time",
    "checkpoint_identity",
    "coefficient_identity",
    "learned_terms",
    "full_learned_terms",
    "learned_scale",
    "initial_state",
    "ground_reference_identity",
    "ground_bitstring",
    "steps",
    "integrator",
    "statevector_integrator",
)


def canonical_hash(value: object) -> str:
    def normalize(item: object) -> object:
        if isinstance(item, Mapping):
            return {
                str(key): normalize(value)
                for key, value in sorted(item.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(item, (list, tuple)):
            return [normalize(value) for value in item]
        if isinstance(item, np.ndarray):
            return {
                "shape": list(item.shape),
                "dtype": str(item.dtype),
                "sha256": hashlib.sha256(item.tobytes()).hexdigest(),
            }
        if isinstance(item, complex):
            return [item.real, item.imag]
        if isinstance(item, np.generic):
            return item.item()
        return item

    encoded = json.dumps(normalize(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def hamiltonian_identity(h0: object, h1: object) -> str:
    return canonical_hash(
        {
            "h0": sorted((label, complex(value)) for label, value in h0.terms.items()),
            "h1": sorted((label, complex(value)) for label, value in h1.terms.items()),
        }
    )


def ground_reference_identity(
    *,
    ground_energy: float,
    ground_bitstring: str,
    reference_path: Path | None,
) -> str:
    source: dict[str, object] = {
        "ground_energy": float(ground_energy),
        "ground_bitstring": str(ground_bitstring),
    }
    if reference_path is not None:
        source["reference_path"] = str(reference_path.resolve())
        source["reference_sha256"] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    return canonical_hash(source)


def schedule_parameters_identity(learned: Mapping[str, object]) -> str:
    return canonical_hash(
        {
            "schedule_source": learned.get("schedule_source"),
            "tau": np.asarray(learned["tau"], dtype=np.float64),
            "lambda": (
                None
                if learned.get("lambda") is None
                else np.asarray(learned["lambda"], dtype=np.float64)
            ),
            "d_lambda_dt": (
                None
                if learned.get("d_lambda_dt") is None
                else np.asarray(learned["d_lambda_dt"], dtype=np.float64)
            ),
            "d_lambda_d_tau": (
                None
                if learned.get("d_lambda_d_tau") is None
                else np.asarray(learned["d_lambda_d_tau"], dtype=np.float64)
            ),
            "time_normalization": learned.get("time_normalization"),
        }
    )


def schedule_identity(learned: Mapping[str, object], *, learned_scale: float) -> str:
    return canonical_hash(
        {
            "schedule_parameters_identity": schedule_parameters_identity(learned),
            "learned_scale": float(learned_scale),
        }
    )


def build_validation_identity(
    *,
    h0: object,
    h1: object,
    learned: Mapping[str, object],
    coefficient_path: Path,
    ground_energy: float,
    ground_bitstring: str,
    total_time: float,
    steps: int,
    learned_scale: float,
    mps_integrator: str,
    ground_reference_path: Path | None = None,
) -> dict[str, object]:
    checkpoint = checkpoint_identity(coefficient_path)
    return {
        "n_qubits": int(h0.n_qubits),
        "hamiltonian_identity": hamiltonian_identity(h0, h1),
        "schedule_identity": schedule_identity(learned, learned_scale=learned_scale),
        "schedule_parameters_identity": schedule_parameters_identity(learned),
        "total_time": float(total_time),
        "checkpoint_identity": checkpoint,
        "coefficient_identity": checkpoint,
        "learned_terms": int(learned["selected_terms"]),
        "full_learned_terms": int(learned["available_terms"]),
        "learned_scale": float(learned_scale),
        "initial_state": "+" * int(h0.n_qubits),
        "ground_reference_identity": ground_reference_identity(
            ground_energy=ground_energy,
            ground_bitstring=ground_bitstring,
            reference_path=ground_reference_path,
        ),
        "ground_bitstring": str(ground_bitstring),
        "steps": int(steps),
        "integrator": str(mps_integrator),
        "statevector_integrator": "rk4_renormalized",
    }


def validation_identity_from_settings(settings: Mapping[str, object]) -> dict[str, object]:
    return {key: settings[key] for key in VALIDATION_IDENTITY_KEYS}
