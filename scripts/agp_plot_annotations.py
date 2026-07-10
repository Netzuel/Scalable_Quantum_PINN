"""Shared plot annotations for AGP benchmark figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence


PHYSICAL_METHODS: tuple[tuple[str, str], ...] = (
    ("learned_sparse_agp", "PINN AGP"),
    ("kipu_dqfm_l1", "nested l=1"),
    ("no_cd", "no CD"),
)


def _finite_float(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return None
    return parsed


def _format_number(value: object) -> str | None:
    parsed = _finite_float(value)
    if parsed is None:
        return None
    return f"{parsed:.6g}"


def find_physical_summary_for_images_dir(images_dir: Path) -> Path | None:
    """Find the nearest physical-validation summary for an HCD figure folder."""

    images_dir = Path(images_dir)
    run_dir = images_dir.parent
    candidates: list[Path] = []
    search_roots = [run_dir]
    cursor = run_dir
    for _ in range(3):
        cursor = cursor.parent
        if cursor == cursor.parent:
            break
        search_roots.append(cursor)

    seen: set[Path] = set()
    for root in search_roots:
        for pattern in (
            "Models_Data/physical_validation_summary.json",
            "*/Models_Data/physical_validation_summary.json",
            "*/*/Models_Data/physical_validation_summary.json",
            "*/*/*/Models_Data/physical_validation_summary.json",
            "*/*/*/*/Models_Data/physical_validation_summary.json",
        ):
            for candidate in root.glob(pattern):
                resolved = candidate.resolve()
                if candidate.is_file() and resolved not in seen:
                    candidates.append(candidate)
                    seen.add(resolved)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def physical_footer_lines_from_summary(summary_path: Path) -> list[str]:
    with Path(summary_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        return []
    return physical_footer_lines(payload)


def physical_footer_lines(payload: Mapping[str, object]) -> list[str]:
    results = payload.get("results", {})
    if not isinstance(results, Mapping):
        return []

    ground_energy = _format_number(payload.get("ground_energy"))
    if ground_energy is None:
        for row in results.values():
            if isinstance(row, Mapping):
                ground_energy = _format_number(row.get("ground_energy"))
                if ground_energy is not None:
                    break

    metric_parts: list[str] = []
    for method, label in PHYSICAL_METHODS:
        row = results.get(method)
        if not isinstance(row, Mapping):
            continue
        energy = _format_number(row.get("final_energy"))
        fidelity = _format_number(row.get("ground_state_fidelity"))
        if energy is None and fidelity is None:
            continue
        fields = []
        if energy is not None:
            fields.append(f"E_final={energy}")
        if fidelity is not None:
            fields.append(f"F0={fidelity}")
        metric_parts.append(f"{label}: " + ", ".join(fields))

    if not metric_parts:
        return []
    header = (
        f"Final-time physical check: real E0={ground_energy}"
        if ground_energy is not None
        else "Final-time physical check"
    )
    return [header, " | ".join(metric_parts)]


def _run_metadata_for_images_dir(images_dir: Path) -> tuple[str | None, int | None]:
    run_dir = Path(images_dir).parent
    candidates = [
        run_dir / "Models_Data" / "config.json",
        *run_dir.glob("*/Models_Data/config.json"),
        *run_dir.glob("*/*/Models_Data/config.json"),
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        physical = payload.get("physical", {})
        physical = physical if isinstance(physical, Mapping) else {}
        parameters = physical.get("parameters", {})
        parameters = parameters if isinstance(parameters, Mapping) else {}
        system = physical.get("system", parameters.get("system"))
        n_qubits = physical.get("n_qubits", parameters.get("num_qubits"))
        parsed_qubits = None
        try:
            if n_qubits is not None:
                parsed_qubits = int(n_qubits)
        except (TypeError, ValueError):
            parsed_qubits = None
        return (str(system) if system is not None else None, parsed_qubits)
    return (None, None)


def physical_footer_lines_for_images_dir(images_dir: Path) -> list[str]:
    summary_path = find_physical_summary_for_images_dir(images_dir)
    if summary_path is None:
        system, n_qubits = _run_metadata_for_images_dir(images_dir)
        if system and system != "TransverseIsingDriverProblem":
            q_text = f", q={n_qubits}" if n_qubits is not None else ""
            return [
                "Final-time physical check: unavailable for this run",
                f"Stored run is {system}{q_text}; no diagonal-Ising physical_validation_summary.json was produced.",
            ]
        return [
            "Final-time physical check: unavailable for this run",
            "No compatible physical_validation_summary.json was found beside this HCD figure.",
        ]
    return physical_footer_lines_from_summary(summary_path)


def footer_bottom_margin(default: float, footer_lines: Sequence[str]) -> float:
    if not footer_lines:
        return default
    return max(default, 0.30 + 0.035 * max(0, len(footer_lines) - 1))


def draw_physical_footer(fig, footer_lines: Sequence[str], *, fontsize: float = 8.2) -> None:
    if not footer_lines:
        return
    base_y = 0.035
    line_gap = 0.035
    for offset, line in enumerate(reversed(list(footer_lines))):
        fig.text(
            0.5,
            base_y + line_gap * offset,
            line,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="0.18",
        )
