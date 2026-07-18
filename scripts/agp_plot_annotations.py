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

PHYSICAL_TABLE_METHODS: tuple[tuple[str, str], ...] = (
    ("no_cd", "No counterdiabatic term"),
    ("kipu_dqfm_l1", "Nested commutator l=1"),
    ("learned_sparse_agp", "PINN sparse AGP"),
)

_TENSOR_NETWORK_BACKENDS = frozenset(
    ("quimb_mps", "quimb_product_formula", "tenpy_tdvp_mpo")
)


def _physical_method_result(results: Mapping[str, object], method: str) -> Mapping[str, object]:
    candidates = ("kipu_dqfm_l1", "nested_l1") if method == "kipu_dqfm_l1" else (method,)
    for candidate in candidates:
        row = results.get(candidate)
        if isinstance(row, Mapping):
            return row
    return {}


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _learned_support_counts(payload: Mapping[str, object]) -> tuple[int | None, int | None]:
    """Return the deployed and available learned-term counts for a validation row."""

    candidates: list[Mapping[str, object]] = []
    certification_resolution = payload.get("certification_resolution", {})
    if isinstance(certification_resolution, Mapping):
        identity = certification_resolution.get("validation_identity", {})
        if isinstance(identity, Mapping):
            candidates.append(identity)
    resolution_results = payload.get("resolution_results", [])
    if isinstance(resolution_results, Sequence) and resolution_results:
        final_resolution = resolution_results[-1]
        if isinstance(final_resolution, Mapping):
            settings = final_resolution.get("settings", {})
            if isinstance(settings, Mapping):
                candidates.append(settings)
            candidates.append(final_resolution)
    validation_identity = payload.get("validation_identity", {})
    if isinstance(validation_identity, Mapping):
        candidates.append(validation_identity)
    truncation = payload.get("learned_agp_truncation", {})
    if isinstance(truncation, Mapping):
        candidates.append(
            {
                "learned_terms": truncation.get("selected_terms"),
                "full_learned_terms": truncation.get("available_terms"),
            }
        )

    for candidate in candidates:
        selected = _positive_int(candidate.get("learned_terms"))
        available = _positive_int(candidate.get("full_learned_terms"))
        if selected is not None and available is not None:
            return selected, available
    return None, _positive_int(payload.get("full_learned_terms"))


def _learned_support_status(payload: Mapping[str, object]) -> str:
    selected, available = _learned_support_counts(payload)
    if selected is None or available is None:
        return "unknown"
    return "full" if selected == available else "truncated"


def _complete_tn_protocol_rows(payload: Mapping[str, object]) -> bool:
    results = payload.get("results", {})
    if not isinstance(results, Mapping):
        return False
    for method, _label in PHYSICAL_TABLE_METHODS:
        row = _physical_method_result(results, method)
        diagnostics = row.get("mps_diagnostics", {})
        if not isinstance(diagnostics, Mapping) or diagnostics.get("status") != "ok":
            return False
        completed = _positive_int(diagnostics.get("completed_steps"))
        planned = _positive_int(diagnostics.get("steps"))
        if completed is None or planned is None or completed < planned:
            return False
    return True


def _converged_full_support_tn(payload: Mapping[str, object]) -> bool:
    if str(payload.get("backend", "")) not in _TENSOR_NETWORK_BACKENDS:
        return False
    if payload.get("execution_mode") == "preflight_only":
        return False
    convergence = payload.get("convergence", {})
    compression = payload.get("compression", {})
    return (
        _learned_support_status(payload) == "full"
        and isinstance(convergence, Mapping)
        and convergence.get("status") == "pass"
        and isinstance(compression, Mapping)
        and compression.get("status") == "pass"
        and _complete_tn_protocol_rows(payload)
    )


def _certified_hcd_results(payload: Mapping[str, object]) -> Mapping[str, object]:
    """Expose only complete canonical or explicitly qualified TN rows in HCD plots."""

    results = payload.get("results", {})
    results = results if isinstance(results, Mapping) else {}
    if payload.get("execution_mode") == "preflight_only":
        return {}
    if str(payload.get("backend", "")) not in _TENSOR_NETWORK_BACKENDS:
        return results
    certification = payload.get("certification", {})
    certification = certification if isinstance(certification, Mapping) else {}
    certified_full_support = (
        certification.get("status") == "pass"
        and _learned_support_status(payload) == "full"
    )
    if not certified_full_support and not _converged_full_support_tn(payload):
        return {}
    return results


def _summary_claim_priority(payload: Mapping[str, object]) -> int:
    """Rank saved summaries by physical claim strength, not filename alone."""

    if payload.get("execution_mode") == "preflight_only":
        return 0
    certification = payload.get("certification", {})
    certification = certification if isinstance(certification, Mapping) else {}
    support_status = _learned_support_status(payload)
    if certification.get("status") == "pass":
        if support_status == "full":
            return 6
        if support_status == "unknown":
            return 4
    if _converged_full_support_tn(payload):
        return 5
    if str(payload.get("backend", "")) not in _TENSOR_NETWORK_BACKENDS:
        if support_status == "full":
            return 5
        if support_status == "unknown":
            return 3
    if payload.get("execution_mode") == "validation_status":
        return 1
    return 2


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
        found_at_root = False
        for pattern in (
            "Models_Data/physical_validation_summary.json",
            "Models_Data/hydrogen_physical_validation_summary.json",
            "Models_Data/mps_physical_validation_summary.json",
            "*/Models_Data/physical_validation_summary.json",
            "*/Models_Data/hydrogen_physical_validation_summary.json",
            "*/Models_Data/mps_physical_validation_summary.json",
            "*/*/Models_Data/physical_validation_summary.json",
            "*/*/Models_Data/hydrogen_physical_validation_summary.json",
            "*/*/Models_Data/mps_physical_validation_summary.json",
            "*/*/*/Models_Data/physical_validation_summary.json",
            "*/*/*/Models_Data/hydrogen_physical_validation_summary.json",
            "*/*/*/Models_Data/mps_physical_validation_summary.json",
            "*/*/*/*/Models_Data/physical_validation_summary.json",
            "*/*/*/*/Models_Data/hydrogen_physical_validation_summary.json",
            "*/*/*/*/Models_Data/mps_physical_validation_summary.json",
        ):
            try:
                for candidate in root.glob(pattern):
                    try:
                        resolved = candidate.resolve()
                        if candidate.is_file() and resolved not in seen:
                            candidates.append(candidate)
                            seen.add(resolved)
                            found_at_root = True
                    except OSError:
                        continue
            except OSError:
                continue
        if found_at_root:
            break
    if not candidates:
        return None
    existing: list[tuple[int, int, float, Path]] = []
    for candidate in candidates:
        try:
            summary_priority = {
                "mps_physical_validation_summary.json": 1,
                "hydrogen_physical_validation_summary.json": 2,
                "physical_validation_summary.json": 3,
            }.get(candidate.name, 0)
            payload = _load_mapping(candidate) or {}
            claim_priority = _summary_claim_priority(payload)
            existing.append(
                (claim_priority, summary_priority, candidate.stat().st_mtime, candidate)
            )
        except OSError:
            continue
    if not existing:
        return None
    return max(existing, key=lambda item: (item[0], item[1], item[2]))[3]


def _load_mapping(path: Path) -> dict[str, object] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _summary_applies_to_images_dir(
    payload: Mapping[str, object],
    *,
    summary_path: Path,
    images_dir: Path,
) -> bool:
    trained_raw = payload.get("trained_run")
    if not trained_raw:
        return True
    trained_run = Path(str(trained_raw))
    if not trained_run.is_absolute():
        resolved = None
        for parent in summary_path.parents:
            candidate = parent / trained_run
            if candidate.is_dir():
                resolved = candidate
                break
        if resolved is None:
            return True
        trained_run = resolved
    run_dir = Path(images_dir).parent.resolve()
    trained_run = trained_run.resolve()
    return run_dir == trained_run or run_dir in trained_run.parents


def _config_for_images_dir(images_dir: Path) -> dict[str, object] | None:
    run_dir = Path(images_dir).parent
    candidates = [
        run_dir / "Models_Data" / "config.json",
        *run_dir.glob("*/Models_Data/config.json"),
        *run_dir.glob("*/*/Models_Data/config.json"),
    ]
    cursor = run_dir
    for _ in range(8):
        candidates.append(cursor / "config.json")
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    fallback = None
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen or not candidate.is_file():
            continue
        seen.add(resolved)
        payload = _load_mapping(candidate)
        if payload is None:
            continue
        fallback = payload if fallback is None else fallback
        if isinstance(payload.get("scalable_validation"), Mapping):
            return payload
    return fallback


def physical_comparison_payload_for_images_dir(images_dir: Path) -> dict[str, object]:
    """Load physical results, or a scalable exact-reference-only fallback."""

    summary_path = find_physical_summary_for_images_dir(images_dir)
    if summary_path is not None:
        payload = _load_mapping(summary_path)
        if payload is not None and _summary_applies_to_images_dir(
            payload,
            summary_path=summary_path,
            images_dir=images_dir,
        ):
            return payload

    config = _config_for_images_dir(images_dir) or {}
    physical = config.get("physical", {})
    physical = physical if isinstance(physical, Mapping) else {}
    parameters = physical.get("parameters", {})
    parameters = parameters if isinstance(parameters, Mapping) else {}
    scalable = config.get("scalable_validation", {})
    scalable = scalable if isinstance(scalable, Mapping) else {}
    pinn_final_energy = _finite_float(scalable.get("pinn_final_energy"))
    results = (
        {"learned_sparse_agp": {"final_energy": pinn_final_energy}}
        if pinn_final_energy is not None
        else {}
    )
    return {
        "n_qubits": parameters.get("num_qubits", physical.get("n_qubits")),
        "system": parameters.get("system", physical.get("system")),
        "ground_energy": scalable.get("ground_energy"),
        "results": results,
        "availability_note": scalable.get(
            "reason",
            "Final-time dynamical energies and fidelities were not computed for this run.",
        ),
    }


def physical_comparison_rows(payload: Mapping[str, object]) -> list[dict[str, object]]:
    """Normalize exact, nested-l1, and learned-AGP metrics for table export."""

    results = payload.get("results", {})
    results = results if isinstance(results, Mapping) else {}
    ground_energy = _finite_float(payload.get("ground_energy"))
    if ground_energy is None:
        for row in results.values():
            if isinstance(row, Mapping):
                ground_energy = _finite_float(row.get("ground_energy"))
                if ground_energy is not None:
                    break

    rows: list[dict[str, object]] = [
        {
            "method": "Exact ground state",
            "final_energy": ground_energy,
            "energy_error": 0.0 if ground_energy is not None else None,
            "ground_state_fidelity": 1.0 if ground_energy is not None else None,
        }
    ]
    for key, label in PHYSICAL_TABLE_METHODS:
        source = _physical_method_result(results, key)
        diagnostics = source.get("mps_diagnostics", {})
        diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
        completed_steps = diagnostics.get("completed_steps")
        planned_steps = diagnostics.get("steps")
        diagnostic_status = str(diagnostics.get("status", "ok"))
        explicit_partial = diagnostic_status != "ok" or (
            completed_steps is not None
            and planned_steps is not None
            and int(completed_steps) < int(planned_steps)
        )
        result_is_complete = not explicit_partial and (
            str(payload.get("backend", "")) != "tenpy_tdvp_mpo"
            or (
                completed_steps is not None
                and planned_steps is not None
                and int(completed_steps) >= int(planned_steps)
            )
        )
        final_energy = _finite_float(source.get("final_energy")) if result_is_complete else None
        energy_error = _finite_float(source.get("energy_error")) if result_is_complete else None
        if energy_error is None and final_energy is not None and ground_energy is not None:
            energy_error = abs(final_energy - ground_energy)
        rows.append(
            {
                "method": label,
                "final_energy": final_energy,
                "energy_error": energy_error,
                "ground_state_fidelity": (
                    _finite_float(source.get("ground_state_fidelity")) if result_is_complete else None
                ),
            }
        )
    return rows


def _table_number(value: object) -> str:
    parsed = _finite_float(value)
    return "not computed" if parsed is None else f"{parsed:.7g}"


def physical_validation_note(payload: Mapping[str, object]) -> str:
    """Return an explicit qualification for tensor-network table metrics."""

    backend = str(payload.get("backend", ""))
    if backend not in {"quimb_mps", "quimb_product_formula", "tenpy_tdvp_mpo"}:
        return ""
    convergence = payload.get("convergence", {})
    convergence = convergence if isinstance(convergence, Mapping) else {}
    convergence_status = str(convergence.get("status", "not_tested")).replace("_", " ")
    certification = payload.get("certification", {})
    certification = certification if isinstance(certification, Mapping) else {}
    certified = str(certification.get("status", "not_tested")) == "pass"
    if backend in {"quimb_mps", "quimb_product_formula"} and convergence_status == "pass":
        return "MPS convergence: pass."
    if backend in {"quimb_mps", "quimb_product_formula"}:
        return f"MPS convergence: {convergence_status}; physical comparison is diagnostic only."
    full_terms = payload.get("full_learned_terms")
    deployed_terms = None
    ablation = False
    if full_terms is None:
        resolution_results = payload.get("resolution_results", [])
        if isinstance(resolution_results, Sequence) and resolution_results:
            final_resolution = resolution_results[-1]
            if isinstance(final_resolution, Mapping):
                full_terms = final_resolution.get("full_learned_terms")
                deployed_terms = final_resolution.get("learned_terms")
                ablation = bool(final_resolution.get("ablation", False))
                settings = final_resolution.get("settings", {})
                if isinstance(settings, Mapping):
                    deployed_terms = settings.get("learned_terms", deployed_terms)
    support = f"; full learned terms={full_terms}" if full_terms is not None else ""
    if ablation:
        support += f"; ablation deployed/available={deployed_terms}/{full_terms}"
    partial_rows: list[str] = []
    results = payload.get("results", {})
    if isinstance(results, Mapping):
        for protocol, row in results.items():
            if not isinstance(row, Mapping):
                continue
            diagnostics = row.get("mps_diagnostics", {})
            if not isinstance(diagnostics, Mapping):
                continue
            completed = diagnostics.get("completed_steps")
            planned = diagnostics.get("steps")
            protocol_status = str(diagnostics.get("status", "ok"))
            if protocol_status != "ok" or (
                completed is not None and planned is not None and int(completed) < int(planned)
            ):
                partial_rows.append(f"{protocol}={protocol_status}")
    partial_note = (
        f"; partial/not feasible final-time data ({', '.join(partial_rows)})"
        if partial_rows
        else ""
    )
    if certified:
        return (
            f"Backend: tenpy_tdvp_mpo{support}; convergence: {convergence_status}; "
            "certification: pass."
        )
    return (
        f"Backend: tenpy_tdvp_mpo{support}; convergence: {convergence_status}; "
        f"physical comparison is diagnostic only{partial_note}."
    )


def plot_physical_comparison_table(
    images_dir: Path,
    payload: Mapping[str, object] | None = None,
) -> Path:
    """Export an availability-aware exact/nested/PINN comparison table."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    normalized = dict(payload) if payload is not None else physical_comparison_payload_for_images_dir(images_dir)
    rows = physical_comparison_rows(normalized)
    n_qubits = normalized.get("n_qubits")
    q_text = f"q={n_qubits} " if n_qubits is not None else ""
    availability_note = str(normalized.get("availability_note", "")).strip()
    if not availability_note and any(row["final_energy"] is None for row in rows[1:]):
        availability_note = "Final-time dynamical energies and fidelities were not computed for this run."

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
    validation_note = physical_validation_note(normalized)
    fig, ax = plt.subplots(figsize=(8.6, 3.0 if validation_note else 2.8))
    ax.axis("off")
    column_labels = ["Method", r"$E(T)$", r"$|E(T)-E_0|$", r"$F_0(T)$"]
    cell_text = [
        [
            str(row["method"]),
            _table_number(row["final_energy"]),
            _table_number(row["energy_error"]),
            _table_number(row["ground_state_fidelity"]),
        ]
        for row in rows
    ]
    table = ax.table(
        cellText=cell_text,
        colLabels=column_labels,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.35, 0.20, 0.22, 0.20],
        bbox=[0.015, 0.24, 0.97, 0.58],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for column in range(len(column_labels)):
        header = table[(0, column)]
        header.set_facecolor("#E8E8E8")
        header.set_text_props(weight="bold", color="0.12")
        header.set_edgecolor("0.65")
    method_colors = ("#777777", "#009E73", "#D55E00", "#0072B2")
    for row_index, color in enumerate(method_colors, start=1):
        for column in range(len(column_labels)):
            cell = table[(row_index, column)]
            cell.set_edgecolor("0.78")
            cell.set_facecolor("#F7F7F7" if column else color)
        table[(row_index, 0)].set_text_props(color="white", weight="bold")

    ax.set_title(
        f"{q_text}final-time ground-state comparison",
        fontsize=14,
        pad=10,
    )
    note = availability_note or (
        r"$E_0$ is the exact final-Hamiltonian ground energy; "
        r"$F_0(T)$ is final-state ground-space fidelity."
    )
    if validation_note:
        note = f"{note}\n{validation_note}"
    fig.text(0.5, 0.075, note, ha="center", va="center", fontsize=8.5, color="0.25", wrap=True)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02)
    output = images_dir / "physical_method_comparison_table.pdf"
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return output


def physical_footer_lines_from_summary(summary_path: Path) -> list[str]:
    with Path(summary_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        return []
    return physical_footer_lines(payload)


def physical_footer_lines(payload: Mapping[str, object]) -> list[str]:
    results = _certified_hcd_results(payload)

    ground_energy = _format_number(payload.get("ground_energy"))
    if ground_energy is None:
        for row in results.values():
            if isinstance(row, Mapping):
                ground_energy = _format_number(row.get("ground_energy"))
                if ground_energy is not None:
                    break

    metric_parts: list[str] = []
    for method, label in PHYSICAL_METHODS:
        row = _physical_method_result(results, method)
        if not row:
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


def _hamiltonian_context_lines(system: str | None, n_qubits: int | None) -> list[str]:
    q = int(n_qubits) if n_qubits is not None else None
    if system == "TransverseIsingDriverProblem":
        upper = str(q) if q is not None else "q"
        edge_upper = str(q - 1) if q is not None else "q-1"
        return [
            rf"$H_{{\mathrm{{initial}}}}=-\sum_{{i=1}}^{{{upper}}}X_i$",
            (
                rf"$H_{{\mathrm{{final}}}}=-\sum_{{i=1}}^{{{upper}}}h_iZ_i"
                rf"-\sum_{{i=1}}^{{{edge_upper}}}J_iZ_iZ_{{i+1}}$"
            ),
        ]
    if system == "Hidrogen":
        return [
            (
                r"$H_{\mathrm{initial}}=\Pi_{\{I,Z\}}(H_{\mathrm{final}})"
                r"=\sum_{P\in\mathcal{S}_{IZ}}c_P P$"
            ),
            r"$H_{\mathrm{final}}=\sum_{P\in\mathcal{S}_{H_2}}c_P P$",
        ]
    if system == "TransverseFieldSpinHUBO":
        upper = str(q) if q is not None else "q"
        return [
            rf"$H_{{\mathrm{{initial}}}}=-\sum_{{i=1}}^{{{upper}}}X_i$",
            r"$H_{\mathrm{final}}=\sum_{S\in\mathcal{S}_{\mathrm{HUBO}}}c_S"
            r"\prod_{i\in S}Z_i$",
        ]
    return [
        r"$H_{\mathrm{initial}}=\sum_P c_P^{(0)}P$",
        r"$H_{\mathrm{final}}=\sum_P c_P^{(1)}P$",
    ]


def _tn_hcd_validation_line(payload: Mapping[str, object]) -> str | None:
    if str(payload.get("backend", "")) not in _TENSOR_NETWORK_BACKENDS:
        return None
    if payload.get("execution_mode") == "preflight_only":
        return "TN validation: preflight only; final-time metrics not certified."

    selected, available = _learned_support_counts(payload)
    support = available if selected == available else None
    support_text = f"full K={support}" if support is not None else "support not certified"
    certification = payload.get("certification", {})
    certification = certification if isinstance(certification, Mapping) else {}
    if certification.get("status") == "pass" and support is not None:
        return f"TN validation: {support_text}; timestep/bond/MPO gates pass; certified."
    if certification.get("status") == "pass":
        return "TN validation: support not certified; reduced/unknown-support metrics omitted."
    if _converged_full_support_tn(payload):
        statevector = payload.get("statevector_agreement", {})
        statevector = statevector if isinstance(statevector, Mapping) else {}
        if statevector.get("status") != "pass":
            return (
                f"TN validation: {support_text}; timestep/bond/MPO gates pass; "
                "exact full-support check not tested."
            )
    return f"TN validation: {support_text}; final-time metrics not certified."


def hcd_context_lines_for_images_dir(images_dir: Path) -> list[str]:
    """Return Hamiltonian and energy lines for an HCD connection summary."""

    config = _config_for_images_dir(images_dir) or {}
    physical = config.get("physical", {})
    physical = physical if isinstance(physical, Mapping) else {}
    parameters = physical.get("parameters", {})
    parameters = parameters if isinstance(parameters, Mapping) else {}
    system_raw = parameters.get("system", physical.get("system"))
    n_qubits_raw = parameters.get("num_qubits", physical.get("n_qubits"))
    try:
        n_qubits = int(n_qubits_raw) if n_qubits_raw is not None else None
    except (TypeError, ValueError):
        n_qubits = None
    system = str(system_raw) if system_raw is not None else None

    payload = physical_comparison_payload_for_images_dir(images_dir)
    ground_energy = _format_number(payload.get("ground_energy"))
    results = _certified_hcd_results(payload)
    ground_line = (
        rf"$E_0(H_{{\mathrm{{final}}}})={ground_energy}$"
        if ground_energy is not None
        else r"$E_0(H_{\mathrm{final}})=$ not computed"
    )

    def method_text(key: str, label: str) -> str:
        row = _physical_method_result(results, key)
        energy = _format_number(row.get("final_energy"))
        fidelity = _format_number(row.get("ground_state_fidelity"))
        energy_text = rf"$E(T)={energy}$" if energy is not None else r"$E(T)$=not computed"
        fidelity_text = (
            rf"$F_0(T)={fidelity}$" if fidelity is not None else r"$F_0(T)$=not computed"
        )
        return f"{label}: {energy_text}, {fidelity_text}"

    baseline_line = " | ".join(
        (
            method_text("no_cd", "no CD"),
            method_text("kipu_dqfm_l1", r"nested $l=1$"),
        )
    )
    pinn_line = method_text("learned_sparse_agp", "PINN AGP")

    lines = [
        *_hamiltonian_context_lines(system, n_qubits),
        ground_line,
        baseline_line,
        pinn_line,
    ]
    tn_validation_line = _tn_hcd_validation_line(payload)
    if tn_validation_line is not None:
        lines.append(tn_validation_line)
    return lines


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
    line_gap = 0.052
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
