from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "tests" / "q20" / "sweep_test" / "config.json"
RUN_DIR = DEFAULT_CONFIG.parent


def configure_run_dir(config_path: Path) -> None:
    global RUN_DIR
    RUN_DIR = config_path.resolve().parent


def coupled_output_roots() -> list[Path]:
    roots: list[Path] = []
    config_path = RUN_DIR / "config.json"
    if config_path.is_file():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        output_root = payload.get("coupled_curriculum", {}).get("output_root")
        if output_root:
            path = Path(str(output_root))
            roots.append(path if path.is_absolute() else RUN_DIR / path)
    roots.append(RUN_DIR / "runs" / "coupled_curriculum")

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(root)
    return unique


def latest_summary() -> Path:
    candidates = sorted(
        (
            path
            for root in coupled_output_roots()
            for path in root.glob("*/Models_Data/coupled_curriculum_summary_residual_*.json")
        ),
        key=lambda path: (path.stat().st_mtime, str(path)),
    )
    if not candidates:
        roots = ", ".join(str(root) for root in coupled_output_roots())
        raise FileNotFoundError(f"No coupled curriculum summary found under configured roots: {roots}.")
    return candidates[-1]


def gate(value: float | None, threshold: float, *, lower_is_better: bool = True) -> dict[str, object]:
    if value is None:
        return {"status": "not tested", "value": None, "threshold": threshold}
    passed = value <= threshold if lower_is_better else value >= threshold
    return {"status": "pass" if passed else "fail", "value": value, "threshold": threshold}


def pct_improvement(previous: float | None, current: float | None) -> float | None:
    if previous is None or current is None or previous == 0.0:
        return None
    return (previous - current) / previous


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a sparse AGP diagnostic run against AGP_CERTIFICATION_CRITERIA.md.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--target-train", type=float, default=0.10)
    parser.add_argument("--target-holdout", type=float, default=0.10)
    parser.add_argument("--target-unseen", type=float, default=1.0)
    parser.add_argument("--target-probe", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    configure_run_dir(args.config)
    summary_path = args.summary or latest_summary()
    payload = json.loads(summary_path.read_text())
    rows = list(payload.get("rows", []))
    rounds = list(payload.get("rounds", []))
    if not rows:
        raise RuntimeError(f"No rows found in {summary_path}.")
    final_row = rows[-1]
    if rounds:
        accepted_rounds = [row for row in rounds if bool(row.get("agp_growth_accepted", False))]
        final_round = accepted_rounds[-1] if accepted_rounds else rounds[-1]
    else:
        final_round = {}

    oracle = payload.get("projected_linear_oracle") or {}
    pruning_candidates = None
    run_dir = final_round.get("run_dir")
    if run_dir:
        pruned_path = summary_path.parents[1] / str(run_dir) / "Models_Data" / "pruned_support_candidates.json"
        if pruned_path.is_file():
            pruning_candidates = json.loads(pruned_path.read_text())

    checks = {
        "training_residual": gate(
            float(final_row.get("training_final_relative_residual", final_round.get("training_final_relative_residual")))
            if final_row.get("training_final_relative_residual", final_round.get("training_final_relative_residual")) is not None
            else None,
            args.target_train,
        ),
        "holdout_residual": gate(
            float(final_row["holdout_relative_residual"]) if "holdout_relative_residual" in final_row else None,
            args.target_holdout,
        ),
        "unseen_residual": gate(
            float(final_row["unseen_relative_residual"]) if "unseen_relative_residual" in final_row else None,
            args.target_unseen,
        ),
        "probe_gate_residual": gate(
            float(final_row["probe_gate_relative_residual"]) if "probe_gate_relative_residual" in final_row else None,
            args.target_probe,
        ),
        "probe_watch_residual": gate(
            float(final_row["probe_watch_relative_residual"]) if "probe_watch_relative_residual" in final_row else None,
            args.target_probe,
        ),
        "probe_test_residual": gate(
            float(final_row["probe_test_relative_residual"]) if "probe_test_relative_residual" in final_row else None,
            args.target_probe,
        ),
        "k_sweep_plateau": {"status": "not tested", "note": "Requires at least two comparable K runs."},
        "q_sweep_plateau": {"status": "not tested", "note": "Requires evaluation on increasing residual probe sizes."},
        "top_terms_stability_across_k": {"status": "not tested"},
        "top_terms_stability_across_seeds": {"status": "not tested"},
        "prune_and_retest": {
            "status": "not tested" if pruning_candidates is None else "fail",
            "note": (
                "Pruned support candidates exist but must be retrained or re-evaluated on fixed probes."
                if pruning_candidates is not None
                else "Run scripts/diagnostics/agp_prune_support.py after training, then retest retained supports."
            ),
        },
        "projected_linear_oracle": {
            "status": "not tested"
            if not oracle
            else ("pass" if float(oracle.get("oracle_relative_residual", float("inf"))) <= args.target_probe else "fail"),
            "oracle_relative_residual": oracle.get("oracle_relative_residual"),
            "note": "Support-capacity diagnostic; passing is not full certification.",
        },
        "physical_validation": {"status": "not tested"},
    }

    hard_statuses = [row["status"] for row in checks.values()]
    simultaneous_keys = [
        "training_residual",
        "holdout_residual",
        "unseen_residual",
        "probe_gate_residual",
        "probe_watch_residual",
        "probe_test_residual",
    ]
    simultaneous_statuses = [checks[key]["status"] for key in simultaneous_keys]
    if all(status == "pass" for status in hard_statuses):
        claim_level = "certified_sparse_agp_for_this_path_and_tolerance"
    elif any(status == "pass" for status in hard_statuses) and not any(
        status == "fail" for status in simultaneous_statuses
    ):
        claim_level = "candidate_robust_sparse_agp"
    else:
        claim_level = "projected_sparse_agp_experiment"

    report = {
        "criteria_file": "AGP_CERTIFICATION_CRITERIA.md",
        "summary": str(summary_path),
        "claim_level": claim_level,
        "checks": checks,
        "decision": {
            "certified": claim_level == "certified_sparse_agp_for_this_path_and_tolerance",
            "reason": "Any fail or not-tested gate downgrades the claim level.",
        },
        "round_stop": payload.get("curriculum_stop"),
        "source_decision": payload.get("decision"),
    }
    output = args.output or (summary_path.parent / "certification_summary.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "claim_level": claim_level, "certified": report["decision"]["certified"]}, indent=2))


if __name__ == "__main__":
    main()
