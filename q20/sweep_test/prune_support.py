from __future__ import annotations

import argparse
import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent


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


def latest_coupled_run() -> Path:
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
    return candidates[-1].parents[1]


def accepted_run_from_summary(summary_path: Path) -> Path:
    payload = json.loads(summary_path.read_text())
    rounds = payload.get("rounds", [])
    accepted = [row for row in rounds if bool(row.get("agp_growth_accepted", False))]
    if accepted:
        run_dir = accepted[-1]["run_dir"]
    elif rounds:
        run_dir = rounds[-1]["run_dir"]
    else:
        rows = payload.get("rows", [])
        run_dir = rows[-1]["run_dir"] if rows else "runs/agp_unknown"
    return summary_path.parents[1] / str(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pruned AGP support candidates from coefficient importance.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--epsilons", default="1e-2,1e-3,1e-4")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.run_dir is None:
        coupled_dir = latest_coupled_run()
        summary_paths = sorted((coupled_dir / "Models_Data").glob("coupled_curriculum_summary_residual_*.json"))
        if not summary_paths:
            raise FileNotFoundError(f"No coupled summary found in {coupled_dir / 'Models_Data'}.")
        run_dir = accepted_run_from_summary(summary_paths[-1])
    else:
        run_dir = args.run_dir
    importance_path = run_dir / "Models_Data" / "coefficient_importance.json"
    if not importance_path.is_file():
        raise FileNotFoundError(f"Missing coefficient importance file: {importance_path}")
    importance = json.loads(importance_path.read_text())
    terms = list(importance.get("all_terms", []))
    if not terms:
        raise RuntimeError(f"No all_terms found in {importance_path}.")
    max_importance = max(float(row.get("importance", row.get("rms", 0.0))) for row in terms)
    epsilons = [float(item) for item in args.epsilons.replace(",", " ").split()]
    pruned = []
    for epsilon in epsilons:
        threshold = epsilon * max_importance
        retained = [
            str(row["label"])
            for row in terms
            if float(row.get("importance", row.get("rms", 0.0))) >= threshold
        ]
        pruned.append(
            {
                "epsilon": epsilon,
                "threshold": threshold,
                "retained_terms": len(retained),
                "retained_fraction": len(retained) / max(len(terms), 1),
                "labels": retained,
            }
        )
    output = args.output or (run_dir / "Models_Data" / "pruned_support_candidates.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_run_dir": str(run_dir),
        "importance_file": str(importance_path),
        "coefficient_definition": importance.get("coefficient_definition", "d_lambda_dt * C_P(t)"),
        "ranking_metric": importance.get("ranking_metric", "rms_over_time"),
        "initial_terms": len(terms),
        "max_importance": max_importance,
        "pruned_supports": pruned,
        "retest_rule": (
            "Retrain or re-evaluate each retained support on fixed holdout/probe bases. "
            "A pruned support passes only if residuals worsen by less than 5% to 10%."
        ),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "initial_terms": len(terms), "epsilons": epsilons}, indent=2))


if __name__ == "__main__":
    main()
