from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from projected_sparse_training_common import plot_connection_summary, rank_coefficients  # noqa: E402


def hcd_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("hcd_connection_summary.pdf")
        if ".git" not in path.parts
    )


def coefficient_priority(path: Path) -> tuple[int, int, float]:
    parts = set(path.parts)
    if "adaptive_temporal_refinement" in parts:
        stage_priority = 0
    elif "temporal_refinement" in parts:
        stage_priority = 1
    elif "rounds" in parts:
        stage_priority = 2
    else:
        stage_priority = 3
    round_index = -1
    for part in path.parts:
        if part.startswith("round_"):
            try:
                round_index = int(part.split("_", 1)[1])
            except ValueError:
                round_index = -1
    return (stage_priority, -round_index, -path.stat().st_mtime)


def find_coefficient_export(run_dir: Path) -> Path | None:
    direct = run_dir / "Models_Data" / "final_agp_coefficients.pt"
    if direct.is_file():
        return direct
    summaries = sorted(
        [
            *run_dir.glob("**/Models_Data/physical_validation_summary.json"),
            *run_dir.glob("**/Models_Data/mps_physical_validation_summary.json"),
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for summary_path in summaries:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(summary, dict) or not summary.get("trained_run"):
            continue
        trained_run = Path(str(summary["trained_run"]))
        if not trained_run.is_absolute():
            trained_run = summary_path.parents[2] / trained_run
        try:
            trained_run.resolve().relative_to(run_dir.resolve())
        except ValueError:
            continue
        selected = trained_run / "Models_Data" / "final_agp_coefficients.pt"
        if selected.is_file():
            return selected
    descendants = sorted(
        run_dir.glob("**/Models_Data/final_agp_coefficients.pt"),
        key=coefficient_priority,
    )
    return descendants[0] if descendants else None


def regenerate_one(pdf_path: Path) -> Path:
    images_dir = pdf_path.parent
    run_dir = images_dir.parent
    coefficient_path = find_coefficient_export(run_dir)
    if coefficient_path is None:
        raise FileNotFoundError(f"missing coefficient export under: {run_dir}")
    payload = torch.load(coefficient_path, map_location="cpu")
    coefficients = payload.get("counterdiabatic_coefficients")
    if coefficients is None:
        coefficients = payload["d_lambda_dt"] * payload["agp_coefficients"]
    labels = [str(label) for label in payload["pauli_labels"]]
    ranked = rank_coefficients(coefficients, labels)
    plot_connection_summary(ranked, len(labels[0]), images_dir)
    return coefficient_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate every saved HCD connection-summary PDF.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Exit non-zero if any existing HCD summary cannot be regenerated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    regenerated: list[str] = []
    failures: list[dict[str, str]] = []
    for pdf_path in hcd_paths(root):
        try:
            coefficient_path = regenerate_one(pdf_path)
        except Exception as exc:  # pragma: no cover - CLI report path
            failures.append({"pdf": str(pdf_path.relative_to(root)), "error": str(exc)})
            continue
        regenerated.append(str(pdf_path.relative_to(root)))
        print(f"regenerated_hcd={pdf_path.relative_to(root)} coefficients={coefficient_path.relative_to(root)}")

    summary = {
        "root": str(root),
        "regenerated_count": len(regenerated),
        "failure_count": len(failures),
        "failures": failures,
    }
    print(json.dumps(summary, indent=2))
    if failures and args.require_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
