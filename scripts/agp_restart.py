from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


RUN_DIR = Path.cwd()
DEFAULT_CONFIG = Path("config.json")
LEGACY_SCRATCH_DIRS = ("Images", "Models_Data", "__pycache__")


def configure_run_dir(config_path: Path) -> None:
    global RUN_DIR
    RUN_DIR = config_path.resolve().parent


def configured_paths(config_path: Path | None = None) -> list[Path]:
    config_path = (RUN_DIR / "config.json") if config_path is None else config_path
    if not config_path.is_file():
        return [RUN_DIR / "runs"]
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = payload.get("summary", {})
    return [RUN_DIR / str(summary.get("runs_dir", "runs/"))]


def reset(config_path: Path | None = None) -> None:
    if config_path is not None:
        configure_run_dir(config_path)
    for path in configured_paths(config_path):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    for name in LEGACY_SCRATCH_DIRS:
        path = RUN_DIR / name
        if path.exists():
            shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean generated artifacts for a configured AGP study.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    reset(args.config.resolve())


if __name__ == "__main__":
    main()
