from __future__ import annotations

import json
import shutil
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
LEGACY_SCRATCH_DIRS = ("Images", "Models_Data", "__pycache__")


def configured_paths() -> list[Path]:
    config_path = RUN_DIR / "config.json"
    if not config_path.is_file():
        return [RUN_DIR / "runs"]
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = payload.get("summary", {})
    return [RUN_DIR / str(summary.get("runs_dir", "runs/"))]


def reset() -> None:
    for path in configured_paths():
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    for name in LEGACY_SCRATCH_DIRS:
        path = RUN_DIR / name
        if path.exists():
            shutil.rmtree(path)


if __name__ == "__main__":
    reset()
