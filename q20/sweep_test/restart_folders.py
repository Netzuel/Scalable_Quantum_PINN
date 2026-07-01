from __future__ import annotations

import json
import shutil
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent


def configured_paths() -> list[Path]:
    config_path = RUN_DIR / "config.json"
    if not config_path.is_file():
        return [RUN_DIR / "runs", RUN_DIR / "Images", RUN_DIR / "Models_Data"]
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = payload.get("summary", {})
    return [
        RUN_DIR / str(summary.get("runs_dir", "runs/")),
        RUN_DIR / str(summary.get("path_images", "Images/")),
        RUN_DIR / str(summary.get("path_data", "Models_Data/")),
    ]


def reset() -> None:
    for path in configured_paths():
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    reset()
