from __future__ import annotations

import json
import shutil
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent


def folders_to_reset() -> tuple[str, str]:
    config_path = RUN_DIR / "config.json"
    if not config_path.is_file():
        return ("Images", "Models_Data")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    export = payload.get("training", {}).get("export", {})
    return (
        str(export.get("path_images", "Images/")),
        str(export.get("path_data", "Models_Data/")),
    )


def reset() -> None:
    for name in folders_to_reset():
        path = RUN_DIR / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    reset()
