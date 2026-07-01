from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

from projected_sparse_training_common import ProjectedTrainingConfig, main_for_config


if __name__ == "__main__":
    main_for_config(
        ProjectedTrainingConfig(system="Hidrogen", n_qubits=20, distance="1_0"),
        Path(__file__).resolve().parent,
    )
