from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

from full_pauli_training_common import TrainingConfig, main_for_config


if __name__ == "__main__":
    main_for_config(
        TrainingConfig(system="Hidrogen", n_qubits=2, distance="1_0"),
        Path(__file__).resolve().parent,
    )
