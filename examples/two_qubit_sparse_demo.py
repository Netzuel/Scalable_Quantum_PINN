"""Small smoke example for the sparse AGP PINN.

This intentionally trains only for a few steps. It verifies the scalable code
path without trying to reproduce the original 500000-epoch manuscript runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import ScalableAGPPINN
from utils import all_local_pauli_labels, transverse_field_ising_problem


def main() -> None:
    torch.manual_seed(7)
    h_initial, h_final = transverse_field_ising_problem(2)
    agp_labels = all_local_pauli_labels(2, max_weight=2)
    model = ScalableAGPPINN(
        h_initial,
        h_final,
        agp_labels,
        hidden_width=16,
        hidden_layers=2,
        max_closure_weight=2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    t = torch.linspace(0.0, 1.0, 32)[:, None]
    for step in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = model.loss(t)
        loss.backward()
        optimizer.step()
        if step in {0, 19}:
            print(
                f"step={step:03d} loss={loss.item():.6e} "
                f"residual={diagnostics['residual'].item():.6e}"
            )
    print(f"basis_size={len(model.basis_labels)} agp_terms={len(model.agp_labels)}")


if __name__ == "__main__":
    main()
