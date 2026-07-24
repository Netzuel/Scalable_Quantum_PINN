# q15 Retained Normalized Variational-Action PINN

This retained v6 benchmark uses the conventional independent-output PAU PINN,
the projected loss plus the normalized variational-action
loss with `beta_action=0.1`, normalized time `tau`, `K=32768`, `Q=65536`, 15
feedback rounds, 2048 active gates, and `T=1`.

Canonical physical evaluation deploys all 32768 learned AGP terms with exact
statevector evolution.

```bash
conda run -n torch-mps python scripts/agp_size_intensive_study.py --qubits 15 --clean --train --validate
```

The benchmark is trained independently from scratch and writes under
`runs/size_extensive_variational_action_v6/`.

```text
final energy = -19.1160784
exact energy = -19.25
energy error = 0.1339216
ground fidelity = 0.9768832
```
