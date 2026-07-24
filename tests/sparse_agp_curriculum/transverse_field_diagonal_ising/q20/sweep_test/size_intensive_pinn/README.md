# q20 Retained Normalized Variational-Action PINN

This retained v6 benchmark trains the conventional independent-output PAU PINN from
scratch with the retained absolute projected loss plus the normalized
variational-action auxiliary loss at `beta_action=0.1`. It uses normalized time
`tau`, `K=58368`, `Q=116736`, 20 feedback rounds, 3840 active gates, 4096 new
residual equations per round, 512 support swaps per round, and fixed `T=1`.
Normalized time still enforces `d/dt=(1/T)d/dtau` explicitly.

Physical evaluation uses convergence-gated tensor-network TDVP and deploys all
58368 learned AGP terms. Only the learned PINN AGP protocol is evaluated.

```bash
conda run -n torch-mps python scripts/agp_size_intensive_study.py --qubits 20 --clean --train --validate
```

The fine all-K tensor-network result passes independent timestep, MPS-bond, MPO
completeness, and compression gates:

```text
final energy = -25.8392797
exact energy = -26.0
energy error = 0.1607203
ground fidelity = 0.9764755
```
