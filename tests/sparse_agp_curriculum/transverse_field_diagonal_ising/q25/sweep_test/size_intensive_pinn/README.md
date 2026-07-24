# q25 Retained Normalized Variational-Action PINN

This retained v6 benchmark trains the conventional independent-output PAU PINN
from scratch. It uses the retained absolute projected loss plus the normalized
variational-action loss with `beta_action=0.1`, normalized time `tau`,
`K=91136`, `Q=182272`, 25 feedback rounds, 5888 active gates, 5120 new
residual equations per round, 512 support swaps per round, and fixed `T=1`.
Normalized time still enforces `d/dt=(1/T)d/dtau` explicitly.

Physical evaluation uses convergence-gated tensor-network TDVP and deploys all
91136 learned AGP terms. Only the learned PINN AGP protocol is evaluated.

```bash
conda run -n torch-mps python scripts/agp_size_intensive_study.py --qubits 25 --clean --train --validate
```

The fine all-K tensor-network result passes independent timestep, MPS-bond, MPO
completeness, and compression gates:

```text
final energy = -32.4501703
exact energy = -32.75
energy error = 0.2998297
ground fidelity = 0.9547459
```

The q20-to-q25 fidelity drop is `0.0217296`; this remains a known scaling
limitation of the retained methodology.
