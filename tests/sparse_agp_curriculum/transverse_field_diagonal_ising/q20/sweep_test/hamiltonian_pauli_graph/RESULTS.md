# q20 Hamiltonian-Pauli Graph Results

The candidate was trained independently from scratch with `K=32768`,
`Q=81920`, and 20 feedback rounds. Adaptive temporal refinement was selected
from projected holdout residual alone (`0.0271755`); no final-state energy or
fidelity entered training or checkpoint selection.

| Method | Final energy | Energy error | Ground-state fidelity |
|---|---:|---:|---:|
| no CD | -3.2396367 | 22.7603633 | 0.0000196521 |
| nested commutator l=1 | -12.2273803 | 13.7726197 | 0.008074661 |
| graph PINN AGP, full K | -16.2237519 | 9.7762481 | 0.119483734 |
| retained PINN AGP, full K | -25.6478383 | 0.3521617 | 0.937712840 |

The full-support tensor-network timestep, state/bond, MPO-compression, and
source-completeness gates pass, so the tensor-network numerical result is
certified. The AGP candidate as a methodology is not certified: its fixed
unseen projected-residual gate remains `not tested` because the generated probe
pool contained no active-reference terms. The graph candidate is not promoted
over the retained PINN.

Generated artifacts are under the independent `runs/` tree. The main plot
surfaces match the retained pipeline: coefficient/support maps, loss plot,
`hcd_connection_summary.pdf`, and `physical_method_comparison_table.pdf`.
