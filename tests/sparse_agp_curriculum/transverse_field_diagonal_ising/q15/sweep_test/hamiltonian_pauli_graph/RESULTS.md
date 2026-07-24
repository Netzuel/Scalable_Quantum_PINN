# q15 Hamiltonian-Pauli Graph Results

The candidate was trained independently from scratch with `K=32768`,
`Q=65536`, and 15 feedback rounds. The temporal-refinement checkpoint was
selected from projected holdout residual alone (`0.0487475`); no final-state
energy or fidelity entered training or checkpoint selection.

| Method | Final energy | Energy error | Ground-state fidelity |
|---|---:|---:|---:|
| no CD | -2.3932406 | 16.8567594 | 0.000287405 |
| nested commutator l=1 | -9.0861275 | 10.1638725 | 0.025913405 |
| graph PINN AGP, full K | -12.4333648 | 6.8166352 | 0.158574129 |
| retained PINN AGP, full K | -19.1097006 | 0.1402994 | 0.964651000 |

The full-support tensor-network state/bond and MPO-compression gates pass, but
the timestep gate fails narrowly and exact statevector agreement is not tested.
The graph result is diagnostic and is not promoted over the retained PINN.

Generated artifacts are under the independent `runs/` tree. The main plot
surfaces match the retained pipeline: coefficient/support maps, loss plot,
`hcd_connection_summary.pdf`, and `physical_method_comparison_table.pdf`.
