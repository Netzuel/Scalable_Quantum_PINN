# q15 Hamiltonian-Pauli Graph Candidate

This is an isolated from-scratch graph-coefficient candidate for the retained
q15 diagonal-Ising sparse-AGP curriculum. It keeps `K=32768`, `Q=65536`, 15
feedback rounds, fixed-K support swaps, learned calibration/schedule, and both
temporal refinements. It changes only the coefficient architecture and its
output lineage.

All generated files stay below this folder's `runs/`. The retained parent
`sweep_test/runs/` tree is neither read as initialization nor overwritten.

```bash
conda run -n torch-mps python scripts/agp_restart.py --config tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q15/sweep_test/hamiltonian_pauli_graph/config.json
conda run --no-capture-output -n torch-mps python scripts/agp_holdout_feedback.py --config tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q15/sweep_test/hamiltonian_pauli_graph/config.json
conda run --no-capture-output -n torch-mps python tests/sparse_agp_curriculum/scripts/agp_physical_validation.py --config tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q15/sweep_test/hamiltonian_pauli_graph/config.json
conda run --no-capture-output -n torch-mps python -u tests/sparse_agp_curriculum/scripts/agp_mps_validation.py --config tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q15/sweep_test/hamiltonian_pauli_graph/config.json
```

Canonical q15 comparison uses exact statevector evolution. Tensor-network
evaluation is an additional backend calibration, not a replacement for the
exact q15 result. This candidate's full-`K` result is therefore diagnostic; see
`RESULTS.md`. The statevector command is a 1,024/2,048-term ablation selected
from the same temporal checkpoint; it cannot certify the full-`K` deployment.
