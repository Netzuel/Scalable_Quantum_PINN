# q15 Expressive Hamiltonian-Pauli Factor-Graph Candidate

This independent candidate retains the q15 benchmark curriculum (`K=32768`,
`Q=65536`, 15 rounds) and replaces only the coefficient body with the signed,
factor-aware graph-v2 architecture. It starts from random weights and cannot
reuse retained-PINN, graph-v1, q20, or other-system checkpoints.

Generated artifacts remain below this folder's ignored `runs/` directory.

```bash
conda run -n torch-mps python scripts/agp_restart.py --config tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q15/sweep_test/hamiltonian_pauli_factor_graph/config.json
conda run --no-capture-output -n torch-mps python scripts/agp_holdout_feedback.py --config tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q15/sweep_test/hamiltonian_pauli_factor_graph/config.json
conda run --no-capture-output -n torch-mps python -u tests/sparse_agp_curriculum/scripts/agp_mps_validation.py --config tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q15/sweep_test/hamiltonian_pauli_factor_graph/config.json
```

Promotion requires full-support fidelity at least `0.95` for both q15 and q20,
with q20 no more than `0.01` below q15. Fidelity is never used in training or
checkpoint selection.
