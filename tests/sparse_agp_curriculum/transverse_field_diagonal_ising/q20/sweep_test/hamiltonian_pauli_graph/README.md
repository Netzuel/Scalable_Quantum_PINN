# q20 Hamiltonian-Pauli Graph Candidate

This is an isolated from-scratch graph-coefficient candidate for the retained
q20 diagonal-Ising sparse-AGP curriculum. It keeps `K=32768`, `Q=81920`, 20
feedback rounds, fixed-K support swaps, learned calibration/schedule, and both
temporal refinements. It changes only the coefficient architecture and its
output lineage.

All generated files stay below this folder's `runs/`. The retained parent
`sweep_test/runs/` tree is neither read as initialization nor overwritten.

```bash
conda run -n torch-mps python scripts/agp_restart.py --config tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test/hamiltonian_pauli_graph/config.json
conda run --no-capture-output -n torch-mps python scripts/agp_holdout_feedback.py --config tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test/hamiltonian_pauli_graph/config.json
conda run --no-capture-output -n torch-mps python -u tests/sparse_agp_curriculum/scripts/agp_mps_validation.py --config tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test/hamiltonian_pauli_graph/config.json
```

Canonical q20 comparison uses every learned AGP term in the convergence-gated
tensor-network validation ladder. See `RESULTS.md` for the completed comparison.
