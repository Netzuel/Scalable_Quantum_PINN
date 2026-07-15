# Transverse-Field to Diagonal-Ising Benchmarks

This scenario is the tractable physical playground used to evaluate the sparse
AGP curriculum across system sizes. It uses

```text
H_initial = -sum_i X_i
H_final   = -sum_i h_i Z_i - sum_i J_i Z_i Z_{i+1}
H_AD(lambda) = (1 - lambda) H_initial + lambda H_final.
```

The final Hamiltonian is diagonal in the computational basis, so its exact
ground energy and ground bitstrings are available from the structure-aware
solver under `scripts/numerical_solver/`. This makes the scenario useful for
physical validation without making those exact targets part of the PINN loss.

## Studies

- `q15/sweep_test/`: retained curriculum with exact statevector validation and
  tensor-network calibration.
- `q20/sweep_test/`: retained curriculum with canonical full-support MPS
  validation.
- `q156/sweep_test/`: retained large-system curriculum and scalable MPS
  validation status.

Each `sweep_test/` keeps its `config.json`, local documentation, and ignored
`runs/` tree together. Reusable training code remains in the repository-level
`scripts/` directory; benchmark-family validation entrypoints remain in
`tests/sparse_agp_curriculum/scripts/`.
