# Sparse AGP Curriculum Benchmarks

This framework retains sparse-AGP curriculum benchmarks grouped first by
physical scenario and then by qubit count. The current retained scenario uses

```text
H_AD(lambda) = (1 - lambda) H_initial + lambda H_final.
```

The retained playground uses a transverse-field driver and diagonal open-chain
Ising target at `q=15`, `q=20`, and `q=156`. The spin-HUBO scenario extends the
same curriculum to a nonlocal mixed one-, two-, and three-spin objective at
`q=24`. Exact statevector validation is retained through `q=15`; converged
tensor-network validation is required above that threshold.

## Layout

```text
sparse_agp_curriculum/
  transverse_field_diagonal_ising/
    q15/sweep_test/    retained q=15 configuration and local runs
    q20/sweep_test/    retained q=20 configuration and local runs
    q156/sweep_test/   retained q=156 configuration and local runs
  transverse_field_spin_hubo/
    run_002_hamiltonian_341/q24/sweep_test/
                       q=24 nonlocal spin-HUBO configuration and local runs
  ground_truth/     curated exact validation data
  scripts/          Python entrypoints specific to this benchmark family
```

The shared sparse-AGP training and curriculum implementation remains in the
repository-level `scripts/` folder. This avoids maintaining scenario-specific
copies of reusable training code.

## Framework Scripts

- `build_driver_problem_hamiltonian.py`: generate the analytic Ising Hamiltonian pair used by the retained studies and optional grid studies.
- `agp_physical_validation.py`: compare no CD, nested-commutator `l=1`, and the learned AGP where statevector validation is configured.
- `agp_qubit_grid_benchmark.py`: prepare and orchestrate optional multi-q Ising studies.
- `agp_regenerate_hcd_summaries.py`: regenerate HCD connection-summary figures.
- `spin_hubo_benchmark.py`: convert tuple-keyed spin objectives into sparse
  Pauli pairs and exactly enumerate diagonal objectives through `q=24`.

## Exact Final-Hamiltonian Ground Truth

The diagonal-Ising benchmark has a dependency-free exact solver under
`scripts/numerical_solver/`. It exports the ground energy, all retained ground
bitstrings, degeneracy, first distinct excitation, and spectral gap for
`q=2..156`. Dynamic programming and the closed-form ferromagnetic solution are
cross-checked against exhaustive `2**q` enumeration through `q=20`.
