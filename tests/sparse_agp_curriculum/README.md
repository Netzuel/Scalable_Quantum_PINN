# Sparse AGP Curriculum Benchmarks

This framework retains the fixed-support sparse-AGP curriculum studies for
`q=15` and `q=20`. Both use

```text
H_AD(lambda) = (1 - lambda) H_initial + lambda H_final.
```

They do not use the same physical Hamiltonian. The `q15` study is the
transverse-field-to-diagonal-Ising playground with exact final-state physical
diagnostics. The retained `q20` study uses the sparse hydrogen Hamiltonian and
is currently limited to projected sparse-AGP diagnostics. This distinction
prevents physical claims from being transferred between the studies.

## Layout

```text
sparse_agp_curriculum/
  q15/sweep_test/   retained q=15 configuration and local runs
  q20/sweep_test/   retained q=20 configuration and local runs
  ground_truth/     curated exact validation data
  scripts/          Python entrypoints specific to this benchmark family
```

The shared sparse-AGP training and curriculum implementation remains in the
repository-level `scripts/` folder. This avoids maintaining scenario-specific
copies of reusable training code.

## Framework Scripts

- `build_driver_problem_hamiltonian.py`: generate the analytic Ising Hamiltonian pair used by `q15` and optional grid studies.
- `agp_physical_validation.py`: compare no CD, nested-commutator `l=1`, and the learned AGP where statevector validation is configured.
- `agp_qubit_grid_benchmark.py`: prepare and orchestrate optional multi-q Ising studies.
- `agp_regenerate_hcd_summaries.py`: regenerate HCD connection-summary figures.

## Exact Final-Hamiltonian Ground Truth

The diagonal-Ising benchmark has a dependency-free exact solver under
`scripts/numerical_solver/`. It exports the ground energy, all retained ground
bitstrings, degeneracy, first distinct excitation, and spectral gap for
`q=2..156`. Dynamic programming and the closed-form ferromagnetic solution are
cross-checked against exhaustive `2**q` enumeration through `q=20`.
