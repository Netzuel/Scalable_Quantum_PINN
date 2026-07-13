# Curated Ground Truth

The `diagonal_ising/` folder contains exact final-Hamiltonian energies,
computational-basis ground states, degeneracies, first distinct excitation
energies, and spectral gaps for the diagonal-Ising validation family.

These are deterministic benchmark data, not training artifacts. Regenerate
them with `scripts/numerical_solver/solve_driver_problem_grid.py`. Exact
exhaustive validation is required through `q=20`; larger instances use the same
validated `O(q)` dynamic-programming recurrence and closed-form check.
