# Exact Diagonal-Ising Ground-State Solver

This folder provides the ground-truth solver for the retained
`TransverseIsingDriverProblem` benchmark:

```text
H_initial = -sum_i X_i
H_final   = -sum_i h_i Z_i - sum_i J_i Z_i Z_{i+1}.
```

Only `H_final` is solved here. It is diagonal in the computational basis, so a
bitstring completely specifies an eigenstate. Characters follow the stored
Pauli-label order: bit `0` has `Z=+1`, and bit `1` has `Z=-1`.

## Solver Choice

Gurobi supports binary variables and quadratic objectives through its Python
API, while IBM CPLEX supports mixed-integer quadratic programming through its
Python APIs. The Ising problem can therefore be sent to either solver after the
exact substitution `z_i = 1 - 2 x_i`.

For this benchmark, a proprietary general-purpose optimizer is unnecessary.
The interaction graph is an open path, so Bellman dynamic programming solves
the exact ground state and first distinct excitation in `O(q)` time. This is
stronger and more reproducible than requiring a Gurobi or CPLEX installation.
The curated JSON also exports the exact QUBO constant, linear coefficients,
and quadratic coefficients for optional external Gurobi or CPLEX checks.

References:

- [Gurobi Python API overview](https://docs.gurobi.com/projects/optimizer/en/current/reference/python/overview.html)
- [Gurobi Python model API](https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html)
- [IBM CPLEX Python API setup](https://www.ibm.com/docs/en/icos/22.1.1?topic=cplex-setting-up-python-api)
- [IBM CPLEX Python optimization capabilities](https://www.ibm.com/docs/en/icos/22.1.0?topic=users-why-python)
- [Scott and Sorkin, dynamic programming for Ising/CSP structure](https://arxiv.org/abs/cs/0604079)

## Exactness Checks

The current coefficients satisfy `h_i > 0` and `J_i > 0`. Every term in
`H_final` is therefore minimized simultaneously by

```text
|psi_0> = |00...0>,
E0(q) = -sum_i h_i - sum_i J_i = 1 - 1.35 q.
```

The export command requires agreement among:

1. exact path dynamic programming;
2. the termwise ferromagnetic closed form;
3. the QUBO energy evaluated on every reported ground bitstring;
4. exhaustive enumeration of all `2**q` bitstrings for `2 <= q <= 20`.

Failure of any required check aborts the export.

## Files

- `ising_ground_state.py`: parsing, Ising/QUBO energies, exact dynamic programming, closed-form checks, and exhaustive enumeration.
- `solve_driver_problem_grid.py`: validated multi-q export command.

Run the retained benchmark export with:

```bash
conda run --no-capture-output -n torch-mps python \
  scripts/numerical_solver/solve_driver_problem_grid.py \
  --min-qubits 2 \
  --max-qubits 156 \
  --validation-max-qubits 20
```

Curated ground-truth artifacts are written under
`tests/sparse_agp_curriculum/ground_truth/diagonal_ising/`.
