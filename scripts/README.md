# Scripts

This folder contains reusable AGP workflow entrypoints. Scripts are configured
with a study `config.json`; they should not be forked per qubit count.

Core workflow:

- `agp_baseline_train.py`: train or summarize a fixed-support projected AGP baseline.
- `agp_holdout_feedback.py`: run fixed-`K` residual holdout-feedback fine-tuning, including configured joint AGP scale/gate calibration.
- `agp_holdout_study.py`: evaluate completed runs on common holdout residual bases.
- `agp_evaluate_holdout.py`: evaluate one trained run on an additional holdout basis.
- `agp_residual_calibration.py`: optional residual-only continuation for a completed sparse AGP.
- `agp_restart.py`: remove generated artifacts for one configured study.
- `agp_support.py`: support-selection helpers.
- `agp_plot_annotations.py`: shared physical-metric footer annotations for HCD summary plots.

Exact numerical validation:

- `numerical_solver/ising_ground_state.py`: diagonal-Ising parsing, QUBO conversion, exact dynamic programming, and exhaustive checks.
- `numerical_solver/solve_driver_problem_grid.py`: export exact ground-state data over a configured qubit range.

Shared implementation:

- `full_pauli_training_common.py`
- `projected_sparse_training_common.py`

Optional diagnostics live under `scripts/diagnostics/`. They are not the
default benchmark methodology.

Hamiltonian-specific generation, statevector validation, grid orchestration,
and HCD figure regeneration live under
`tests/sparse_agp_curriculum/scripts/`.
