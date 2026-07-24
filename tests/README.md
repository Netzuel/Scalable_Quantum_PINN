# Tests

This folder is reserved for unit and regression tests plus benchmark
configuration folders.

Allowed Python files are the root `test_*.py` unit tests and `__init__.py`.
Training code, diagnostics, and reusable helpers belong under the repository-level
`scripts/` folder. Framework-specific entrypoints may live beside their study
configurations when they are not reusable outside that benchmark family.

The retained normalized variational-action v6 benchmark configurations are:

- `sparse_agp_curriculum/transverse_field_diagonal_ising/q15/sweep_test/size_intensive_pinn/`
- `sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test/size_intensive_pinn/`
- `sparse_agp_curriculum/transverse_field_diagonal_ising/q25/sweep_test/size_intensive_pinn/`

Legacy retained references and shared benchmark-family code remain under:

- `sparse_agp_curriculum/transverse_field_diagonal_ising/q156/sweep_test/`
- `sparse_agp_curriculum/scripts/`

Generated artifacts may exist locally under each study's ignored `runs/`
folder, but they should not be committed.
