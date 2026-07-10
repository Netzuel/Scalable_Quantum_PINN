# Tests

This folder is reserved for unit and regression tests plus benchmark
configuration folders.

Allowed Python files are the root `test_*.py` unit tests and `__init__.py`.
Training code, diagnostics, and reusable helpers belong under `scripts/`.

Benchmark studies are configuration-only folders:

- `q15/sweep_test/`
- `q20/sweep_test/`
- `diagonal_ising_grid/q*/`

Generated artifacts may exist locally under each study's ignored `runs/`
folder, but they should not be committed.
