# Repository Guidelines

## Project Scope

This repository is the scalable successor to `Quantum_PINN`. The purpose is to
test whether physics-informed neural networks can learn useful adiabatic gauge
potentials for larger qubit systems without dense matrices and without emitting
the full `4**N` Pauli basis.

The final MLST manuscript PDF from the original work is retained under
`docs/` for context. Do not copy old result folders, checkpoints, or
scratch runs into this repository.

## Core Rule

Keep the main reusable Python surface in:

- `models.py`: PINN architectures and differentiable loss wrappers.
- `utils.py`: sparse Pauli algebra, Hamiltonian helpers, and basis utilities.

Avoid introducing dense `2**N x 2**N` matrices in library code. If a dense
matrix is needed for a tiny diagnostic comparison, keep it in a clearly named
test or example and document the qubit limit.

## Physics Target

The working ansatz is:

```text
H_AD(lambda) = (1 - lambda) H_initial + lambda H_final
A_lambda(t) = sum_j a_j(t) P_j
```

The sparse loss evaluates:

```text
[ i dH_AD/dlambda - [A_lambda, H_AD], H_AD ] = 0
```

in Pauli-coordinate space over a chosen ansatz and residual basis. This is a
projected/local exactness statement, not a generic exact AGP guarantee.

## Scalability Discipline

- Treat the AGP support as an explicit research choice.
- Prefer low-locality, geometry-aware, symmetry-aware, or adaptively discovered
  Pauli supports.
- Track both `agp_terms` and residual `basis_size` in every experiment.
- Never silently expand to the full `4**N` basis for large `N`.
- For every large-`N` sparse AGP task, re-read
  `AGP_CERTIFICATION_CRITERIA.md` before accepting a result or making a
  sufficiency claim. Mark every certification gate as `pass`, `fail`, or
  `not tested`.
- Do not commit generated `results/`, `outputs/`, `runs/`, `.pt`, `.h5`, or
  notebook checkpoint files.

## Validation Commands

Use the `torch-mps` conda environment for Python commands:

```bash
conda run -n torch-mps python -m py_compile models.py utils.py
conda run -n torch-mps python examples/two_qubit_sparse_demo.py
conda run -n torch-mps python -m unittest discover -s tests
```

Keep validation inside the environment. Do not bypass `torch-mps` just because
the default shell Python happens to work.
