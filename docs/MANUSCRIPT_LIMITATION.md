# Original Manuscript Limitation

The MLST 2024 `Quantum_PINN` manuscript demonstrated a PINN route to
counterdiabatic driving for small systems. The important limitation was not the
PINN idea itself, but the representation:

- `A_CD(t)` was treated as a dense matrix.
- The Pauli decomposition was emitted over a full basis.
- The Euler-Lagrange loss was evaluated with matrix commutators.
- The output dimension scaled as `1 + 2 * 4**N` for `N` qubits.

That representation is impossible to scale to genuinely large qubit counts.

The new project keeps the physics-informed loss idea but changes the
computational object. It learns AGP coefficients on selected Pauli supports and
computes commutators through sparse Pauli algebra.

## What Was Carried Over

- The Hamiltonian interpolation:

```text
H_AD(t) = (1 - lambda(t)) H_initial + lambda(t) H_final
```

- The counterdiabatic correction:

```text
H(t) = H_AD(t) + dot(lambda)(t) A_lambda(t)
```

- The least-action / Euler-Lagrange AGP condition:

```text
[ i dH_AD/dlambda - [A_lambda, H_AD], H_AD ] = 0
```

## What Was Intentionally Not Carried Over

- Historical `Results/` folders and paper artifacts.
- Dense-matrix AGP output heads.
- Full Pauli-basis coefficient output by default.
- CUDA-device-specific training scripts from the original repository.

