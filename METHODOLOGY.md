# Sparse AGP PINN Methodology

The canonical detailed methodology is
`docs/CURRENT_SPARSE_AGP_METHODOLOGY.md`. This file is the concise repository
entrypoint.

## Current Benchmark

The retained benchmark is
`size_extensive_normalized_variational_action_conventional_pinn_v6`.
Each Hamiltonian and qubit size is trained independently from scratch. No
cross-system checkpoint initialization is allowed.

The current diagonal-Ising benchmark configurations are:

```text
q15/sweep_test/size_intensive_pinn/config.json
q20/sweep_test/size_intensive_pinn/config.json
q25/sweep_test/size_intensive_pinn/config.json
```

under
`tests/sparse_agp_curriculum/transverse_field_diagonal_ising/`.

## Representation

For `q <= 8`, the exact-output regime uses all `4**q` Pauli strings. For
`q > 8`, the network emits a bounded support of `K` Pauli coefficients and
never materializes the full basis.

```text
H_AD(lambda) = (1 - lambda) H_initial + lambda H_final
A_lambda(tau) = sum_{P in S_AGP} C_P(tau) P
tau = (t - t_initial) / T
T = 1
```

The current benchmark uses a conventional independent-output quadratic/QRes
network with four hidden layers of width 96 and trainable Padé activations. The
schedule, global CD scale, and soft Pauli gates are trained jointly.

## Training Objective

The projected Euler-Lagrange residual is

```text
R(A) = [i dH_AD/dlambda - [A_lambda, H_AD], H_AD].
```

The retained v6 objective adds the normalized variational action:

```text
G(A) = i dH_AD/dlambda - [A_lambda, H_AD]
L_action = ||G(A)||^2 / max(||i dH_AD/dlambda||^2, eps)
L_total = L_projected + 0.1 L_action + L_regularization
```

The action and residual are computed in sparse Pauli coordinates. Exact final
energy, fidelity, observables, and bitstrings are excluded from training,
hyperparameter selection, and checkpoint selection.

## Curriculum And Scaling

The fixed-`K` holdout-feedback curriculum adds hard residual equations and
swaps weak support terms for residual-derived candidates while preserving the
output count. The retained benchmark uses:

| q | K | Q | Rounds |
|---:|---:|---:|---:|
| 15 | 32,768 | 65,536 | 15 |
| 20 | 58,368 | 116,736 | 20 |
| 25 | 91,136 | 182,272 | 25 |

Uniform and adaptive temporal-refinement stages continue training on the final
support without using physical ground truth.

## Physical Validation

All benchmark claims use fixed physical duration `T=1`.

```text
q <= 15: exact statevector evolution
q > 15:  convergence-gated tensor-network evolution
```

Canonical tensor-network evaluation deploys every learned AGP term. Top-term
truncations are ablations only.

| q | Energy error | Ground fidelity | Validation |
|---:|---:|---:|---|
| 15 | 0.1339216 | 0.9768832 | exact statevector |
| 20 | 0.1607203 | 0.9764755 | certified all-K TN |
| 25 | 0.2998297 | 0.9547459 | certified all-K TN |

The q20-to-q25 fidelity drop is `0.0217296`. It is a known limitation of the
current benchmark, not evidence of unrestricted AGP support sufficiency.

See `Rules.md` and `AGP_CERTIFICATION_CRITERIA.md` before training,
validation, or interpreting a result.
