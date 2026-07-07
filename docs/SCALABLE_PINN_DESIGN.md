# Scalable PINN Design

## Starting Point

The original `Quantum_PINN` manuscript learned:

```text
lambda(t), A_CD(t), C(t)
```

for small Hamiltonians by explicitly producing dense matrix entries for
`A_CD(t)` and coefficients over the full Pauli-product basis. This was viable
for two qubits and partially useful for four, but the output and loss became
unscalable.

## New Representation

This repository changes the representation:

```text
A_lambda(t) = sum_{P in S} a_P(t) P
```

where `S` is a selected sparse support, for example all one- and two-local
Pauli strings on a hardware graph. The neural network output dimension is
`1 + |S|`, not `1 + 2 * 4**N`.

For the low-size regime, `q <= 8`, the repository can deliberately choose
`S = {I,X,Y,Z}^q` and train all `4**q` AGP coefficients exactly in Pauli
coordinates. For `q > 8`, the default training path is projected sparse.

## Loss Without Dense Matrices

The Euler-Lagrange residual is evaluated through Pauli algebra:

```text
R(lambda) = [ i dH_AD/dlambda - [A_lambda, H_AD], H_AD ].
```

The implementation stores only sparse Pauli labels and structure constants.
The residual is differentiable because commutator coefficients are precomputed
and applied to PyTorch tensors.

## Exactness Statement

The implementation can be exact only with respect to the operator basis being
used. If the selected AGP support and residual closure contain the true AGP and
all residual terms, then a zero residual is an exact AGP certificate for that
problem. If the true AGP contains longer Pauli strings outside the support, the
method learns a projected/local approximation.

This distinction is the central research discipline for the repository.

## Immediate Experiment Path

1. Use full-basis training only for the low-size exact regime, `q <= 8`.
2. For `q > 8`, start from endpoint-commutator AGP candidates and generated
   commutator residual bases.
3. Use adaptive support growth: train, inspect residual Pauli terms, expand the
   AGP support, and retrain while transferring shared network weights.
4. Track scaling of `agp_terms`, residual-basis size, and training time as
   qubit count increases.
5. Compare learned sparse AGPs against exact small-qubit solutions only for
   validation.
