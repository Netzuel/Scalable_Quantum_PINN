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

1. Start with transverse-field Ising chains using local one- and two-body AGP
   supports.
2. Track scaling of `agp_terms`, residual-basis size, and training time as
   qubit count increases.
3. Compare learned sparse AGPs against exact small-qubit solutions only for
   validation.
4. Add adaptive support growth: train, inspect residual Pauli terms, expand the
   support, and retrain.
5. Move to chemistry-style Hamiltonians only after the sparse workflow is
   stable on structured spin chains.

