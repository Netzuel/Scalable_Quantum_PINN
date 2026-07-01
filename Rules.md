# AGP Support Rules

This document defines the working rules for deciding whether a sparse Pauli
support is large enough to describe the adiabatic gauge potential (AGP) in this
repository.

## Scope

For `q <= 8`, full-basis AGP training is allowed:

```text
AGP support = {I, X, Y, Z}^q
```

For `q > 8`, the full basis is not a computational object. The AGP support is a
research choice and must be validated by residual generalization, not by the
training loss alone.

## Core Rule

Do not declare an AGP support sufficient only because the training residual is
small on the residual basis used during optimization.

A support is considered sufficient only if it also generalizes to a larger
holdout residual basis and remains stable under pruning of small learned
coefficients.

## Definitions

The learned AGP is

```text
A_lambda(t) = sum_{P in S_AGP} C_P(t) P
```

The direct counterdiabatic coefficient plotted and ranked is

```text
dot(lambda)(t) C_P(t)
```

The importance of a Pauli string is measured by

```text
I_P = RMS_tau(dot(lambda)(t) C_P(t)).
```

The normalized residual is

```text
relative_residual = ||R(A)||^2 / (||R(A=0)||^2 + eps),
```

where

```text
R(A) = [i dH_AD/dlambda - [A_lambda, H_AD], H_AD].
```

## Minimum-Support Protocol

To find the minimum useful support for a Hamiltonian path and qubit count:

1. Choose a sequence of AGP support sizes:

```text
K_1 < K_2 < ... < K_m
```

2. Keep everything else fixed:

```text
H_initial, H_final, schedule, tau grid, residual training basis, model size,
optimizer, seed, and training budget.
```

3. Train each support size.

4. Evaluate each trained model on a larger holdout residual basis:

```text
K_train_residual < K_holdout_residual.
```

5. Rank coefficients by `I_P`.

6. Select the smallest `K_i` satisfying all acceptance criteria below.

## Acceptance Criteria

A support size `K_i` is accepted as the current minimum only when all of these
conditions hold.

### 1. Training Residual Plateau

Increasing the support to the next size gives only a small improvement:

```text
(relative_residual(K_i) - relative_residual(K_{i+1}))
/ relative_residual(K_i) < 5% to 10%.
```

This must not be the only criterion.

### 2. Holdout Residual Pass

The trained model must remain good on a larger residual basis not used for
optimization:

```text
holdout_relative_residual <= target_residual.
```

A practical first target is

```text
target_residual = 0.05 to 0.10.
```

If the holdout residual becomes much worse than the training residual, the AGP
support is not robust, even if the training loss is small.

### 3. Unseen-Residual Pass

Compute the residual only on holdout Pauli strings absent from the training
residual basis. This is the strictest diagnostic:

```text
unseen_relative_residual <= target_unseen_residual.
```

The initial target can be looser than the full holdout target, but it must not
be orders of magnitude larger than one.

### 4. Top-Term Stability

The important learned terms must be stable across support sizes. Compare the
top `5%` to `10%` terms ranked by `I_P`.

A practical target is

```text
top-set overlap with the next/largest support >= 85% to 90%
```

using overlap relative to the smaller top set.

### 5. Prune-and-Retest

After training a large candidate support, prune small coefficients:

```text
keep P if I_P >= epsilon * max_Q I_Q
```

with a first sweep over

```text
epsilon in {1e-2, 1e-3, 1e-4}.
```

Then rebuild the support with only the retained terms and re-evaluate on the
large holdout residual basis. The pruned model is accepted only if

```text
holdout residual increases by < 5% to 10%.
```

### 6. Physical Validation

Whenever feasible, validate the final counterdiabatic Hamiltonian on physical
observables, not only on the Euler-Lagrange residual. Examples include final
energy, excitation density, local observables, or approximate state-evolution
metrics.

## General Rule

There is no reliable formula of the form

```text
K_min = f(q)
```

that depends only on the number of qubits. The minimum AGP support depends on
the Hamiltonian path, locality, spectrum, interpolation, residual projection,
and target accuracy.

The operational rule is therefore:

```text
K_min(q, H_path, tolerance) is the smallest AGP support size that passes
training-residual plateau, holdout residual, unseen-residual, top-term
stability, prune-and-retest, and physical-validation checks.
```

For large systems, the support should be discovered by an iterative loop:

```text
train -> evaluate holdout residual -> add high-residual/generated terms
-> retrain -> rank coefficients -> prune small terms -> retest.
```

## Current q=20 Lesson

The q=20 support-size sweep showed that increasing AGP terms from `576` to
`2048` monotonically improved the training-projection residual. However, the
`1536`-term AGP failed an `8192`-term holdout residual test:

```text
training relative residual: 2.37e-2
holdout relative residual:  1.55
unseen-only relative:       2.85e2
```

Therefore, `1536` is a good support for the `2048`-term training projection, but
it is not a robust AGP support under the larger holdout criterion.

The next q=20 step is to train larger supports and/or adapt the AGP support using
the high-residual holdout strings, then repeat the holdout test.

## Reporting Requirements

Every large-`q` experiment should report:

- `q`
- `4**q`
- `AGP terms`
- AGP fraction of full basis
- training residual terms
- holdout residual terms
- unseen residual terms
- final training relative residual
- final holdout relative residual
- unseen-only relative residual
- top-term overlap across support sizes
- pruning threshold and retained terms
- physical validation metric, when available

Without these diagnostics, the result should be described as a projected sparse
AGP experiment, not as evidence that the AGP support is sufficient.
