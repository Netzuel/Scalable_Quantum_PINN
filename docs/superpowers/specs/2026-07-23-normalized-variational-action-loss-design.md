# Normalized Variational-Action Auxiliary Loss

> **Outcome (2026-07-24):** Implemented and promoted by explicit user decision
> to the retained v6 benchmark after independently trained q15, q20, and q25
> runs all exceeded `0.95` fidelity. The q20-to-q25 `0.0217296` fidelity drop
> remains a reported scaling limitation.

## Objective

Test whether a standard variational AGP action removes the physically poor
stationary branches admitted by the projected Euler-Lagrange residual, without
using exact energies, ground states, fidelities, or cross-system initialization.

The controlled sequence is:

```text
q20 all-K TN at T=1
  -> continue only if F >= 0.95
q15 exact statevector at T=1, with all K whenever computationally feasible
  -> continue only if F >= 0.95
q25 all-K TN at T=1
```

## Loss

For

```text
G_lambda = i dH_AD/dlambda - [A_lambda, H_AD],
```

the retained projected residual is

```text
L_EL = ||[G_lambda, H_AD]||^2.
```

The candidate adds

```text
L_action = ||G_lambda||^2 / max(||i dH_AD/dlambda||^2, eps),
L_total = L_EL + beta_action L_action + retained regularizers.
```

The denominator is detached. This makes the auxiliary term dimensionless across
qubit counts and Hamiltonian coefficient scales while preserving the standard
variational gradient. The implementation reports raw, reference, and relative
action values even when `beta_action=0`.

## Configuration And Compatibility

- Add `variational_action` to `ProjectedSparseLossWeights`.
- Read `training.loss.variational_action` into every projected sparse training
  stage, including baseline, feedback, temporal refinement, and adaptive
  temporal refinement.
- A missing key defaults to zero, preserving every retained checkpoint and
  historical configuration.
- Candidate configurations use a fixed predeclared nonzero weight and independent
  output PAU networks. No graph architecture is involved.
- Physical duration is fixed at `T=1`; duration reparameterization is forbidden.
- q20 and q25 canonical evaluation deploys every learned term under the existing
  convergence-gated tensor-network ladder.

## Fail-Fast Promotion

The candidate is not promoted merely because its projected losses improve.

1. q20 must reach all-K tensor-network ground-state fidelity `F >= 0.95`.
2. If q20 passes, q15 is cleaned and trained independently from scratch. It
   must reach exact-statevector fidelity `F >= 0.95`; if an all-K exact
   evolution is computationally unavailable, its claim remains `not tested`
   until the project rulebook permits a defensible canonical substitute.
3. If q15 passes, q25 is cleaned and trained independently from scratch and
   must reach all-K tensor-network fidelity `F >= 0.95`.
4. Any failure stops the sequence. Existing retained benchmarks remain intact.

## Verification

- Unit-test the action formula on a two-level Pauli system.
- Unit-test backward compatibility at zero action weight.
- Unit-test configuration propagation into baseline and feedback settings.
- Compile modified Python files and run focused projected-loss and benchmark
  layout tests before training.
- Record raw physical metrics and certification gates without rewriting failed
  candidates as retained results.
