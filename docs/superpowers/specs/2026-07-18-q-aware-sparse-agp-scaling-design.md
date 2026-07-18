# Q-Aware Sparse AGP Scaling Design

## Objective

Improve the physical accuracy of the sparse PINN AGP as the number of qubits
grows, using q156 as the immediate stress test without specializing the method
to that Hamiltonian or using its exact ground state during training.

The retained q156 tensor-network result is numerically certified, but its raw
ground-state fidelity is much lower than q15 and q20. The loss is not explained
by raw system size alone: the per-qubit log-infidelity and energy error per
qubit also degrade. The first implementation must therefore distinguish a
checkpoint/export mismatch from a genuine training-scale defect before any
expensive retraining.

## Non-Negotiable Constraints

- Training remains self-supervised and uses projected AGP residuals and frozen
  probes only. Exact energies and ground states are evaluation targets, never
  training labels or model-selection inputs.
- q156 canonical validation uses convergence-gated tensor-network dynamics.
- Canonical learned-AGP validation deploys every retained checkpoint term.
- Reduced-support runs are labeled ablations and cannot become canonical.
- Changes must be config-controlled and backward compatible. Existing retained
  q15, q20, and q156 artifacts keep their original semantics.
- The implementation must avoid dense many-body operators.

## Stage 1: Deployment Attribution

The retained q15 and q20 validations use their adaptive temporal-refinement
checkpoints with 64 exported time samples. The retained q156 validation uses
round 20 with 16 samples, although a q156 adaptive checkpoint also exists.
This mismatch is resolved before changing the loss.

1. Evaluate the existing q156 adaptive checkpoint with the same 8,192-term,
   24-step, bond-32 TN ablation used for the retained round-20 source.
2. Compare it only with the source-matched round-20 8,192-term result.
3. If adaptive refinement improves fidelity by at least `1e-3` and reduces
   absolute energy error by at least `5e-2`, it becomes the attribution
   champion. These margins exceed the observed coarse/fine numerical drift.
4. If it fails, reconstruct round 20 at a denser export cadence and repeat the
   same attribution test. Resampling writes a new immutable candidate artifact;
   it never overwrites the retained checkpoint.
5. Run the expensive all-32,768-term TN convergence ladder only for the
   attribution champion. Attribution results remain ablations regardless of
   their numerical quality.

## Stage 2: Dimensionless Projected Objective

The current optimizer minimizes an absolute projected residual while merely
reporting its quotient relative to the zero-AGP reference. The reference norm
and residual support scale with the system. Add a config-selectable objective:

```text
absolute:             L_res = ||R(A)||^2
reference_normalized: L_res = ||R(A)||^2 / stopgrad(max(||R(0)||^2, eps))
```

Both modes continue to export absolute residual, zero-AGP reference residual,
relative residual, and per-term diagnostics. The default remains `absolute` so
all old configurations reproduce their original objective.

## Stage 3: Dimensionless Gate Budget

The current gate budget divides the active-count error by all `K` labels. With
fixed `K` and a much larger active requirement, this makes the regularizer too
weak. Add a config-selectable denominator:

```text
support: calibration_budget = ((sum(g) - K_active) / K)^2
target:  calibration_budget = ((sum(g) - K_active) / K_active)^2
```

The default remains `support`. The q-aware candidate uses `target`, making a
fixed relative active-budget error comparable across q and K.

## Stage 4: Q-Aware Resource Policy

Support and residual budgets are derived from intensive densities rather than
copied constants. The policy is deterministic and config-driven:

```text
active_terms(q)   = clip(round(active_terms_per_qubit * q), min, K)
residual_terms(q) = clip(round(residual_terms_per_qubit * q), min, Q_available)
swap_terms(q)     = clip(round(swap_terms_per_qubit * q), min, active_terms(q))
```

For the first q156 candidate, the densities are anchored to the retained q20
benchmark, where `K_active/q = 102.4` and `Q/q = 4096`. Memory limits may cap Q,
but every realized budget and cap reason must be persisted. No physical target
chooses these values.

Candidate and residual reservoirs are stratified by Pauli locality and spatial
coverage. Each stratum receives a deterministic minimum quota before remaining
slots are filled by the existing residual-importance ranking. This prevents a
large system from spending a fixed budget on a small spatial region while
preserving the established importance score.

## Model Selection And Promotion

Frozen projected probes choose among q-aware training candidates. Probe labels,
seeds, and partitions are created before training and are immutable. Exact
ground truth is opened only after the candidate is frozen.

A new q156 benchmark is promoted only if:

- full-support identity proves all 32,768 checkpoint terms were deployed;
- all canonical tensor-network operator and dynamics gates pass;
- raw ground-state fidelity is higher than the retained q156 canonical value;
- absolute final-energy error is lower than the retained q156 canonical value;
- fidelity density and energy error per qubit are reported;
- q15 and q20 regression checks show no material methodology regression when
  the new modes are enabled with their q-aware budgets.

If a candidate fails, its run remains a clearly labeled diagnostic and the
retained benchmark/configuration is unchanged.

## Verification

Unit tests cover objective normalization, zero-reference protection, budget
normalization, q-aware budget resolution, deterministic stratification, and
backward-compatible defaults. Integration tests verify that the settings flow
through every curriculum and refinement path. Physical verification follows
the reduced-cost attribution gate, then the canonical full-K TN ladder.

