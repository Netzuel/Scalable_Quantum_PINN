# AGP Certification Criteria

This document is the required checklist for deciding whether a sparse
Pauli-coordinate AGP result is reliable enough to be treated as a robust
counterdiabatic object, rather than only as a projected sparse experiment.

Read this file before:

- claiming that an AGP support is sufficient;
- reporting a minimum support size `K_min`;
- interpreting a large-`q` training run as representative of the unrestricted
  AGP;
- accepting a curriculum step that changes either the AGP support or the
  residual basis.

The core rule is:

```text
No sparse AGP is certified unless support-size adequacy, support-content
quality, and training representativeness all pass at the same time.
```

For `q <= 8`, the full Pauli basis can be used exactly. For `q > 8`, the full
`4**q` basis is not a computational object in this repository. Large-`q`
results must therefore be treated as projected sparse AGPs until the criteria
below are satisfied.

## Definitions

The AGP is represented as

```text
A_lambda(t) = sum_{P in S_AGP} C_P(t) P,
K = |S_AGP|.
```

The direct counterdiabatic Hamiltonian is

```text
H_CD(t) = dot(lambda)(t) A_lambda(t).
```

Coefficient importance is measured on the direct counterdiabatic coefficient:

```text
I_P = RMS_tau(dot(lambda)(tau) C_P(tau)).
```

The Euler-Lagrange residual is

```text
R(A) = [i dH_AD/dlambda - [A_lambda, H_AD], H_AD].
```

The normalized residual on any residual basis `B` is

```text
relative_residual_B = ||R(A)||^2_B / (||R(A=0)||^2_B + eps).
```

Values below one mean the learned AGP improves over the no-AGP baseline on that
same projected basis. Values above one mean the learned AGP is worse than no AGP
on that projected basis.

## Required Residual Bases

Every large-`q` experiment must separate these bases:

```text
S_R_train:
    residual equations used directly in the loss.

S_R_holdout:
    larger residual pool used to diagnose generalization and to identify hard
    missed equations.

S_R_unseen:
    holdout strings absent from S_R_train.

S_R_probe_gate:
    fixed disjoint residual pool used to accept or reject curriculum steps.

S_R_probe_watch:
    second fixed disjoint residual pool used to accept or reject curriculum
    steps. This catches probe-gate over-specialization before the final
    external diagnostic is consulted.

S_R_probe_test:
    fixed disjoint residual pool reported after every round but never used for
    accepting steps.
```

The training residual can move with the curriculum. The probe bases must not be
quietly regenerated after every support update, otherwise generalization is
measured on a moving target.

## Gate 1: Enough Number Of Terms

This gate asks whether the chosen size `K` is large enough.

Pass only if all checks below hold.

### 1.1 Training Residual Pass

The final active training residual must be low:

```text
training_relative_residual <= target_train.
```

Initial practical target:

```text
target_train <= 0.05 to 0.10.
```

This is necessary but never sufficient.

### 1.2 Holdout Residual Pass

The trained model must remain good on a larger residual basis not used directly
in the loss:

```text
holdout_relative_residual <= target_holdout.
```

Initial practical target:

```text
target_holdout <= 0.05 to 0.10.
```

The holdout residual should not be much larger than the training residual. A
large train-holdout gap means the AGP overfits the projected training equations.

### 1.3 Unseen Residual Pass

The residual restricted to holdout strings absent from the training loss is the
strictest diagnostic:

```text
unseen_relative_residual <= target_unseen.
```

Initial practical target:

```text
target_unseen <= 1.0.
```

This target is looser than the full holdout target because unseen terms can be
hard, but it must not be orders of magnitude larger than one.

### 1.4 Fixed Probe Pass

All fixed probes should pass:

```text
probe_gate_relative_residual <= 1.0
probe_watch_relative_residual <= 1.0
probe_test_relative_residual <= 1.0
```

The `probe_gate` and `probe_watch` bases protect training decisions. The
`probe_test` basis is a true external diagnostic. A candidate can be considered
methodologically useful if it improves these values but remains above one;
however, it is not certified.

### 1.5 K-Sweep Plateau

Train nearby larger supports under the same Hamiltonian path, schedule,
residual construction, model size, optimizer, seed policy, and training budget.
Increasing `K` must give only a small improvement on holdout and probe metrics:

```text
(metric(K) - metric(K_next)) / metric(K) < 0.05 to 0.10.
```

This plateau must be measured on holdout/probe residuals, not only on the
training residual.

### 1.6 Q-Sweep Plateau

Evaluate the same trained AGP on larger residual probes:

```text
Q_1 < Q_2 < ... < Q_m.
```

The reported residuals should change by less than `5%` to `10%` when `Q`
increases. If the residual rises strongly with larger `Q`, the basis was too
small to reveal missing directions.

## Gate 2: Good Enough Terms

This gate asks whether the selected Pauli strings are the right strings, not
only whether there are enough of them.

Pass only if all checks below hold.

### 2.1 Top-Term Stability Across K

Rank Pauli strings by `I_P`. Compare the top `5%` to `10%` terms between
nearby support sizes.

Acceptance target:

```text
top-set overlap with next/largest support >= 85% to 90%
```

Overlap is measured relative to the smaller top set.

### 2.2 Top-Term Stability Across Seeds

Repeat training with different seeds, or at least with independently initialized
weights when runtime permits. The dominant terms should remain stable:

```text
top-set overlap across seeds >= 80% to 90%.
```

The exact threshold can be relaxed for exploratory runs, but instability must be
reported.

### 2.3 Curriculum Proposal Exhaustion

When adaptive support expansion is enabled, newly proposed terms must stop being
useful. A support is not final while residual-driven proposals keep entering and
materially improving fixed holdout/probe metrics.

Acceptance requires one of:

```text
new AGP terms are rejected by fixed probes,
or new accepted AGP terms have negligible I_P,
or accepted AGP expansion improves holdout/probe metrics by < 5% to 10%.
```

### 2.4 Prune And Retest

After training a large candidate support, prune small coefficients:

```text
keep P if I_P >= epsilon * max_Q I_Q.
```

Sweep at least:

```text
epsilon in {1e-2, 1e-3, 1e-4}.
```

Retrain or re-evaluate the retained support. The pruned support passes only if

```text
holdout/probe residuals worsen by < 5% to 10%.
```

This proves that negligible coefficients are actually disposable.

### 2.5 Rejected-Term Audit

If candidate terms are rejected, record why. Rejection is trustworthy only when
the candidate damages fixed holdout/probe metrics or fails to improve them, not
when it merely had too little training budget.

## Gate 3: Correctly Trained And Representative

This gate asks whether the selected support has been optimized well enough to
represent the projected AGP.

Pass only if all checks below hold.

### 3.1 Simultaneous Residual Pass

The final accepted model must pass the residual metrics simultaneously:

```text
training_relative_residual <= target_train
holdout_relative_residual <= target_holdout
unseen_relative_residual <= target_unseen
probe_gate_relative_residual <= target_probe_gate
probe_watch_relative_residual <= target_probe_watch
probe_test_relative_residual <= target_probe_test
```

Initial practical targets:

```text
target_train      <= 0.05 to 0.10
target_holdout    <= 0.05 to 0.10
target_unseen     <= 1.0
target_probe_gate <= 1.0
target_probe_watch <= 1.0
target_probe_test <= 1.0
```

Passing only the training loss is not meaningful for certification.

### 3.2 No Moving-Target Shortcut

A decreasing residual is not enough if the residual basis changes every round.
The fixed `probe_gate`, `probe_watch`, and `probe_test` bases must be evaluated
across all rounds, and their selection rules must be reported. A result cannot
be certified from a moving holdout basis alone.

### 3.3 Coefficient Regularity

The learned functions `dot(lambda) C_P(tau)` should be smooth enough to be
physically interpretable. Large high-frequency oscillations, seed-dependent
spikes, or endpoint artifacts must be diagnosed before claiming success.

The direct counterdiabatic coefficients should vanish at endpoints when the
fixed schedule has

```text
dot(lambda)(0) = dot(lambda)(1) = 0.
```

This endpoint vanishing comes from `dot(lambda)`, not from forcing `C_P(t)` to
be zero.

### 3.4 Optimizer And Budget Check

A failed support is not conclusive if optimization was undertrained. Before
declaring a support insufficient, check at least one of:

```text
longer training budget,
smaller learning rate continuation,
independent seed,
or optimizer variant.
```

If these materially improve fixed holdout/probe metrics, the earlier result was
an optimization failure, not a support failure.

### 3.5 Physical Validation

When feasible, validate the final `H_CD(t)` on observables beyond the
Euler-Lagrange residual, for example:

```text
final energy,
excitation probability,
local observables,
approximate state-evolution metrics,
or known small-q dense/full-basis comparisons.
```

For large `q`, these checks may be approximate. If they are unavailable, report
that the result is residual-certified only, not physically validated.

When an MPS or other tensor-network approximation is used, physical validation
passes only if all of the following are reported and checked:

```text
small-q agreement with an exact statevector at matching full learned support,
time-step convergence,
bond-dimension and cutoff convergence,
full learned-support size and active coefficient count,
state norm and peak bond,
and final observable differences across the numerical ladder.
```

The canonical MPS validation must use every term in the retained learned AGP
checkpoint. A top-term or pruned deployment is an ablation and cannot satisfy
this physical-validation gate on behalf of the full trained model. If the full
learned support cannot be evolved under a convergence ladder, mark this gate
`not tested`.

Do not call an MPS result exact. A converged full-learned-support MPS evolution
certifies the trained sparse protocol under the tested numerical ladder; it
does not certify Pauli strings absent from the learned support or the full
`4**q` basis.

## Claim Levels

Use precise language when reporting results.

### Projected Sparse AGP Experiment

Use this when only training residuals, or a limited subset of diagnostics, have
passed.

Allowed claim:

```text
The model learned an AGP on the selected projected residual basis.
```

Forbidden claim:

```text
The AGP support is sufficient.
```

### Candidate Robust Sparse AGP

Use this when training, holdout, unseen, and at least one fixed probe improve
substantially, but one or more strict thresholds remain above target.

Allowed claim:

```text
The support is promising, but not certified under the fixed-probe criteria.
```

### Certified Sparse AGP For This Path And Tolerance

Use this only when all gates in this document pass at the chosen tolerance.

Allowed claim:

```text
For this Hamiltonian path, schedule, residual construction, and tolerance, the
support is sufficient under the sparse certification protocol.
```

This is still not a proof that the unrestricted full-basis AGP contains no other
terms. It is a controlled sparse certificate.

## Minimum Acceptable Report

Every large-`q` run must report:

- `q`
- `4**q`
- `K` and `K / 4**q`
- AGP support selection rule
- residual training basis size and selection rule
- holdout basis size and selection rule
- unseen basis size
- `probe_gate` size and selection rule
- `probe_watch` size and selection rule
- `probe_test` size and selection rule
- training, holdout, unseen, `probe_gate`, `probe_watch`, and `probe_test`
  relative residuals
- K-sweep plateau result
- Q-sweep plateau result
- top-term overlap across K
- top-term overlap across seeds, when available
- pruning threshold and retained term count
- physical validation metric, when available
- exact claim level from the section above

## Current q20 Interpretation

The latest q20 transverse-Ising run completed all 20 feedback rounds with the
requested fixed support and residual budget:

```text
q                              = 20
4**q                           = 1099511627776
K                              = 32768
K / 4**q                       = 2.980232238770e-08
Q_requested = Q_effective      = 81920
rounds                         = 20
round_20_training_residual     = 1.085813e-03
round_20_holdout_residual      = 5.519648e-02
round_20_unseen_residual       = 3.638261e-06 absolute
round_20_unseen_relative       = not tested (zero reference)
adaptive_training_residual     = 2.029059e-03
adaptive_holdout_residual      = 5.687243e-02
```

The training and holdout gates pass their configured `0.1` thresholds. The
relative unseen gate cannot be evaluated because its reference residual is
zero; the absolute value above is reported instead. Fixed probe gates, K/Q
plateaus, cross-seed support stability, pruning, coefficient regularity, and
physical validation were not tested. Support swaps were still active in round
20, so proposal exhaustion was not established. Therefore this run is a
completed projected sparse AGP experiment, not a certified AGP support.

## Operational Rule For Future Work

Before accepting, publishing, or summarizing a large-`q` AGP as reliable, read
this checklist and explicitly mark every gate as:

```text
pass
fail
not tested
```

Any `fail` or `not tested` entry downgrades the claim level.
