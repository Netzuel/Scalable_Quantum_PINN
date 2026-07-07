# Sparse AGP PINN Methodology

This document describes the working methodology for scalable counterdiabatic
AGP learning in this repository. The goal is to learn useful time-dependent
adiabatic gauge potential coefficients without constructing dense
`2**q x 2**q` matrices and without emitting the full `4**q` Pauli basis when
`q` is large.

The concrete running example is `q=20`, but the methodology is intended for
larger systems as well.

## 1. Problem Statement

For `q` qubits, the full Pauli-product basis contains

```text
4**q
```

operator strings. For `q=20`,

```text
4**20 = 1,099,511,627,776
```

terms. It is therefore not practical to make the PINN output one coefficient
for every possible Pauli string.

The Hamiltonian path is assumed to be given:

```text
H_AD(lambda) = (1 - lambda) H_initial + lambda H_final
```

where `H_initial` and `H_final` are input operators decomposed in sparse Pauli
coordinates. The schedule is fixed:

```text
lambda = lambda(t)
```

and in the current experiments it is the sinusoidal schedule already used in the
counterdiabatic protocols.

The AGP is represented as

```text
A_lambda(t) = sum_{P in S_AGP} C_P(t) P
```

where `S_AGP` is a selected sparse AGP support with

```text
K = |S_AGP| << 4**q.
```

The PINN receives time as input and outputs the `K` time-dependent coefficients

```text
C_P(t),  P in S_AGP.
```

The direct counterdiabatic Hamiltonian contribution is

```text
H_CD(t) = dot(lambda)(t) A_lambda(t)
```

so plots and coefficient importance diagnostics usually rank

```text
dot(lambda)(t) C_P(t).
```

## 2. Sparse Pauli-Coordinate Loss

The AGP is trained by minimizing the Euler-Lagrange residual in Pauli-coordinate
space:

```text
R(A) = [i dH_AD/dlambda - [A_lambda, H_AD], H_AD].
```

The exact AGP would make this residual vanish in the full Pauli basis. For large
`q`, we do not evaluate the residual on all `4**q` Pauli strings. Instead, we
evaluate it on a selected residual basis:

```text
S_R_train,    |S_R_train| = R_train.
```

The training loss is the squared residual norm projected onto that residual
basis:

```text
L_train = ||R(A)||^2_{S_R_train}.
```

The implementation computes this symbolically from Pauli multiplication,
commutators, and sparse coefficient tensors. It does not build dense Hilbert
space matrices.

## 3. Choosing the AGP Support `K`

The AGP support is a fixed-budget object. The full Pauli basis is available only
for small enough systems:

```text
q <= 7: K_full = 4**q is still a trainable output budget.
q = 8: K_full = 4**8 is possible as an exact-output diagnostic, but expensive.
q > 7: the active trainable AGP support is capped at K_active = 4**7.
```

For large `q`, the default active AGP support is therefore:

```text
K_active = 4**7 = 16384.
```

This is not a claim that the exact AGP contains only 16384 terms. It is the
maximum active output budget that the method is designed to optimize.

The first support-selection rule is:

1. Load `H_initial` and `H_final` in sparse Pauli form.
2. Compute the symbolic endpoint commutator:

```text
[H_initial, H_final].
```

3. Rank the generated Pauli strings by the absolute value of their commutator
coefficient.
4. Keep the largest `K_active` terms as the initial active AGP pool.

For the q20 support-refinement run:

```text
q = 20
K_active = 16384
K_explore <= 4**8 = 65536 generated outside candidates
```

This means the PINN outputs 16384 coefficient functions, not `4**20`
coefficient functions.

Important caveat: this does not prove that these are the best possible 16384
Pauli strings out of the full basis. It is a physically motivated initial pool.
The curriculum is responsible for improving that pool by looking for strong
outside candidates generated from missed residual directions.

## 3.1 Fixed-Budget Support Refinement

For `q > 7`, the AGP support should not grow without bound. Each curriculum
iteration keeps exactly `K_active` trainable terms:

```text
|S_AGP| = K_active.
```

The iteration is:

1. Train or fine-tune the current active support.
2. Rank active terms by direct counterdiabatic coefficient importance:

```text
I_P = RMS_tau(dot(lambda)(tau) C_P(tau)).
```

3. Evaluate fixed residual bases and identify hard missed residual equations.
4. Generate outside AGP candidates from inverse double-commutator paths.
5. Score those candidates by their projected ability to reduce the current
   residual.
6. Replace weak active terms by stronger outside terms.
7. Accept the replacement only if the fixed validation quotients improve or
   remain within configured tolerances.

The goal is not to discover the perfect unrestricted AGP. For q20 and certainly
for q156, most Pauli strings are never enumerated. The goal is to make the
fixed active support progressively more representative of the important AGP
directions that can be found from the Hamiltonian and the observed residuals.

## 4. Training Residual Basis

Once `S_AGP` is fixed, the loss needs a residual basis. The first training stage
uses a projected residual basis generated from:

```text
H_initial, H_final, S_AGP
```

through symbolic commutator closure and coefficient ranking.

For the current q20 baseline example:

```text
K = 16384 AGP terms
R_train = 8192 residual terms
```

The PINN is trained to reduce the Euler-Lagrange residual only on those 8192
residual equations.

This is a projected training problem. A low training residual means:

```text
The learned AGP satisfies the selected 2048 residual equations well.
```

It does not automatically mean:

```text
The learned AGP satisfies all missing residual equations.
```

## 5. Holdout Residual Basis `Q`

To test whether the AGP generalizes beyond the training projection, we evaluate
the trained model on a larger holdout residual basis:

```text
S_R_holdout,    |S_R_holdout| = Q,    Q > R_train.
```

For the q20 study:

```text
Q = 8192.
```

The holdout residual basis is not used to optimize the initial model. It is used
as a diagnostic after training.

The holdout residual is split into two parts:

```text
seen residual terms   = S_R_holdout intersect S_R_train
unseen residual terms = S_R_holdout \ S_R_train
```

The unseen residual is the most important diagnostic. If it is large, the model
has learned an AGP that works on the training equations but fails on residual
directions that were not included in the loss.

The normalized diagnostic is

```text
relative_residual = ||R(A)||^2 / (||R(A=0)||^2 + eps).
```

This compares the learned AGP against the no-AGP baseline on the same residual
projection.

## 6. Why Larger `K` Can Look Worse on Holdout

Increasing `K` gives the network more coefficient functions. On the training
projection, the loss should generally improve because the model has more
degrees of freedom.

However, if the residual training basis is too small, a larger AGP support can
overfit the selected residual equations. It may cancel the seen residual
directions while creating large residual components in unseen Pauli directions.

Therefore:

```text
larger K improves the projected training loss
```

does not necessarily imply

```text
larger K improves the holdout residual.
```

This is why every large-q sparse AGP experiment must report both training and
holdout residuals.

## 7. Holdout-Feedback Fine-Tuning

The next step is to use the holdout residual as feedback.

The procedure is:

1. Train the PINN with fixed `K` AGP terms and an initial residual basis.
2. Evaluate the trained model on the larger holdout residual basis `Q`.
3. Rank the unseen holdout residual strings by their RMS residual over time.
4. Add the largest unseen residual strings to the training residual basis.
5. Keep the AGP support fixed.
6. Warm-start from the trained PINN weights.
7. Fine-tune using the enlarged residual training basis.
8. Re-evaluate on the same holdout basis.

In formula form:

```text
S_AGP stays fixed.
S_R_train grows.
```

For the q20 run:

```text
q = 20
K = 1024
Q = 8192
initial R_train = 2048
feedback additions = 1024
new R_train = 3072
```

The AGP still has exactly 1024 output coefficients. What changes is the set of
Euler-Lagrange equations used to constrain those coefficients.

This is a curriculum-learning strategy:

```text
train on an initial projected physics loss
evaluate where the physics loss fails
add the hardest missed equations
fine-tune
repeat
```

## 8. Iterative Curriculum

The one-round feedback procedure can be repeated for several iterations:

```text
for i = 1, ..., N_feedback:
    train or fine-tune on S_R_train
    evaluate on S_R_holdout
    identify largest unseen residual terms
    add them to S_R_train
    warm-start and continue
```

For example, with five iterations:

```text
i = 0:
    K = 1024
    R_train = 2048
    Q = 8192

i = 1:
    add 1024 high-residual holdout terms
    R_train = 3072

i = 2:
    add another batch of high-residual holdout terms
    R_train = 4096

i = 3:
    R_train = 5120

i = 4:
    R_train = 6144

i = 5:
    R_train = 7168
```

The exact batch size is a configuration choice. It should be chosen according to
memory, runtime, and the desired strictness of the projected physics loss.

The expected behavior is:

```text
unseen residual should decrease across feedback iterations
holdout residual should decrease or remain stable
coefficient ranking should become more stable
```

If unseen residual stops improving, the fixed AGP support may be insufficient.
At that point, increasing only the residual basis is no longer enough.

## 9. Fixed AGP Support vs AGP Support Expansion

The feedback run described above keeps `S_AGP` fixed. This answers the question:

```text
Can the chosen K AGP terms satisfy a stricter residual basis?
```

It does not answer:

```text
Are we missing important AGP terms outside S_AGP?
```

To answer that stronger question, we need an AGP expansion test:

1. Start with a trained sparse AGP.
2. Evaluate holdout residuals.
3. Use persistent high-residual strings to propose new AGP candidate terms.
4. Retrain with `K + M` AGP terms.
5. Check whether the new AGP terms acquire large
   `RMS_tau(dot(lambda) C_P)` values.
6. Check whether holdout residuals improve materially.

If newly introduced AGP terms become important, the original support was
incomplete. If they remain small and residuals do not improve, the original
support is more defensible.

Thus there are two levels of curriculum:

```text
residual curriculum:
    keep AGP support fixed, add missing residual equations

AGP-support curriculum:
    add new AGP candidate terms when fixed support is no longer enough
```

The current q20 feedback experiment implements the first level.

The exploratory q20 experiment implements a coupled version:

```text
residual curriculum:
    grow the Euler-Lagrange residual equations enforced in the loss

AGP-support curriculum:
    grow the Pauli strings allowed in A_lambda
```

The AGP growth is not random. After each round, the code inspects the largest
remaining holdout residual components and proposes new AGP strings through
symbolic inverse-commutator paths:

```text
P -> [P, H_AD] -> [[P, H_AD], H_AD]
```

If such a path can generate a high-RMS residual Pauli string, the candidate `P`
receives a score proportional to the residual RMS and to the endpoint
Hamiltonian coefficients participating in the path. The best candidates are
added to the AGP support, the old network output rows are copied into the
expanded output layer, and the new rows are initialized near zero before
fine-tuning.

The coupled curriculum now uses a step-level validation gate. Five residual
sets are kept separate:

```text
S_R_train:
    residual equations used directly in the loss

S_R_feedback:
    fixed larger residual pool used to choose hard residual equations

S_R_probe:
    deprecated shorthand for the two fixed probe pools below

S_R_probe_gate:
    fixed disjoint residual pool used to accept or reject a trained curriculum
    step

S_R_probe_watch:
    second fixed disjoint residual pool used to accept or reject a trained
    curriculum step

S_R_probe_test:
    fixed disjoint residual pool reported after every round but never used for
    accept/reject decisions
```

The AGP proposal score is allowed to inspect the feedback spectrum plus the
fixed validation probe spectra. In the robust q20 configuration, a proposed AGP
term must be supported by either `S_R_probe_gate` or `S_R_probe_watch`. A
proposed support expansion is only accepted after a trained candidate round is
evaluated. The gate checks feedback residuals plus both relative and absolute
residuals on `probe_gate` and `probe_watch`. If a candidate worsens these beyond
configured tolerances, the whole step is rejected and the code retries smaller
residual and AGP batches before falling back to an AGP-only or no-op round.

This distinction matters because when `S_AGP` changes, a residual basis
generated from the current AGP support is a moving target. A decreasing moving
"unseen" curve can be misleading, and an increasing one can mix true
generalization failure with basis drift. The probe-gate, probe-watch, and
probe-test bases are cleaner diagnostics: all are selected once and kept
disjoint from feedback and training labels. The gate and watch probes
participate in decisions; the probe-test basis remains an external diagnostic.

New AGP rows are also warmed up before full fine-tuning. During this short
warm-up, hidden layers and old output rows are frozen, and gradients are allowed
only on the newly introduced output rows. The full network is then unfrozen for
the rest of the round. This reduces abrupt changes in previously learned
coefficients when `K` grows.

The current coupled implementation also adds a trust-region retention term on
the already accepted AGP coefficient functions. During fine-tuning, the model is
penalized if the old coefficient functions drift too far from the previously
accepted network on the common AGP support. This does not freeze learning; it
turns support growth into a smaller local update instead of a full rewrite of
the old AGP.

## 10. q20 Feedback Configuration

The current q20 default configuration is:

```text
q = 20
K = 1024
Q = auto
initial R_train = 2048
feedback iterations = 10
feedback additions per iteration = 1024 residual terms
unseen residual batches after final iteration = 1
fine-tuning epochs per iteration = 1000
learning rate = 1e-5
```

The automatic residual budget is:

```text
Q = initial R_train + (feedback iterations + unseen residual batches after final iteration)
    * feedback additions per iteration
```

For the default q20 configuration:

```text
Q = 2048 + (10 + 1) * 1024 = 13312
```

The residual training basis therefore follows the curriculum:

```text
i = 0: R_train = 2048
i = 1: R_train = 3072
i = 2: R_train = 4096
i = 3: R_train = 5120
i = 4: R_train = 6144
i = 5: R_train = 7168
i = 6: R_train = 8192
i = 7: R_train = 9216
i = 8: R_train = 10240
i = 9: R_train = 11264
i = 10: R_train = 12288
```

The final feedback round still leaves `13312 - 12288 = 1024` configured holdout
residual equations unseen. Therefore the unseen residual in the final plot is a
real diagnostic on the selected holdout pool, not an empty-set zero.

The improved coupled configuration separates three decisions:

1. how large the initial AGP support should be,
2. which and how many new terms may enter per curriculum round,
3. how the already accepted coefficient functions are improved without losing
   probe generalization.

For the initial support, `K` is no longer treated as a blind fixed number. The
default q20 coupled run uses an endpoint-commutator coverage rule:

```text
choose the smallest rounded K such that

sum_{j <= K} |c_j([H_initial, H_final])|^2
/
sum_j |c_j([H_initial, H_final])|^2
>= target coverage,
```

with the current q20 bounds

```text
target coverage = 0.99
minimum K       = 1536
maximum K       = 2048
rounding        = multiples of 64
```

For the current q20 Hamiltonian this resolves to the configured maximum
`K=2048`, because the endpoint commutator L2 coverage at `K=2048` is about
`0.9885`, just below the configured `0.99` target. The chosen Pauli strings are
still the largest endpoint-commutator strings; only the number of strings is
selected automatically.

The default coupled configuration is now:

```text
q = 20
initial K = auto endpoint-commutator coverage, resolving to 2048 here
maximum K = 4096
AGP proposals per iteration = 128
Q = auto = 9728
probe-gate residual terms = 8192
probe-watch residual terms = 8192
probe-test residual terms = 16384
initial R_train = 4096
feedback iterations = 10
residual additions per iteration = 512 residual terms
support-admission schedule =
    (R=512, A=128), (R=512, A=64), (R=256, A=64),
    (R=256, A=32), (R=128, A=32), (R=128, A=16),
    (R=128, A=0), (R=0, A=32), (R=0, A=16)
new AGP row warm-up = 400 epochs per accepted growth attempt
trust-region weight = 1e-2
fine-tuning epochs per iteration = 2000
learning rate = 5e-6
```

Candidate AGP terms are still generated by symbolic inverse-commutator paths,
but ranking is now validation-probe-aware. The candidate score combines
selection residual support with fixed `probe_gate` and `probe_watch` support.
Terms that appear useful to both moving and fixed sources receive a
source-diversity bonus, and by default a term must have nonzero validation-probe
support to be admitted into the proposed batch. This makes the proposal stage
less likely to chase terms that only repair one moving selection pool.

The gate is also stricter. A candidate step must satisfy feedback, probe-gate,
and probe-watch worsening bounds. If either validation probe is still larger
than the no-AGP baseline, the candidate must improve it. In the default
configuration:

```text
probe-gate improvement is required while probe_gate_relative_residual > 1.0
candidate_probe_gate <= 0.99 * previous_probe_gate
probe-watch improvement is required while probe_watch_relative_residual > 1.0
candidate_probe_watch <= 0.99 * previous_probe_watch
```

Therefore, if every round passes the probe gate and finds enough useful AGP
candidates:

```text
i = 0: K = 1536
i = 1: K <= 1600
i = 2: K <= 1664
i = 3: K <= 1728
i = 4: K <= 1792
i = 5: K <= 1856
i = 6: K <= 1920
i = 7: K <= 1984
i = 8: K <= 2048
```

This gives the method exploratory capability: the endpoint-selected initial
support is no longer final, but the expansion is still symbolic, sparse, and
driven by the remaining projected Euler-Lagrange residual.

For the coupled configuration, the accepted residual training basis grows more
slowly:

```text
R_train = 2048 + 256 * accepted_residual_steps
```

If all ten residual steps are accepted, the final training residual basis has
`4608` terms and the automatic `Q=4864` budget leaves `256` configured feedback
residual equations unseen.

The first one-round feedback test produced:

```text
baseline:
    train relative residual   = 0.04249
    holdout relative residual = 0.06863
    unseen relative residual  = 1.29722

after one feedback round:
    train relative residual   = 0.06554
    holdout relative residual = 0.06584
    unseen relative residual  = 0.31017
```

The training residual became larger because the training projection became
harder: it now includes 3072 residual equations instead of 2048. This is not a
failure. The key result is that the unseen residual decreased substantially:

```text
1.29722 -> 0.31017
```

This means the same 1024 AGP coefficients became more robust when constrained by
important residual equations that were initially missing from the loss.

## 11. Acceptance Criteria

A sparse AGP support should not be accepted only because the training residual
is small.

The full certification checklist is maintained in
`AGP_CERTIFICATION_CRITERIA.md`. Future large-`q` results must be classified
with that checklist before they are called sufficient or representative.

A robust candidate should satisfy:

1. Low training residual on the active residual basis.
2. Low holdout residual on a larger residual basis.
3. Low unseen residual on holdout strings absent from training.
4. Stable important coefficient ranking across feedback iterations.
5. Limited improvement when adding more residual equations.
6. Limited improvement when adding new AGP candidate terms.
7. Physical validation whenever possible, such as final energy, excitation
   density, or approximate state-evolution metrics.

For current q20 diagnostics, a practical first threshold is:

```text
holdout_relative_residual <= 0.10
unseen_relative_residual <= 1.0
```

The one-round feedback run passed these two thresholds. The current default
configuration now continues this process for ten feedback iterations.

## 12. What Must Be Reported

Every large-q experiment should report:

```text
q
4**q
K = number of AGP terms
Q = holdout residual basis size
R_train at each feedback iteration
selection rule for S_AGP
selection rule for S_R_train
selection rule for S_R_holdout
selection rule and size for S_R_probe_gate
selection rule and size for S_R_probe_watch
selection rule and size for S_R_probe_test
number of feedback iterations
number of residual terms added per iteration
training relative residual
holdout relative residual
unseen relative residual
probe_gate relative residual
probe_watch relative residual
probe_test relative residual
top learned AGP coefficients
least important learned AGP coefficients
coefficient-ranking stability
whether AGP support was fixed or expanded
```

Without these diagnostics, the result should be described as a projected sparse
AGP experiment, not as evidence that the unrestricted AGP has been solved.

## 13. Current Implementation

The retained q20 workflow is configured under:

```text
tests/q20/sweep_test/
```

Main scripts:

```text
scripts/agp_baseline_train.py
    trains the fixed-support q20 baseline

scripts/agp_holdout_study.py
    evaluates trained supports on a common holdout residual basis

scripts/agp_holdout_feedback.py
    performs holdout-feedback fine-tuning at fixed K

scripts/diagnostics/agp_coupled_curriculum.py
    performs holdout-feedback fine-tuning while expanding K
```

The default feedback command reads the configured `K`, `Q=auto`, `i=10`
curriculum from `tests/q20/sweep_test/config.json`. If the cleaned folder has no baseline
checkpoint, this command trains the baseline first:

```bash
conda run --no-capture-output -n torch-mps python scripts/agp_holdout_feedback.py \
  --config tests/q20/sweep_test/config.json
```

The feedback summary exports round-wise residual plots and, for the final round,
copies these two AGP-structure plots into the main feedback `Images/` folder:

```text
hcd_coefficient_support_map.pdf
hcd_connection_summary.pdf
```

The coupled curriculum additionally exports:

```text
coupled_curriculum_support_growth.pdf
coupled_curriculum_residuals_vs_agp_terms.pdf
coupled_curriculum_probe_gate.pdf
hcd_coefficient_support_map.pdf
hcd_connection_summary.pdf
```

## 14. Interpretation

The current methodology does not claim to solve the full `4**q` AGP problem.
Instead, it builds a scalable projected approach:

```text
choose a physically motivated sparse AGP support
train on symbolic Pauli residual equations
test against a larger holdout residual basis
feed the missed residual equations back into training
iterate until holdout/unseen residuals stabilize
expand AGP support only when fixed support no longer improves
```

This gives a practical route to large-qubit AGP learning while keeping every
step symbolic and sparse.
