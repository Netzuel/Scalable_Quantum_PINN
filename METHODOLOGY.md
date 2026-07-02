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

The AGP support is selected before training. In the current q20 experiments, the
rule is:

1. Load `H_initial` and `H_final` in sparse Pauli form.
2. Compute the symbolic endpoint commutator:

```text
[H_initial, H_final].
```

3. Rank the generated Pauli strings by the absolute value of their commutator
coefficient.
4. Keep the largest `K` terms.

For the example run:

```text
q = 20
K = 1024
```

This means the PINN outputs 1024 coefficient functions, not `4**20`
coefficient functions.

Important caveat: this does not prove that these are the best possible 1024
Pauli strings out of the full basis. It is a physically motivated sparse
selection rule based on endpoint noncommutativity.

## 4. Training Residual Basis

Once `S_AGP` is fixed, the loss needs a residual basis. The first training stage
uses a projected residual basis generated from:

```text
H_initial, H_final, S_AGP
```

through symbolic commutator closure and coefficient ranking.

For the q20 baseline example:

```text
K = 1024 AGP terms
R_train = 2048 residual terms
```

The PINN is trained to reduce the Euler-Lagrange residual only on those 2048
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

The coupled curriculum now uses a step-level validation gate. Four residual
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

S_R_probe_test:
    fixed disjoint residual pool reported after every round but never used for
    accept/reject decisions
```

The AGP proposal score is allowed to inspect both the feedback spectrum and the
probe-gate spectrum, but a proposed support expansion is only accepted after a
trained candidate round is evaluated. The gate checks both the relative and
absolute probe-gate residuals, plus the feedback residual. If a candidate
worsens these beyond configured tolerances, the whole step is rejected and the
code retries smaller residual batches before falling back to an AGP-only or
no-op round.

This distinction matters because when `S_AGP` changes, a residual basis
generated from the current AGP support is a moving target. A decreasing moving
"unseen" curve can be misleading, and an increasing one can mix true
generalization failure with basis drift. The probe-gate and probe-test bases are
cleaner diagnostics: both are selected once, kept disjoint from feedback and
training labels, and evaluated consistently after every round. Only the
probe-gate basis participates in decisions, leaving the probe-test basis as an
external diagnostic.

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

The default coupled configuration is:

```text
q = 20
initial K = 1024
maximum K = 1664
AGP proposals per iteration = 64
Q = auto = 4864
probe-gate residual terms = 2048
probe-test residual terms = 4096
initial R_train = 2048
feedback iterations = 10
residual additions per iteration = 256 residual terms
residual backtracking counts = 256, 128, 64, 0
new AGP row warm-up = 100 epochs per accepted growth attempt
trust-region weight = 1e-4
fine-tuning epochs per iteration = 1000
learning rate = 1e-5
```

Therefore, if every round passes the probe gate and finds enough useful AGP
candidates:

```text
i = 0: K = 1024
i = 1: K = 1088
i = 2: K = 1152
i = 3: K = 1216
i = 4: K = 1280
i = 5: K = 1344
i = 6: K = 1408
i = 7: K = 1472
i = 8: K = 1536
i = 9: K = 1600
i = 10: K = 1664
```

This gives the method exploratory capability: the initial `K=1024` ansatz is no
longer final, but the expansion is still symbolic, sparse, and driven by the
remaining projected Euler-Lagrange residual.

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
number of feedback iterations
number of residual terms added per iteration
training relative residual
holdout relative residual
unseen relative residual
top learned AGP coefficients
least important learned AGP coefficients
coefficient-ranking stability
whether AGP support was fixed or expanded
```

Without these diagnostics, the result should be described as a projected sparse
AGP experiment, not as evidence that the unrestricted AGP has been solved.

## 13. Current Implementation

The q20 workflow lives under:

```text
q20/sweep_test/
```

Main scripts:

```text
training_script.py
    trains the fixed-support q20 baseline

holdout_study.py
    evaluates trained supports on a common holdout residual basis

holdout_feedback_training.py
    performs holdout-feedback fine-tuning at fixed K

coupled_curriculum_training.py
    performs holdout-feedback fine-tuning while expanding K
```

The default feedback command reads the `K=1024`, `Q=auto`, `i=10` curriculum
from `q20/sweep_test/config.json`. If the cleaned folder has no baseline
checkpoint, this command trains the baseline first:

```bash
conda run --no-capture-output -n torch-mps python q20/sweep_test/holdout_feedback_training.py
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
