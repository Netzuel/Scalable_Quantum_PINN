# q20 Sparse AGP Curriculum

This folder configures the q20 sparse AGP curriculum under `tests/`. The
default methodology mirrors the current retained q15 pipeline while keeping the
q20 trainable AGP output budget fixed at the q7 cap:

```text
K = 4**7 = 16384 trainable AGP outputs
Q = 18432 generated residual holdout terms
i = 15 feedback iterations
```

For `q > 7`, the code does not attempt to enumerate or train the full `4**q`
Pauli basis. The research choice is the explicit AGP subset `S_AGP` with
`|S_AGP| = 16384`. For this q20 hydrogen Hamiltonian, the initial support is
selected from the largest generated symbolic endpoint-commutator terms in
`|[H_initial, H_final]|`. This keeps support construction tractable for the
large sparse hydrogen Hamiltonian while the support-swap curriculum supplies the
exploratory replacement mechanism.

The current neural architecture is configured entirely in `config.json`: a
quadratic/QRes AGP coefficient network with width 96, four hidden layers, and
trainable PAU activations. The PAU feedback stage warm-starts from a width-96
SiLU baseline checkpoint. The schedule correction is trained jointly with the
AGP and calibration variables using the bounded envelope

```text
lambda(t) = sin^2(pi t / 2T) + tau^2 (1 - tau)^2 A_sched tanh(u_theta(tau)).
```

Feedback keeps the AGP support size fixed at `K=16384`. Each round evaluates a
larger residual holdout basis, adds the highest-RMS unseen residual equations
to the training residual basis, swaps 256 weak AGP strings for hard
residual-derived candidates, remaps retained output rows by Pauli label, and
fine-tunes the same coefficient functions. After round 15, the current q20 run
uses a temporal-refinement continuation and then an adaptive temporal-refinement
continuation on the fixed support and residual basis.

With the current q20 config:

```text
Q0 = 2048
add_residual_terms_per_iteration = 1024
i = 15
Q_holdout(auto) = 2048 + (15 + 1) * 1024 = 18432
```

The final trained round therefore uses 17408 residual equations and leaves one
1024-term unseen residual batch for the final diagnostic. Unseen relative
residuals are reported only when the AGP=0 reference residual on the unseen
subset is nonzero. If that reference is zero, the quotient is not physically
meaningful and the summary stores `null` for the quotient plus the absolute
unseen residual.

Generated artifacts are ignored by git and written under:

```text
runs/baselines/agp_16384/
runs/fixed_k_holdout_feedback_trainable_schedule_w96_l4_pau_support_swap_adaptive_temporal_refinement_v1/agp_16384_residual_18432_add_1024_rounds_15/
runs/support_sweep_summary/
```

The top-level `Images/` and `Models_Data/` scratch folders are not created.
For completed feedback runs, canonical report figures are kept in the run-level
`Images/` folder. Per-round checkpoints and data are kept under `rounds/`;
per-round figure folders are pruned by default to avoid repeated copies of the
same diagnostics.

## Clean

Clean generated artifacts and recreate only the run root:

```bash
conda run -n torch-mps python scripts/agp_restart.py --config tests/sparse_agp_curriculum/q20/sweep_test/config.json
```

## Train

Run the default end-to-end pipeline. If the baseline
`runs/baselines/agp_16384/` checkpoint is missing, this command trains it first
and then executes the fifteen holdout-feedback rounds plus the two
post-curriculum temporal-refinement stages:

```bash
conda run --no-capture-output -n torch-mps python scripts/agp_holdout_feedback.py \
  --config tests/sparse_agp_curriculum/q20/sweep_test/config.json
```

Train only the baseline `K=16384` AGP model:

```bash
conda run --no-capture-output -n torch-mps python scripts/agp_baseline_train.py \
  --config tests/sparse_agp_curriculum/q20/sweep_test/config.json
```

Rebuild only the baseline summary from completed runs:

```bash
conda run -n torch-mps python scripts/agp_baseline_train.py \
  --config tests/sparse_agp_curriculum/q20/sweep_test/config.json \
  --summary-only
```

Evaluate a trained support on a larger residual holdout basis without
retraining:

```bash
conda run --no-capture-output -n torch-mps python scripts/agp_evaluate_holdout.py \
  --config tests/sparse_agp_curriculum/q20/sweep_test/config.json \
  --trained-run tests/sparse_agp_curriculum/q20/sweep_test/runs/baselines/agp_16384 \
  --residual-top-k 18432 \
  --device cpu
```

Run the holdout study across all trained support sizes and rebuild the summary
plots:

```bash
conda run --no-capture-output -n torch-mps python scripts/agp_holdout_study.py \
  --config tests/sparse_agp_curriculum/q20/sweep_test/config.json \
  --residual-top-k 18432 \
  --device cpu
```

## Certification Note

Unlike q15, this q20 hydrogen study has no full statevector physical validation
step in the default pipeline. It should therefore be reported as a projected
sparse AGP experiment unless the certification gates in
`AGP_CERTIFICATION_CRITERIA.md` are explicitly evaluated and passed.

## Latest Run Diagnostics

The latest cleaned end-to-end run used:

```text
run = runs/fixed_k_holdout_feedback_trainable_schedule_w96_l4_pau_support_swap_adaptive_temporal_refinement_v1/agp_16384_residual_18432_add_1024_rounds_15/
K = 16384
Q = 18432
i = 15
```

The baseline trained well on its initial 2048 residual equations, but it did not
generalize to the larger holdout pool:

```text
baseline training relative residual = 0.010413
baseline holdout relative residual  = 4901.547
baseline unseen relative residual   = 22695.164
```

The fixed-K support-swap feedback curriculum corrected that failure. The final
round-15 diagnostics were:

```text
training relative residual = 0.011567
holdout relative residual  = 0.016104
unseen relative residual   = 0.037473
```

The post-curriculum uniform temporal-refinement stage improved training and
holdout residuals:

```text
training relative residual = 0.008906
holdout relative residual  = 0.013615
unseen relative residual   = 0.039642
```

The final adaptive temporal-refinement stage, which is the configured final
checkpoint, slightly improved training and holdout residuals again while
slightly worsening the unseen residual:

```text
training relative residual = 0.008717
holdout relative residual  = 0.013334
unseen relative residual   = 0.042667
```

Under `AGP_CERTIFICATION_CRITERIA.md`, the current q20 status is:

| Gate | Status | Evidence |
|---|---|---|
| Training residual | pass | adaptive temporal-refinement training relative residual `0.008717` |
| Holdout residual | pass | adaptive temporal-refinement holdout relative residual `0.013334` |
| Unseen residual | pass | adaptive temporal-refinement unseen relative residual `0.042667` with finite AGP=0 reference |
| Fixed `probe_gate` / `probe_watch` / `probe_test` residuals | not tested | no fixed disjoint probe bases were evaluated in this run |
| K-sweep plateau | not tested | only `K = 16384` was trained |
| Q-sweep plateau | not tested | only `Q = 18432` was evaluated |
| Top-term stability across K and seeds | not tested | no independent K/seed stability study was run |
| Prune-and-retest | not tested | no residual prune sweep was run |
| Physical validation | not tested | no scalable q20 physical-observable validation is configured |

The correct claim level is therefore:

```text
Projected sparse AGP experiment with strong residual validation on the selected
q20 projected residual basis.
```

## Optional Diagnostics

The fixed-budget residual/AGP-support refinement scripts are retained as
diagnostics, but they are not the default q20 methodology. They can be useful
after the fixed-`K` feedback run to check whether excluded AGP terms look
promising enough to justify a separate support-selection experiment.

Run the diagnostic fixed-budget support-refinement curriculum:

```bash
conda run --no-capture-output -n torch-mps python scripts/diagnostics/agp_coupled_curriculum.py \
  --config tests/sparse_agp_curriculum/q20/sweep_test/config.json
```

With the current config this writes under:

```text
runs/diagnostic_fixed_budget_support_refinement_v1/
```

Build pruned support candidates from the final coefficient ranking:

```bash
conda run -n torch-mps python scripts/diagnostics/agp_prune_support.py \
  --config tests/sparse_agp_curriculum/q20/sweep_test/config.json
```

Classify the latest diagnostic coupled result against
`AGP_CERTIFICATION_CRITERIA.md`:

```bash
conda run -n torch-mps python scripts/diagnostics/agp_certify_coupled.py \
  --config tests/sparse_agp_curriculum/q20/sweep_test/config.json
```

The certification script writes `certification_summary.json` and marks each gate
as `pass`, `fail`, or `not tested`. Any `fail` or `not tested` downgrades the
claim level.
