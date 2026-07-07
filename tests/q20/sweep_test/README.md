# q20 Sparse AGP Curriculum

This folder configures the q20 sparse AGP curriculum under `tests/`. The
default methodology is the fixed-`K` holdout-feedback curriculum:

```text
K = 4**7 = 16384 trainable AGP outputs
i = 10 feedback iterations
```

For `q > 7`, the code does not attempt to enumerate or train the full `4**q`
Pauli basis. The research choice is the explicit AGP subset `S_AGP` with
`|S_AGP| = 16384`. The baseline support is selected from the largest generated
symbolic endpoint-commutator terms in `|[H_initial, H_final]|`, which is the
matrix-free proxy used here for initially important AGP directions.

The feedback curriculum keeps this AGP support fixed. Each round evaluates the
current model on a larger residual holdout basis, selects the largest unseen
residual equations, adds those equations to the training residual basis, and
fine-tunes the same coefficient functions:

```text
round 0: train fixed S_AGP on Q0 residual equations
round r: add hard residual equations from holdout, keep S_AGP unchanged
```

With the current config:

```text
Q0 = 2048
add_residual_terms_per_iteration = 1024
i = 10
Q_holdout(auto) = 2048 + (10 + 1) * 1024 = 13312
```

The final trained round therefore uses 12288 residual equations and leaves one
1024-term unseen residual batch for the final diagnostic. This is the intended
monotone-feedback test: the learned AGP is judged by whether adding hard
projected residual equations improves the residual quotients relative to
`AGP = 0`.

Generated artifacts are ignored by git and written under:

```text
runs/baselines/agp_16384/
runs/fixed_k_holdout_feedback_v2/agp_16384_residual_13312_add_1024_rounds_10/
runs/support_sweep_summary/Images/
runs/support_sweep_summary/Models_Data/
```

The top-level `Images/` and `Models_Data/` scratch folders are not created.
For completed feedback runs, canonical report figures are kept in the run-level
`Images/` folder. Per-round checkpoints and data are kept under `rounds/`;
per-round figure folders are pruned by default to avoid repeated copies of the
same diagnostics.

## Clean

Clean generated artifacts and recreate only the run root:

```bash
conda run -n torch-mps python scripts/agp_restart.py --config tests/q20/sweep_test/config.json
```

## Train

Run the default end-to-end pipeline. If the baseline
`runs/baselines/agp_16384/` checkpoint is missing, this command trains it first
and then executes the ten holdout-feedback rounds:

```bash
conda run --no-capture-output -n torch-mps python scripts/agp_holdout_feedback.py \
  --config tests/q20/sweep_test/config.json
```

Train only the baseline `K=16384` AGP model:

```bash
conda run --no-capture-output -n torch-mps python scripts/agp_baseline_train.py \
  --config tests/q20/sweep_test/config.json
```

Rebuild only the baseline summary from completed runs:

```bash
conda run -n torch-mps python scripts/agp_baseline_train.py \
  --config tests/q20/sweep_test/config.json \
  --summary-only
```

Evaluate a trained support on a larger residual holdout basis without
retraining:

```bash
conda run --no-capture-output -n torch-mps python scripts/agp_evaluate_holdout.py \
  --config tests/q20/sweep_test/config.json \
  --trained-run tests/q20/sweep_test/runs/baselines/agp_16384 \
  --residual-top-k 13312 \
  --device cpu
```

Run the holdout study across all trained support sizes and rebuild the summary
plots:

```bash
conda run --no-capture-output -n torch-mps python scripts/agp_holdout_study.py \
  --config tests/q20/sweep_test/config.json \
  --residual-top-k 13312 \
  --device cpu
```

## Optional Diagnostics

The fixed-budget residual/AGP-support refinement scripts are retained as
diagnostics, but they are not the default q20 methodology. They can be useful
after the fixed-`K` feedback run to check whether excluded AGP terms look
promising enough to justify a separate support-selection experiment.

Run the diagnostic fixed-budget support-refinement curriculum:

```bash
conda run --no-capture-output -n torch-mps python tests/q20/sweep_test/coupled_curriculum_training.py
```

With the current config this writes under:

```text
runs/diagnostic_fixed_budget_support_refinement_v1/
```

Build pruned support candidates from the final coefficient ranking:

```bash
conda run -n torch-mps python tests/q20/sweep_test/prune_support.py
```

Classify the latest diagnostic coupled result against
`AGP_CERTIFICATION_CRITERIA.md`:

```bash
conda run -n torch-mps python tests/q20/sweep_test/certify_sparse_agp.py
```

The certification script writes `certification_summary.json` and marks each gate
as `pass`, `fail`, or `not tested`. Any `fail` or `not tested` downgrades the
claim level.
