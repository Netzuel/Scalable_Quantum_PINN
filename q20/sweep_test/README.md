# q20 Sparse AGP Curriculum

This folder runs the q20 sparse AGP curriculum outside `tests/`.

The default configuration is:

```text
K = 1024 AGP output terms
Q = auto holdout residual terms
i = 10 holdout-feedback iterations
```

The AGP support is fixed during feedback. The curriculum adds the largest unseen
holdout residual equations back into the training residual basis and fine-tunes
the same `K=1024` coefficient functions. In automatic mode,

```text
Q = initial_residual_terms + (i + unseen_batches_after_final_iteration)
    * add_residual_terms_per_iteration
```

For the default q20 settings this resolves to `Q = 13312`, leaving 1024
configured holdout residual equations unseen after round 10.

The coupled curriculum extends this by also growing the AGP support:

```text
K = 1024 -> at most 1664 by default
propose 64 AGP terms per round
try up to 256 residual equations per round
```

New AGP terms are proposed from high-RMS holdout residual strings using symbolic
inverse-commutator paths of the form `P -> [P, H] -> [[P, H], H]`.
The current coupled run is gated at the whole-step level. It keeps two fixed
probe bases disjoint from the feedback residual pool:

```text
probe_gate = validation basis used for accept/reject decisions
probe_test = diagnostic basis reported but not used for decisions
```

Candidate residual/AGP expansions are accepted only if the feedback residual and
the `probe_gate` residual remain within configured relative and absolute
worsening tolerances. Rejected steps are retried with smaller residual batches
(`256 -> 128 -> 64 -> 0` by default), and the accepted trajectory is rolled back
if no safe step is found. A small trust-region penalty keeps previously learned
AGP coefficient functions from drifting during fine-tuning.

Generated artifacts are ignored by git and written to:

```text
runs/agp_<N>/
Images/
Models_Data/
```

Clean generated artifacts and recreate empty output folders:

```bash
conda run -n torch-mps python q20/sweep_test/restart_folders.py
```

Train the baseline `K=1024` AGP model:

```bash
conda run --no-capture-output -n torch-mps python q20/sweep_test/training_script.py
```

Rebuild only the baseline summary from completed runs with:

```bash
conda run -n torch-mps python q20/sweep_test/training_script.py --summary-only
```

Evaluate a trained support on a larger residual holdout basis without retraining:

```bash
conda run --no-capture-output -n torch-mps python q20/sweep_test/evaluate_holdout_residual.py \
  --trained-run q20/sweep_test/runs/agp_1536 \
  --residual-top-k 8192 \
  --device cpu
```

Run the full holdout study across all trained support sizes and rebuild the
summary plots. By default this evaluates every trained support on the same
8192-term holdout residual basis, generated from the union of the trained AGP
supports:

```bash
conda run --no-capture-output -n torch-mps python q20/sweep_test/holdout_study.py \
  --residual-top-k 8192 \
  --device cpu
```

Run the default ten-iteration holdout-feedback curriculum. If the baseline
`runs/agp_1024/` checkpoint is missing, this command trains it first using
`config.json`.

```bash
conda run --no-capture-output -n torch-mps python q20/sweep_test/holdout_feedback_training.py
```

The feedback summary exports the round-wise residual plots plus the final-round
`hcd_coefficient_support_map.pdf` and `hcd_connection_summary.pdf` in the main
feedback `Images/` folder.

Run the coupled residual/AGP-support curriculum:

```bash
conda run --no-capture-output -n torch-mps python q20/sweep_test/coupled_curriculum_training.py
```

The coupled summary exports all feedback plots plus:

```text
coupled_curriculum_support_growth.pdf
coupled_curriculum_residuals_vs_agp_terms.pdf
coupled_curriculum_probe_gate.pdf
```
