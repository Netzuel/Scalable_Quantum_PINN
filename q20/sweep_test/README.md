# q20 Support-Size Sweep

This folder runs a fixed-support q=20 AGP sweep outside `tests/`.

The controlled variable is the AGP output support size:

```text
576, 768, 1024, 1536, 2048
```

Adaptive support growth is disabled so each run keeps exactly its requested
number of AGP terms. The q=20 Hamiltonian, residual projection size, architecture,
time grid, seed, and optimizer are kept fixed.

Generated artifacts are ignored by git and written to:

```text
runs/agp_<N>/
Images/
Models_Data/
```

Run the full sweep with:

```bash
conda run --no-capture-output -n torch-mps python q20/sweep_test/training_script.py
```

Rebuild only the sweep summary from completed runs with:

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

Run one holdout-feedback training pass. This keeps the AGP support fixed, adds
the largest unseen holdout residual strings to the training residual basis, and
fine-tunes from the existing checkpoint:

```bash
conda run --no-capture-output -n torch-mps python q20/sweep_test/holdout_feedback_training.py \
  --base-agp-terms 1024 \
  --rounds 1 \
  --add-residual-terms 1024 \
  --epochs-per-round 1000 \
  --lr 1e-5 \
  --device mps
```
