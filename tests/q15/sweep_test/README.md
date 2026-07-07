# q15 Sparse AGP Physical Validation

This folder runs a q15 sparse AGP curriculum above the exact-output regime
(`q > 8`) while keeping the trainable AGP support fixed:

```text
K = 4**7 = 16384 trainable AGP outputs
i = 10 holdout-feedback iterations
```

The Hamiltonian path is a driver-to-problem transverse Ising example:

```text
H_initial = - sum_i X_i
H_final   = sum_i h_i Z_i + sum_i J_i Z_i Z_{i+1}
lambda(t) = sin^2(pi t / 2T)
```

The baseline support is selected from a bounded nested-commutator Krylov pool
seeded by the order-1 direction `i[H, dH]`. This never enumerates the full
`4**15` basis; the support is a deliberate sparse research choice.

Feedback keeps the AGP support fixed. Each round evaluates a larger residual
holdout basis, adds the highest-RMS unseen residual equations to the training
residual basis, and fine-tunes the same coefficient functions.

The residual-feedback budget is fitted automatically after the generated
holdout pool is known. If the requested schedule would exhaust the generated
residual labels before the requested `i` iterations, the script keeps `i`
fixed and reduces `add_residual_terms_per_iteration` so the final round still
has an unseen residual batch. For the current q15 Hamiltonian, the generated
pool has 6737 residual labels, so the default request
`add_residual_terms_per_iteration = 1024` is fitted to `426`.

Unseen relative residuals are reported only when the AGP=0 reference residual
on the unseen subset is nonzero. If that reference is zero, the quotient is
not physically meaningful and the summary stores `null` for the quotient plus
the absolute unseen residual and per-term residual.

Generated artifacts are ignored by git and written under:

```text
runs/baselines/agp_16384/
runs/fixed_k_holdout_feedback_v1/agp_16384_residual_6737_add_426_rounds_10/
runs/support_sweep_summary/
```

The top-level `Images/` and `Models_Data/` scratch folders are not created.
For completed feedback runs, canonical report figures are kept in the run-level
`Images/` folder. Per-round figure folders are pruned by default.

## Prepare Hamiltonian

Generate the q15 analytic driver/problem Hamiltonian and update the repository
Pauli-decomposition index:

```bash
conda run -n torch-mps python scripts/build_driver_problem_hamiltonian.py --update-index
```

## Clean

Clean generated artifacts and recreate only the run root:

```bash
conda run -n torch-mps python scripts/agp_restart.py --config tests/q15/sweep_test/config.json
```

## Train

Run the default end-to-end pipeline. If the baseline
`runs/baselines/agp_16384/` checkpoint is missing, this command trains it first
and then executes the ten holdout-feedback rounds:

```bash
conda run --no-capture-output -n torch-mps python scripts/agp_holdout_feedback.py \
  --config tests/q15/sweep_test/config.json
```

Train only the baseline `K=16384` AGP model:

```bash
conda run --no-capture-output -n torch-mps python scripts/agp_baseline_train.py \
  --config tests/q15/sweep_test/config.json
```

## Physical Validation

After training, run the q15 statevector diagnostic:

```bash
conda run --no-capture-output -n torch-mps python scripts/agp_physical_validation.py \
  --config tests/q15/sweep_test/config.json
```

This compares final observables for:

```text
no_cd
nested_l1
learned_sparse_agp
```

The diagnostic reports final energy, excitation probability, ground-state
fidelity, `<Z_i>` RMSE, nearest-neighbor `<Z_i Z_{i+1}>` RMSE, and improvement
ratios relative to the no-CD evolution. The statevector code is intentionally
kept as a script-level diagnostic because it is not a scalable library path.
