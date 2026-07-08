# q15 Sparse AGP Physical Validation

This folder runs a q15 sparse AGP curriculum above the exact-output regime
(`q > 8`) while keeping the trainable AGP support fixed:

```text
K = 32768 trainable AGP outputs
Q = 65536 generated residual holdout terms
i = 15 holdout-feedback iterations
```

The Hamiltonian path is a driver-to-problem transverse Ising example:

```text
H_initial = - sum_i X_i
H_final   = sum_i h_i Z_i + sum_i J_i Z_i Z_{i+1}
lambda(t) = sin^2(pi t / 2T) + tau^2 (1 - tau)^2 A_sched tanh(u_theta(tau))
```

The schedule correction is trained jointly with the AGP and calibration
variables. The envelope enforces `lambda(0)=0`, `lambda(T)=1`, and zero endpoint
derivatives by construction.

The current retained neural architecture is configured entirely in
`config.json`: a quadratic/QRes AGP coefficient network with width 96, four
hidden layers, and SiLU activations, plus a trainable schedule MLP with width
32, two hidden layers, and tanh activations.

The baseline support is selected from a bounded nested-commutator Krylov pool
seeded by the order-1 direction `i[H, dH]`. This never enumerates the full
`4**15` basis; the support is a deliberate sparse research choice.

Feedback keeps the AGP support fixed. Each round evaluates a larger residual
holdout basis, adds the highest-RMS unseen residual equations to the training
residual basis, and fine-tunes the same coefficient functions.

The current residual-feedback budget uses `intermediate_top_k = 32768`,
`Q = 65536`, and `add_residual_terms_per_iteration = 3072`. The final trained
round therefore uses 50176 residual equations and keeps a 15360-term unseen
residual batch for the final diagnostic.

Unseen relative residuals are reported only when the AGP=0 reference residual
on the unseen subset is nonzero. If that reference is zero, the quotient is
not physically meaningful and the summary stores `null` for the quotient plus
the absolute unseen residual and per-term residual.

Generated artifacts are ignored by git and written under:

```text
runs/baselines/agp_32768/
runs/fixed_k_holdout_feedback_trainable_schedule_w96_l4_v1/agp_32768_residual_65536_add_3072_rounds_15/
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
`runs/baselines/agp_32768/` checkpoint is missing, this command trains it first
and then executes the fifteen holdout-feedback rounds:

```bash
conda run --no-capture-output -n torch-mps python scripts/agp_holdout_feedback.py \
  --config tests/q15/sweep_test/config.json
```

Train only the baseline `K=32768` AGP model:

```bash
conda run --no-capture-output -n torch-mps python scripts/agp_baseline_train.py \
  --config tests/q15/sweep_test/config.json
```

## Physical Validation

The configured AGP calibration is trained jointly during the baseline and all
holdout-feedback curriculum rounds. It trains a global AGP scale plus soft
Pauli gates plus the bounded schedule correction using only the projected
Euler-Lagrange residual and schedule regularizers; it does not use the final
ground-state energy, fidelity, or exact final observables.

After training, run the q15 statevector diagnostic:

```bash
conda run --no-capture-output -n torch-mps python scripts/agp_physical_validation.py \
  --config tests/q15/sweep_test/config.json
```

This compares final observables for:

```text
no_cd
kipu_dqfm_l1
learned_sparse_agp
```

The diagnostic reports final energy, excitation probability, ground-state
fidelity, `<Z_i>` RMSE, nearest-neighbor `<Z_i Z_{i+1}>` RMSE, and improvement
ratios relative to the no-CD evolution. The q15 ground truth is used only here,
after training, to benchmark the learned AGP.

The learned AGP row uses the exported learned schedule grid. The no-CD and
Kipu/DQFM l=1 rows use the fixed reference `sin^2(pi t / 2T)` schedule.

The statevector code is intentionally kept as a script-level diagnostic because
it is not a scalable library path.
