# Q-Aware Sparse AGP Scaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attribute the q156 fidelity loss, add dimensionless q-aware sparse-AGP training controls, retrain q156, and promote only a full-support convergence-certified physical improvement.

**Architecture:** First compare existing checkpoint/export variants under one inexpensive tensor-network resolution. Then add backward-compatible loss and budget policies in the shared model/training surface, followed by deterministic q-aware support allocation. Frozen projected probes select a training candidate; exact ground truth is used only by the final full-K tensor-network evaluation.

**Tech Stack:** Python 3.10, PyTorch, NumPy, physics-tenpy, JSON, `unittest`, Matplotlib.

## Global Constraints

- Run every Python command with `conda run -n torch-mps`.
- Do not use exact ground energy or ground state in training or candidate selection.
- q156 canonical validation must use all 32,768 learned terms and pass independent TN convergence gates.
- Label reduced-support evaluations as ablations.
- Keep existing configuration behavior unchanged unless a new mode is explicitly enabled.
- Do not commit generated runs, checkpoints, or result directories.

---

### Task 1: Attribute Checkpoint And Export Effects

**Files:**
- Read: `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q156/sweep_test/config.json`
- Read: `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q156/sweep_test/runs/fixed_k_holdout_feedback_trainable_schedule_w96_l4_pau_support_swap_adaptive_temporal_refinement_v1/agp_32768_residual_69855_add_3072_rounds_20/`
- Create when needed: `scripts/diagnostics/agp_resample_checkpoint.py`
- Test when needed: `tests/test_agp_resample_checkpoint.py`

**Interfaces:**
- Consumes: retained round-20 and adaptive `final_agp_coefficients.pt` artifacts.
- Produces: source-matched 8,192-term attribution summaries and one immutable attribution champion.

- [ ] Run the adaptive checkpoint with `steps=24`, `max_bond=32`, `learned_terms=8192`, and protocol `learned_sparse_agp` under an ablation output directory.
- [ ] Compare with retained round-20 top-8,192 metrics and apply `delta_fidelity >= 1e-3` plus `delta_energy_error <= -5e-2`.
- [ ] If adaptive fails, write a failing bounded-q test proving dense checkpoint resampling preserves labels, calibration, schedule endpoints, and requested sample count.
- [ ] Run `conda run -n torch-mps python -m unittest tests.test_agp_resample_checkpoint -v` and confirm the new test fails because the resampling interface is absent.
- [ ] Implement a config-driven resampler that reconstructs the saved model and writes a new coefficient artifact without mutating the retained run.
- [ ] Re-run the focused test, then repeat the 8,192-term attribution comparison for dense round 20.
- [ ] Persist an attribution JSON recording candidate paths, source hashes, TN settings, metric deltas, and the champion decision.

### Task 2: Add Reference-Normalized Residual Training

**Files:**
- Modify: `models.py`
- Modify: `scripts/projected_sparse_training_common.py`
- Modify: `scripts/agp_holdout_feedback.py`
- Modify: `scripts/diagnostics/agp_coupled_curriculum.py`
- Test: `tests/test_sparse_pauli.py`
- Test: `tests/test_agp_joint_calibration.py`

**Interfaces:**
- Produces: `ProjectedSparseLossWeights.residual_objective: str` with values `absolute` and `reference_normalized`.
- Consumes: `training.loss.residual_objective` from experiment JSON.

- [ ] Add a failing test where two otherwise identical residual/reference pairs scaled by different constants yield different absolute objectives but equal `reference_normalized` objectives.
- [ ] Add failing settings tests proving the default is `absolute` and the explicit JSON value reaches all projected loss-weight construction paths.
- [ ] Run `conda run -n torch-mps python -m unittest tests.test_sparse_pauli tests.test_agp_joint_calibration -v` and verify failures are caused by the missing field.
- [ ] Implement `reference_normalized` as `residual_loss / reference_loss.detach().clamp_min(eps)` while retaining every existing diagnostic in physical units.
- [ ] Reject unknown objective names with `ValueError` before optimization.
- [ ] Pass the setting through baseline, feedback, and refinement loss construction.
- [ ] Re-run both focused test modules and confirm they pass.

### Task 3: Add Target-Normalized Gate Budgets And Q-Aware Counts

**Files:**
- Modify: `models.py`
- Modify: `scripts/projected_sparse_training_common.py`
- Create: `scripts/agp_resource_policy.py`
- Test: `tests/test_agp_joint_calibration.py`
- Create: `tests/test_agp_resource_policy.py`

**Interfaces:**
- Produces: `ProjectedSparseLossWeights.calibration_budget_normalization: str`.
- Produces: `resolve_resource_budget(spec: int | dict, *, q: int, capacity: int, name: str) -> ResourceBudget`.
- Consumes: integer legacy budgets or `{mode, per_qubit, minimum, maximum}` objects.

- [ ] Add failing gate tests showing `support` divides count error by K and `target` divides it by the clipped active target.
- [ ] Add failing policy tests for legacy integers, per-qubit scaling, min/max/cap clipping, invalid negative densities, and deterministic provenance.
- [ ] Run `conda run -n torch-mps python -m unittest tests.test_agp_joint_calibration tests.test_agp_resource_policy -v` and verify the intended failures.
- [ ] Implement both budget denominators with `support` as the backward-compatible default.
- [ ] Implement the pure resource-policy module and persist requested, realized, capacity, and clipping reason.
- [ ] Resolve active, residual, and swap counts once per run before support construction; reject inconsistent capacities.
- [ ] Re-run the focused tests and confirm they pass.

### Task 4: Add Geometry-Stratified Reservoir Allocation

**Files:**
- Create: `scripts/agp_stratified_support.py`
- Modify: `scripts/agp_holdout_feedback.py`
- Test: `tests/test_agp_stratified_support.py`
- Test: `tests/test_agp_support_swap.py`

**Interfaces:**
- Produces: `stratified_ranked_selection(candidates, budget, *, q, locality_quotas, spatial_bins, seed) -> StratifiedSelection`.
- Consumes: existing ranked candidate rows containing `label` and importance score.

- [ ] Add failing tests for exact budget size, no duplicates, deterministic output, minimum locality quotas, spatial-bin coverage, and graceful redistribution from empty strata.
- [ ] Run `conda run -n torch-mps python -m unittest tests.test_agp_stratified_support tests.test_agp_support_swap -v` and confirm the missing selector causes failure.
- [ ] Implement deterministic quota allocation, then fill unused slots by the unchanged global importance ranking.
- [ ] Integrate the selector behind `support.stratification.enabled=false` by default and export per-stratum requested/realized counts.
- [ ] Re-run the focused tests and confirm they pass.

### Task 5: Train And Select The q156 Candidate

**Files:**
- Modify: `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q156/sweep_test/config.json`
- Modify: `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q156/sweep_test/README.md`

**Interfaces:**
- Consumes: reference-normalized residual, target-normalized gate budget, q-aware counts, stratified reservoirs, immutable frozen probes.
- Produces: a separately named q156 candidate run with 20 curriculum rounds plus configured temporal refinements.

- [ ] Create a candidate config that preserves K=32,768 and architecture/schedule settings while resolving the active count at `102.4` terms per qubit and the residual count at `4096` terms per qubit, both subject to measured capacity.
- [ ] Validate the candidate config and print all resolved budgets before training.
- [ ] Run the complete 20-round curriculum and both retained temporal refinements in `torch-mps`.
- [ ] Verify all rounds completed, frozen probes remained immutable, all requested K labels are present, and exported coefficients are finite.
- [ ] Select between training candidates only from frozen projected-probe metrics and record the decision before physical validation.

### Task 6: Canonical Full-K Physical Validation

**Files:**
- Use: `tests/sparse_agp_curriculum/scripts/agp_mps_validation.py`
- Modify: `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q156/sweep_test/RESULTS.md`
- Modify: `docs/CURRENT_SPARSE_AGP_METHODOLOGY.md` only if promoted.

**Interfaces:**
- Consumes: frozen attribution/training champion and all 32,768 coefficient labels.
- Produces: convergence-gated no-CD, nested-l1, and learned-AGP final energy/fidelity metrics.

- [ ] Run the configured full-support TN preflight and verify source identity and action-error gates.
- [ ] Run independent time-step, MPO-bond, and MPS-bond/cutoff resolutions for all canonical protocols.
- [ ] Verify full-K identity, norm drift, operator action error, compression gates, and inter-resolution metric convergence.
- [ ] Compare against retained q156 raw fidelity and energy error; also report fidelity density and energy error per qubit.
- [ ] Run q15 and q20 regression checks with the enabled q-aware modes and reject material regressions.
- [ ] Promote config, results, plots, and methodology only if every design gate passes; otherwise restore the retained benchmark pointers and label the candidate diagnostic.

### Task 7: Repository Verification

**Files:**
- Verify: `models.py`
- Verify: `utils.py`
- Verify: `scripts/`
- Verify: `tests/`

**Interfaces:**
- Produces: reproducible code and documentation with generated artifacts excluded from version control.

- [ ] Run `conda run -n torch-mps python -m py_compile models.py utils.py scripts/projected_sparse_training_common.py scripts/agp_resource_policy.py scripts/agp_stratified_support.py`.
- [ ] Run `conda run -n torch-mps python examples/two_qubit_sparse_demo.py`.
- [ ] Run all focused modules from Tasks 2-4.
- [ ] Run `conda run -n torch-mps python -m unittest discover -s tests`.
- [ ] Run `git diff --check`, inspect `git status --short`, and confirm no generated checkpoints or result directories are staged.
