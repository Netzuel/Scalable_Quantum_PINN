# Normalized Variational-Action Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a normalized variational AGP action to the conventional sparse PINN and execute the gated q20, q15, q25 fixed-duration benchmark sequence.

**Architecture:** `models.py` computes the action from the generator already built for the Euler-Lagrange residual. Existing settings/config loaders propagate one new scalar weight through every training stage. Candidate study configs remain isolated from retained runs and use the existing all-support physical validators.

**Tech Stack:** Python, PyTorch, sparse Pauli coordinates, `unittest`, exact statevector evolution, TeNPy TDVP/MPO, `torch-mps`.

## Global Constraints

- Use normalized time and fixed physical duration `T=1`.
- Do not use ground truth in training or checkpoint selection.
- Do not use cross-system checkpoints or parameters.
- Canonical q20/q25 tensor-network validation deploys all learned `K` terms.
- Stop immediately when a required `F >= 0.95` gate fails.
- Preserve unrelated dirty-worktree changes and retained benchmark artifacts.

---

### Task 1: Variational-Action Loss

**Files:**
- Modify: `models.py`
- Test: `tests/test_sparse_pauli.py`

**Interfaces:**
- Consumes: `generator` and `reference_generator` tensors already built by `ProjectedSparseAGPPINN.loss`.
- Produces: `ProjectedSparseLossWeights.variational_action: float` and diagnostics `variational_action`, `reference_variational_action`, and `relative_variational_action`.

- [ ] Write a failing test that manually evaluates both generator norms on a two-level sparse Pauli model and checks the reported relative action.
- [ ] Run `conda run -n torch-mps python -m unittest tests.test_sparse_pauli.TestProjectedSparsePINN.test_variational_action_matches_generator_norm`.
- [ ] Add the weight and normalized action term without changing behavior when the weight is zero.
- [ ] Rerun the focused test and existing projected sparse loss tests.

### Task 2: Configuration Propagation

**Files:**
- Modify: `scripts/projected_sparse_training_common.py`
- Modify: `scripts/agp_baseline_train.py`
- Modify: `scripts/agp_holdout_feedback.py`
- Test: `tests/test_agp_support_swap.py`

**Interfaces:**
- Consumes: `training.loss.variational_action`.
- Produces: `ProjectedRunSettings.variational_action_weight` supplied to every `ProjectedSparseLossWeights` construction.

- [ ] Write failing configuration tests for a nonzero action weight and zero default.
- [ ] Run the focused configuration tests and confirm the missing propagation.
- [ ] Add the settings field, parser, and constructor arguments in baseline, feedback, temporal, and adaptive paths.
- [ ] Rerun focused tests and compile all modified modules.

### Task 3: Isolated Gated Study

**Files:**
- Modify: `scripts/agp_size_intensive_study.py`
- Modify: `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/size_intensive_pinn_study.json`
- Modify: `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test/size_intensive_pinn/config.json`
- Modify only after q20 passes: corresponding q15 and q25 candidate configs.
- Test: `tests/test_agp_benchmark_layout.py`

**Interfaces:**
- Consumes: candidate loss weight and physical summaries.
- Produces: isolated candidate roots and fail-fast gate status.

- [ ] Write failing tests requiring `T=1`, a positive action weight, isolated output roots, and q20-first execution.
- [ ] Add the candidate configuration and explicit fidelity gate.
- [ ] Clean only the q20 candidate root, train from scratch, and run all-K q20 TN validation.
- [ ] Stop and diagnose if q20 fidelity is below `0.95`.
- [ ] Only after q20 passes, clean/train/validate q15 independently.
- [ ] Only after q15 passes, clean/train/validate q25 independently.

### Task 4: Verification And Documentation

**Files:**
- Modify after results: `docs/superpowers/specs/2026-07-23-normalized-variational-action-loss-design.md`
- Modify after promotion only: `docs/CURRENT_SPARSE_AGP_METHODOLOGY.md`

**Interfaces:**
- Consumes: completed physical summaries and certification records.
- Produces: retained or rejected methodology diagnosis.

- [ ] Run focused unit tests and `python -m py_compile` in `torch-mps`.
- [ ] Verify no learned-support truncation and no duration change occurred.
- [ ] Record q20/q15/q25 metrics and stop reason.
- [ ] Update the current methodology only if every required gate passes.
