# Block-Balanced Residual Objective Implementation Plan

> **Archived rejected experiment:** The q156 candidate completed, failed its
> frozen-active projected gate, and produced much worse diagnostic physical
> metrics than the retained benchmark. Its code, configuration, and generated
> artifacts were removed. This file is retained only as a methodological
> record; see the q156 `RESULTS.md` for measured outcomes.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a sparse qubit-block CVaR residual objective and determine whether it improves the certified q156 full-support ground-state fidelity.

**Architecture:** `ProjectedSparseAGPPINN` precomputes sparse residual-term-to-qubit incidence buffers and aggregates normalized per-block residuals with differentiable `scatter_add` and top-k CVaR. Existing configuration parsers propagate opt-in block settings through baseline, curriculum, and refinement stages; all old configurations retain identical defaults. The q156 experiment uses a separately named clean candidate lineage and the existing full-`K` TDVP certification ladder.

**Tech Stack:** Python 3.10, PyTorch in `torch-mps`, sparse Pauli-coordinate algebra, `unittest`, TeNPy two-site TDVP, JSON experiment configuration.

## Global Constraints

- Do not use exact energy, fidelity, or observables during training or projected candidate selection.
- Do not construct dense `2**q x 2**q` matrices or a dense `q x Q` block-incidence matrix.
- Preserve existing behavior unless `residual_objective` is `block_balanced_reference_normalized`.
- Keep q156 `K=32768` fixed during training and deploy every retained term during canonical validation.
- Use `conda run -n torch-mps` for every Python command.
- Preserve the retained q156 benchmark outside the cleaned scenario until promotion is decided.

---

### Task 1: Sparse Block Objective

**Files:**
- Modify: `models.py`
- Test: `tests/test_sparse_pauli.py`

**Interfaces:**
- Produces: `residual_block_incidence(labels, n_qubits) -> tuple[Tensor, Tensor, Tensor, Tensor]`
- Produces: `block_balanced_residual_objective(residual, reference, block_indices, term_indices, shares, covered_mask, tail_fraction, tail_weight, reference_floor) -> tuple[Tensor, dict[str, Tensor]]`
- Extends: `ProjectedSparseLossWeights` with tail fraction, tail weight, and reference floor.

- [ ] Add tests proving equal support sharing, deterministic worst-tail selection, finite zero-reference handling, gradient flow, and equality with a hand-computed two-block example.
- [ ] Run `conda run -n torch-mps python -m unittest tests.test_sparse_pauli` and confirm the new tests fail before implementation.
- [ ] Implement sparse incidence buffers and the block-balanced objective without dense block matrices.
- [ ] Integrate the new objective into `ProjectedSparseAGPPINN.loss()` and export the five block diagnostics.
- [ ] Rerun `tests.test_sparse_pauli` and confirm all tests pass.

### Task 2: Configuration And Curriculum Propagation

**Files:**
- Modify: `scripts/projected_sparse_training_common.py`
- Modify: `scripts/agp_baseline_train.py`
- Modify: `scripts/agp_holdout_feedback.py`
- Modify: `scripts/diagnostics/agp_coupled_curriculum.py`
- Test: `tests/test_agp_joint_calibration.py`
- Test: `tests/test_agp_support_swap.py`

**Interfaces:**
- Extends: `ProjectedRunSettings` with `residual_block_tail_fraction`, `residual_block_tail_weight`, and `residual_block_reference_floor`.
- Accepts: `training.loss.residual_objective = block_balanced_reference_normalized`.

- [ ] Add parser tests for the new objective and numeric range validation, plus a curriculum test proving every training stage receives identical block settings.
- [ ] Run the focused parser/curriculum tests and confirm failure before implementation.
- [ ] Parse and validate the settings in both common and baseline configuration paths.
- [ ] Pass the settings through baseline, feedback, temporal refinement, adaptive refinement, and coupled-curriculum loss construction.
- [ ] Rerun the focused tests and confirm compatibility with absent settings.

### Task 3: Candidate Configuration And Documentation

**Files:**
- Create: `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q156/sweep_test/config_block_balanced_candidate.json`
- Modify: `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q156/sweep_test/README.md`

**Interfaces:**
- Consumes: corrected q-aware v2 settings from `config_q_aware_candidate.json`.
- Produces: isolated `runs/block_balanced_candidate_v1/{baselines,feedback}` lineage.

- [ ] Copy the corrected q-aware configuration and change only the block objective, candidate name, output roots, and physical-validation target placeholders.
- [ ] Set tail fraction `0.15`, tail weight `1.0`, and reference floor `1e-6`.
- [ ] Add README commands for reset, training, projected-gate inspection, coefficient resampling when required, and full-support TDVP validation.
- [ ] Validate JSON parsing and resolved q-aware budgets without training.

### Task 4: Verification Before Long Training

**Files:**
- Test: all affected repository tests

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: a training-ready, backward-compatible implementation.

- [ ] Compile every modified Python module in `torch-mps`.
- [ ] Run `conda run --no-capture-output -n torch-mps python -m unittest tests.test_sparse_pauli tests.test_agp_joint_calibration tests.test_agp_support_swap`.
- [ ] Run `conda run --no-capture-output -n torch-mps python -m unittest discover -s tests`.
- [ ] Run `conda run --no-capture-output -n torch-mps python examples/two_qubit_sparse_demo.py` and confirm finite decreasing loss.
- [ ] Run `git diff --check`.

### Task 5: Clean q156 Candidate Training

**Files:**
- Move generated: `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q156/sweep_test/runs/`
- Generate: candidate baseline, rounds 1-20, and both temporal refinements.

**Interfaces:**
- Consumes: `config_block_balanced_candidate.json`.
- Produces: immutable fixed probes, checkpoints, coefficient exports, and projected-gate summary.

- [ ] Move the retained q156 `runs/` tree to ignored `tmp/q156_retained_before_block_balanced_20260720/` and record byte/file counts.
- [ ] Reset the q156 scenario without creating top-level `Images/` or `Models_Data/` directories.
- [ ] Run the candidate feedback pipeline to completion with all 20 rounds and refinements.
- [ ] Verify immutable probe lifecycle, finite coefficients, exact `K=32768`, and complete round history.
- [ ] Reject before physical evolution if holdout or fixed-active unseen gates fail.

### Task 6: Full-Support Physical Comparison

**Files:**
- Generate: full-support resampled export if the selected checkpoint cadence is insufficient.
- Generate: full-support TDVP preflight and independent timestep/state convergence summaries.
- Modify on promotion: q156 `config.json`, `RESULTS.md`, `README.md`, and common methodology.

**Interfaces:**
- Consumes: the projected-gate-selected candidate and every one of its 32,768 terms.
- Produces: matched no-CD, nested-l1, and PINN energy/fidelity table.

- [ ] Resolve the accepted training champion exclusively from frozen projected metrics.
- [ ] Export a provenance-complete 257-point coefficient tensor without optimizer steps when needed.
- [ ] Run the full-source MPO preflight and require source/hash, coefficient, action, Hermiticity, and workspace gates to pass.
- [ ] Run the 24/48-step and bond-32/64 TDVP ladder with all 32,768 terms.
- [ ] Compare fidelity against `0.2591562697` and energy error against `7.7486769` using the promotion margins in the design.
- [ ] Promote and regenerate plots only on a pass; otherwise restore the archived retained q156 run and record the candidate as rejected.

### Task 7: Final Verification And Diagnosis

**Files:**
- Modify: `docs/CURRENT_SPARSE_AGP_METHODOLOGY.md` only if promoted or to record a rejected tested variant.
- Modify: q156 `RESULTS.md` with the final certified comparison.

**Interfaces:**
- Produces: final methodology status and reproducible evidence paths.

- [ ] Regenerate `hcd_connection_summary.pdf` and `physical_method_comparison_table.pdf` for the retained result.
- [ ] Visually inspect the PDFs and verify their numbers against the canonical JSON summary.
- [ ] Rerun focused tests, full tests, compilation, and `git diff --check`.
- [ ] Report whether fidelity improved, the numerical convergence evidence, and whether the candidate became the retained benchmark.
