# Spin-HUBO Sparse-AGP Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import and train the approved q24 spin-HUBO Hamiltonian, then compare no-CD, nested-commutator l=1, and the full learned sparse AGP with convergence-gated MPS dynamics.

**Architecture:** Reusable utilities convert a spin-polynomial JSON snapshot into a sparse Pauli Hamiltonian pair and exactly solve small diagonal objectives by Walsh-Hadamard transform. The existing curriculum consumes the generated pair unchanged, while the MPS validator is generalized from an all-zero Ising target to any non-degenerate computational-basis ground bitstring.

**Tech Stack:** Python 3.10, NumPy, PyTorch, quimb MPS, Matplotlib, JSON, unittest.

## Global Constraints

- Use `conda run -n torch-mps python` for every Python command.
- Do not construct dense `2**q x 2**q` operators in reusable code.
- Keep q20 parity at `K=32768`, `Q=81920`, and 20 curriculum rounds.
- Do not use final-state ground truth in training or hyperparameter selection.
- Canonical MPS validation must deploy all 32,768 learned terms at every resolution.
- Mark every unperformed certification gate `not tested`.
- Keep generated `runs/`, checkpoints, and figures ignored by git.

---

### Task 1: Spin-Polynomial Conversion And Exact Oracle

**Files:**
- Create: `tests/test_spin_hubo_benchmark.py`
- Create: `tests/sparse_agp_curriculum/scripts/spin_hubo_benchmark.py`

**Interfaces:**
- Produces: `load_spin_polynomial(path)`, `spin_polynomial_to_pauli_pair(...)`, `evaluate_spin_energy(...)`, and `exact_walsh_ground_reference(...)`.

- [ ] Write tests for support parsing, spin-to-Pauli conversion, bit ordering, exact energy, degeneracy, and invalid inputs.
- [ ] Run the focused unittest and confirm failures are caused by the missing module.
- [ ] Implement the minimal structured converter and bounded q<=24 exact oracle.
- [ ] Run the focused test module and confirm it passes.

### Task 2: Arbitrary Ground-Bitstring MPS Metrics

**Files:**
- Modify: `tests/test_agp_mps_validation.py`
- Modify: `tests/sparse_agp_curriculum/scripts/agp_mps_validation.py`

**Interfaces:**
- Renames: `diagonal_ising_mps_metrics(...)` to `diagonal_pauli_mps_metrics(...)`, retaining a compatibility alias.
- Produces: energy, ground fidelity, target-relative `Z_i` RMSE, and target-relative nearest-neighbor `Z_i Z_(i+1)` RMSE.

- [ ] Write a failing test using a nonzero product-state ground bitstring and mixed diagonal Z terms.
- [ ] Run the focused test and verify the current hard-coded all-zero observable target fails.
- [ ] Generalize metric targets and reject non-diagonal final terms explicitly.
- [ ] Run all MPS validation tests and confirm they pass.

### Task 3: q24 Scenario

**Files:**
- Create: `tests/sparse_agp_curriculum/transverse_field_spin_hubo/run_002_hamiltonian_341/q24/sweep_test/config.json`
- Create: `tests/sparse_agp_curriculum/transverse_field_spin_hubo/run_002_hamiltonian_341/q24/sweep_test/README.md`
- Create: `tests/sparse_agp_curriculum/transverse_field_spin_hubo/run_002_hamiltonian_341/q24/sweep_test/Hamiltonians_to_use/...`
- Create: `tests/sparse_agp_curriculum/transverse_field_spin_hubo/run_002_hamiltonian_341/q24/sweep_test/ground_reference.json`

**Interfaces:**
- Consumes: the common training and MPS entrypoints.
- Produces: a self-contained source snapshot, Pauli pair, exact reference, provenance hash, and q20-parity configuration.

- [ ] Generate the scenario from the approved source instance.
- [ ] Validate every JSON file and independently re-evaluate the stored ground bitstring energy.
- [ ] Run configuration preflight and support-generation smoke checks without training.

### Task 4: End-To-End Curriculum

**Files:**
- Generated only: q24 `sweep_test/runs/`.

**Interfaces:**
- Produces: the round-20 checkpoint, holdout-feedback summary, uniform temporal refinement, adaptive temporal refinement, and final AGP coefficients.

- [ ] Clean generated q24 artifacts with `scripts/agp_restart.py`.
- [ ] Run `scripts/agp_holdout_feedback.py` with the q24 config.
- [ ] Monitor every round and refinement stage; diagnose non-finite loss, exhausted residual pools, or support-size drift immediately.
- [ ] Verify the final checkpoint contains exactly 32,768 learned labels and the requested effective residual budget is recorded.

### Task 5: Full-Support MPS Comparison

**Files:**
- Generated only: retained checkpoint `mps_validation/`.

**Interfaces:**
- Produces: coarse/fine protocol metrics, convergence assessment, machine-readable JSON/CSV, and `physical_method_comparison_table.pdf`.

- [ ] Run a cost-estimation resolution with all learned terms.
- [ ] Run the configured coarse and fine resolutions for all three protocols.
- [ ] Require identical learned support and grouping across the ladder.
- [ ] Export exact-reference, no-CD, nested-l1, and learned-PINN rows.

### Task 6: Verification And Diagnosis

**Files:**
- Modify: q24 `README.md` with measured results and certification table.

**Interfaces:**
- Consumes: training and MPS summaries.
- Produces: conservative result diagnosis and reproducible retained documentation.

- [ ] Run `py_compile`, the two focused unittest modules, the full unittest suite, and the two-qubit sparse demo.
- [ ] Verify JSON artifacts, learned/evaluated support counts, state norms, convergence deltas, and PDF rendering.
- [ ] Confirm no required process remains active and generated artifacts remain ignored.
- [ ] Classify every AGP certification gate as `pass`, `fail`, or `not tested` without claiming global support sufficiency.
