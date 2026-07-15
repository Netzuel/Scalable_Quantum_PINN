# MPS Dynamical Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a convergence-controlled MPS backend for final-state energy and fidelity, validate it against q15 exact evolution, and run it for q156.

**Architecture:** A framework-level script uses quimb `CircuitMPS` to apply symmetric Pauli rotations generated from the same Hamiltonians, schedules, and learned coefficient exports as the existing statevector validator. Metrics and convergence status are exported in the existing run layout and consumed by the shared physical comparison table.

**Tech Stack:** Python 3.10, NumPy, PyTorch checkpoint loading, quimb 1.11.2, Matplotlib, unittest.

## Global Constraints

- Never construct a dense `2**q x 2**q` Hamiltonian.
- Dense statevectors are test-only and limited to tiny q.
- Use `conda run -n torch-mps python` for every Python command.
- Use the same MPS propagator for no-CD, nested `l=1`, and learned AGP.
- Export pass, fail, and not-tested/converged states explicitly.
- Do not commit generated `runs/` or checkpoint artifacts.

---

### Task 1: Tensor-Network Evolution Primitives

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/sparse_agp_curriculum/scripts/agp_mps_validation.py`
- Create: `tests/test_agp_mps_validation.py`

**Interfaces:**
- Produces: `pauli_rotation_gate(label, angle)`, `evolve_mps_protocol(...)`, and `mps_protocol_metrics(...)`.

- [ ] Write failing tests for one- and two-qubit Pauli rotations, no-CD evolution, all-zero amplitude, and local Ising energy.
- [ ] Run the focused unittest module and confirm missing-interface failures.
- [ ] Add optional `tensor-network = ["quimb==1.11.2"]` dependency.
- [ ] Implement the minimal quimb-backed primitives without large dense objects.
- [ ] Run the focused module and confirm all primitive tests pass.

### Task 2: Configuration and Result Contract

**Files:**
- Modify: `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q15/sweep_test/config.json`
- Modify: `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q156/sweep_test/config.json`
- Modify: `scripts/agp_plot_annotations.py`
- Test: `tests/test_agp_mps_validation.py`

**Interfaces:**
- Consumes: `tensor_network_validation` config block.
- Produces: `mps_physical_validation_summary.json` and convergence rows.

- [ ] Write failing tests for config parsing, convergence classification, and comparison-table MPS fallback.
- [ ] Implement configuration defaults and strict availability/convergence status.
- [ ] Add JSON, CSV, and publication-style PDF exporters.
- [ ] Confirm the comparison table only consumes MPS metrics when they are marked converged.

### Task 3: q15 Calibration

**Files:**
- Generated: q15 retained run `Models_Data/mps_physical_validation_summary.json`
- Generated: q15 retained run `Images/mps_physical_validation_convergence.pdf`

**Interfaces:**
- Consumes: retained q15 adaptive-refinement coefficient export and exact physical summary.
- Produces: per-protocol MPS-versus-statevector energy/fidelity deltas.

- [ ] Run a small convergence grid first to establish runtime and bond growth.
- [ ] Increase bond dimension, steps, and learned-term count to the configured q15 validation point.
- [ ] Require energy and fidelity agreement within explicit tolerances.
- [ ] Stop and diagnose rather than proceeding to q156 if q15 validation fails.

### Task 4: q156 Convergence Study

**Files:**
- Generated: q156 retained run `Models_Data/mps_physical_validation_summary.json`
- Generated: q156 retained run `Images/mps_physical_validation_convergence.pdf`
- Regenerated: q156 `Images/physical_method_comparison_table.pdf`

**Interfaces:**
- Consumes: q156 round-20 learned coefficients, exact `E0=-209.6`, and unique all-zero target.
- Produces: no-CD, nested-l1, and PINN final-state energy/fidelity with convergence status.

- [ ] Run ascending term/bond/step settings and retain every row.
- [ ] Monitor maximum bond dimension, truncation estimate, runtime, and norm.
- [ ] Select the highest converged setting without using ground-truth energy or fidelity for training.
- [ ] Regenerate the canonical physical comparison table.

### Task 5: Verification and Documentation

**Files:**
- Modify: `docs/CURRENT_SPARSE_AGP_METHODOLOGY.md`
- Modify: q15/q156 `README.md` and q156 `RESULTS.md`

- [ ] Run py_compile and the full unittest suite.
- [ ] Render and inspect the MPS convergence and comparison PDFs.
- [ ] Verify generated runs remain gitignored and no process remains active.
- [ ] Report q15 validation deltas and q156 convergence status without overstating certification.
