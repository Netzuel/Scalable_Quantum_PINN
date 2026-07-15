# HCD Connection Summary Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Hamiltonian expressions and provenance-backed ground/PINN energies to `hcd_connection_summary.pdf`, then regenerate q15 and q20.

**Architecture:** Keep discovery and formatting in `scripts/agp_plot_annotations.py`, while the shared plotting function only consumes and renders context lines. Reuse the existing regeneration script so study folders do not acquire Python forks.

**Tech Stack:** Python 3.10, Matplotlib, PyTorch artifact loading, `unittest`, Poppler rendering, `torch-mps` conda environment.

## Global Constraints

- Do not introduce dense Hamiltonian matrices in reusable library code.
- Do not invent missing dynamical metrics; render `not computed` explicitly.
- Preserve all unrelated dirty-worktree changes.
- Execute every Python command through `conda run -n torch-mps`.

---

### Task 1: Context-line contract

**Files:**
- Modify: `tests/test_agp_physical_validation.py`
- Modify: `scripts/agp_plot_annotations.py`

**Interfaces:**
- Produces: `hcd_context_lines_for_images_dir(images_dir: Path) -> list[str]`

- [ ] **Step 1: Write failing tests**

Create temporary q15 and q20 study/run layouts. Assert that q15 emits the
transverse-Ising expressions plus numeric `E_0` and `E_PINN(T)`. Assert that
q20 emits the diagonal-projection and molecular-Pauli expressions, preserves a
configured FCI ground energy, and emits `E_PINN(T)=not computed` without a
compatible dynamical summary.

- [ ] **Step 2: Verify RED**

Run:

```bash
conda run -n torch-mps python -m unittest tests.test_agp_physical_validation.AGPPhysicalValidationTests.test_hcd_context_lines_include_hamiltonians_and_energies
```

Expected: import or assertion failure because the context-line function does
not exist.

- [ ] **Step 3: Implement the formatter**

Add system-aware Hamiltonian expression helpers and an energy formatter. Load
the nearest compatible summary/config and return exactly four lines in the
order specified by the design.

- [ ] **Step 4: Verify GREEN**

Run the focused test command again. Expected: one passing test.

### Task 2: Figure integration

**Files:**
- Modify: `scripts/projected_sparse_training_common.py`
- Modify: `tests/test_sparse_pauli.py`

**Interfaces:**
- Consumes: `hcd_context_lines_for_images_dir(images_dir)`
- Produces: updated `hcd_connection_summary.pdf`

- [ ] **Step 1: Extend the export regression test**

Patch the context-line helper during export and assert that it is consumed by
the connection-summary path.

- [ ] **Step 2: Verify RED**

Run the focused sparse-Pauli export test and confirm it fails against the old
footer integration.

- [ ] **Step 3: Integrate the four-line band**

Replace the old physical footer call with the new context helper, increase the
figure height, and reserve sufficient bottom margin.

- [ ] **Step 4: Verify GREEN**

Run both focused test modules and confirm zero failures.

### Task 3: q20 exact-energy provenance

**Files:**
- Modify: `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test/config.json`
- Modify: `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test/README.md`

**Interfaces:**
- Produces: configured `scalable_validation.ground_energy` with method and
  source metadata.

- [ ] **Step 1: Add a failing config-contract assertion**

Assert the q20 config records `ground_energy=-1.1400734808760409`, method
`PySCF FCI`, and explicitly marks the PINN dynamical energy `not_tested`.

- [ ] **Step 2: Verify RED**

Run the config-contract test and confirm the missing key fails.

- [ ] **Step 3: Add the validated metadata and README note**

Record the H2/cc-pVDZ FCI value and the command/provenance distinction between
exact ground energy and the unperformed full-qubit PINN evolution.

- [ ] **Step 4: Verify GREEN**

Run the config-contract test and confirm it passes.

### Task 4: Regeneration and visual verification

**Files:**
- Regenerate ignored artifacts under `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q15/sweep_test/runs/`
- Regenerate ignored artifacts under `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test/runs/`

- [ ] **Step 1: Run focused and repository tests**

```bash
conda run -n torch-mps python -m py_compile scripts/agp_plot_annotations.py scripts/projected_sparse_training_common.py tests/sparse_agp_curriculum/scripts/agp_regenerate_hcd_summaries.py
conda run -n torch-mps python -m unittest tests.test_agp_physical_validation tests.test_sparse_pauli tests.test_agp_benchmark_layout
```

- [ ] **Step 2: Regenerate both studies**

```bash
conda run -n torch-mps python tests/sparse_agp_curriculum/scripts/agp_regenerate_hcd_summaries.py --root tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q15/sweep_test --require-all
conda run -n torch-mps python tests/sparse_agp_curriculum/scripts/agp_regenerate_hcd_summaries.py --root tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test --require-all
```

- [ ] **Step 3: Render canonical PDFs**

Use `pdftoppm -png -f 1 -singlefile` on each adaptive-refinement summary.

- [ ] **Step 4: Inspect and correct layout**

Confirm the four lines are readable, centered, unclipped, source-correct, and
do not collide with either panel. Repeat regeneration after any layout fix.
