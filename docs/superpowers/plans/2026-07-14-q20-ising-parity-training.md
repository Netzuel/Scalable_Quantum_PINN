# Q20 Ising Parity Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the q20 hydrogen sweep with the accepted q15 transverse-Ising methodology adapted to q=20, K=32,768, requested Q=81,920, and twenty feedback rounds, then complete and verify the full training workflow.

**Architecture:** Treat the retained q15 configuration as the source template and test parity for every methodology-bearing section. Keep the shared fixed-K training engine unchanged unless a failing preparation test proves a q20-specific defect; remove only the obsolete q20 hydrogen validator and generated q20 hydrogen runs. Training uses the shared `agp_holdout_feedback.py` orchestration and records requested versus generated/effective residual budgets.

**Tech Stack:** Python 3, PyTorch/MPS, NumPy, sparse Pauli-coordinate algebra, JSON configurations, `unittest`, Matplotlib.

## Global Constraints

- Run every Python command through `conda run -n torch-mps`; use `--no-capture-output` for training.
- Never construct dense `2**20 x 2**20` Hamiltonian matrices.
- Preserve q15, q156, and all unrelated processes and generated runs.
- Keep K fixed at 32,768 throughout baseline, feedback, and refinement.
- Request Q=81,920; record any generated/effective cap without disguising it as the requested budget.
- Complete twenty feedback rounds plus uniform and adaptive temporal refinement.
- Report every certification gate as `pass`, `fail`, or `not tested`.
- Do not commit generated `runs/`, checkpoints, PDFs, HDF5 files, or notebook checkpoints.

---

### Task 1: Lock the q15-to-q20 methodology contract in tests

**Files:**
- Modify: `tests/test_agp_benchmark_layout.py`
- Test: `tests/test_agp_benchmark_layout.py`

**Interfaces:**
- Consumes: q15 and q20 `config.json` mappings.
- Produces: `test_q20_matches_q15_ising_lineage_for_twenty_rounds`, the regression contract for later tasks.

- [ ] **Step 1: Replace the hydrogen-specific q20 layout assertion with a failing parity test**

```python
def test_q20_matches_q15_ising_lineage_for_twenty_rounds(self):
    q15 = json.loads((ROOT / "tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q15/sweep_test/config.json").read_text())
    q20 = json.loads((ROOT / "tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test/config.json").read_text())

    self.assertEqual(q20["physical"]["parameters"]["system"], "TransverseIsingDriverProblem")
    self.assertEqual(q20["physical"]["parameters"]["num_qubits"], 20)
    self.assertEqual(q20["default_pipeline"]["agp_terms"], 32768)
    self.assertEqual(q20["holdout_feedback"]["base_agp_terms"], 32768)
    self.assertEqual(q20["holdout_feedback"]["holdout_residual_top_k"], 81920)
    self.assertEqual(q20["holdout_feedback"]["iterations"], 20)
    self.assertEqual(q20["holdout_feedback"]["add_residual_terms_per_iteration"], 3072)
    self.assertEqual(q20["holdout_feedback"]["unseen_residual_batches_after_final_iteration"], 1)

    for key in ("neural", "support", "support_sweep", "agp_calibration", "training", "schedule_optimization"):
        self.assertEqual(q20[key], q15[key])
    for key in ("baseline_neural", "pau_transfer_stability", "support_swap", "temporal_refinement", "adaptive_temporal_refinement"):
        self.assertEqual(q20["holdout_feedback"][key], q15["holdout_feedback"][key])
```

- [ ] **Step 2: Run the test and verify the current hydrogen config fails**

Run:

```bash
conda run -n torch-mps python -m unittest tests.test_agp_benchmark_layout.AGPBenchmarkLayoutTests.test_q20_matches_q15_ising_lineage_for_twenty_rounds
```

Expected: failure because q20 currently reports `Hidrogen`, K=16,384, and fifteen rounds.

- [ ] **Step 3: Add physical-validation resource assertions**

```python
validation = q20["physical_validation"]
self.assertEqual(validation["statevector_qubits"], 20)
self.assertEqual(validation["evolution_steps"], 96)
self.assertEqual(validation["learned_top_terms"], 2048)
self.assertLessEqual(validation["learned_action_cache_size"], 16)
self.assertNotIn("scalable_validation", q20)
self.assertNotIn("hydrogen_energy_validation", q20)
```

- [ ] **Step 4: Keep the test red until Task 2 changes the configuration**

Run the same command and confirm it still fails for the intended configuration mismatch.

### Task 2: Replace q20 configuration and documentation with the approved Ising design

**Files:**
- Modify: `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test/config.json`
- Modify: `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test/README.md`
- Test: `tests/test_agp_benchmark_layout.py`

**Interfaces:**
- Consumes: accepted q15 configuration and Task 1 parity assertions.
- Produces: a q20 configuration accepted by `scripts/agp_holdout_feedback.py`.

- [ ] **Step 1: Replace q20 config with the q15 template and exact approved adaptations**

Set these fields after copying the q15 mapping:

```json
{
  "physical": {"parameters": {"system": "TransverseIsingDriverProblem", "num_qubits": 20}},
  "default_pipeline": {"name": "q20_fixed_k_holdout_feedback_physical_validation", "agp_terms": 32768},
  "holdout_feedback": {
    "base_agp_terms": 32768,
    "holdout_residual_top_k": 81920,
    "iterations": 20,
    "add_residual_terms_per_iteration": 3072,
    "unseen_residual_batches_after_final_iteration": 1
  },
  "physical_validation": {
    "statevector_qubits": 20,
    "evolution_steps": 96,
    "learned_top_terms": 2048,
    "learned_top_terms_sweep": [1024, 2048],
    "learned_action_cache_size": 8
  }
}
```

Retain every other q15 methodology value verbatim, including the SiLU baseline, PAU transfer stability, 256-term support swaps, learned schedule, calibration, optimizer, seed, and both refinement stages.

- [ ] **Step 2: Rewrite README around q20 Ising provenance and the 20-round command**

Document K=32,768, requested Q=81,920, effective-Q capping, twenty rounds, expected output layout, training command, and the distinction between run completion and certification.

```bash
conda run --no-capture-output -n torch-mps python scripts/agp_holdout_feedback.py \
  --config tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test/config.json
```

- [ ] **Step 3: Run the parity test and verify it passes**

Run the Task 1 command. Expected: one test passes.

- [ ] **Step 4: Validate JSON and configuration loading**

```bash
conda run -n torch-mps python -c 'import json; from pathlib import Path; p=Path("tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test/config.json"); c=json.loads(p.read_text()); assert c["physical"]["parameters"]["num_qubits"] == 20; assert c["holdout_feedback"]["iterations"] == 20'
```

Expected: exit 0.

### Task 3: Remove obsolete q20 hydrogen-only validation surfaces

**Files:**
- Delete: `tests/sparse_agp_curriculum/scripts/agp_hydrogen_energy_validation.py`
- Modify: `tests/test_agp_physical_validation.py`
- Modify: `tests/test_agp_benchmark_layout.py`
- Verify: `scripts/agp_plot_annotations.py`

**Interfaces:**
- Consumes: q20 Ising config from Task 2.
- Produces: no retained q20 reference to the hydrogen-only validator; generic molecular Hamiltonian support remains intact.

- [ ] **Step 1: Update script-inventory tests to expect no hydrogen-only validator**

Remove `agp_hydrogen_energy_validation.py` from `expected_framework` and the allowed tests-tree path set.

- [ ] **Step 2: Remove validator imports and validator-specific unit tests**

Delete tests for `compile_pauli_terms`, `apply_compiled_pauli_sum`, `evolve_hydrogen_pinn`, and its CLI defaults. Retain generic plot-payload tests for molecular summaries because generic molecular support is outside the q20 cleanup scope.

- [ ] **Step 3: Delete the unreferenced validator**

Delete `tests/sparse_agp_curriculum/scripts/agp_hydrogen_energy_validation.py` with a patch-based deletion.

- [ ] **Step 4: Prove q20 contains no obsolete references while general hydrogen support remains**

```bash
! rg -n 'Hidrogen|hydrogen_energy|pinn_final_energy|scalable_validation' tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test
rg -n 'Hidrogen' tests/test_full_pauli_pinn.py Hamiltonians_to_use/pauli_decompositions/index.json
```

Expected: first command finds nothing; second confirms reusable molecular support remains.

- [ ] **Step 5: Run affected unit tests**

```bash
conda run -n torch-mps python -m unittest tests.test_agp_physical_validation tests.test_agp_benchmark_layout
```

Expected: all affected tests pass.

### Task 4: Clean old q20 runs and verify training preparation

**Files:**
- Remove generated contents: `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test/runs/`
- Verify: `Hamiltonians_to_use/pauli_decompositions/TransverseIsingDriverProblem/20_qubits/distance_1_0.json`
- Verify: `Hamiltonians_to_use/pauli_decompositions/index.json`

**Interfaces:**
- Consumes: cleaned q20 config and shared Hamiltonian loader.
- Produces: an empty q20 run root and a verified sparse Ising Hamiltonian pair load.

- [ ] **Step 1: Verify no active process owns the q20 run tree**

```bash
ps -Ao pid,etime,command | rg 'q20/sweep_test|agp_holdout_feedback.py' | rg -v 'rg '
```

Expected: no active q20 training process. Do not stop the unrelated q156 validation process.

- [ ] **Step 2: Remove only obsolete generated q20 runs**

Remove `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test/runs/`, recreate the empty directory only if the training entrypoint requires it, and confirm q15/q156 run trees are unchanged.

- [ ] **Step 3: Load the q20 Hamiltonian and assert sparse structure**

```bash
conda run -n torch-mps python -c 'from pathlib import Path; from scripts.projected_sparse_training_common import load_pauli_hamiltonian_pair; h0,h1=load_pauli_hamiltonian_pair(Path("Hamiltonians_to_use/pauli_decompositions/index.json"), system="TransverseIsingDriverProblem", n_qubits=20, distance="1_0"); assert h0.n_qubits == h1.n_qubits == 20; assert set().union(*(set(k) for k in h1.terms)) <= {"I","Z"}; print(len(h0.terms), len(h1.terms))'
```

Expected: exit 0 with a sparse transverse driver and diagonal Ising final Hamiltonian.

- [ ] **Step 4: Run compilation and the complete unit suite**

```bash
conda run -n torch-mps python -m py_compile models.py utils.py scripts/agp_holdout_feedback.py scripts/projected_sparse_training_common.py
conda run -n torch-mps python -m unittest discover -s tests
```

Expected: all tests pass before training starts.

### Task 5: Complete twenty feedback rounds and both refinement stages

**Files:**
- Generate (ignored): `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test/runs/baselines/agp_32768/`
- Generate (ignored): `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test/runs/fixed_k_holdout_feedback_trainable_schedule_w96_l4_pau_support_swap_adaptive_temporal_refinement_v1/`

**Interfaces:**
- Consumes: Task 4 prepared q20 configuration.
- Produces: baseline checkpoint, rounds `round_01` through `round_20`, temporal refinement, adaptive temporal refinement, and final holdout-feedback summary.

- [ ] **Step 1: Launch training with unbuffered progress**

```bash
conda run --no-capture-output -n torch-mps python scripts/agp_holdout_feedback.py \
  --config tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test/config.json
```

- [ ] **Step 2: Monitor every milestone and revalidate quiet periods**

For each round, require a `Models_Data/final_agp_coefficients.pt` checkpoint and a corresponding row in the eventual summary. During quiet intervals, inspect process CPU/elapsed time and newest artifact modification time before diagnosing a stall.

- [ ] **Step 3: Verify twenty rounds rather than trusting process exit alone**

```bash
find tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test/runs -type f -path '*/rounds/round_*/Models_Data/final_agp_coefficients.pt' | sort
```

Expected: twenty distinct round checkpoints ending in `round_20`.

- [ ] **Step 4: Verify both post-curriculum continuations**

Require these files under the retained run:

```text
temporal_refinement/Models_Data/final_agp_coefficients.pt
adaptive_temporal_refinement/Models_Data/final_agp_coefficients.pt
```

### Task 6: Audit final artifacts and certification status

**Files:**
- Inspect generated final summary under q20 `Models_Data/`.
- Regenerate generated HCD summaries under q20 `Images/` if necessary.
- Modify: `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test/README.md` only if final effective-Q or status fields need recording.

**Interfaces:**
- Consumes: completed Task 5 outputs.
- Produces: evidence-backed training completion and gate classification.

- [ ] **Step 1: Parse the final summary and assert contract fields**

Use a `torch-mps` Python probe to assert:

```python
assert config["default_pipeline"]["agp_terms"] == 32768
assert summary["residual_budget"]["requested_holdout_residual_top_k"] == 81920
assert summary["residual_budget"]["feedback_iterations"] == 20
assert len(summary["rounds"]) == 20
assert summary["rounds"][-1]["round"] == 20
assert summary["temporal_refinement"]["enabled"] is True
assert summary["adaptive_temporal_refinement"]["enabled"] is True
```

- [ ] **Step 2: Re-read `AGP_CERTIFICATION_CRITERIA.md` and classify every gate**

Record training, holdout, unseen, fixed probes, K/Q plateau, support stability, prune-and-retest, and physical-validation gates as `pass`, `fail`, or `not tested`. Do not equate twenty completed rounds with certification.

- [ ] **Step 3: Regenerate q20 HCD connection summaries**

```bash
conda run -n torch-mps python tests/sparse_agp_curriculum/scripts/agp_regenerate_hcd_summaries.py \
  --root tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test --require-all
```

Expected: zero failures and Ising Hamiltonian annotations.

- [ ] **Step 4: Run final verification**

```bash
conda run -n torch-mps python -m py_compile models.py utils.py scripts/agp_holdout_feedback.py scripts/projected_sparse_training_common.py
conda run -n torch-mps python -m unittest discover -s tests
git diff --check
```

Expected: compilation succeeds, all tests pass, and no whitespace errors are reported.

- [ ] **Step 5: Report exact completion evidence**

Report K, requested/effective Q, completed rounds, refinement completion, final residual metrics, certification-gate table, and artifact paths. Keep the persistent goal active until every Task 5 and Task 6 assertion is proven.
