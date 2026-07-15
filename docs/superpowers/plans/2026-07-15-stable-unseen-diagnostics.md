# Stable Unseen Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add immutable active/null unseen probes whose metrics remain scientifically defined through every fixed-K feedback round.

**Architecture:** A small pure helper module owns probe configuration, deterministic stratification, and metric calculation. The existing holdout-feedback runner generates and persists the probes before round 1, evaluates them against every checkpoint without exposing them to training, and adds separate fixed-probe curves while retaining the moving holdout diagnostics.

**Tech Stack:** Python 3.10, PyTorch, NumPy, sparse Pauli-coordinate models, JSON, Matplotlib, `unittest`.

## Global Constraints

- Use `conda run -n torch-mps` for every Python command.
- Keep `probe_gate`, `probe_watch`, and `probe_test` unchanged.
- Fixed unseen labels must be disjoint from every training, feedback-candidate, and certification-probe label.
- Never define a relative residual by adding epsilon to a zero reference.
- Do not change the PINN loss, architecture, fixed-K support swaps, schedule, optimizer, or checkpoint lineage.
- Do not commit generated `runs/`, checkpoints, or rendered result artifacts.

---

### Task 1: Pure Fixed-Probe Selection And Metrics

**Files:**
- Create: `scripts/agp_residual_probes.py`
- Create: `tests/test_agp_residual_probes.py`

**Interfaces:**
- Consumes: ordered candidate labels, per-label AGP=0 RMS values, excluded labels, and residual/reference tensors.
- Produces: `FixedUnseenProbeConfig`, `select_fixed_unseen_probes(...)`, and `fixed_unseen_metrics(...)`.

- [ ] **Step 1: Write failing tests for deterministic, disjoint stratification**

```python
def test_fixed_unseen_selection_is_disjoint_and_deterministic(self):
    labels = ["XIII", "YIII", "ZIII", "IXII", "IYII", "IZII"]
    rms = np.asarray([2.0, 0.0, 1.0, 1.0e-15, 3.0, 0.0])
    config = FixedUnseenProbeConfig(
        enabled=True,
        active_terms=2,
        null_terms=2,
        reference_rms_threshold=1.0e-12,
        seed=11,
        candidate_multiplier=4,
    )

    first = select_fixed_unseen_probes(labels, rms, excluded_labels={"XIII", "IZII"}, config=config)
    second = select_fixed_unseen_probes(labels, rms, excluded_labels={"XIII", "IZII"}, config=config)

    self.assertEqual(first, second)
    self.assertEqual(set(first["active_labels"]), {"ZIII", "IYII"})
    self.assertEqual(set(first["null_labels"]), {"YIII", "IXII"})
    self.assertFalse(set(first["active_labels"]) & set(first["null_labels"]))
```

- [ ] **Step 2: Write failing tests for active ratios and null leakage**

```python
def test_fixed_unseen_metrics_separate_active_ratio_from_null_leakage(self):
    residual = torch.tensor([[2.0, 1.0, 3.0, 4.0]])
    reference = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    metrics = fixed_unseen_metrics(
        residual=residual,
        reference=reference,
        active_indices=[0, 1],
        null_indices=[2, 3],
        reference_floor=1.0e-12,
    )

    self.assertAlmostEqual(metrics["active_relative"], 1.0)
    self.assertEqual(metrics["active_status"]["reason"], "finite_reference")
    self.assertAlmostEqual(metrics["null_absolute_per_term"], 12.5)
    self.assertAlmostEqual(metrics["null_scaled"], 5.0)


def test_fixed_unseen_metrics_never_invents_zero_reference_ratio(self):
    metrics = fixed_unseen_metrics(
        residual=torch.ones((1, 2)),
        reference=torch.zeros((1, 2)),
        active_indices=[],
        null_indices=[0, 1],
        reference_floor=1.0e-12,
    )
    self.assertIsNone(metrics["active_relative"])
    self.assertEqual(metrics["active_status"]["reason"], "empty_subset")
    self.assertTrue(np.isfinite(metrics["null_absolute_per_term"]))
    self.assertIsNone(metrics["null_scaled"])
```

- [ ] **Step 3: Run the tests and confirm the module is missing**

Run: `conda run -n torch-mps python -m unittest tests.test_agp_residual_probes -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'agp_residual_probes'`.

- [ ] **Step 4: Implement the pure helper module**

```python
@dataclass(frozen=True)
class FixedUnseenProbeConfig:
    enabled: bool = False
    active_terms: int = 0
    null_terms: int = 0
    reference_rms_threshold: float = 1.0e-12
    seed: int = 0
    candidate_multiplier: int = 4


def select_fixed_unseen_probes(
    labels: Sequence[str],
    reference_rms: np.ndarray,
    *,
    excluded_labels: Collection[str],
    config: FixedUnseenProbeConfig,
) -> dict[str, object]:
    if len(labels) != len(reference_rms):
        raise ValueError("labels and reference_rms must have equal length")
    excluded = set(excluded_labels)
    rows = [(str(label), float(value)) for label, value in zip(labels, reference_rms, strict=True) if label not in excluded]
    active = sorted((row for row in rows if row[1] > config.reference_rms_threshold), key=lambda row: (-row[1], row[0]))
    null = sorted((row for row in rows if row[1] <= config.reference_rms_threshold), key=lambda row: (row[1], row[0]))
    active = active[: config.active_terms]
    null = null[: config.null_terms]
    return {
        "active_labels": [label for label, _ in active],
        "null_labels": [label for label, _ in null],
        "active_reference_rms": [value for _, value in active],
        "null_reference_rms": [value for _, value in null],
        "requested_active_terms": config.active_terms,
        "requested_null_terms": config.null_terms,
        "status": "complete" if len(active) == config.active_terms and len(null) == config.null_terms else "insufficient_candidates",
    }
```

Implement `fixed_unseen_metrics(...)` with `norm_sq_subset(...)`; return `None` plus an explicit status for empty/zero-reference active subsets, and return finite per-term null leakage without an epsilon quotient.

- [ ] **Step 5: Run the focused tests**

Run: `conda run -n torch-mps python -m unittest tests.test_agp_residual_probes -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit the pure component**

```bash
git add scripts/agp_residual_probes.py tests/test_agp_residual_probes.py
git commit -m "Add stable unseen residual probes"
```

### Task 2: Curriculum Initialization, Persistence, And Evaluation

**Files:**
- Modify: `scripts/agp_holdout_feedback.py`
- Modify: `tests/test_agp_support_swap.py`

**Interfaces:**
- Consumes: `holdout_feedback.fixed_unseen_probes` and the ordered residual candidate universe.
- Produces: `Models_Data/fixed_unseen_probe_labels.json` and fixed-probe fields on every feedback row.

- [ ] **Step 1: Write failing tests for config parsing and resume identity**

```python
def test_fixed_unseen_settings_are_read_from_feedback_config(self):
    settings = fixed_unseen_probe_settings_from_feedback({
        "fixed_unseen_probes": {
            "enabled": True,
            "active_terms": 4096,
            "null_terms": 4096,
            "reference_rms_threshold": 1.0e-12,
            "seed": 11,
            "candidate_multiplier": 8,
        }
    })
    self.assertTrue(settings.enabled)
    self.assertEqual(settings.active_terms, 4096)
    self.assertEqual(settings.null_terms, 4096)


def test_persisted_fixed_probe_rejects_changed_labels(self):
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "fixed_unseen_probe_labels.json"
        save_fixed_unseen_probe(path, {"active_labels": ["XI"], "null_labels": ["YI"]})
        with self.assertRaisesRegex(ValueError, "immutable fixed unseen probe"):
            load_or_validate_fixed_unseen_probe(path, expected_excluded_labels={"XI"})
```

- [ ] **Step 2: Run the tests and verify the new interfaces are absent**

Run: `conda run -n torch-mps python -m unittest tests.test_agp_support_swap -v`

Expected: FAIL on missing fixed-probe helpers.

- [ ] **Step 3: Add initialization and persistence before round 1**

Implement these functions in `scripts/agp_holdout_feedback.py`:

```python
def fixed_unseen_probe_settings_from_feedback(feedback: Mapping[str, object]) -> FixedUnseenProbeConfig: ...

def save_fixed_unseen_probe(path: Path, payload: Mapping[str, object]) -> None: ...

def load_or_validate_fixed_unseen_probe(
    path: Path,
    *,
    expected_excluded_labels: Collection[str],
) -> dict[str, object]: ...

def build_fixed_unseen_probe(
    *,
    candidate_labels: list[str],
    excluded_labels: set[str],
    reference_rms: np.ndarray,
    settings: FixedUnseenProbeConfig,
) -> dict[str, object]: ...

def evaluate_fixed_unseen_probe(
    *,
    run_dir: Path,
    config_payload: dict[str, object],
    probe_metadata: Mapping[str, object],
    intermediate_top_k: int,
    device: torch.device,
) -> dict[str, object]: ...
```

Request `Q + candidate_multiplier * (active_terms + null_terms)` residual labels initially. Reserve the first `Q` labels for the moving holdout/feedback universe and select fixed probes only from the tail. If either partition is short, double the generated candidate request deterministically up to the configured generator/resource cap; stop only when both budgets are filled or persist `insufficient_candidates` with realized counts. Exclude the baseline residual labels and the union of every certification-probe label. Persist the selected labels and reference RMS values before evaluating round 0.

- [ ] **Step 4: Evaluate the immutable labels for every checkpoint**

Merge these keys into every baseline, curriculum, temporal-refinement, and adaptive-refinement row:

```python
row.update({
    "fixed_unseen_active_terms": probe["active_terms"],
    "fixed_unseen_active_residual": probe["active_residual"],
    "fixed_unseen_active_reference_residual": probe["active_reference_residual"],
    "fixed_unseen_active_relative": probe["active_relative"],
    "fixed_unseen_active_status": probe["active_status"],
    "fixed_unseen_null_terms": probe["null_terms"],
    "fixed_unseen_null_absolute_per_term": probe["null_absolute_per_term"],
    "fixed_unseen_null_scaled": probe["null_scaled"],
})
```

Assert before each evaluation that neither fixed partition intersects the checkpoint's training residual labels.

- [ ] **Step 5: Run curriculum-focused tests**

Run: `conda run -n torch-mps python -m unittest tests.test_agp_support_swap tests.test_agp_benchmark_layout -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit curriculum integration**

```bash
git add scripts/agp_holdout_feedback.py tests/test_agp_support_swap.py
git commit -m "Persist unseen probes across feedback rounds"
```

### Task 3: Reporting, Plotting, And Certification Semantics

**Files:**
- Modify: `scripts/agp_holdout_feedback.py`
- Modify: `scripts/agp_holdout_study.py`
- Modify: `tests/test_agp_residual_probes.py`
- Modify: `tests/test_agp_physical_validation.py`

**Interfaces:**
- Consumes: fixed-probe fields on feedback rows.
- Produces: `holdout_feedback_fixed_unseen_probes.pdf` and an explicit fixed-unseen certification gate.

- [ ] **Step 1: Write failing tests for the decision and plot data**

```python
def test_fixed_active_probe_is_the_stable_unseen_gate(self):
    row = {
        "holdout_relative_residual": 0.05,
        "unseen_relative_residual": None,
        "fixed_unseen_active_relative": 0.8,
        "fixed_unseen_active_status": {"valid": True, "reason": "finite_reference"},
    }
    decision = feedback_threshold_decision([row], holdout_threshold=0.1, unseen_threshold=1.0)
    self.assertEqual(decision["status"], "found_feedback_round")
    self.assertEqual(decision["unseen_gate_source"], "fixed_unseen_active")


def test_plot_series_keeps_null_leakage_when_moving_ratio_is_undefined(self):
    series = fixed_unseen_plot_series([{
        "feedback_round": 7,
        "fixed_unseen_active_relative": 0.4,
        "fixed_unseen_null_scaled": 0.02,
    }])
    self.assertEqual(series["active_relative"].tolist(), [0.4])
    self.assertEqual(series["null_scaled"].tolist(), [0.02])
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `conda run -n torch-mps python -m unittest tests.test_agp_residual_probes tests.test_agp_physical_validation -v`

Expected: FAIL on missing decision/plot helpers.

- [ ] **Step 3: Implement explicit reporting semantics**

Keep the current moving-unseen fields for diagnosis. Use the fixed active quotient as the stable unseen certification value when enabled and complete. If the active pool is unavailable, emit `not_tested` with its reason; never fall back to an epsilon ratio.

Implement the reporting interfaces used by the tests:

```python
def feedback_threshold_decision(
    rows: Sequence[Mapping[str, object]],
    *,
    holdout_threshold: float,
    unseen_threshold: float,
) -> dict[str, object]: ...

def fixed_unseen_plot_series(rows: Sequence[Mapping[str, object]]) -> dict[str, np.ndarray]: ...
```

Add a two-panel publication plot:

```python
axes[0].semilogy(rounds, active_relative, marker="o", label="fixed active quotient")
axes[0].axhline(thresholds.unseen, linestyle=":", color="0.5")
axes[1].semilogy(rounds, null_absolute_per_term, marker="s", label="null absolute / term")
axes[1].semilogy(rounds, null_scaled, marker="^", label="null scaled")
```

Do not connect undefined values. Label the old series `moving unseen quotient` and include its status reason in JSON.

- [ ] **Step 4: Run focused plotting and summary tests**

Run: `conda run -n torch-mps python -m unittest tests.test_agp_residual_probes tests.test_agp_physical_validation -v`

Expected: all tests PASS and temporary PDFs are nonempty.

- [ ] **Step 5: Commit reporting changes**

```bash
git add scripts/agp_holdout_feedback.py scripts/agp_holdout_study.py tests/test_agp_residual_probes.py tests/test_agp_physical_validation.py
git commit -m "Report fixed active and null unseen diagnostics"
```

### Task 4: General Configuration And Documentation

**Files:**
- Modify: `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q15/sweep_test/config.json`
- Modify: `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test/config.json`
- Modify: `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q156/sweep_test/config.json`
- Modify: `tests/sparse_agp_curriculum/transverse_field_spin_hubo/run_002_hamiltonian_341/q24/sweep_test/config.json`
- Modify: `docs/CURRENT_SPARSE_AGP_METHODOLOGY.md`
- Modify: `AGP_CERTIFICATION_CRITERIA.md`
- Modify: `tests/test_agp_benchmark_layout.py`

**Interfaces:**
- Consumes: the new general config block.
- Produces: reproducible future-run defaults without reclassifying historical artifacts.

- [ ] **Step 1: Add layout tests for all retained scenarios**

```python
for payload in (q15, q20, q156, q24):
    probes = payload["holdout_feedback"]["fixed_unseen_probes"]
    self.assertTrue(probes["enabled"])
    self.assertGreater(probes["active_terms"], 0)
    self.assertGreater(probes["null_terms"], 0)
    self.assertGreaterEqual(probes["candidate_multiplier"], 1)
```

- [ ] **Step 2: Add the general configuration**

```json
"fixed_unseen_probes": {
  "enabled": true,
  "active_terms": 4096,
  "null_terms": 4096,
  "reference_rms_threshold": 1e-12,
  "seed": 11,
  "candidate_multiplier": 8
}
```

For small full-basis cases, cap requested probe sizes to the available disjoint labels and preserve an explicit `insufficient_candidates` status. Do not claim that existing completed runs contain these probes.

- [ ] **Step 3: Document the active/null interpretation**

Update the methodology and certification documents to distinguish:

```text
moving unseen quotient: curriculum diagnostic, possibly undefined
fixed active quotient: stable relative unseen gate
fixed null leakage: absolute AGP-induced residual in zero-reference directions
```

- [ ] **Step 4: Run layout and compile checks**

Run: `conda run -n torch-mps python -m unittest tests.test_agp_benchmark_layout -v`

Run: `conda run -n torch-mps python -m py_compile scripts/agp_residual_probes.py scripts/agp_holdout_feedback.py scripts/agp_holdout_study.py`

Expected: all commands PASS.

- [ ] **Step 5: Commit config and documentation**

```bash
git add AGP_CERTIFICATION_CRITERIA.md docs/CURRENT_SPARSE_AGP_METHODOLOGY.md tests/sparse_agp_curriculum/*/*/sweep_test/config.json tests/sparse_agp_curriculum/transverse_field_spin_hubo/run_002_hamiltonian_341/q24/sweep_test/config.json tests/test_agp_benchmark_layout.py
git commit -m "Configure stable unseen diagnostics"
```

### Task 5: Backfill The Completed q24 Checkpoints Without Retraining

**Files:**
- Modify only if needed: `scripts/agp_holdout_feedback.py`
- Generated only: q24 run summaries and PDFs under ignored `runs/` paths.

**Interfaces:**
- Consumes: existing baseline, round 1-20, temporal, and adaptive checkpoints.
- Produces: one immutable q24 probe manifest and metrics for every available checkpoint.

- [ ] **Step 1: Add a diagnostics-only CLI mode**

Add `--refresh-fixed-unseen-only`. It must load existing checkpoints, build or validate the persisted probe manifest, update summaries/plots, and refuse to train or overwrite checkpoints.

- [ ] **Step 2: Run the q24 diagnostics refresh**

Run:

```bash
conda run --no-capture-output -n torch-mps python scripts/agp_holdout_feedback.py \
  --config tests/sparse_agp_curriculum/transverse_field_spin_hubo/run_002_hamiltonian_341/q24/sweep_test/config.json \
  --refresh-fixed-unseen-only
```

Expected: every available feedback round has finite fixed active/null metrics; any missing pool is explicitly `insufficient_candidates`, not blank.

- [ ] **Step 3: Verify probe immutability and artifacts**

Run a `conda run -n torch-mps python -c` check that loads all row metrics, verifies identical probe-manifest hashes, verifies no fixed label occurs in any training residual support, and confirms the PDF is nonempty.

- [ ] **Step 4: Run the full focused suite**

Run: `conda run -n torch-mps python -m unittest tests.test_agp_residual_probes tests.test_agp_support_swap tests.test_agp_benchmark_layout tests.test_agp_physical_validation -v`

Expected: all tests PASS.
