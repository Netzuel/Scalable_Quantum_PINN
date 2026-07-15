# Compressed-MPO Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the q24 full-support product-formula bottleneck with a controlled, evaluation-only compressed-MPO evolution backend that includes all learned AGP terms.

**Architecture:** Pure NumPy utilities factor the learned direct-CD coefficient matrix in time and choose a deterministic qubit order. TeNPy builds exact finite-state-machine MPOs from every Pauli term; a Hilbert-Schmidt SVD sweep compresses those MPOs with measured discarded weight before a finite-MPS two-site TDVP evolution. The existing validation CLI dispatches to this backend and preserves the current result/table schema.

**Tech Stack:** Python 3.10, NumPy, PyTorch checkpoint loading, `physics-tenpy==1.1.0`, quimb 1.11.2 reference backend, Matplotlib, JSON, `unittest`.

## Global Constraints

- Use `conda run -n torch-mps` for every Python command.
- TeNPy remains optional; training and residual-only workflows must import without it.
- Every learned Pauli label enters the exact MPO builder before compression.
- No coefficient-ranked or top-term truncation may certify the learned method.
- Never construct a dense `2**q x 2**q` operator outside tiny tests.
- Report temporal, static-MPO, dynamic-MPO, MPS, and timestep errors separately.
- A failed resource cap is `not_feasible`; an incomplete ladder is `not_tested`.
- Preserve the existing quimb evaluator as a reference and ablation backend.
- Do not commit generated `runs/`, checkpoints, or rendered result artifacts.

---

### Task 1: Optional Dependency, Temporal Factorization, And Qubit Ordering

**Files:**
- Modify: `pyproject.toml`
- Create: `scripts/agp_mpo_backend.py`
- Create: `tests/test_agp_mpo_backend.py`

**Interfaces:**
- Consumes: full learned labels, sampled direct-CD coefficients, Hamiltonian terms, and candidate order names.
- Produces: `TemporalFactorization`, `factor_direct_cd_coefficients(...)`, `permute_pauli_label(...)`, and `select_qubit_order(...)`.

- [ ] **Step 1: Add the pinned optional dependency**

```toml
tensor-network = [
  "quimb==1.11.2",
  "physics-tenpy==1.1.0"
]
```

- [ ] **Step 2: Write failing factorization and ordering tests**

```python
def test_temporal_factorization_uses_all_terms_and_meets_norm_target(self):
    tau = np.linspace(0.0, 1.0, 9)
    factors = np.stack([np.sin(np.pi * tau), np.sin(2.0 * np.pi * tau)], axis=1)
    modes = np.asarray([[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 0.25, 2.0]])
    direct = factors @ modes
    result = factor_direct_cd_coefficients(tau, direct, retained_norm=0.999999)
    self.assertEqual(result.rank, 2)
    self.assertEqual(result.static_modes.shape, (2, 4))
    self.assertGreaterEqual(result.retained_norm_fraction, 0.999999)
    np.testing.assert_allclose(result.reconstruct(), direct, atol=1.0e-11)


def test_qubit_order_is_deterministic_and_permutation_safe(self):
    terms = [("XIIX", 4.0), ("IXXI", 1.0), ("ZIZI", 2.0)]
    first = select_qubit_order(terms, n_qubits=4, candidates=("native", "spectral"))
    second = select_qubit_order(terms, n_qubits=4, candidates=("native", "spectral"))
    self.assertEqual(first.order, second.order)
    self.assertEqual(sorted(first.order), [0, 1, 2, 3])
    self.assertEqual(unpermute_pauli_label(permute_pauli_label("XYZI", first.order), first.order), "XYZI")
```

- [ ] **Step 3: Run the tests and verify failure**

Run: `conda run -n torch-mps python -m unittest tests.test_agp_mpo_backend -v`

Expected: FAIL with missing `agp_mpo_backend`.

- [ ] **Step 4: Implement temporal SVD and ordering**

```python
@dataclass(frozen=True)
class TemporalFactorization:
    tau: np.ndarray
    temporal_factors: np.ndarray
    static_modes: np.ndarray
    singular_values: np.ndarray
    rank: int
    retained_norm_fraction: float
    max_abs_error: float
    endpoint_max_abs_error: float

    def reconstruct(self) -> np.ndarray:
        result = self.temporal_factors @ self.static_modes
        result[0] = 0.0
        result[-1] = 0.0
        return result
```

Choose the smallest rank whose cumulative squared singular values meet `retained_norm`. Require finite arrays and zero the reconstructed endpoint direct-CD vectors. Build a weighted interaction graph from every non-identity support and score native/reversed/spectral candidates by maximum cut count, then mean cut count, then candidate name.

Represent the ordering result explicitly:

```python
@dataclass(frozen=True)
class QubitOrderSelection:
    order: tuple[int, ...]
    candidate: str
    max_cut_terms: int
    mean_cut_terms: float
```

- [ ] **Step 5: Run focused tests and compile**

Run: `conda run -n torch-mps python -m unittest tests.test_agp_mpo_backend -v`

Run: `conda run -n torch-mps python -m py_compile scripts/agp_mpo_backend.py`

Expected: all commands PASS.

- [ ] **Step 6: Commit factorization utilities**

```bash
git add pyproject.toml scripts/agp_mpo_backend.py tests/test_agp_mpo_backend.py
git commit -m "Add temporal factorization for MPO evaluation"
```

### Task 2: Exact Full-Support MPO Construction And Controlled Compression

**Files:**
- Modify: `scripts/agp_mpo_backend.py`
- Modify: `tests/test_agp_mpo_backend.py`

**Interfaces:**
- Consumes: `Sequence[tuple[str, complex]]`, qubit order, MPO bond/cutoff limits.
- Produces: `build_exact_pauli_mpo(...)`, `compress_mpo_hilbert_schmidt(...)`, and `probe_mpo_compression(...)`.
- Test-only helpers: `dense_pauli_sum(...)` and `mpo_to_dense(...)`, both restricted to `q <= 4`.

- [ ] **Step 1: Write failing dense-equivalence tests at q <= 4**

```python
def test_exact_pauli_mpo_contains_every_input_term(self):
    terms = [("XI", 0.3), ("YZ", -0.7), ("ZZ", 0.2)]
    mpo, metadata = build_exact_pauli_mpo(terms, n_qubits=2, order=(0, 1))
    np.testing.assert_allclose(mpo_to_dense(mpo), dense_pauli_sum(terms), atol=1.0e-12)
    self.assertEqual(metadata["input_terms"], 3)
    self.assertEqual(metadata["dropped_terms"], 0)


def test_compressed_mpo_reports_and_respects_operator_error(self):
    terms = [("XII", 0.3), ("IYZ", -0.7), ("ZZI", 0.2), ("XYZ", 0.1)]
    exact, _ = build_exact_pauli_mpo(terms, n_qubits=3, order=(0, 1, 2))
    compressed, diagnostics = compress_mpo_hilbert_schmidt(exact, max_bond=16, cutoff=1.0e-13)
    relative = np.linalg.norm(mpo_to_dense(exact) - mpo_to_dense(compressed)) / np.linalg.norm(mpo_to_dense(exact))
    self.assertLess(relative, 1.0e-11)
    self.assertGreaterEqual(diagnostics["discarded_weight"], 0.0)
    self.assertEqual(diagnostics["post_bonds"], list(compressed.chi))
```

- [ ] **Step 2: Run the tests and verify missing interfaces**

Run: `conda run -n torch-mps python -m unittest tests.test_agp_mpo_backend -v`

Expected: FAIL on missing MPO helpers.

- [ ] **Step 3: Build exact sparse MPOs with TeNPy's finite-state graph**

Use a charge-free `SpinHalfSite`, add exact Pauli operators named `X`, `Y`, and `Z`, convert every non-identity label to a `TermList` entry, then call:

```python
term_list = TermList(operator_terms, strengths)
graph = MPOGraph.from_term_list(term_list, sites, bc="finite", insert_all_id=True)
mpo = graph.build_MPO()
```

Reject duplicate labels only after summing their coefficients exactly. Drop only exact numerical zeros at the configured arithmetic-zero tolerance and report their labels/count.

- [ ] **Step 4: Implement a Hilbert-Schmidt SVD sweep**

Treat each MPO tensor as an MPS tensor with local dimension four. Sweep left-to-right, SVD each `(left_bond * 4, right_bond)` matrix, keep singular values according to `max_bond` and the relative squared-weight cutoff, and absorb `S @ Vh` into the next tensor. Return per-bond discarded squared weights and their sum.

- [ ] **Step 5: Add deterministic action probes**

Compare exact and compressed MPO actions on fixed product states and seeded random MPS states. Report:

```text
relative_action_error = ||H_exact|psi> - H_compressed|psi>|| / ||H_exact|psi>||
```

The probe set and seed are configuration values. A zero denominator is reported as `not_tested`, not epsilon-clamped.

- [ ] **Step 6: Run MPO tests**

Run: `conda run -n torch-mps python -m unittest tests.test_agp_mpo_backend -v`

Expected: exact dense equivalence and compressed-action tests PASS.

- [ ] **Step 7: Commit the MPO component**

```bash
git add scripts/agp_mpo_backend.py tests/test_agp_mpo_backend.py
git commit -m "Build and compress full-support Pauli MPOs"
```

### Task 3: Time-Dependent MPO Assembly And Two-Site TDVP

**Files:**
- Modify: `scripts/agp_mpo_backend.py`
- Modify: `tests/test_agp_mpo_backend.py`

**Interfaces:**
- Consumes: compressed `H_initial`, `H_final`, temporal mode MPOs, schedule/factor interpolants, and MPS settings.
- Produces: `prepare_tdvp_operators(...)`, `evolve_protocol_tdvp(...) -> tuple[MPS, dict[str, object]]`, and `evolve_protocol_expm_mpo(...) -> tuple[MPS, dict[str, object]]`.
- Test-only helpers: `dense_midpoint_evolution(...)` and `state_overlap_dense(...)`, both restricted to `q <= 4`.

- [ ] **Step 1: Write failing one- and two-qubit evolution tests**

```python
def test_tdvp_no_cd_matches_dense_midpoint_evolution(self):
    h0 = [("X", -1.0)]
    h1 = [("Z", -1.0)]
    state, diagnostics = evolve_protocol_tdvp(
        h0_terms=h0,
        h1_terms=h1,
        cd_factorization=None,
        total_time=1.0,
        steps=128,
        mps_max_bond=8,
        mps_cutoff=1.0e-13,
        mpo_max_bond=16,
        mpo_cutoff=1.0e-13,
    )
    reference = dense_midpoint_evolution(h0, h1, steps=4096)
    self.assertGreater(abs(state_overlap_dense(state, reference)) ** 2, 1.0 - 1.0e-7)
    self.assertEqual(diagnostics["steps"], 128)


def test_tdvp_learned_path_includes_every_mode_and_term(self):
    result = prepare_tdvp_operators(
        labels=("XI", "YI", "XZ"),
        static_modes=np.asarray([[1.0, 2.0, 3.0], [0.5, 0.25, -0.1]]),
        temporal_factors=np.ones((5, 2)),
        n_qubits=2,
        order=(0, 1),
        mpo_max_bond=16,
        mpo_cutoff=1.0e-13,
    )
    self.assertEqual(result.diagnostics["learned_input_terms"], 3)
    self.assertEqual(result.diagnostics["temporal_rank"], 2)
    self.assertEqual(result.diagnostics["support_fraction"], 1.0)


def test_expm_mpo_comparison_backend_matches_tdvp_on_one_qubit(self):
    settings = {
        "h0_terms": [("X", -1.0)],
        "h1_terms": [("Z", -1.0)],
        "cd_factorization": None,
        "total_time": 1.0,
        "steps": 128,
        "mps_max_bond": 8,
        "mps_cutoff": 1.0e-13,
        "mpo_max_bond": 16,
        "mpo_cutoff": 1.0e-13,
    }
    tdvp_state, _ = evolve_protocol_tdvp(**settings)
    expm_state, _ = evolve_protocol_expm_mpo(**settings)
    self.assertGreater(abs(tdvp_state.overlap(expm_state)) ** 2, 1.0 - 1.0e-7)
```

- [ ] **Step 2: Run and verify failure**

Run: `conda run -n torch-mps python -m unittest tests.test_agp_mpo_backend -v`

Expected: FAIL on missing TDVP functions.

- [ ] **Step 3: Implement MPO linear combinations**

Construct exact block-sum MPO tensors for:

```text
H(t) = (1 - lambda(t)) H_initial + lambda(t) H_final + sum_r f_r(t) V_r.
```

Apply the scalar coefficient to each component's first tensor, block-diagonalize interior virtual bonds, and concatenate the final left bonds. Optionally compress this dynamic sum with the same Hilbert-Schmidt sweep and record a separate dynamic-MPO discarded weight.

Build the nested-commutator `l=1` direct-CD samples on the same temporal grid and factor them through the same path. This keeps no CD, nested `l=1`, and learned AGP on one integrator while retaining every term in each declared operator.

- [ ] **Step 4: Implement midpoint two-site TDVP**

At each midpoint, construct an `MPOModel` from the assembled MPO and evolve one step with `TwoSiteTDVPEngine`. Use:

```python
options = {
    "dt": dt,
    "N_steps": 1,
    "trunc_params": {"chi_max": mps_max_bond, "svd_min": mps_cutoff},
    "lanczos_params": {"N_max": lanczos_max},
}
```

Record norm drift, accumulated TDVP truncation error, peak/final MPS bond, static/dynamic MPO bonds, midpoint build time, and evolution time. Permute the initial state and ground bitstring consistently with the chosen qubit order.

Return operator preparation through:

```python
@dataclass
class PreparedTDVPOperators:
    sites: list[object]
    h0_mpo: object
    h1_mpo: object
    cd_mode_mpos: list[object]
    temporal_factorization: TemporalFactorization | None
    order: tuple[int, ...]
    diagnostics: dict[str, object]
```

- [ ] **Step 5: Add physical metrics and exact small-q agreement**

Compute final energy with the exact final MPO and fidelity as squared overlap with the exact product-state ground bitstring. Compare q <= 4 TDVP results with independent dense midpoint evolution for no CD, nested l=1, and a synthetic learned AGP.

Implement `evolve_protocol_expm_mpo(...)` with TeNPy's `TimeDependentExpMPOEvolution` as a comparison integrator selected by `integrator: "expm_mpo"`. Keep `integrator: "tdvp"` as the default and report each integrator independently; the ExpMPO result does not replace the TDVP convergence ladder.

- [ ] **Step 6: Run TDVP tests**

Run: `conda run -n torch-mps python -m unittest tests.test_agp_mpo_backend -v`

Expected: all tests PASS with no dense operator construction in production helpers.

- [ ] **Step 7: Commit TDVP evolution**

```bash
git add scripts/agp_mpo_backend.py tests/test_agp_mpo_backend.py
git commit -m "Add full-support MPO TDVP evolution"
```

### Task 4: Existing Validation CLI Integration And Certification Ladder

**Files:**
- Modify: `tests/sparse_agp_curriculum/scripts/agp_mps_validation.py`
- Modify: `tests/test_agp_mps_validation.py`
- Modify: `scripts/agp_plot_annotations.py`
- Modify: `tests/test_agp_physical_validation.py`

**Interfaces:**
- Consumes: `tensor_network_validation.mpo_backend` and `resolutions`.
- Produces: the existing `mps_physical_validation_summary.json` schema with additional MPO diagnostics.

- [ ] **Step 1: Write failing dispatch, cache, and certification tests**

```python
def test_mpo_cache_key_includes_every_numerical_axis(self):
    settings = {
        "backend": "tenpy_tdvp_mpo",
        "steps": 24,
        "temporal_retained_norm": 0.9999,
        "mpo_max_bond": 64,
        "mpo_cutoff": 1.0e-10,
        "mps_max_bond": 32,
        "mps_cutoff": 1.0e-9,
        "learned_terms": 32768,
    }
    previous = [{"settings": dict(settings), "results": {"learned_sparse_agp": {"final_energy": -1.0}}}]
    self.assertIsNotNone(cached_protocol_result(previous, settings=settings, protocol="learned_sparse_agp"))
    self.assertIsNone(
        cached_protocol_result(previous, settings={**settings, "mpo_max_bond": 128}, protocol="learned_sparse_agp")
    )


def test_certification_requires_temporal_mpo_mps_and_timestep_gates(self):
    certification = validation_certification(
        convergence={"status": "pass"},
        compression={"status": "pass"},
        statevector_agreement={"status": "not_tested"},
        require_convergence=True,
        require_compression=True,
        require_statevector=False,
    )
    self.assertEqual(certification["status"], "pass")
    self.assertIn("mpo_compression", certification["required_gates"])
```

- [ ] **Step 2: Run existing validation tests and confirm failure**

Run: `conda run -n torch-mps python -m unittest tests.test_agp_mps_validation tests.test_agp_physical_validation -v`

Expected: FAIL on missing MPO dispatch/certification fields.

- [ ] **Step 3: Add backend dispatch without breaking quimb**

Parse:

```json
"mpo_backend": {
  "name": "tenpy_tdvp_mpo",
  "qubit_order_candidates": ["native", "spectral"],
  "temporal_grid_points": 257,
  "action_probe_seed": 11,
  "action_probe_product_states": 4,
  "action_probe_random_mps": 2,
  "action_error_max": 0.001,
  "resource_caps": {"max_build_seconds": 3600, "max_peak_memory_gb": 24}
}
```

Dispatch `tenpy_tdvp_mpo` cases to the new backend. Keep `quimb_product_formula` as an explicit legacy backend. For the learned protocol, enforce `learned_terms == len(full learned labels)` unless the output is marked `ablation`.

- [ ] **Step 4: Expand resolution and convergence records**

Each resolution records:

```text
temporal_retained_norm, temporal_rank, temporal reconstruction error,
static/dynamic MPO max bonds, MPO discarded weight and action error,
MPS max bond and cutoff, timestep, runtime, norm drift,
final energy and ground-state fidelity.
```

Certification passes only when every required compression gate passes and the final two resolutions satisfy energy/fidelity tolerances. q <= 15 additionally requires matching full-support statevector results.

- [ ] **Step 5: Preserve table behavior and label unresolved data**

The table continues to compare no CD, nested l=1, and learned AGP. Add a note identifying `tenpy_tdvp_mpo`, full learned term count, and convergence status. `not_feasible` and `not_tested` never render as certified physical metrics.

- [ ] **Step 6: Run validation tests**

Run: `conda run -n torch-mps python -m unittest tests.test_agp_mpo_backend tests.test_agp_mps_validation tests.test_agp_physical_validation -v`

Expected: all tests PASS, including legacy quimb tests.

- [ ] **Step 7: Commit CLI integration**

```bash
git add tests/sparse_agp_curriculum/scripts/agp_mps_validation.py tests/test_agp_mps_validation.py scripts/agp_plot_annotations.py tests/test_agp_physical_validation.py
git commit -m "Integrate compressed MPO physical validation"
```

### Task 5: q15 Calibration And q24 Speed Preflight

**Files:**
- Modify: `tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q15/sweep_test/config.json`
- Modify: `tests/sparse_agp_curriculum/transverse_field_spin_hubo/run_002_hamiltonian_341/q24/sweep_test/config.json`
- Generated only: ignored validation outputs.

**Interfaces:**
- Consumes: retained q15/q24 checkpoints and exact ground references.
- Produces: q15 statevector agreement and q24 one-step speed/compression diagnostics.

- [ ] **Step 1: Configure a q15 full-support calibration case**

Use the same learned support as its matching statevector row. Add two MPO resolutions that vary temporal tolerance, MPO bond/cutoff, MPS bond/cutoff, and steps while preserving the support.

- [ ] **Step 2: Run q15 calibration**

Run the existing validation entrypoint with `--config` pointing to q15. Expected: energy/fidelity agreement within the configured statevector tolerances and compression action error below `1e-3`.

- [ ] **Step 3: Configure q24 preflight before the ladder**

Add a named `speed_preflight` case with one midpoint, all 32,768 learned labels, temporal retained norm `0.9999`, MPO bond cap 64, and MPS bond cap 8. Set `preflight_only: true` so it cannot certify physical dynamics.

- [ ] **Step 4: Run q24 preflight and compare runtime**

Run:

```bash
conda run --no-capture-output -n torch-mps python \
  tests/sparse_agp_curriculum/scripts/agp_mps_validation.py \
  --config tests/sparse_agp_curriculum/transverse_field_spin_hubo/run_002_hamiltonian_341/q24/sweep_test/config.json \
  --preflight-only
```

Expected: all 32,768 terms enter the MPO builder, compression diagnostics are finite, and one-step wall time is below the measured 2,502.7-second product-formula baseline. Otherwise mark `not_feasible` and do not start the full ladder.

- [ ] **Step 5: Commit calibrated configs**

```bash
git add tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q15/sweep_test/config.json tests/sparse_agp_curriculum/transverse_field_spin_hubo/run_002_hamiltonian_341/q24/sweep_test/config.json
git commit -m "Configure compressed MPO validation ladder"
```

### Task 6: q24 Convergence Ladder, Artifacts, And Methodology

**Files:**
- Modify: `docs/CURRENT_SPARSE_AGP_METHODOLOGY.md`
- Modify: `Rules.md`
- Modify: `AGP_CERTIFICATION_CRITERIA.md`
- Modify: `tests/sparse_agp_curriculum/transverse_field_spin_hubo/run_002_hamiltonian_341/q24/sweep_test/README.md`
- Generated only: ignored q24 validation outputs.

**Interfaces:**
- Consumes: a passing speed preflight.
- Produces: a full q24 convergence result or an explicit `not_feasible`/`not_tested` status.

- [ ] **Step 1: Run the configured q24 ladder only after preflight passes**

Run the validation entrypoint without `--preflight-only`. Monitor each protocol and resolution; persist progress after every protocol so an interruption resumes safely.

- [ ] **Step 2: Verify every numerical gate**

Use a `conda run -n torch-mps python -c` check to assert:

```text
learned input terms == 32768 at every resolution,
temporal retained norm meets target,
compression action errors meet tolerance,
no resource cap was silently relaxed,
successive energy/fidelity deltas meet thresholds,
certification agrees with the individual gates.
```

- [ ] **Step 3: Regenerate the comparison table and inspect the PDF**

Render `physical_method_comparison_table.pdf` and verify it names the MPO backend, full support, exact ground reference, and convergence status without overlapping text.

- [ ] **Step 4: Update methodology and q24 status conservatively**

Document temporal SVD, exact sparse MPO construction, Hilbert-Schmidt compression, action probes, TDVP, and the numerical ladder. If any gate fails, retain the physical result as diagnostic and state `not tested` or `not feasible`; do not promote it to a pass.

- [ ] **Step 5: Run repository verification**

Run:

```bash
conda run -n torch-mps python -m py_compile models.py utils.py scripts/agp_residual_probes.py scripts/agp_mpo_backend.py scripts/agp_holdout_feedback.py tests/sparse_agp_curriculum/scripts/agp_mps_validation.py
conda run -n torch-mps python examples/two_qubit_sparse_demo.py
conda run -n torch-mps python -m unittest discover -s tests
```

Expected: all commands PASS.

- [ ] **Step 6: Commit documentation and retained source changes**

```bash
git add Rules.md AGP_CERTIFICATION_CRITERIA.md docs/CURRENT_SPARSE_AGP_METHODOLOGY.md tests/sparse_agp_curriculum/transverse_field_spin_hubo/run_002_hamiltonian_341/q24/sweep_test/README.md
git commit -m "Document compressed MPO validation methodology"
```
