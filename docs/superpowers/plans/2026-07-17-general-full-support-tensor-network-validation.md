# General Full-Support Tensor-Network Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace uncontrolled post-hoc learned-AGP MPO compression with a general, full-K, convergence-gated tensor-network evaluation framework and validate it on exact small systems plus the retained q24 HUBO checkpoint.

**Architecture:** Add full-support provenance and a measured-complexity backend router, then make a positioned joint time-Pauli tensor train the canonical MPS path. Every requested midpoint reconstructs all learned coefficients. Contiguous windows may split adaptively when rank or workspace gates fail, but no split changes `K`. Exact small-system and independent multi-resolution gates prevent uncertified energies or fidelities from reaching canonical artifacts.

**Tech Stack:** Python 3.10, NumPy, PyTorch checkpoint loading, physics-tenpy 1.1.0, quimb reference utilities, Matplotlib, JSON, `unittest`.

## Global Constraints

- Run every Python command through `conda run -n torch-mps`.
- Canonical learned dynamics must use all `K` checkpoint terms.
- Keep top-term selection only as an explicitly labeled ablation.
- Do not construct dense operators outside bounded exact tests.
- Keep training independent of optional tensor-network imports.
- Preserve unrelated existing worktree changes.
- Publish physical metrics only after operator and dynamics gates pass.
- q24 is an integration test, not a source of fixed algorithmic constants.

---

### Task 1: Freeze Full-K Provenance And Hamiltonian Profiles

**Files:**
- Modify: `scripts/agp_mpo_backend.py`
- Modify: `tests/sparse_agp_curriculum/scripts/agp_mps_validation.py`
- Modify: `tests/test_agp_mpo_backend.py`
- Modify: `tests/test_agp_mps_validation.py`

**Interfaces:**
- Produces `FullSupportIdentity`, mutation-sensitive checkpoint identities, and `assert_full_support_identity(...)`.
- Consumes ordered labels, sampled coefficient arrays, Hamiltonian terms, and qubit count.

- [ ] Write tests proving that label removal, label reordering without a matching hash, nonfinite coefficients, and accidental coefficient thresholds fail before MPO construction.
- [ ] Run the focused tests and confirm that they fail on missing interfaces.
- [ ] Implement stable SHA-256 identities over labels, shapes, normalized coefficient bytes, and checkpoint identity.
- [ ] Implement profiles for term count, Pauli locality, coefficient scale/dynamic range, interaction cuts, and deterministic candidate orders.
- [ ] Persist profiles and all-K identities in each resolution payload.
- [ ] Re-run focused tests and `py_compile` until they pass.

### Task 2: Add A General Fail-Closed Backend Router

**Files:**
- Create: `scripts/agp_tn_router.py`
- Create: `tests/test_agp_tn_router.py`
- Modify: `tests/sparse_agp_curriculum/scripts/agp_mps_validation.py`

**Interfaces:**
- Produces `TensorNetworkRoute` and `route_tensor_network_backend(profile, policy)`.
- Route names are `exact_statevector`, `direct_full_support_mpo_tdvp`, `adaptive_windowed_full_support_mpo_tdvp`, and `unsupported_topology`.

- [ ] Write failing routing tests for small-q exact, sparse local, dense low-rank, high-cut-width, and resource-cap cases.
- [ ] Verify the expected failures.
- [ ] Implement deterministic routing from measured rank, memory, and work estimates; do not route from density or `q` alone.
- [ ] Export the selected route, rejected alternatives, and reasons.
- [ ] Verify all router tests pass.

### Task 3: Build Direct-Time Full-Support MPOs

**Files:**
- Modify: `scripts/agp_mpo_backend.py`
- Modify: `tests/test_agp_mpo_backend.py`

**Interfaces:**
- Produces `combine_instantaneous_full_support_terms(...)`, `build_direct_time_mpo(...)`, and `DirectMPOCertificate`.
- Consumes all learned labels, interpolated direct-CD coefficients, schedule values, and both Hamiltonian term lists.

- [ ] Write failing tests showing that direct construction contains all source labels, combines duplicates stably, preserves Hermiticity, and matches dense q<=4 matrices.
- [ ] Add a regression where separate temporal modes require a larger block-sum bond than the directly combined operator.
- [ ] Verify failures before implementation.
- [ ] Implement coefficient normalization, stable duplicate summation, sparse Pauli TT-SVD, and scale restoration.
- [ ] Issue exact-identity evidence only when cutoff is zero and no singular direction is discarded; otherwise run action probes.
- [ ] Export source K, combined nonzeros, coefficient hash, ranks, discarded weight, action error, workspace, and build time.
- [ ] Run focused MPO tests until all pass.

### Task 4: Integrate Direct MPOs With Two-Site TDVP

**Files:**
- Modify: `scripts/agp_mpo_backend.py`
- Modify: `tests/sparse_agp_curriculum/scripts/agp_mps_validation.py`
- Modify: `tests/test_agp_mpo_backend.py`
- Modify: `tests/test_agp_mps_validation.py`

**Interfaces:**
- Adds `representation="direct_time_full_support"` to `evolve_protocol_tdvp(...)`.
- Produces per-step `operator_certificates`, state diagnostics, and final metrics.

- [ ] Write failing one- and two-qubit tests comparing no-CD, nested-l1, and learned direct-MPO TDVP against dense midpoint evolution.
- [ ] Write a failure test proving that one bad midpoint action certificate aborts the learned run and suppresses metrics.
- [ ] Verify the tests fail for the intended reason.
- [ ] Build one complete instantaneous MPO at every midpoint and evolve one two-site TDVP step.
- [ ] Preserve all-K identity from checkpoint through every midpoint certificate.
- [ ] Record norm drift, truncation error, state/MPO bonds, time, memory, and failure location.
- [ ] Run focused tests until they pass.

### Task 4B: Amortize Full-K Construction Across Time

**Files:**
- Modify: `scripts/agp_mpo_backend.py`
- Modify: `tests/sparse_agp_curriculum/scripts/agp_mps_validation.py`
- Modify: `tests/test_agp_mpo_backend.py`
- Modify: `tests/test_agp_mps_validation.py`

**Interfaces:**
- Produces `TimePauliTensorTrain`, `build_time_pauli_tensor_train(...)`,
  `slice_time_pauli_mpo(...)`, and `evolve_protocol_time_tensor_tdvp(...)`.

- [ ] Write failing tests for exact small-q slices, arbitrary-length labels,
  full-K identity, coefficient-space gates, and dense-evolution agreement.
- [ ] Factor each complete contiguous midpoint-by-Pauli window in a
  workspace-bounded positioned TT-SVD, with arbitrary-precision Pauli
  encodings.
- [ ] Certify a conservative per-slice coefficient error bound and exact sparse
  action probes at deterministic representative times.
- [ ] Slice one ordinary MPO per TDVP midpoint without mode-wise block
  summation; split failed windows without reducing `K`.
- [ ] Route canonical learned validation through this representation and retain
  direct per-time construction only as a measured fallback/diagnostic.
- [ ] Verify full-K q<=4 dense agreement before launching q24.

### Task 5: Add Independent Operator And Dynamics Ladders

**Files:**
- Modify: `tests/sparse_agp_curriculum/scripts/agp_mps_validation.py`
- Modify: `tests/test_agp_mps_validation.py`

**Interfaces:**
- Produces `assess_independent_mpo_convergence(...)` and a fail-closed `validation_certification(...)` payload.

- [ ] Write failing tests for mismatched K/hash, incomparable orders, incomplete ladders, action-gate failures, and converged two-resolution ladders.
- [ ] Verify failures.
- [ ] Separate operator-bond convergence from time-step and state-bond convergence.
- [ ] Require all canonical protocols, all K terms, and at least two comparable completed resolutions.
- [ ] Suppress result-table rows whenever certification is not passed.
- [ ] Run focused validation tests until they pass.

### Task 6: Validate Across q And Hamiltonian Regimes

**Files:**
- Create: `tests/test_agp_tn_regression_matrix.py`
- Modify: `tests/test_agp_mpo_backend.py`

**Interfaces:**
- Uses synthetic deterministic Pauli sums and independent dense evolution for bounded q.

- [ ] Add q=2,3,4,6 cases spanning local sparse, long-range sparse, dense factorizable, and high-cut-width operators.
- [ ] Add coefficient scales `1e-8`, `1`, and `1e8`, plus mixed dynamic ranges.
- [ ] Compare full-K MPO matrices/actions and TDVP final energies/fidelities with exact statevector results.
- [ ] Verify expected successful and fail-closed routes.
- [ ] Run the complete regression matrix in `torch-mps`.

### Task 7: Run And Diagnose The q24 Full-K Benchmark

**Files:**
- Modify: `tests/sparse_agp_curriculum/transverse_field_spin_hubo/run_002_hamiltonian_341/q24/sweep_test/config.json`
- Modify: `tests/sparse_agp_curriculum/transverse_field_spin_hubo/run_002_hamiltonian_341/q24/sweep_test/README.md`

**Interfaces:**
- Uses the retained 32,768-term adaptive checkpoint and exact diagonal ground reference.

- [ ] Add a direct-path preflight that measures midpoint exact ranks and resource requirements without evolution.
- [ ] Run the preflight and select two feasible resolutions from measured ranks, not guessed constants.
- [ ] Run no CD, nested-l1, and learned sparse AGP with all 32,768 terms.
- [ ] Refine operator bond, MPS bond/cutoff, and time step until gates pass or a certified resource obstruction is reached.
- [ ] Inspect energies, fidelities, norm drift, action errors, and convergence deltas for physical plausibility.
- [ ] Regenerate the comparison table and HCD summary only from canonical passed results.

### Task 8: Documentation And Repository Verification

**Files:**
- Modify: `Rules.md`
- Modify: `AGP_CERTIFICATION_CRITERIA.md`
- Modify: `docs/CURRENT_SPARSE_AGP_METHODOLOGY.md`
- Modify: `scripts/README.md`

**Interfaces:**
- Documents the backend router, direct full-support path, certification schema, and limitations.

- [ ] Update the methodology and rulebook so full-K direct-time validation is canonical and temporal-mode compression is diagnostic unless independently certified.
- [ ] Run `conda run -n torch-mps python -m py_compile models.py utils.py scripts/agp_mpo_backend.py scripts/agp_tn_router.py`.
- [ ] Run `conda run -n torch-mps python examples/two_qubit_sparse_demo.py`.
- [ ] Run focused tensor-network tests, then `conda run -n torch-mps python -m unittest discover -s tests`.
- [ ] Review `git diff --check`, generated-artifact status, and the complete diff without reverting unrelated changes.
