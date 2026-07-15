# MPO Task 4 Report

## Scope

Integrated the compressed-MPO validation surface without changing benchmark
configs or generated validation artifacts.

## Implementation

- Added explicit `tenpy_tdvp_mpo` dispatch while retaining
  `quimb_product_formula` as the legacy default.
- Enforced full learned support for certifiable learned MPO protocols. A reduced
  learned support must be explicitly marked as an `ablation` and cannot certify.
- Expanded cached-resolution identity to the complete normalized settings map,
  including a checkpoint/content SHA-256, coefficient identity, learned scale,
  canonical H0/H1 term hash, ground-reference content hash and bitstring, and
  learned schedule identity. Temporal, MPO, MPS, timestep, order, integrator,
  action-probe, and resource axes therefore all invalidate reuse when changed.
- Recorded MPO temporal factorization, static/dynamic MPO, MPS, timestep,
  runtime, norm, truncation, final-energy, and fidelity diagnostics alongside
  each protocol result.
- Added an MPO compression gate. It requires temporal, static/dynamic MPO, and
  finite action-error interval evidence collected through Task 3
  `probe_mpo_compression` for static and representative dynamic MPOs. The
  recorded diagnostics retain probe status, bounds, seeds, caps, and raw probe
  results. A measured upper bound above tolerance fails the gate;
  `not_feasible`, `not_tested`,
  `not_comparable`, numerical uncertainty, and unresolved backend errors leave
  certification closed.
- Added complete MPO result records with final energy, fidelity, excitation
  probability, and Z/nearest-neighbor ZZ availability/status fields. Baseline
  quotients tolerate unavailable observables.
- MPO certification now requires two completed, comparable non-ablation
  resolutions and convergence for every system size. Every eligible resolution
  must contain `no_cd`, canonical `nested_l1` (including the legacy
  `kipu_dqfm_l1` alias), and `learned_sparse_agp`; an ablation, a reduced
  support, a missing canonical protocol, or a single resolution is
  `not_tested`.
- One shared eligible-resolution identity controls MPO comparability,
  compression, MPS convergence, and timestep convergence. It includes backend
  and integrator, qubit count, full learned support/scale, H0/H1, schedule,
  checkpoint/coefficient, initial state, total time, and ground-reference
  provenance. Interleaved ablations and non-matching physical rows cannot
  contribute to any of those gates.
- q <= 15 statevector agreement now requires a `validation_identity` that
  exactly matches the eligible MPO run's q, Hamiltonian hash, schedule,
  duration, checkpoint/coefficient, support/scale, initial state, ground
  reference/bitstring, steps, and integrator semantics. A missing or mismatched
  identity is `not_tested`.
- Final-time energy and fidelity are suppressed in physical-comparison tables
  when a backend status is not `ok` or completed steps are short. The note
  identifies partial/not-feasible rows and, for an ablation, deployed versus
  available learned terms.
- Required q <= 15 MPO results to have a matching full-support statevector
  reference before certification.
- Kept the comparison-table schema unchanged. MPO tables state backend,
  learned-support count, convergence, and diagnostic-only status unless the
  complete certification ladder passes. Legacy quimb metrics still render when
  old diagnostics lack `completed_steps`; MPO rows and explicit partial/error
  rows suppress final-time metrics. Certified MPO notes retain the actual
  ladder-convergence status.
- Backend-exception rows use the same full metric schema as completed MPO rows,
  with unavailable metrics and explicit observable/final-time statuses.
- Top-level MPO `results`, `full_learned_terms`, and
  `certification_resolution` now publish from the same final eligible
  full-support row. A trailing ablation stays in `resolution_results` but
  cannot replace certified top-level metrics.
- Added `agp_validation_identity.py`, shared by the statevector and MPO CLIs.
  The statevector payload now emits a default and per-learned-variant
  `validation_identity` with canonical H0/H1, schedule parameters, timing,
  checkpoint/coefficient, support/scale, initial state, ground reference, and
  both statevector RK4 and calibrated-MPO integrator semantics.
- Plot/table normalization accepts canonical `nested_l1` as well as legacy
  `kipu_dqfm_l1`, mapping both to the nested-commutator row.

## TDD Evidence

Added failing tests first for real TeNPy MPO resolution output, static/dynamic
action-probe interval aggregation, cache-staleness identities, unavailable
quotients, ablation/single-resolution certification branches, and partial
table suppression. Follow-up adversarial coverage rejects a no-CD-only ladder,
skips interleaved ablations, rejects mismatched statevector physics, preserves
legacy quimb rendering, retains certified-ladder note status, and verifies the
backend-exception result schema. The real resolution test executes the MPO
backend rather than only mocking its summary helpers.

Final coverage adds a trailing-ablation payload-publication test, a
statevector-identity producer/consumer match-and-mismatch test, canonical
nested-l1 rendering, and the framework-script layout allowlist update.

## Verification

```text
conda run -n torch-mps python -m unittest \
  tests.test_agp_mpo_backend tests.test_agp_mps_validation \
  tests.test_agp_physical_validation -v
```

Result: 128 tests passed, including the MPO, MPS, physical-validation, and
layout suites.

```text
conda run -n torch-mps python -m py_compile \
  models.py utils.py scripts/agp_mpo_backend.py \
  tests/sparse_agp_curriculum/scripts/agp_mps_validation.py \
  scripts/agp_plot_annotations.py
git diff --check
```

Result: passed.

## Certification Boundary

No generated physical-validation run was performed. A configured run now
persists bounded static and representative-dynamic probe evidence, but remains
diagnostic until every required convergence, compression, support, and (for
q <= 15) statevector gate passes. It cannot promote incomplete, unresolved, or
reduced-support data to a certified physical claim.
