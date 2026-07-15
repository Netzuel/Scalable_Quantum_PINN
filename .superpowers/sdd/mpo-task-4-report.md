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
  including checkpoint path, size, and nanosecond modification time; temporal,
  MPO, MPS, timestep, order, integrator, and resource axes therefore all
  invalidate reuse when changed.
- Recorded MPO temporal factorization, static/dynamic MPO, MPS, timestep,
  runtime, norm, truncation, final-energy, and fidelity diagnostics alongside
  each protocol result.
- Added an MPO compression gate. It requires temporal, static/dynamic MPO, and
  finite action-error evidence. `not_feasible`, `not_tested`,
  `not_comparable`, numerical uncertainty, and unresolved backend errors leave
  certification closed.
- Required q <= 15 MPO results to have a matching full-support statevector
  reference before certification.
- Kept the comparison-table schema unchanged. MPO tables state backend,
  learned-support count, convergence, and diagnostic-only status unless the
  complete certification ladder passes. Legacy quimb table wording is retained.

## TDD Evidence

Added failing tests first for MPO cache axes/checkpoint identity, certification
requirements, explicit backend parsing/full-support enforcement, and
uncertified MPO table annotations. Before implementation they failed on the
missing public dispatch/full-support APIs and absent table annotation.

## Verification

```text
conda run -n torch-mps python -m unittest \
  tests.test_agp_mpo_backend tests.test_agp_mps_validation \
  tests.test_agp_physical_validation -v
```

Result: 95 tests passed.

```text
conda run -n torch-mps python -m py_compile \
  models.py utils.py scripts/agp_mpo_backend.py \
  tests/sparse_agp_curriculum/scripts/agp_mps_validation.py \
  scripts/agp_plot_annotations.py
git diff --check
```

Result: passed.

## Certification Boundary

No generated physical-validation run was performed. In particular, the
integration reports the action-error field explicitly but keeps its compression
gate `not_tested` until a configured run persists finite MPO action-probe
evidence. It therefore cannot promote an unresolved or reduced-support result
to a certified physical claim.
