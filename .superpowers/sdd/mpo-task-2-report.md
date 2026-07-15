# MPO Task 2 Report

## Status

Complete in commit `Build and compress full-support Pauli MPOs`.

## Files

- `scripts/agp_mpo_backend.py`
- `tests/test_agp_mpo_backend.py`
- `.superpowers/sdd/mpo-task-2-report.md`

## RED Evidence

The initial focused run failed while importing the missing Task 2 interfaces:

```text
ImportError: cannot import name 'build_exact_pauli_mpo'
```

During diff review, a second regression test compared the reported total
discarded squared norm with the tiny-system dense Hilbert-Schmidt error. It
failed with `4.0 != 41.47685570173767`, proving that the finite-state-graph MPO
needed an exact right-canonicalization before the left-to-right truncating SVD
sweep.

## Implementation

- Exact TeNPy finite-state-graph construction combines duplicate labels first,
  permutes every retained label into chain order, includes identity labels, and
  reports all included and arithmetic-zero-dropped labels and coefficients.
- TeNPy imports remain lazy. Importing the preparation module without the
  optional dependency succeeds; calling an MPO operation gives an installation
  hint.
- Compression extracts both finite boundary indices, vectorizes each local
  operator leg pair, right-canonicalizes without truncation, then performs the
  required left-to-right SVD sweep. The cutoff is a per-bond relative cumulative
  discarded squared-weight limit; `max_bond` remains a hard cap with explicit
  cutoff-satisfaction diagnostics.
- Diagnostics include source/effective/canonical/post-compression bonds,
  retained ranks, per-bond relative and absolute discarded weights, total
  Hilbert-Schmidt error, and the source Hilbert-Schmidt norm.
- Product and seeded random MPS probes use overlap contractions only. Requested
  random bond dimension is capped at the finite-chain Schmidt bound and both
  values are reported. Exact zero action denominators return `not_tested`.
- Dense Pauli and MPO helpers reject `q > 4`; no production path constructs a
  dense many-qubit operator.

## GREEN Evidence

Commands:

```text
conda run -n torch-mps python -m unittest tests.test_agp_mpo_backend -v
conda run -n torch-mps python -m py_compile scripts/agp_mpo_backend.py tests/test_agp_mpo_backend.py
git diff --check
```

Result: 29 focused tests passed, compilation passed, and the diff check was
clean. Coverage includes exact dense equivalence, duplicate combination and
cancellation metadata, explicit arithmetic-zero tolerance, qubit-order
equivalence, large-q construction with dense guards, compression error and
resource limits, global Hilbert-Schmidt discarded-weight accounting,
deterministic product/random MPS action errors, zero-denominator status, and
optional-import behavior.

## Scope

Concurrent holdout, methodology, residual-probe, support-swap, and generated
q24 artifact changes were not modified or staged. Repository-wide tests were
not run because the task requested focused Task 2 verification and concurrent
owned work is present outside this component.
