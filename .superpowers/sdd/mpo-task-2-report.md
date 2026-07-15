# MPO Task 2 Report

## Status

Initial implementation: `a2b4fa6 Build and compress full-support Pauli MPOs`.
The high-review resource findings are remediated in a separate follow-up commit.

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

The high-review remediation added two further RED gates:

- The q12/512 compression test initially failed because the API had no
  `workspace_cap_bytes`; the old implementation would also hit the test's fatal
  `Array.to_ndarray()` interception before any SVD.
- The action test failed at the intercepted `exact_mpo.apply_naively()` call,
  and the random resource test failed because `exact_work_cap` did not exist.

## Implementation

- Exact TeNPy finite-state-graph construction combines duplicate labels first,
  permutes every retained label into chain order, includes identity labels, and
  reports all included and arithmetic-zero-dropped labels and coefficients.
- TeNPy imports remain lazy. Importing the preparation module without the
  optional dependency succeeds; calling an MPO operation gives an installation
  hint.
- The exact builder retains immutable combined chain-order Pauli provenance on
  the sparse TeNPy graph MPO. Compression never converts exact graph tensors to
  dense arrays. It performs TT-SVD over one Pauli-coordinate suffix unfolding
  at a time and converts only the retained compressed cores to local operator
  tensors.
- Every local unfolding is preflighted against `workspace_cap_bytes` using a
  conservative estimate that includes index arrays, local matrices, eigensolver
  work, retained values, and compressed output copies. Diagnostics expose the
  cap, conservative peak, explicit-array peak, required bytes, and failed bond.
  A cap violation returns `(None, {"status": "not_feasible", ...})` before the
  prohibited allocation.
- Product probes compute the exact action from every Pauli term by aggregating
  output bitstrings and amplitudes. Compressed action norms and cross terms use
  streamed local transfer contractions and batched amplitude queries.
- Seeded random MPS probes compute exact norms from Pauli-pair expectations and
  cross terms from individual fixed-bond Pauli actions against one bounded
  compressed action. Cases above `exact_work_cap` or the workspace cap return a
  named `not_feasible` probe; no exact MPO action is formed or approximated.
- Requested random bond dimension is capped at the finite-chain Schmidt bound
  and both values are reported. Exact zero action denominators remain
  `not_tested`.
- Dense Pauli and MPO helpers reject `q > 4`; no production path constructs a
  dense many-qubit operator.

## GREEN Evidence

Commands:

```text
conda run -n torch-mps python -m unittest tests.test_agp_mpo_backend -v
conda run -n torch-mps python -m py_compile scripts/agp_mpo_backend.py tests/test_agp_mpo_backend.py
git diff --check
```

Result: 32 focused tests passed, compilation passed, and the diff check was
clean. Coverage includes exact dense equivalence, duplicate combination and
cancellation metadata, explicit arithmetic-zero tolerance, qubit-order
equivalence, large-q construction with dense guards, compression error and
resource limits, global Hilbert-Schmidt discarded-weight accounting,
deterministic product/random MPS action errors, zero-denominator status, and
optional-import behavior.

The adversarial q12/512 test uses labels scrambled across the full 12-site
Pauli address space. With `max_bond=8` and an 8 MiB cap it reports
`peak_workspace_bytes=842240`, `peak_explicit_workspace_bytes=465408`, and
retained bond 8 across the central cuts while a fatal `to_ndarray()` patch is
active. A separate 1 KiB test returns `not_feasible` before allocation.

## Scope

Concurrent holdout, methodology, residual-probe, support-swap, and generated
q24 artifact changes were not modified or staged. Repository-wide tests were
not run because the task requested focused Task 2 verification and concurrent
owned work is present outside this component.
