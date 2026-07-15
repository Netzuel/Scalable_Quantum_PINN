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
- The interrupted remediation left product and random action probes calling
  `_cancellation_safe_nonnegative_difference` and
  `_overlap_squared_difference` without defining them. The focused test run
  failed with `NameError` in both action-error regressions.

The final numerical review added three RED regressions:

- A compressed MPO with exact-reconstruction provenance was mutated through
  public `set_W`; the old provenance-only marker still reported exact zero.
- Product and random tiny-q probes with a `1e-8` off-support perturbation had
  no operation-count diagnostics and could not demonstrate that their
  unresolved bounds enclosed the independently computed dense action error.
- The reviewer reproduction `II=1e16`, `IZ=1`, `ZI=-1e16`, `ZZ=1`, with a
  candidate `XI=1e-8` on `|up,up>` exposed left-to-right sparse aggregation:
  it formed an exact action of one instead of two and reported an interval near
  one although the dense relative error is `5e-9`. The same failure applies
  after uniform rescaling of every coefficient.

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
  work, owned retained cores, retained values, and compressed output copies.
  Retained TT cores are C-contiguous copies, so no returned core keeps an
  eigensolver matrix alive through a view. Diagnostics expose the cap,
  conservative peak, explicit-array peak, required bytes, and failed bond. A
  cap violation returns `(None, {"status": "not_feasible", ...})` before the
  prohibited allocation.
- The q24/2048 regression first sets a one-byte cap while intercepting
  `np.fromiter`; it returns `not_feasible` without creating the coefficient
  buffer. The successful 33 MiB run uses `tracemalloc`, verifies every retained
  core owns its compact buffer, and requires both reported peak and required
  workspace to remain within the cap.
- Product probes bucket Pauli contributions by output bitstring and combine
  their real and imaginary components with `math.fsum`; this keeps a retained
  term-count-bounded workspace under the existing preflight. Diagnostics report
  the aggregation method, operation estimate, condition number, and final
  rounding uncertainty. That uncertainty is propagated into the squared-error
  and exact-action-norm intervals; when the conditioned action denominator has
  no positive lower bound, the result is `numerically_unresolved` with an
  infinite conservative relative upper bound rather than a false finite claim.
  Compressed action norms and cross terms use streamed local transfer
  contractions and batched amplitude queries.
- Seeded random MPS probes compute exact norms from Pauli-pair expectations and
  cross terms from individual fixed-bond Pauli actions against one bounded
  compressed action. Their construction and action workspace are estimated
  before invoking the MPS constructor. Cases above `exact_work_cap` or the
  workspace cap return a named `not_feasible` probe; no exact MPO action is
  formed or approximated.
- Product probes calculate their output-amplitude difference directly.
  Random-MPS probes use the corresponding overlap roundoff bound. Identical
  actions report exactly zero only through the source-object or exact
  reconstruction provenance criterion, while a relative `1e-10` perturbation
  remains detectable.
- Exact-reconstruction evidence now records a SHA-256 fingerprint of the
  bounded compressed MPO tensor payloads, leg labels, shapes, and dtypes.
  Verification recomputes that fingerprint before exact zero is used; a public
  `set_W` mutation invalidates the certificate. The exact finite-state graph is
  never fingerprinted or densified: only compression-created tensors whose
  bonds remain within the recorded cap are eligible.
- Cancellation handling is scale-relative: it has no unit-scale floor. A
  subtractive norm or overlap within its roundoff bound is not reported as
  zero. Unless the compared MPO is the source object or carries an exact
  reconstruction provenance marker, the probe is `numerically_unresolved` with
  `relative_action_error=None`. Its squared-error interval is
  `[max(0, d - B), max(0, d) + B]`, where `d` is the computed signed
  difference and `B` uses the binary64 `gamma_n = n eps / (1 - n eps)` model.
  The operation estimate is derived from MPO bond dimensions and chain length;
  every probe reports the estimate, model, assumptions, interval endpoints,
  and arithmetic uncertainty.
- The single-site random probe creates a seeded normalized local state with
  `MPS.from_product_state`; it never calls the no-bond random-unitary
  evolution path.
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

Result: 43 focused tests passed, compilation passed, and the diff check was
clean. Coverage includes exact dense equivalence, duplicate combination and
cancellation metadata, explicit arithmetic-zero tolerance, qubit-order
equivalence, large-q construction with dense guards, compression error and
resource limits, global Hilbert-Schmidt discarded-weight accounting,
deterministic product/random MPS action errors, cancellation-safe zero action
errors with a detectable small perturbation, scale-rescaled error probes,
numerically-unresolved leakage bounds, single-site seeded random MPS probes,
zero-denominator status, mutation-safe exact-identity evidence, adversarial
dense product/random bound enclosure, the reviewer cancellation-conditioned
product action at three global scales, and optional-import behavior.

The adversarial q12/512 test uses labels scrambled across the full 12-site
Pauli address space. With `max_bond=8` and an 8 MiB cap it reports
`peak_workspace_bytes=842240`, `peak_explicit_workspace_bytes=465408`, and
retained bond 8 across the central cuts while a fatal `to_ndarray()` patch is
active. The q24/2048 test extends that evidence with an owned-core,
tracemalloc-bounded successful compression and a one-byte preflight gate. A
separate random-MPS test verifies a one-byte cap returns `not_feasible` without
calling its constructor.

## Scope

Concurrent holdout, methodology, residual-probe, support-swap, and generated
q24 artifact changes were not modified or staged. Repository-wide tests were
not run because the task requested focused Task 2 verification and concurrent
owned work is present outside this component.
