# MPO Task 3 Report

## Status

Complete. Implements full-support time-dependent MPO evolution with default
two-site TDVP, explicit single-site `q=1` handling, and a TeNPy 1.1.0
`TimeDependentExpMPOEvolution` comparison backend. Review remediation is
included below.

## Implementation

- Added `PreparedTDVPOperators`, `prepare_tdvp_operators`,
  `evolve_protocol_tdvp`, and `evolve_protocol_expm_mpo`.
- Static `H_initial`, `H_final`, and every temporal CD mode are compressed once.
  Midpoint TDVP assembles their scalar linear combination as local MPO blocks.
- Learned and nested-commutator `l=1` direct-CD paths retain every declared
  label and every factor mode. Resource failures return explicit
  `not_feasible` diagnostics without a dense fallback.
- The ExpMPO comparison rebuilds a provenance-preserving Pauli MPO at each
  midpoint because TeNPy requires identity-channel metadata for exponential
  MPO construction; it does not construct a dense operator.
- Added final-energy, ground-fidelity, norm, truncation, bond, timing, order,
  and resource diagnostics. Dense midpoint helpers are test-only and reject
  `q > 4`.

## Review Remediation

- Ground fidelity now uses canonical-form-aware TeNPy overlap on safe MPS
  copies; `q=1` uses TeNPy's direct finite contraction because its canonical
  form routine intentionally rejects a one-site chain. Norm drift uses the
  same physical self-overlap, never `psi.norm` bookkeeping.
- Dynamic TDVP block-sum planning derives every local source and output shape,
  including interior cross-product virtual bonds, scaling temporaries, output
  copies, conversion copies, and a safety margin before tensor conversion or
  allocation. Allocation failures return `not_feasible`. ExpMPO independently
  preflights its provenance-bounded midpoint graph before aggregation/build.
- ExpMPO remains an exact Pauli-graph comparator while TDVP consumes the
  compressed block sum. Diagnostics certify an integrator comparison only when
  every static component has a verified lossless identity certificate;
  otherwise they report `not_comparable` rather than agreement.
- Evolution diagnostics now expose static/dynamic discarded weight, action and
  error status, actual per-step truncation weights, and physical norm drift.
  The ExpMPO path captures the `TruncationError` returned by TeNPy's direct
  `evolve` call, which its time-dependent `run_evolution` path otherwise drops.
- Tests use a self-contained `q <= 4` dense oracle: test-owned Pauli matrices,
  ordering, schedules, temporal interpolation, nested-`l=1` construction, and
  Hermitian eigendecomposition midpoint propagators. Regressions cover q2
  learned-CD fidelity with reordered qubits, lossy q3 non-comparability, q4
  2 MiB preflight failure for TDVP and ExpMPO, and forced bond-1 truncation for
  both engines.

## RED Evidence

Before implementation:

```text
conda run -n torch-mps python -m unittest tests.test_agp_mpo_backend.TDVPEvolutionTests -v
```

failed with missing `prepare_tdvp_operators`, `evolve_protocol_tdvp`,
`evolve_protocol_expm_mpo`, and dense-reference helper APIs.

## Verification

```text
conda run -n torch-mps python -m unittest tests.test_agp_mpo_backend -v
conda run -n torch-mps python -m py_compile scripts/agp_mpo_backend.py tests/test_agp_mpo_backend.py
git diff --check
```

Result: all 55 focused MPO tests passed; compilation and diff checks passed.
