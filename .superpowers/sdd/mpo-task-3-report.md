# MPO Task 3 Report

## Status

Complete. Implements full-support time-dependent MPO evolution with default
two-site TDVP, explicit single-site `q=1` handling, and a TeNPy 1.1.0
`TimeDependentExpMPOEvolution` comparison backend.

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

Result: all 51 focused MPO tests passed; compilation and diff checks passed.
