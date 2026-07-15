# Task 4 Report: General Configuration And Documentation

## Scope

Configured stable unseen diagnostics for the four retained sweep studies and
documented their certification semantics. The configuration applies to future
runs only; historical artifacts without a persisted fixed-probe manifest remain
uncertified for these gates.

## RED Evidence

Command:

```text
conda run -n torch-mps python -m unittest tests.test_agp_benchmark_layout.AGPBenchmarkLayoutTests.test_retained_sweeps_configure_stable_unseen_probes -v
```

Result before configuration: failed in all four subtests with
`KeyError: 'fixed_unseen_probes'`.

## GREEN Evidence

The new layout test passed after adding the config block:

```text
Ran 1 test in 0.001s
OK
```

The complete owned layout test passed after accommodating the existing
`tests/test_agp_mpo_backend.py` file in the test-tree allowlist:

```text
conda run -n torch-mps python -m unittest tests.test_agp_benchmark_layout -v
Ran 19 tests in 0.061s
OK
```

The plan's compile command passed:

```text
conda run -n torch-mps python -m py_compile scripts/agp_residual_probes.py scripts/agp_holdout_feedback.py scripts/agp_holdout_study.py
```

All four retained JSON configs also parsed successfully.

## Changes

- Added the compatible `holdout_feedback.fixed_unseen_probes` block with
  `enabled=true`, 4096 active terms, 4096 null terms, threshold `1e-12`, seed
  11, and candidate multiplier 8.
- Added layout coverage for q15, q20, q156, and q24 spin-HUBO.
- Documented moving-unseen undefined status, fixed-active stable relative
  gating, fixed-null absolute leakage, `insufficient_candidates`, and the
  future-run-only/historical-artifact boundary.
- Kept the optional parser-level candidate cap out of the general configs; the
  current parser already supports compatible cap aliases when a resource cap is
  needed.

## Commit

`Configure stable unseen diagnostics`

The commit contains this report and the owned Task 4 changes only. Pre-existing
q24/MPS work, MPO work, and `pyproject.toml` changes remain unstaged.

## Review Fixes

The normal runner now checks the target feedback run root before creating a
fixed-probe manifest. If fixed probes are enabled and the root already contains
a summary, round/refinement directory, checkpoint, or feedback spectrum without
`fixed_unseen_probe_labels.json`, it fails closed with an error requiring a new
run root. An explicit diagnostics-only refresh remains the future path for
backfilling historical metrics and is described as certification-ineligible.
Existing manifests still support valid resumes, and empty roots still support
new runs. Disabled legacy fixed-probe configurations remain compatible.

The retained layout regression now requires `candidate_multiplier == 8` rather
than only checking that it is positive. The MPO files from `dd90286` were not
modified or removed.

### Review-Fix RED

The new historical-run regression initially failed during import because
`assert_fixed_unseen_manifest_lifecycle` did not yet exist.

### Review-Fix GREEN

```text
conda run -n torch-mps python -m unittest tests.test_agp_residual_probes tests.test_agp_benchmark_layout tests.test_agp_support_swap -v
Ran 53 tests in 1.589s
OK
```

```text
conda run -n torch-mps python -m py_compile scripts/agp_residual_probes.py scripts/agp_holdout_feedback.py scripts/agp_holdout_study.py tests/test_agp_benchmark_layout.py tests/test_agp_support_swap.py
```

Both commands passed.
