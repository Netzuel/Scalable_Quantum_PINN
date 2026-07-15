# Task 5 Report: Diagnostic Fixed-Unseen Backfill

## Implementation

- Added `--refresh-fixed-unseen-only` to `scripts/agp_holdout_feedback.py`.
- Historical refreshes load retained checkpoints, create or validate an immutable
  manifest with `certification_eligible=false` and
  `provenance=diagnostic_backfill`, then update only diagnostic summaries,
  plots, and the manifest.
- The refresh refuses to train a missing baseline or incomplete feedback run,
  refuses to reclassify a certification-eligible manifest, and checks that
  every discovered checkpoint has been evaluated.
- Historical selection excludes the union of all retained checkpoint residual
  supports. Normal feedback additions also exclude fixed unseen labels.
- Manifests carry a content hash. Fixed active certification is `not_tested`
  for diagnostic backfills with reason `historical_diagnostic_backfill`.

## q24 Evidence

Command:

```bash
conda run --no-capture-output -n torch-mps python scripts/agp_holdout_feedback.py \
  --config tests/sparse_agp_curriculum/transverse_field_spin_hubo/run_002_hamiltonian_341/q24/sweep_test/config.json \
  --refresh-fixed-unseen-only
```

Runtime: about 28 minutes of CPU evaluation.

Generated ignored artifact root:

```text
tests/sparse_agp_curriculum/transverse_field_spin_hubo/run_002_hamiltonian_341/q24/sweep_test/runs/
fixed_k_holdout_feedback_trainable_schedule_w96_l4_pau_support_swap_adaptive_temporal_refinement_v1/
agp_32768_residual_81920_add_3072_rounds_20/
```

Verified results:

- 23 checkpoints were evaluated: baseline, rounds 1 through 20, temporal
  refinement, and adaptive temporal refinement.
- The diagnostic manifest has zero fixed active labels and 4096 fixed null
  labels, with `status=insufficient_candidates`.
- Every checkpoint has an explicit finite null metric; the unavailable active
  partition is represented by the manifest's explicit insufficiency status.
- The manifest content hash is valid, all fixed labels are absent from every
  checkpoint residual support, and the fixed-unseen PDF is nonempty (18,550
  bytes).
- SHA-256 values for the 22 checkpoints within the retained q24 feedback run
  matched their pre-refresh snapshot. The refresh did not overwrite a
  checkpoint.
- The fixed-unseen certification decision is `not_tested` with reason
  `historical_diagnostic_backfill`.

## Validation

```bash
conda run -n torch-mps python -m py_compile scripts/agp_holdout_feedback.py scripts/agp_holdout_study.py
conda run -n torch-mps python -m unittest \
  tests.test_agp_residual_probes \
  tests.test_agp_support_swap \
  tests.test_agp_benchmark_layout \
  tests.test_agp_physical_validation -v
```

## Review Follow-Up

- Refresh now performs a read-only preflight before normal setup: a missing or
  incomplete historical summary, or any missing expected stage checkpoint,
  fails before manifest creation, model setup, or a training call.
- Current-schema manifests require a valid `manifest_sha256`. Eligible manifests
  require `pre_training_fixed_probe`/`true`; diagnostic backfills require
  `diagnostic_backfill`/`false`. The loader and certification gate reject every
  inconsistent combination. Legacy manifests remain `not_tested`.
- TDD regression coverage verifies early CLI refusal with unchanged checkpoint
  bytes, incomplete-history no-write behavior with normal resumability,
  checkpoint-stage preflight, artifact preservation, and fail-closed manifest
  integrity/provenance checks.
- Focused follow-up validation completed with 44 tests in 1.755 seconds under
  `torch-mps`. The q24 metric evaluation was not rerun; its existing diagnostic
  manifest was revalidated under the strengthened loader: 23 stage checkpoints
  are present, all fixed labels remain disjoint from checkpoint residual
  support, all 23 null rows are finite, the empty active partition is covered by
  manifest `status=insufficient_candidates`, and the PDF remains nonempty at
  18,550 bytes.
- Final low-review regression coverage verifies the certification gate returns
  `not_tested/invalid_fixed_unseen_manifest_hash` for a tampered current-schema
  manifest and `not_tested/legacy_fixed_unseen_manifest` for schema 1. Focused
  residual tests: 14 passed in 0.688 seconds under `torch-mps`; no q24 command
  was rerun.
