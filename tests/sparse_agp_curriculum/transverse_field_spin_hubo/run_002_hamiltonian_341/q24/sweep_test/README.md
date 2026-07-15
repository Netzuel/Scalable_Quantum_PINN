# Q24 Spin-HUBO Fixed-K Holdout-Feedback Study

## Hamiltonian

```text
H_initial = -sum_i X_i
H_final   = sum_S c_S product_(i in S) Z_i
H_AD      = (1 - lambda) H_initial + lambda H_final
```

The final objective has 24 one-body, 72 two-body, and 120 three-body terms.
The tracked `spin_pm1` coefficients map directly to Pauli-Z eigenvalues.

## Exact Reference

An exhaustive Walsh-Hadamard evaluation of all `2**24` assignments gives:

```text
ground energy    = -28.101465646808336
ground bitstring = 000011100010110000010010
degeneracy       = 1
```

This exact reference is validation-only and does not enter training.

## Retained Configuration

The training methodology and budget match the retained q20 benchmark:

```text
q                         = 24
K                         = 32768 fixed AGP outputs
Q_requested               = 81920 residual holdout labels
feedback iterations       = 20
residual additions/round  = 3072
support swaps/round       = 256 from round 2
baseline                  = width-96, four-layer SiLU
feedback network          = width-96, four-layer PAU
```

The run also includes the retained uniform and adaptive temporal-refinement
stages. Training uses projected residual quantities and regularizers only.

## Commands

From the repository root:

```bash
conda run -n torch-mps python scripts/agp_restart.py \
  --config tests/sparse_agp_curriculum/transverse_field_spin_hubo/run_002_hamiltonian_341/q24/sweep_test/config.json

conda run --no-capture-output -n torch-mps python scripts/agp_holdout_feedback.py \
  --config tests/sparse_agp_curriculum/transverse_field_spin_hubo/run_002_hamiltonian_341/q24/sweep_test/config.json

conda run --no-capture-output -n torch-mps python tests/sparse_agp_curriculum/scripts/agp_mps_validation.py \
  --config tests/sparse_agp_curriculum/transverse_field_spin_hubo/run_002_hamiltonian_341/q24/sweep_test/config.json
```

For q=24, canonical dynamical validation requires the exact diagonal ground
reference and a two-resolution quimb MPS ladder. Every one of the 32,768
learned AGP terms must be deployed in the PINN row. A reduced-term deployment
is an ablation and cannot satisfy the physical-validation gate.

## Current Status

The end-to-end training run completed. Round 20 used all requested residual
labels and retained the fixed `K=32768` AGP support:

```text
round-20 training relative residual     = 0.00546703
round-20 holdout relative residual      = 0.09397036
round-20 unseen absolute residual       = 0.14483541
round-20 unseen relative residual       = not tested (zero reference)
adaptive training relative residual     = 0.00715475
adaptive holdout relative residual      = 0.12782112
```

The adaptive checkpoint passes the training-residual target but fails the
configured `0.1` holdout target. Support swaps were still active in round 20.
Fixed probes, K/Q plateaus, cross-seed stability, pruning, and proposal
exhaustion were not tested. The claim level is therefore **projected sparse AGP
experiment**, not a certified sparse AGP.

## Full-Support MPS Cost Diagnostic

The full learned support contains 15,025 occupied-support groups; 11,025 are
single-Pauli groups. One symmetric time step at bond 8 still requires 30,058
group applications and took 2,502.7 seconds on the reference machine. The
configured 24/48-step, bond-32/64 ladder was therefore not completed.

The following equal-resolution numbers use one time step, bond 8, cutoff
`1e-6`, and all 32,768 learned terms. They are a computational diagnostic only:

| Method | Final energy | Energy error | Ground fidelity |
| --- | ---: | ---: | ---: |
| No CD | -4.879474 | 23.221991 | 2.421700e-6 |
| Nested commutator l=1 | -6.508609 | 21.592857 | 1.294986e-5 |
| Learned sparse AGP | -3.856186 | 24.245280 | 1.235953e-11 |

These values do not establish that l=1 outperforms the learned AGP, because
time-step, bond-dimension, and cutoff convergence were not demonstrated. The
physical-validation certification gate is `not tested`.

Generated checkpoints and diagnostics remain in the ignored local `runs/`
tree. The consolidated diagnostic is under the adaptive checkpoint at
`mps_validation_diagnostic/`, including
`Images/physical_method_comparison_table.pdf` and the corresponding JSON.

## Certification Summary

```text
q, 4**q, K, K/4**q           = 24, 281474976710656, 32768, 1.164153218269e-10
training residual gate        = pass
holdout residual gate         = fail (final adaptive checkpoint)
unseen relative residual gate = not tested (zero reference)
fixed probe gates             = not tested
K/Q plateau gates             = not tested
support stability gates       = not tested
proposal exhaustion           = fail (swaps active in round 20)
prune-and-retest               = not tested
physical MPS validation       = not tested (no convergence ladder)
claim level                   = projected sparse AGP experiment
```
