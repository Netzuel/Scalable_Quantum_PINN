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
  --config tests/sparse_agp_curriculum/transverse_field_spin_hubo/run_002_hamiltonian_341/q24/sweep_test/config.json \
  --preflight-only

conda run --no-capture-output -n torch-mps python tests/sparse_agp_curriculum/scripts/agp_mps_validation.py \
  --config tests/sparse_agp_curriculum/transverse_field_spin_hubo/run_002_hamiltonian_341/q24/sweep_test/config.json
```

For q=24, canonical dynamical validation requires the exact diagonal ground
reference and independent TeNPy TDVP timestep/state ladders. Every one of the 32,768
learned AGP terms must be deployed in the PINN row. A reduced-term deployment
is an ablation and cannot satisfy the physical-validation gate. The isolated
one-step preflight establishes operator accuracy before the named `8 -> 12`
timestep pair and bond `16 -> 32` state pair are considered.

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

## Full-Support TDVP Preflight

The TDVP backend feeds exact Pauli-coordinate provenance into a positioned,
workspace-bounded time-Pauli TT-SVD while retaining all 32,768 learned terms.
The spectral qubit order is used and the finite time axis is inserted at chain
position 12. A measured window ladder established that MPO bond 1024 cannot
share multiple q24 midpoint operators under the configured error tolerance:

```text
window samples   coefficient bound   workspace      result
8                2.6288e-1           14.68 GB       fail
2                1.6245e-2            4.37 GB       fail
1                3.2261e-7            2.61 GB       pass
```

The accepted one-slice preflight also passes the exact sparse full-source
action gate:

```text
learned terms accounted                = 32768 / 32768
full-support SHA-256                    = c1eda30e...09136f4
MPO action-error upper bound            = 0.00448206
configured action-error maximum         = 0.01
operator build time                     = 496.14 s
norm drift                              = 3.55e-15
```

The following one-step values remain a computational diagnostic only:

| Method | Final energy | Energy error | Ground fidelity |
| --- | ---: | ---: | ---: |
| No CD | -4.437531 | 23.663935 | 2.117314e-6 |
| Nested commutator l=1 | -6.729172 | 21.372293 | 9.195138e-6 |
| Learned sparse AGP | -1.425268 | 26.676197 | 1.053858e-6 |

These values do not compare physical performance because one midpoint is
intentionally too coarse. Unlike the superseded temporal-mode block-sum result,
the represented operator is controlled; physical promotion now depends on the
independent multi-resolution dynamics gates.

Generated checkpoints and diagnostics remain in the ignored local `runs/`
tree. The diagnostic is isolated under the adaptive checkpoint at
`mpo_validation/preflight/`. The canonical `mpo_validation/` status contains
the exact ground reference, suppresses diagnostic method values, and records
physical validation as `not tested` until that ladder completes.

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
physical TDVP validation      = not tested (independent dynamics ladder pending)
claim level                   = projected sparse AGP experiment
```
