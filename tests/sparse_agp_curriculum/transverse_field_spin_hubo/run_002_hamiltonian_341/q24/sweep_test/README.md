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

For q=24, canonical dynamical validation uses the exact diagonal ground
reference and a two-resolution quimb MPS ladder. Every one of the 32,768
learned AGP terms must be deployed in the PINN row.

## Current Status

The scenario is configured for end-to-end training. Generated checkpoints,
figures, and MPS summaries remain under the ignored local `runs/` tree. Results
and certification gates will be added after the complete run and convergence
ladder finish.
