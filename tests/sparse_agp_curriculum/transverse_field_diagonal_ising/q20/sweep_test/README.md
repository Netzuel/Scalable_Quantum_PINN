# Q20 Fixed-K Ising Holdout-Feedback Study

This folder is the 20-qubit continuation of the accepted q15
`TransverseIsingDriverProblem` benchmark. It uses the same Hamiltonian family,
network, learned schedule, SiLU-to-PAU transfer, fixed-K support swaps,
calibration losses, and temporal-refinement stages. It is not a molecular
hydrogen benchmark.

## Hamiltonian

```text
H_initial = -sum_i X_i
H_final   = -sum_i h_i Z_i - sum_i J_i Z_i Z_(i+1)
H_AD      = (1 - lambda) H_initial + lambda H_final
```

The sparse Pauli decomposition is indexed by
`Hamiltonians_to_use/pauli_decompositions/index.json` under
`TransverseIsingDriverProblem_20_qubits_1_0`.

## Retained Configuration

```text
q                         = 20
K                         = 32768 fixed AGP outputs
Q_requested               = 81920 residual holdout labels
feedback iterations       = 20
residual additions/round  = 3072
support swaps/round       = 256 from round 2
baseline                  = width-96, four-layer SiLU
feedback network          = width-96, four-layer PAU
```

`Q_requested` is a ceiling. In the completed run, sparse commutator generation
provided all 81,920 requested residual strings, so `Q_effective = 81,920` and
the per-round addition remained 3,072.

K stays fixed at 32,768. Hard residual-derived strings replace weak AGP strings
through support swaps rather than increasing the output dimension.

## Training

From the repository root:

```bash
conda run --no-capture-output -n torch-mps python scripts/agp_holdout_feedback.py \
  --config tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q20/sweep_test/config.json
```

The command trains or reuses the SiLU baseline, runs twenty PAU feedback
rounds, then executes uniform and adaptive temporal refinement on the fixed
round-20 support.

Expected generated layout:

```text
runs/
  baselines/agp_32768/
  fixed_k_holdout_feedback_trainable_schedule_w96_l4_pau_support_swap_adaptive_temporal_refinement_v1/
    agp_32768_residual_81920_add_3072_rounds_20/
      rounds/round_01/ ... rounds/round_20/
      temporal_refinement/
      adaptive_temporal_refinement/
      Models_Data/holdout_feedback_summary_residual_81920.json
```

Generated runs, checkpoints, and plots are intentionally ignored by Git.

## Latest Completed Run

The 20-round run and both temporal-refinement stages completed successfully.

```text
full Pauli basis                    = 1099511627776
K / 4**q                            = 2.980232238770e-08
round-20 training relative residual = 1.085813e-03
round-20 holdout relative residual  = 5.519648e-02
round-20 unseen absolute residual   = 3.638261e-06
round-20 unseen relative residual   = not defined (zero reference)
adaptive training relative residual = 2.029059e-03
adaptive holdout relative residual  = 5.687243e-02
```

Certification-gate status for this run:

| Gate | Status | Evidence |
| --- | --- | --- |
| Training relative residual | pass | adaptive value below `0.1` |
| Holdout relative residual | pass | adaptive value below `0.1` |
| Unseen relative residual | not tested | zero reference makes the ratio undefined |
| Fixed probe gate/watch/test | not tested | no fixed-probe run |
| K and Q plateaus | not tested | no sweep |
| Support stability across K/seeds | not tested | single K and seed |
| Proposal exhaustion | not tested | 256 support swaps remained active in round 20 |
| Prune and retest | not tested | no pruning run |
| Coefficient regularity | not tested | no regularity audit |
| Physical validation | pass | full-support q20 MPS ladder converged for all three protocols |

The justified claim is therefore a completed projected sparse AGP experiment,
not a certified AGP support.

## Physical Validation

The exact diagonal-Ising target is available at q20, but the project validation
backend policy uses exact statevectors only through q15 and tensor networks for
`q > 15`. The old top-term q20 statevector configuration is retained only for
provenance and is not the canonical validator.

Physical validation is not part of the loss. The completed canonical MPS
validation targets the `residual_81920` adaptive-temporal checkpoint and keeps
all 32,768 learned AGP terms. Terms sharing the same occupied-qubit support are
combined before local exponentiation; no coefficient-ranked pruning or
numerical threshold was used.

The exact target is `E0=-26.0` with unique ground bitstring `00...0`. The
fine-resolution results are:

| Method | Final energy | Energy error | Ground fidelity |
| --- | ---: | ---: | ---: |
| no CD | -3.2390823 | 22.7609177 | 1.95723e-05 |
| nested commutator l=1 | -12.2268621 | 13.7731379 | 0.00806762 |
| learned sparse AGP | -25.6453873 | 0.3546127 | 0.93648534 |

The 24-step/bond-32/cutoff-`1e-9` and
48-step/bond-64/cutoff-`1e-10` ladders agree within the configured tolerances.
For the PINN row, the energy delta is `0.00678613` and the fidelity delta is
`0.00212781`; the fine run reached peak bond 57 below its cap of 64. The
machine-readable evidence and comparison PDF are under the retained
checkpoint's `mps_validation/` directory.

## Certification Discipline

Completing twenty rounds does not certify the AGP. Interpret the final artifacts
using `AGP_CERTIFICATION_CRITERIA.md` and mark every gate as `pass`, `fail`, or
`not tested`, including:

- training, holdout, and unseen residuals;
- fixed probe-gate, probe-watch, and probe-test residuals;
- K- and Q-sweep plateaus;
- support stability across seeds;
- prune-and-retest;
- physical validation.

The maximum justified claim remains a projected sparse AGP unless all required
gates are explicitly evaluated and passed.
