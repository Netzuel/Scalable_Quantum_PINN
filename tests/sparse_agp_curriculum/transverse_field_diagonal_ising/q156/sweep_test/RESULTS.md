# q156 Run Results

## Run Contract

The clean end-to-end run completed with:

```text
q = 156
K = 32768
requested Q_max = 81920
realized Q = 69855 distinct generated residual strings
i = 20 feedback rounds
initial training residual terms = 4096
added per round = 3072
final training residual terms = 65536
final unseen residual terms = 4319
K / 4^q = 3.927274772238e-90
```

The realized residual pool was smaller than the requested cap, but it was large
enough to complete all 20 rounds without reusing the final unseen tail.

## Residual Results

| Checkpoint | Training relative residual | Holdout relative residual | Unseen absolute residual |
|---|---:|---:|---:|
| Round 6 | 0.0141992 | 0.0320740 | 1.01717 |
| Round 20 | 0.00343378 | 0.0561529 | 9.64943e-6 |
| Temporal refinement | 0.00355959 | 0.0566945 | 9.07505e-9 |
| Adaptive temporal refinement | 0.00887972 | 0.0567062 | 3.17420e-9 |

Round 20 is the balanced curriculum endpoint: it passes the configured
training and holdout gates, includes all 20 fresh feedback batches, and retains
4319 terms outside training. The temporal continuations reduce the absolute
residual on that final tail but slightly worsen the normalized training and
holdout metrics.

The unseen relative quotient is undefined for every reported final checkpoint
because the AGP=0 reference residual on the final unseen tail is exactly zero.
The run therefore reports the absolute unseen residual and per-term residual
instead of dividing by zero. At round 20 the unseen residual per term is
`2.23418e-9`.

## Exact Final-Hamiltonian Reference

The final diagonal Ising problem has:

```text
exact ground energy = -209.6
ground-state degeneracy = 1
ground bitstring = 156 zeroes
```

This reference validates the final classical optimization problem. It does not
provide the evolved q156 quantum state. The final dynamics were therefore
computed with a bounded-bond MPS approximation rather than a dense statevector.

## MPS Dynamical Validation

The retained comparison uses the same 2048 learned terms at both numerical
resolutions. This deployment support retains `0.9555395` of the trained
coefficient RMS norm.

| Method | Final energy | Energy error | Ground fidelity | `<Z_i>` RMSE | `<Z_i Z_{i+1}>` RMSE |
|---|---:|---:|---:|---:|---:|
| no CD | -26.2569071 | 183.343093 | 3.85877e-37 | 0.970099 | 0.841677 |
| Kipu/DQFM l=1 | -97.7452309 | 111.854769 | 1.42713e-16 | 0.846395 | 0.424831 |
| learned sparse AGP | -188.153516 | 21.4464841 | 0.0207493 | 0.146486 | 0.0926501 |

This historical reduced-support ablation compares 24 steps, bond 32, cutoff `1e-9` against
48 steps, bond 64, cutoff `1e-10`. Successive-resolution differences are:

| Method | Energy difference | Fidelity difference | Gate |
|---|---:|---:|---|
| no CD | 0.0225031 | 1.13717e-38 | pass |
| Kipu/DQFM l=1 | 0.00915165 | 1.00548e-18 | pass |
| learned sparse AGP | 0.0279652 | 5.40831e-5 | pass |

Peak fine-resolution MPS bonds are 4, 6, and 23 respectively. The reduced-
support trajectory is numerically converged, but it is not canonical physical
validation of the trained 32768-term PINN operator. It remains a deployment
ablation, not an exact q156 statevector or a global-support sufficiency proof.

## Learned-Support Deployment Diagnostic

The same trained round-20 checkpoint was evolved with progressively larger
coefficient-RMS-ranked subsets. All support points below use 24 time steps,
bond 32, cutoff `1e-9`, and the same learned schedule and AGP coefficients.

| Learned terms | Retained RMS norm | Final energy | Energy error | Ground fidelity | `<Z_i>` RMSE | `<Z_i Z_{i+1}>` RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 0.9555395 | -188.125551 | 21.4744493 | 0.0206952 | 0.146810 | 0.0927282 |
| 4096 | 0.9926054 | -198.459098 | 11.1409023 | 0.1391134 | 0.0683913 | 0.0528208 |
| 8192 | 0.9998588 | -201.309670 | 8.29032994 | 0.2359579 | 0.0501650 | 0.0410057 |
| 16384 | 0.9999999 | -201.381192 | 8.21880754 | 0.2397062 | 0.0498045 | 0.0406469 |
| 32768 | 1.0000000 | -201.381402 | 8.21859765 | 0.2397203 | 0.0498032 | 0.0406462 |

Increasing the deployment from 2048 to 8192 terms is physically material:
the coarse fidelity rises by a factor of `11.40`, and the energy error falls by
`61.4%`. The 8192-to-16384 improvement is only `1.59%` in fidelity and `0.86%`
in energy error. The 16384-to-32768 change is negligible. The coefficient-
ranked deployment ablation therefore reaches the configured `5%` within-output
dynamical plateau at 8192 terms.

The 8192-term endpoint was independently rerun with 48 steps, bond 64, and
cutoff `1e-10`. Its fine result is:

| Final energy | Energy error | Ground fidelity | `<Z_i>` RMSE | `<Z_i Z_{i+1}>` RMSE | Peak bond |
|---:|---:|---:|---:|---:|---:|
| -201.326646 | 8.27335359 | 0.2361107 | 0.0499150 | 0.0409758 | 25 |

The coarse/fine differences are `0.0169764` in energy and `1.52810e-4` in
fidelity, so the 8192-term ablation passes the existing MPS convergence
tolerances. This validates only the truncation-sensitivity study inside the
trained 32768-term output. The all-32768-term trajectory has one coarse result
but no fine-resolution partner, so canonical full-model physical validation is
`not tested`. The study also does not establish sufficiency relative to Pauli
strings outside the trained output.

## Certification

| Gate | Status | Evidence |
|---|---|---|
| Training residual | pass | Round 20 relative residual 0.00343378 |
| Generated holdout residual | pass | Round 20 relative residual 0.0561529 |
| Fresh final unseen tail | pass | 4319 terms were never added to training |
| Unseen relative residual | not tested | AGP=0 denominator is zero; quotient is undefined |
| Unseen absolute residual | diagnostic | Round 20 total 9.64943e-6; per term 2.23418e-9 |
| Twenty-round curriculum | pass | All rounds completed with 3072 fresh terms per round |
| Fixed-K support swaps | pass | 256 replacements per round from rounds 2 through 20 |
| K-sweep plateau | not tested | Only K=32768 was trained |
| Q-sweep plateau | not tested | Only one generated residual pool was evaluated |
| Seed/support stability | not tested | No multi-seed overlap study was run |
| Fixed external probe basis | not tested | No independent probe family was evaluated |
| q156 dense-statevector validation | not tested | A 2^156 statevector is unavailable |
| Reduced-support q156 MPS ablation | pass | the 2048- and 8192-term deployments pass their 24-to-48-step convergence gates |
| Full learned-support q156 MPS validation | not tested | all 32768 terms were run only at the coarse resolution |
| Learned-output deployment support plateau | pass | ablation only: 8192-to-16384 gains are below 5%; the coarse 32768 endpoint confirms the plateau |

The defensible claim is **promising projected sparse AGP behavior with
converged reduced-support q156 MPS ablations**. Canonical physical validation
of the complete learned PINN operator is not tested, and the missing
independent probes and K/Q/seed sweeps also prevent a global-support reliability
claim.

## Artifacts

The complete ignored run is stored under:

```text
runs/fixed_k_holdout_feedback_trainable_schedule_w96_l4_pau_support_swap_adaptive_temporal_refinement_v1/
  agp_32768_residual_69855_add_3072_rounds_20/
```

Canonical run-level PDFs:

```text
Images/hcd_connection_summary.pdf
Images/hcd_coefficient_support_map.pdf
Images/physical_method_comparison_table.pdf
Images/holdout_feedback_added_terms.pdf
Images/holdout_feedback_relative_residuals.pdf
Images/holdout_feedback_residual_spectrum.pdf
Images/holdout_feedback_seen_unseen_residuals.pdf
```

The machine-readable curriculum summary is:

```text
Models_Data/holdout_feedback_summary_residual_69855.json
```

The retained tensor-network outputs are:

```text
rounds/round_20/mps_validation/Models_Data/mps_physical_validation_summary.json
rounds/round_20/mps_validation/Images/physical_method_comparison_table.pdf
rounds/round_20/mps_support_sweep/Models_Data/support_sweep_summary.json
```
