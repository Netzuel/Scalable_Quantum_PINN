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

## Q-Aware Candidate Diagnosis

The separately named q-aware v1 candidate completed 20 rounds plus both
temporal refinements, but it is not certification-eligible. Its fixed-probe
manifest was written after the baseline checkpoint and therefore fails the
current lifecycle contract with `missing_pretraining_lifecycle`.

The original evaluator also failed to restore the learned schedule. After that
loader defect was corrected, a round-19 diagnostic over all 69,855 generated
residual labels gave:

| Metric | Value | Required | Status |
|---|---:|---:|---|
| Generated holdout relative residual | 0.3049245 | <= 0.10 | fail |
| Fixed-active unseen relative residual | 1.3637672 | <= 1.0 | fail |
| Moving unseen relative residual | 1.3826552 | diagnostic | diagnostic |
| Fixed-null scaled leakage | 4.69132e-4 | diagnostic | diagnostic |

The candidate was therefore rejected before tensor-network evolution. The
retained training checkpoint and curriculum remain unchanged. The isolated v2
configuration fixes the evaluator, probe lifecycle, formal-probe exclusion,
q-aware resource accounting, and residual/support stratification, but it has
not been trained.

## Block-Balanced Candidate Diagnosis

The q-aware v2 curriculum was trained end to end with a block-balanced
reference-normalized objective. Each loss evaluation combined the mean
per-qubit residual with the worst 15% qubit-block tail. The run kept
`K=32768`, completed all 20 rounds, realized 67,297 generated residual strings,
and retained 3,021 final unseen terms.

| Checkpoint | Training relative residual | Holdout relative residual | Frozen-active relative residual | Gate status |
|---|---:|---:|---:|---|
| Round 20 | 6.86894e-4 | 6.84697e-4 | 1.57992 | fail |
| Temporal refinement | 1.01447e-3 | 1.07421e-3 | 1.48088 | fail |
| Adaptive refinement | 3.67005e-3 | 1.03489e-3 | 1.35409 | fail |

No checkpoint passed both the generated-holdout threshold (`0.10`) and the
frozen-active threshold (`1.0`), so no projected champion was selected. Exact
ground-state information was not used for training or checkpoint selection.

For diagnosis only, the deterministic round-20 endpoint was resampled at 257
time points with zero optimizer steps and evolved with all 32,768 learned AGP
terms:

| Resolution | Steps | MPS bond | Final energy | Energy error | Ground fidelity |
|---|---:|---:|---:|---:|---:|
| Time coarse | 24 | 64 | -116.5807631 | 93.0192369 | 3.40274e-14 |
| State coarse | 48 | 32 | -146.4896411 | 63.1103589 | 1.23538e-9 |
| Fine diagnostic | 48 | 64 | -146.4893298 | 63.1106702 | 1.23696e-9 |

The all-term MPO source-completeness and action gates passed. The fine
trajectory completed, but exceeded the configured 21,600-second resource cap
(`29,680` seconds) and was therefore ineligible for formal convergence. Its
numerical agreement with the bond-32 trajectory is close, while the 24-to-48
step energy shift is about `29.91`, far above the `0.1` tolerance. The
candidate is also substantially worse than the retained full-support result
(`E(T)=-201.8513231`, fidelity `0.2591563`). It was rejected and the retained
q156 benchmark was restored unchanged. Its implementation, candidate JSON, and
generated artifacts were removed; this Markdown diagnosis is the retained
experiment record.

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

## Full-Support Tensor-Network Dynamical Validation

The canonical comparison deploys every one of the 32,768 learned AGP terms.
The coefficients are compressed as a joint-time full-support MPO and evolved
with two-site TDVP. No coefficient threshold or ranked support truncation is
used.

| Method | Final energy | Energy error | Ground fidelity | `<Z_i>` RMSE | `<Z_i Z_{i+1}>` RMSE |
|---|---:|---:|---:|---:|---:|
| no CD | -26.2623879 | 183.3376121 | 4.00991e-37 | 0.970088 | 0.841646 |
| Kipu/DQFM l=1 | -97.7266362 | 111.8733638 | 1.41209e-16 | 0.846403 | 0.424950 |
| learned sparse AGP | -201.8513231 | 7.7486769 | 0.2591563 | 0.045929 | 0.039075 |

The numerical ladder changes one approximation axis at a time:

| Gate | Coarse/fine settings | PINN energy difference | PINN fidelity difference | Status |
|---|---|---:|---:|---|
| Timestep | 24/48 steps, MPS bond 64 | 0.0186256 | 0.000729451 | pass |
| State bond | bonds 32/64, 48 steps | 2.62134e-6 | 3.03579e-7 | pass |

The learned dynamic MPO has peak bond 102, and the fine learned trajectory
reaches the configured MPS bond cap of 64. Learned-source completeness passes,
the sampled learned-MPO action error is zero, and the maximum reported static
action errors are `2.39139e-5` for no CD and `4.40030e-5` for nested l=1,
below the configured `1e-3` threshold. All required q156 tensor-network gates
therefore pass. This validates the numerical deployment of the complete
trained output; it is not an exact q156 statevector or a proof of sufficiency
relative to Pauli strings outside the learned support.

The retained physical source is a 257-point resampling of the frozen round-20
checkpoint with zero optimization steps. Its provenance-correct export has the
same canonical physical hash as the export used by the completed tensor-network
ladder: `a1cdb1ce8c8b80121f883a1453614bcde20efd5c9c7698aad916c2155447b951`.
Relative to the previous 16-point deployment, the denser representation raises
fidelity from `0.2394617` to `0.2591563` and reduces energy error from
`8.2098541` to `7.7486769`; it changes deployment accuracy, not the trained AGP.

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
trained 32768-term output. The separate canonical ladder above now supplies
the independent timestep and state-bond partners for the all-32768-term
trajectory. Neither study establishes sufficiency relative to Pauli strings
outside the trained output.

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
| Full learned-support q156 TN validation | pass | all 32768 terms pass independent 24/48-step and bond-32/64 convergence gates |
| Full-support MPO/source fidelity | pass | source completeness passes and learned sampled action error is zero |
| Learned-output deployment support plateau | pass | ablation only: 8192-to-16384 gains are below 5%; the coarse 32768 endpoint confirms the plateau |

The defensible claim is **projected sparse AGP behavior with converged,
full-output q156 tensor-network dynamics**. The missing independent probes and
K/Q/seed sweeps still prevent a global-support reliability claim, and no q156
exact-statevector oracle exists for the full time evolution.

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

The retained canonical tensor-network outputs are:

```text
rounds/round_20/deployment_attribution/dense_257_export/mpo_validation/Models_Data/mps_physical_validation_summary.json
rounds/round_20/deployment_attribution/dense_257_export/mpo_validation/Images/physical_method_comparison_table.pdf
```

The historical support-deployment ablations remain under:

```text
rounds/round_20/mps_support_sweep/Models_Data/support_sweep_summary.json
```
