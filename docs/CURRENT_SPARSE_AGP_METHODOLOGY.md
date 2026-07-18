# Current Sparse AGP Methodology

This document records the current general methodology used in this repository
to learn sparse adiabatic gauge potentials (AGPs) for arbitrary qubit counts
`q`, without dense `2**q x 2**q` matrices and without enumerating the full
`4**q` Pauli basis for large systems.

The current retained q15 study is a benchmark instance of this methodology, not
the methodology itself. Its role is to provide a physically checkable
above-exact-output testbed where final energy, fidelity, and simple observables
can be compared against known ground-truth diagnostics after training. For
larger `q`, the same training pipeline is used with different configuration
values, but full-basis or full-state validation may no longer be accessible.

For `q > 8`, the full Pauli basis is not treated as a computational object in
this repository. The trainable AGP support size `K`, the residual holdout pool
size `Q`, and the number of curriculum iterations `i` are explicit research
choices recorded in the test configuration.

## Goal

The goal is to learn an approximate sparse adiabatic gauge potential

```text
A_lambda(t) = sum_{P in S_AGP} C_P(t) P
```

that improves physical counterdiabatic evolution without using final-state
ground-truth observables during training.

When exact or approximate physical validation is feasible, the post-training
diagnostic asks whether the learned AGP improves quantities such as:

- final ground-state energy error;
- final ground-state fidelity;
- final `<Z_i>` expectation RMSE;
- final nearest-neighbor `<Z_i Z_{i+1}>` RMSE.

These observables are used only after training. They are not targets in the
PINN loss.

## Hamiltonian Path

The general Hamiltonian path is:

```text
H_AD(lambda) = (1 - lambda) H_initial + lambda H_final
A_lambda(t) = sum_{P in S_AGP} C_P(t) P
```

`H_initial`, `H_final`, the schedule parameterization, the qubit count `q`, and
the sparse Pauli decompositions are supplied by the active test configuration.

The current retained q15 benchmark instance uses a transverse-driver to
diagonal Ising-problem path:

```text
H_initial = - sum_i X_i
H_final   = sum_i h_i Z_i + sum_i J_i Z_i Z_{i+1}
H_AD(lambda) = (1 - lambda) H_initial + lambda H_final
T = 1
q = 15
```

For this q15 instance, the final Hamiltonian is diagonal, so the exact final
ground energy and ground-state observables are accessible. That accessibility
is diagnostic-only and is not assumed in the general methodology. The training
loop does not use the final ground-state energy, final fidelity, or exact final
observables.

## Sparse AGP Support

The support budget is configured per study:

```text
K = |S_AGP| trainable AGP coefficient functions
Q = generated residual holdout terms
i = holdout-feedback curriculum iterations
```

The current retained q15 benchmark instance uses:

```text
K = 32768 trainable AGP outputs
Q = 65536 generated residual holdout terms
i = 15 holdout-feedback curriculum iterations
```

The initial AGP support is selected from a bounded nested-commutator Krylov pool
seeded by the order-1 commutator direction. The method never enumerates the full
`4**q` basis for large `q`.

The active AGP support has fixed cardinality during holdout-feedback. The
curriculum does not grow the number of neural outputs; instead, it adds hard
residual equations to the training residual basis and, in the retained
support-swap benchmark, replaces weak AGP strings with hard residual-derived
candidates while keeping `K` unchanged.

## Neural Architecture

The current retained methodology uses a quadratic/QRes coefficient network with
a trainable Padé activation unit (PAU):

```text
input = normalized time tau
outputs = K AGP coefficient functions
layer_type = quadratic
hidden width = 96
hidden layers = 4
activation = PAU
```

Each quadratic layer has a linear path plus a multiplicative branch:

```text
y = W_linear x + (W_left x) * (W_right x)
```

The PAU nonlinearity has trainable numerator and denominator coefficients in the
form `P(x) / (1 + |Q(x)|)`. The numerator is initialized with a SiLU-like
polynomial, and the denominator is initialized with small non-zero coefficients.

The PAU feedback run can warm-start from a width-96 SiLU baseline checkpoint
before entering the holdout-feedback curriculum. The `baseline_neural` block in
the active configuration makes that warm-start explicit and reproducible from a
cleaned test folder: the baseline checkpoint is trained with SiLU when missing,
then the feedback rounds load the compatible body weights into the PAU model
while training the PAU parameters.

The output layer is linear in the final hidden representation; no activation is
applied to the emitted AGP coefficients. The learned global scale and soft Pauli
gates are applied after the network output.

## Trainable Scheduling Function

The current retained methodology trains the counterdiabatic schedule jointly
with the AGP coefficient network and calibration variables. The parameterization
follows the constrained-envelope idea used in Section 2.1 of arXiv:2604.18506:
a fixed smooth reference schedule plus a bounded neural correction that vanishes
at the boundaries.

For the current q15 benchmark instance:

```text
lambda_0(t) = sin^2(pi t / 2T)
tau = t / T
lambda(t) = lambda_0(t) + tau^2 (1 - tau)^2 A_sched tanh(u_theta(tau))
A_sched = 2.4
u_theta network = MLP(width=32, hidden_layers=2, activation=tanh)
```

The envelope enforces `lambda(0)=0`, `lambda(T)=1`, and zero endpoint
derivatives by construction. The `tanh` bound keeps the correction controlled;
the schedule loss also penalizes non-monotone segments and large corrections:

```text
L_schedule = 10.0 L_monotonic + 0.0001 L_correction_l2
```

The schedule is trained from the baseline stage through every curriculum round
using only the residual objective and schedule regularizers. It does not use
final ground-state energy, final fidelity, or exact final observables.

## Loss Used During Training

The current sparse PINN loss is based on the Euler-Lagrange residual

```text
R(A) = [i dH_AD/dlambda - [A_lambda, H_AD], H_AD].
```

Training minimizes the squared norm of this residual in a selected Pauli
coordinate residual basis. The current setup also trains:

- a global AGP scale;
- soft Pauli gates that select an active subset of the learned support;
- the bounded schedule correction described above.

Those calibration variables are trained jointly from the baseline stage through
each curriculum round using only the projected residual objective and regularizers.

## Holdout-Feedback Curriculum

Each round:

1. trains or fine-tunes the current PINN on the active residual basis;
2. evaluates a larger generated residual holdout basis;
3. ranks unseen residual equations by RMS residual;
4. adds the hardest unseen residual equations to the training residual basis;
5. ranks AGP strings by `RMS_tau(dot(lambda) C_P(tau))`;
6. replaces 256 weak AGP strings with hard residual-derived candidates;
7. remaps retained output rows and gate logits by Pauli label;
8. fine-tunes the resulting fixed-`K` AGP coefficient functions and calibration
   variables.

Candidate replacements are generated from the largest holdout residual
directions plus their one-commutator closure with the Hamiltonian support. This
gives the method exploratory capability without increasing the output budget or
using final-state ground truth during training.

For the current retained q15 benchmark instance:

```text
add_residual_terms_per_iteration = 3072
support_swap_terms_per_iteration = 256
final train residual equations = 50176
final generated holdout pool = 65536
```

## Stable Unseen Diagnostics

The retained sweep configurations enable an immutable active/null probe
partition for future runs:

```text
active_terms = 4096
null_terms = 4096
reference_rms_threshold = 1e-12
seed = 11
candidate_multiplier = 8
```

The helper requests a bounded candidate tail after reserving the moving
holdout/feedback universe, excludes training and certification-probe labels,
and persists the selected labels and reference RMS values before round 1. The
same labels are evaluated against every later checkpoint. For a small full-basis
case, the requested counts are capped by the available disjoint labels and the
manifest records `insufficient_candidates` with the realized counts; it does
not silently reuse an overlapping basis. The optional parser-level candidate
cap remains available for resource-constrained runs, but is not required in the
general retained configuration block.

These diagnostics have distinct meanings:

- The **moving unseen quotient** is a curriculum diagnostic. Its denominator
  can be zero, so the quotient may be undefined and must carry an explicit
  status rather than an epsilon-clamped value.
- The **fixed active quotient** is the stable relative unseen gate. It is
  evaluated only on fixed probes with finite nonzero reference RMS.
- **Fixed null leakage** is the absolute AGP-induced residual in zero-reference
  directions. It is reported per term (and, when defined, in its configured
  scaled form); it is not treated as a relative quotient.

The four configuration blocks define defaults for future runs only. They do not
retroactively certify completed artifacts. A normal-run fixed-probe manifest is
eligible only when established before training, and a valid resume reuses that
immutable manifest. The diagnostics-only `--refresh-fixed-unseen-only` mode
can backfill one immutable historical probe manifest and evaluate retained
checkpoints without training or changing checkpoint files. Such a manifest is
explicitly marked `certification_eligible=false` with
`provenance=diagnostic_backfill`, so its fixed active and null metrics remain
`not tested` for certification with reason `historical_diagnostic_backfill`.
The current manifest schema requires a valid `manifest_sha256`: normal
certification-eligible manifests additionally require
`provenance=pre_training_fixed_probe`, while diagnostics require
`certification_eligible=false` and `provenance=diagnostic_backfill`. Missing,
invalid, or inconsistent provenance/hash data fails the certification gate
closed; legacy historical manifests remain explicitly `not tested`.

Diagnostic refresh is preflighted against a complete historical summary and
every expected stage checkpoint before it writes a manifest. It refuses an
incomplete history without creating diagnostic artifacts, never trains or
overwrites checkpoints, and preserves existing round-level artifacts while it
regenerates only diagnostic summaries and plots.

## Post-Curriculum Temporal Refinement

After the fixed-K support-swap curriculum finishes, the retained methodology can
run self-supervised continuation stages on the final-round AGP support and
residual basis.

The first continuation uses a denser uniform time-collocation grid:

```text
epochs = 2500
num_points = 64
lr = 3e-6
optimizer = AdamW
```

The retained benchmark then runs an adaptive temporal-refinement continuation.
It scores a dense time grid using only projected Euler-Lagrange residual
quantities, concentrates the final collocation grid near harder time regions,
and keeps the AGP support fixed:

```text
epochs = 1500
dense_points = 257
num_points = 64
lr = 1.5e-6
optimizer = AdamW
difficulty = residual_x_cd_norm
weight_power = 0.5
min_weight = 0.25
max_weight = 4.0
```

Both refinements continue to train the PAU network, learned schedule, global
scale, and soft gates using only the projected residual objective and
regularizers. They do not use final ground-state energy, final fidelity, or
exact final observables.

## Exact Diagonal-Ising Final-State Oracle

For the `TransverseIsingDriverProblem` validation family, `H_final` contains
only `Z_i` and nearest-neighbor `Z_i Z_{i+1}` terms on an open chain. Its exact
ground energy and computational-basis ground bitstrings are therefore available
without constructing a dense Hamiltonian or statevector.

The retained solver under `scripts/numerical_solver/` uses exact path dynamic
programming in `O(q)` time. It cross-checks the Ising-to-QUBO mapping and the
closed-form ferromagnetic result. For `q=2..20`, it additionally enumerates all
`2**q` bitstrings as an independent oracle. The current family has

```text
ground bitstring = 00...0
ground-state degeneracy = 1
E0(q) = 1 - 1.35 q.
```

Curated results through `q=156`, including the first distinct excitation and
spectral gap, are tracked under
`tests/sparse_agp_curriculum/ground_truth/diagonal_ising/`. These final-state
targets support energy and bitstring-probability validation at large `q`, but
they do not make the full time-dependent statevector or exact AGP accessible.

For the current retained q15 benchmark instance, the accepted
adaptive-refinement residual diagnostics were:

```text
training relative residual = 0.003279208
holdout relative residual  = 0.055878535
absolute unseen residual   = 0.000301352
```

The reported unseen relative residual is not meaningful in this run because the
AGP=0 reference residual on the sampled unseen batch is zero. The absolute
unseen residual is still stored, but the quotient is recorded as invalid.

## End-To-End Pipeline

The general pipeline is controlled by one test-local configuration file:

```text
config = tests/<case>/sweep_test/config.json
training entrypoint = scripts/agp_holdout_feedback.py
physical validation entrypoint = tests/sparse_agp_curriculum/scripts/agp_physical_validation.py
optional baseline entrypoint = scripts/agp_baseline_train.py
optional cleanup entrypoint = scripts/agp_restart.py
```

The current retained q15 benchmark instance is:

```text
config = tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q15/sweep_test/config.json
current run root =
  tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q15/sweep_test/runs/
  fixed_k_holdout_feedback_trainable_schedule_w96_l4_pau_support_swap_adaptive_temporal_refinement_v1/
  agp_32768_residual_65536_add_3072_rounds_15/
```

From a cleaned `tests/<case>/sweep_test/` folder, the full retained pipeline is:

1. Build or refresh the sparse Hamiltonian decomposition and index for the
   configured problem.

```bash
conda run -n torch-mps python tests/sparse_agp_curriculum/scripts/build_driver_problem_hamiltonian.py --update-index
```

2. Clean generated artifacts without recreating top-level `Images/` or
   `Models_Data/` folders in the test folder.

```bash
conda run -n torch-mps python scripts/agp_restart.py \
  --config tests/<case>/sweep_test/config.json
```

3. Train the retained end-to-end sparse AGP pipeline. This command trains the
   missing SiLU warm-start baseline if needed, warm-starts the PAU feedback
   model, runs the configured fixed-K holdout-feedback/support-swap curriculum,
   runs the uniform temporal-refinement continuation, and then runs the
   adaptive temporal-refinement continuation.

```bash
conda run --no-capture-output -n torch-mps python scripts/agp_holdout_feedback.py \
  --config tests/<case>/sweep_test/config.json
```

4. Run the post-training physical diagnostic. The script chooses the
   `adaptive_temporal_refinement/` checkpoint when the feedback summary records
   it as enabled, otherwise it falls back to `temporal_refinement/` and then to
   the final feedback round. For the q15 benchmark instance, it compares no-CD,
   Kipu/DQFM `l=1`, and the learned sparse AGP.

```bash
conda run --no-capture-output -n torch-mps python tests/sparse_agp_curriculum/scripts/agp_physical_validation.py \
  --config tests/<case>/sweep_test/config.json
```

The common plotting pipeline exports
`Images/physical_method_comparison_table.pdf`. Its rows compare the exact
final ground state, the nested-commutator `l=1` protocol, and the learned
sparse AGP; its columns report final energy, absolute energy error, and
ground-state fidelity. Values are populated from
`physical_validation_summary.json` when statevector validation is feasible.
At larger `q`, the table preserves an available exact final-Hamiltonian
ground-energy reference and marks unavailable dynamical energies/fidelities as
`not computed` rather than estimating them.

5. Accept or reject the candidate using the strongest available diagnostics for
   the configured `q`. When a physical table is available, the deciding retained
   metrics are final energy error and ground-state fidelity, with local
   observables such as `<Z_i>` and `<Z_i Z_{i+1}>` RMSEs as consistency checks.
   For larger `q`, the diagonal-Ising final energy and ground bitstring remain
   exact, but full statevector evolution is not available. The candidate must
   still be reported at the correct certification level using
   `AGP_CERTIFICATION_CRITERIA.md`.

Generated run artifacts are local and ignored by git. The repository stores the
code, configuration, tests, and this methodology record; it does not commit
`runs/`, checkpoint files, or generated figures.

## Current Benchmark Instance And Certification Status

The current retained physical benchmark instance is q15. It is evidence that
the general methodology can produce a physically useful sparse AGP on an
above-exact-output problem, but it is not a full certification of the support
against the unrestricted `4**15` basis. Under `AGP_CERTIFICATION_CRITERIA.md`,
the current q15 status is:

| Gate | Status | Evidence |
|---|---|---|
| Training residual | pass | adaptive temporal-refinement training relative residual `0.003279208` |
| Holdout residual | pass | adaptive temporal-refinement holdout relative residual `0.055878535`, within the practical `0.05` to `0.10` target band |
| Unseen residual quotient | not tested | quotient invalid because the AGP=0 reference residual on the sampled unseen subset is zero |
| Fixed `probe_gate` / `probe_watch` / `probe_test` residuals | not tested | the current retained pipeline does not yet define fixed disjoint probe bases |
| K-sweep plateau | not tested | current retained run uses `K = 32768`; no formal nearby-K plateau is stored for this adaptive-refinement benchmark |
| Q-sweep plateau | not tested | current validation uses `Q = 65536`; no formal larger-Q plateau is stored |
| Top-term stability across K and seeds | not tested | no formal top-term overlap study is stored for the retained adaptive-refinement run |
| Prune-and-retest | not tested | deployment truncates to the top 2048 terms for statevector validation, but no formal residual prune sweep is stored |
| Physical validation | pass | learned AGP improves energy error, fidelity, and local observable RMSEs against no-CD and Kipu/DQFM `l=1` |

The correct claim level for the current q15 benchmark instance is therefore:

```text
Projected sparse AGP experiment with strong q15 physical validation.
```

It should not be described as a certified globally sufficient support for the
full Pauli basis.

## Physical Validation

The retained backend policy is:

```text
q <= 15: exact statevector evolution
q > 15:  tensor-network evolution
```

This dynamical threshold does not limit exact final-Hamiltonian ground solvers.
At any `q`, use an exact ground energy and ground-state manifold whenever the
operator structure permits it. Canonical tensor-network validation must keep
the complete learned AGP support fixed across its numerical convergence ladder.

After training, the q15 statevector diagnostic compares:

```text
no_cd
kipu_dqfm_l1
learned_sparse_agp
```

The q15 statevector path is intentionally a benchmark diagnostic, not a scalable
large-q library path.

The latest retained adaptive-temporal benchmark result is:

| Method | Energy error | Ground fidelity | `<Z_i>` RMSE | `<Z_i Z_{i+1}>` RMSE |
|---|---:|---:|---:|---:|
| no CD | 16.8582 | 0.000287 | 0.9700 | 0.8411 |
| Kipu/DQFM l=1 | 10.1628 | 0.02594 | 0.8441 | 0.4119 |
| learned sparse AGP + learned schedule + fixed-K support swap + adaptive temporal refinement | 0.1873 | 0.9601 | 0.0144 | 0.0140 |

The previous retained temporal-refinement benchmark had:

```text
energy error = 0.2369
ground fidelity = 0.9549
<Z_i> RMSE = 0.0148
<Z_i Z_{i+1}> RMSE = 0.0124
```

Adaptive temporal refinement therefore improved the primary retained physical
benchmark targets:

```text
energy error improvement ~= 20.9%
ground fidelity gain    ~= 0.0053
```

It also slightly improved `<Z_i>` RMSE, while nearest-neighbor
`<Z_i Z_{i+1}>` RMSE worsened from `0.0124` to `0.0140`. The method is retained
because the stated primary objective for this iteration was lower final energy
error and higher ground-state fidelity.

The older retained fixed-K support-swap benchmark without temporal refinement
had:

```text
energy error = 0.2740
ground fidelity = 0.9478
```

The current benchmark improves over that no-temporal-refinement run:

```text
energy error improvement ~= 31.6%
ground fidelity gain    ~= 0.0124
```

The older retained PAU benchmark without support swaps had:

```text
energy error = 0.7002
ground fidelity = 0.8675
```

The current benchmark improves substantially over that older no-swap PAU run:

```text
energy error improvement ~= 73.2%
ground fidelity gain    ~= 0.0926
```

Earlier architecture/activation candidates from the PAU sweep remain rejected:

| Candidate | Energy error | Ground fidelity | Reason not retained |
|---|---:|---:|---|
| width 128, 4 layers, SiLU | 1.2558 | 0.7410 | worse physical metrics |
| width 96, 4 layers, trainable SiLU | 1.1214 | 0.7789 | improved over SiLU but worse than PAU |

The final adaptive-refinement holdout residual (`0.0559`) is slightly worse
than the previous no-swap PAU holdout residual (`0.0532`). The method is
nevertheless retained because the primary physical validation targets improved:
final energy and ground-state fidelity moved closer to the exact q15 final
ground-state diagnostics. This reinforces that the projected residual is a
necessary diagnostic, not the only benchmark objective.

The learned row uses the exported learned schedule grid from the trained AGP
checkpoint. The no-CD and Kipu/DQFM l=1 rows use the fixed reference
`sin^2(pi t / 2T)` schedule, so the learned result should be interpreted as the
performance of the jointly learned sparse-AGP-plus-schedule protocol. A useful
future attribution control is a no-CD row evolved under the same learned
schedule.

## Current Interpretation

The current retained q15 benchmark instance shows that the jointly learned
sparse AGP and schedule are much more physically useful than no-CD and the
first-order nested-commutator approximator on a physically checkable
above-exact-output problem. The adaptive temporal-refinement stage is retained
because it improved final energy and ground fidelity beyond the uniform
temporal-refinement benchmark without using final-state observables in training.

It does not certify that the selected support is globally sufficient out of the
full `4**q` basis. It also does not prove that a lower projected residual always
maps to a better final physical state.

The next methodological improvements should therefore add physical robustness
or attribution controls without using benchmark-only ground-truth targets.

## Scalable MPS Dynamical Validation

Large-q final-state metrics are evaluated with a fail-closed tensor-network
backend instead of a dense `2**q` statevector. This post-training layer does
not alter AGP learning. It deploys the exported schedule and coefficients and
compares:

```text
no_cd
kipu_dqfm_l1
learned_sparse_agp
```

The canonical PINN row uses every ordered term exported by the retained
checkpoint. A mutation-sensitive hash covers all labels and sampled coefficient
rows. Coefficient-ranked top-term deployments remain ablations and cannot
replace full-support validation.

The preferred learned path constructs the complete sampled coefficient tensor

```text
C[r, p_1, ..., p_q] = coefficient of P(p_1,...,p_q) in H_CD(t_r)
```

and factorizes it as a workspace-bounded tensor train. The finite time axis may
be inserted at a measured position along the Pauli chain; selecting one time
index then produces an ordinary MPO for one TDVP midpoint. This placement can
reduce temporal rank inflation but never changes `K`. If a multi-time window
fails, adaptive execution may split it into smaller contiguous windows. Every
accepted window still contains all learned terms.

An operator is evolvable only after it passes all of:

```text
full-K source and hash completeness,
finite Hermitian coefficients,
conservative coefficient-space error bound,
full-source sparse-action error upper bound,
and workspace/bond limits.
```

A cancellation-limited action estimate may pass when its finite conservative
upper bound is below tolerance. Missing, nonfinite, or unbounded evidence is
`not tested`. Two-site TeNPy TDVP is then used so the MPS bond can grow. The
legacy quimb product-formula path remains a diagnostic/historical backend, not
the canonical fallback after an MPO gate failure.

Every reported large-q result records the time steps, MPS/MPO cutoffs and bonds,
time-axis position and windows, deployed learned-term count and hash, operator
error bounds, workspace, runtime, state norm, truncation error, and final
observables. Timestep and MPS convergence use independent named pairs: the time
pair fixes MPS/MPO settings, while the state pair fixes timestep/MPO settings.
Changing both in one comparison is `not comparable`. Merely finishing one MPS
run is not a pass.

The legacy grouped product-formula backend was calibrated at q15 against the
matching retained 1024-term statevector variant. At 96 steps, bond 128, and
cutoff `1e-12`, learned-protocol differences were:

```text
final-energy difference = 1.401e-4
ground-fidelity difference = 2.658e-5
```

The retained q20 grouped product-formula validation deploys all 32,768 learned
terms from the completed
`Q=81,920` adaptive-temporal checkpoint. These map to 1,828 occupied-qubit
support groups, with no coefficient threshold. Its 24-step/bond-32/cutoff-
`1e-9` and 48-step/bond-64/cutoff-`1e-10` results pass the configured
successive-resolution gate. The fine comparison is:

| Method | Final energy | Energy error | Ground fidelity |
|---|---:|---:|---:|
| no CD | -3.2390823 | 22.7609177 | 1.95723e-05 |
| Kipu/DQFM l=1 | -12.2268621 | 13.7731379 | 0.00806762 |
| learned sparse AGP | -25.6453873 | 0.3546127 | 0.93648534 |

The q20 exact target is `E0=-26` with unique ground bitstring `00...0`. For the
PINN protocol, the coarse-to-fine energy and fidelity changes are `0.00678613`
and `0.00212781`, and the fine evolution reaches peak bond 57 below the cap 64.
This passes the q20 full-support physical-validation gate, but it does not by
itself certify global support sufficiency in the full `4**20` basis.

The q156 retained deployment uses the top 2048 terms, which retain `0.9555395`
of the trained coefficient RMS norm. Its 24-step/bond-32/cutoff-`1e-9` and
48-step/bond-64/cutoff-`1e-10` results agree for all three protocols. The final
fine-resolution comparison is:

| Method | Final energy | Energy error | Ground fidelity |
|---|---:|---:|---:|
| no CD | -26.2569071 | 183.343093 | 3.85877e-37 |
| Kipu/DQFM l=1 | -97.7452309 | 111.854769 | 1.42713e-16 |
| learned sparse AGP | -188.153516 | 21.4464841 | 0.0207493 |

This certifies numerical convergence of the historical 2048-term deployment
ablation under the configured ladder. Under the current full learned-support
rule, it does not pass physical validation for the trained `K=32768` PINN
operator. The full support has been evaluated at one coarse MPS resolution, but
has not yet passed a successive-resolution ladder; canonical q156 full-model
physical validation is therefore `not tested`. Dense-statevector validation
also remains unavailable at q156.

## Non-Retained Probe-Loss Variant

The current retained methodology does not include the previously tested
gauge-error probe loss. That candidate introduced the gauge-error operator

```text
G(A) = dH_AD/dlambda + i [A_lambda, H_AD]
```

and added a self-supervised probe-state penalty of the form

```text
L_probe = E_{tau,s} ||G(A, tau) |phi_s>||^2
          / (||G(0, tau) |phi_s>||^2 + eps)
```

to the projected Euler-Lagrange residual. The idea was physically motivated and
did not use q15 final-state ground-truth metrics during training, but the tested
configuration did not improve the retained benchmark.

The first tested `L_probe` implementation used `alpha = 0.05` with four
deterministic Pauli-stabilizer product probes. It was trained end-to-end under
the same q15 curriculum, but it was not retained because it worsened the final
physical benchmark:

| Method | Energy error | Ground fidelity | `<Z_i>` RMSE | `<Z_i Z_{i+1}>` RMSE |
|---|---:|---:|---:|---:|
| learned sparse AGP with `L_probe` | 3.9766 | 0.3005 | 0.2785 | 0.2118 |

That rejected probe-loss result is superseded by the current retained benchmark
above. It was also worse than the older no-swap PAU benchmark, which had energy
error `0.7002` and ground fidelity `0.8675`.
