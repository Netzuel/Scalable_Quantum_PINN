# Current q15 Benchmark Methodology

This document records the current q15 benchmark used to test whether the
learned sparse AGP is physically useful, not merely whether it reduces a
projected algebraic residual.

The benchmark is intentionally above the exact-output regime. For q15, the full
Pauli basis contains `4**15` strings, so the method must restrict the trainable
AGP support.

## Goal

The goal is to learn an approximate sparse adiabatic gauge potential

```text
A_lambda(t) = sum_{P in S_AGP} C_P(t) P
```

that improves physical counterdiabatic evolution without using final-state
ground-truth observables during training.

The physical validation asks whether the learned AGP improves:

- final ground-state energy error;
- final ground-state fidelity;
- final `<Z_i>` expectation RMSE;
- final nearest-neighbor `<Z_i Z_{i+1}>` RMSE.

These observables are used only after training.

## Hamiltonian Path

The benchmark uses a transverse-driver to diagonal Ising-problem path:

```text
H_initial = - sum_i X_i
H_final   = sum_i h_i Z_i + sum_i J_i Z_i Z_{i+1}
H_AD(lambda) = (1 - lambda) H_initial + lambda H_final
T = 1
q = 15
```

The final Hamiltonian is diagonal, so the exact final ground energy and
ground-state observables are accessible for this benchmark. That accessibility
is diagnostic-only. The training loop does not use the final ground-state
energy, final fidelity, or exact final observables.

## Sparse AGP Support

The current expanded support run uses:

```text
K = 32768 trainable AGP outputs
Q = 65536 generated residual holdout terms
i = 15 holdout-feedback curriculum iterations
```

The initial AGP support is selected from a bounded nested-commutator Krylov pool
seeded by the order-1 commutator direction. The method never enumerates the full
`4**15` basis.

The active AGP support has fixed cardinality during holdout-feedback. The
curriculum does not grow the number of neural outputs; instead, it adds hard
residual equations to the training residual basis and, in the retained
support-swap benchmark, replaces weak AGP strings with hard residual-derived
candidates while keeping `K` unchanged.

## Neural Architecture

The current retained benchmark uses a quadratic/QRes coefficient network with a
trainable Padé activation unit (PAU):

```text
input = normalized time tau
outputs = 32768 AGP coefficient functions
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

The PAU feedback run warm-starts from the retained width-96 SiLU baseline
checkpoint before entering the 15-round holdout-feedback curriculum. The
`baseline_neural` block in `tests/q15/sweep_test/config.json` makes that
warm-start explicit and reproducible from a cleaned q15 folder: the baseline
checkpoint is trained with SiLU when missing, then the feedback rounds load the
compatible body weights into the PAU model while training the PAU parameters.

The output layer is linear in the final hidden representation; no activation is
applied to the emitted AGP coefficients. The learned global scale and soft Pauli
gates are applied after the network output.

## Trainable Scheduling Function

The current retained benchmark trains the counterdiabatic schedule jointly with
the AGP coefficient network and calibration variables. The parameterization
follows the constrained-envelope idea used in Section 2.1 of arXiv:2604.18506:
a fixed smooth reference schedule plus a bounded neural correction that vanishes
at the boundaries.

For this benchmark:

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
using the q15 final-state ground truth during training.

For the current q15 run:

```text
add_residual_terms_per_iteration = 3072
support_swap_terms_per_iteration = 256
final train residual equations = 50176
final generated holdout pool = 65536
```

The final round residual diagnostics were:

```text
training relative residual = 0.001547411
holdout relative residual  = 0.056218036
absolute unseen residual   = 0.000110266
```

The reported unseen relative residual is not meaningful in this run because the
AGP=0 reference residual on the sampled unseen batch is zero. The absolute
unseen residual is still stored, but the quotient is recorded as invalid.

## Physical Validation

After training, the q15 statevector diagnostic compares:

```text
no_cd
kipu_dqfm_l1
learned_sparse_agp
```

The q15 statevector path is intentionally a benchmark diagnostic, not a scalable
large-q library path.

The latest expanded-support result is:

| Method | Energy error | Ground fidelity | `<Z_i>` RMSE | `<Z_i Z_{i+1}>` RMSE |
|---|---:|---:|---:|---:|
| no CD | 16.8582 | 0.000287 | 0.9700 | 0.8411 |
| Kipu/DQFM l=1 | 10.1628 | 0.02594 | 0.8441 | 0.4119 |
| learned sparse AGP + learned schedule + fixed-K support swap | 0.2740 | 0.9478 | 0.0187 | 0.0138 |

The previous retained PAU benchmark without support swaps had:

```text
energy error = 0.7002
ground fidelity = 0.8675
```

The promoted fixed-K support-swap curriculum therefore improved the retained
physical benchmark:

```text
energy error improvement ~= 60.9%
ground fidelity gain    ~= 0.0803
```

Earlier architecture/activation candidates from the PAU sweep remain rejected:

| Candidate | Energy error | Ground fidelity | Reason not retained |
|---|---:|---:|---|
| width 128, 4 layers, SiLU | 1.2558 | 0.7410 | worse physical metrics |
| width 96, 4 layers, trainable SiLU | 1.1214 | 0.7789 | improved over SiLU but worse than PAU |

The final support-swap holdout residual (`0.0562`) is slightly worse than the
previous no-swap PAU holdout residual (`0.0532`). The method is nevertheless
retained because the physical validation improved substantially: final energy,
ground-state fidelity, `<Z_i>`, and `<Z_i Z_{i+1}>` all moved closer to the
exact q15 final ground-state diagnostics. This reinforces that the projected
residual is a necessary diagnostic, not the only benchmark objective.

The learned row uses the exported learned schedule grid from the trained AGP
checkpoint. The no-CD and Kipu/DQFM l=1 rows use the fixed reference
`sin^2(pi t / 2T)` schedule, so the learned result should be interpreted as the
performance of the jointly learned sparse-AGP-plus-schedule protocol. A useful
future attribution control is a no-CD row evolved under the same learned
schedule.

## Current Interpretation

The current method is good enough to show that the jointly learned sparse AGP
and schedule are much more physically useful than no-CD and the first-order
nested-commutator approximator for this benchmark.

It does not certify that the selected support is globally sufficient out of the
full `4**q` basis. It also does not prove that a lower projected residual always
maps to a better final physical state.

The next methodological improvements should therefore add physical robustness
or attribution controls without using benchmark-only ground-truth targets.

## Recommended Next Direction

The most natural next loss is a variational action or stochastic probe-state
loss that acts one level closer to the physical counterdiabatic condition than
the current Euler-Lagrange residual alone.

Define the gauge-error operator:

```text
G(A) = dH_AD/dlambda + i [A_lambda, H_AD].
```

The current residual asks whether the stationarity condition is small:

```text
R(A) = [G(A), H_AD].
```

A complementary physical loss would also penalize the size of `G(A)` itself,
projected onto generated Pauli coordinates or estimated with stochastic probe
states:

```text
L_action = E_tau ||G(A)||^2 / (||G(0)||^2 + eps)
```

or, with scalable probe states `|phi_s>`:

```text
L_probe = E_{tau,s} ||G(A, tau) |phi_s>||^2
          / (||G(0, tau) |phi_s>||^2 + eps).
```

The combined training objective would be:

```text
L_total = L_Euler_Lagrange
        + alpha L_action_or_probe
        + beta L_budget
        + gamma L_binary_gate
        + eta L_scale_l2
        + optional smoothness regularization on C_P(t).
```

This does not use the q15 final ground state. It asks the AGP to reduce a
physically meaningful gauge error across sampled times and sampled probes, while
the existing residual term still enforces the Euler-Lagrange condition.

For q15, the probe loss can be validated against the same final energy and
fidelity table. For larger q, the same idea can be estimated with product-state,
local-shadow, or tensor-network probes without enumerating the full Pauli basis.

## Rejected Probe-Loss Variant

The first tested `L_probe` implementation used `alpha = 0.05` with four
deterministic Pauli-stabilizer product probes. It was trained end-to-end under
the same q15 curriculum, but it was not retained because it worsened the final
physical benchmark:

| Method | Energy error | Ground fidelity | `<Z_i>` RMSE | `<Z_i Z_{i+1}>` RMSE |
|---|---:|---:|---:|---:|
| learned sparse AGP with `L_probe` | 3.9766 | 0.3005 | 0.2785 | 0.2118 |

That rejected probe-loss result is superseded by the retained PAU benchmark
above, with energy error `0.7002` and ground fidelity `0.8675`.
