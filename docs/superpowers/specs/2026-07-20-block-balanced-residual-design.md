# Block-Balanced Residual Objective Design

> **Status: rejected and not retained.** The q156 test failed the frozen-active
> projected gate and materially underperformed the retained benchmark. The
> implementation, candidate configuration, and generated artifacts were
> removed; this design remains only as a written research record.

## Objective

Improve large-system ground-state fidelity without using exact energy, exact
ground-state fidelity, or benchmark observables during training or model
selection. The immediate stress test is the q156 transverse-field to diagonal
Ising path, but the implementation must apply to arbitrary sparse Pauli
Hamiltonians and qubit counts.

## Diagnosis

The retained q156 full-support tensor-network result is numerically converged
but has ground-state fidelity `0.2591563`. Its local observable errors are much
smaller, and deployment already plateaus between 16,384 and 32,768 learned
terms. The principal hypothesis is therefore that a globally averaged projected
residual permits spatially concentrated weak regions. Small local diabatic
errors then compound in the 156-body overlap.

## Loss Definition

For every projected residual Pauli label, let `S_j` be the qubits on which the
label is non-identity. The squared residual contribution of term `j` is divided
equally among the qubits in `S_j`. This produces one residual energy and one
zero-AGP reference energy per covered qubit block:

```text
r_b = mean_t sum_{j: b in S_j} |R_j(t)|^2 / |S_j|
r0_b = mean_t sum_{j: b in S_j} |R0_j(t)|^2 / |S_j|
rho_b = r_b / stopgrad(max(r0_b, epsilon_ref))
```

The reference floor is relative to the mean covered-block reference:

```text
epsilon_ref = reference_floor * mean_b(r0_b)
```

The optimized residual is:

```text
L_block = mean_b(rho_b) + tail_weight * CVaR_tail_fraction({rho_b})
```

`CVaR` is the arithmetic mean of the largest
`ceil(tail_fraction * covered_blocks)` ratios. The top-k selection is
piecewise differentiable: gradients flow through the selected block ratios.
Identity-only residual labels do not define a spatial block; they remain in the
unchanged global diagnostics and are expected to vanish for commutator
residuals.

## Sparse Representation

The model stores a flattened sparse incidence representation rather than a
dense `q x Q` matrix:

```text
block_indices[k]     -> qubit block receiving incidence k
block_term_indices[k] -> residual term contributing incidence k
block_shares[k]       -> 1 / Pauli weight of that residual term
```

At each loss evaluation, the model first averages squared residuals over time,
then uses `scatter_add` to form block energies. This keeps memory proportional
to the total non-identity count of the residual labels.

## Configuration Contract

The existing `training.loss.residual_objective` gains one opt-in value:

```json
{
  "residual_objective": "block_balanced_reference_normalized",
  "residual_block_tail_fraction": 0.15,
  "residual_block_tail_weight": 1.0,
  "residual_block_reference_floor": 1e-6
}
```

Defaults preserve every existing configuration:

```text
residual_objective = absolute
tail_fraction = 0.15
tail_weight = 1.0
reference_floor = 1e-6
```

Validation requires `0 < tail_fraction <= 1`, `tail_weight >= 0`, and
`reference_floor > 0`.

## Diagnostics

Every training history row and checkpoint-compatible loss call records:

```text
block_mean_relative_residual
block_tail_relative_residual
block_max_relative_residual
block_covered_count
block_tail_count
```

The existing absolute residual, zero-AGP reference, global relative residual,
fixed probes, and holdout metrics remain unchanged. The block metrics are
self-supervised training diagnostics, not physical certification by themselves.

## q156 Experiment

The candidate starts from the corrected q-aware v2 methodology and changes only
the residual objective and isolated output roots. It retains:

```text
q = 156
K = 32768
20 feedback rounds
fixed-K support swaps
q-aware Q and probe budgets
PAU width-96 depth-4 network
learned schedule and calibration
uniform and adaptive temporal refinements
```

Before cleaning, the current retained `runs/` directory is moved to an ignored
root-level backup. The q156 scenario is then reset and trained from a clean
baseline so its immutable fixed probes precede every optimizer step.

## Physical Evaluation And Promotion

Training and projected model selection never inspect the exact q156 ground
state. A candidate reaching the physical stage is evaluated afterward with all
32,768 learned terms under the existing source-complete joint time-Pauli MPO and
independent TDVP timestep/state convergence pairs.

The candidate becomes the retained benchmark only if:

1. all required projected and fixed-probe gates pass;
2. full-support tensor-network certification passes;
3. fidelity improves by at least `0.002` over `0.2591562697`; and
4. absolute energy error does not worsen by more than `0.1`.

Otherwise the candidate remains a rejected experiment and the archived retained
q156 benchmark is restored unchanged.
