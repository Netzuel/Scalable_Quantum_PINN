# MPS Dynamical Validation Design

## Objective

Compute final-state energy and ground-state fidelity for large-q sparse-AGP
protocols without constructing a dense `2**q` statevector, and validate the
approximation against the retained q15 exact-statevector benchmark before using
it at q156.

## Architecture

The implementation uses quimb's eager `CircuitMPS` backend. The initial
`|+>**q` product state is evolved under

```text
H_eff(t) = H_AD(lambda(t)) + H_CD(t)
```

with a symmetric Pauli-product formula. Each Pauli exponential is applied as a
small local gate over the string's occupied spatial span. The q156 retained
top-2048 learned support has maximum span seven, so no dense object larger than
`2**7 x 2**7` is created.

The same propagator handles:

- no CD, with the reference sin-squared schedule;
- variational nested-commutator `l=1`, evaluated at each time midpoint;
- the learned PINN AGP, using its exported learned schedule and interpolated
  counterdiabatic coefficients.

## Metrics

The final diagonal-Ising energy is contracted from local `Z_i` and
`Z_i Z_{i+1}` expectations. For the unique all-zero ground state, fidelity is
the squared MPS amplitude of `|0...0>`. The output also records norm,
maximum bond dimension, quimb truncation estimate, retained AGP RMS fraction,
gate count, runtime, and convergence deltas.

## Validation Contract

1. Unit tests compare Pauli rotations and small-system evolution with dense
   reference calculations.
2. q15 MPS results are compared with the stored 96-step RK4 statevector table.
3. q156 is evaluated through a convergence grid in timestep, bond dimension,
   truncation cutoff, and learned-term count.
4. A q156 value is called converged only when the highest two retained settings
   satisfy configured energy and fidelity tolerances. Otherwise it is exported
   as an approximate, unconverged diagnostic.
5. No exact or physical-validation gate is marked passed from an unperformed
   calculation.

## Dependencies

`quimb==1.11.2` is an optional `tensor-network` dependency because the
repository's `torch-mps` environment uses Python 3.10. The newer quimb 1.14
series requires Python 3.11.

## Outputs

Each study writes ignored generated artifacts under its retained run:

```text
Models_Data/mps_physical_validation_summary.json
Models_Data/mps_physical_validation_convergence.csv
Images/mps_physical_validation_convergence.pdf
Images/physical_method_comparison_table.pdf
```

The common comparison table prefers validated statevector data when present,
then converged MPS data, and otherwise continues to show `not computed`.
