# Spin-HUBO Sparse-AGP Benchmark Design

## Objective

Extend the retained sparse-AGP curriculum beyond the diagonal Ising chain to a
frustrated, nonlocal diagonal spin polynomial while preserving the current
self-supervised training methodology and exact-reference validation rules.

## Selected Instance

The benchmark imports
`HAMILTONIANS_SPIN/run_002/hamiltonian_341` from the sibling
`bitstrings_operator` repository. It has 24 qubits and 216 terms:

```text
one-body terms   = 24
two-body terms   = 72
three-body terms = 120
```

It is the requested 20--30 qubit candidate with the lowest native-order MPS
cut-crossing count among the inspected run_001--run_004 instances. Runs 008
and 009 are excluded. The tracked benchmark stores a source snapshot, hash,
and conversion metadata so execution does not depend on the sibling checkout.

## Hamiltonian Path

The initial state remains the exact product ground state `|+>**24` of

```text
H_initial = -sum_i X_i.
```

For the source spin objective

```text
E(s) = sum_S c_S product_(i in S) s_i,  s_i in {-1, +1},
```

the final quantum Hamiltonian is the diagonal Pauli operator

```text
H_final = sum_S c_S product_(i in S) Z_i.
```

The computational mapping is `s_i = +1 <-> |0>` and
`s_i = -1 <-> |1>`. No coefficient rescaling or sign change is applied.

## Exact Ground Reference

Because `H_final` is diagonal and q=24, all `2**24` energies are evaluated by
an in-place Walsh-Hadamard transform. This independently confirms the supplied
simulated-annealing minimum:

```text
ground energy    = -28.101465646808336
ground bitstring = 000011100010110000010010
degeneracy       = 1
```

The exact oracle certifies only the final classical objective. Quantum dynamics
remain subject to tensor-network convergence.

## Training Contract

The q20 retained training budget is transferred unchanged:

```text
K                            = 32768
Q_requested                  = 81920
curriculum rounds            = 20
residual additions per round = 3072
support swaps per round      = 256 from round 2
```

The width-96, four-hidden-layer PAU model, bounded learned schedule, jointly
trained global scale and gates, fixed-K support swaps, uniform temporal
refinement, and adaptive temporal refinement remain unchanged. Exact energy,
ground bitstring, fidelity, and final observables never enter training or model
selection.

## Physical Validation

For q=24, canonical dynamics use quimb MPS evolution. The comparison includes:

```text
no_cd
kipu_dqfm_l1
learned_sparse_agp
```

All methods share the Hamiltonian path, initial state, total time, integrator,
and numerical ladder. The learned row deploys all 32,768 exported AGP terms;
coefficient-ranked truncation is not allowed in the canonical table.

At least two fixed-support resolutions vary timestep, bond cap, and cutoff.
Physical validation passes only when energy and ground-fidelity deltas meet the
configured tolerances. The MPS observable target uses the selected ground
bitstring's `Z_i` and `Z_i Z_(i+1)` eigenvalues rather than assuming all zeros.

## Claim Boundary

The run is a projected sparse-AGP experiment unless every gate in
`AGP_CERTIFICATION_CRITERIA.md` passes. Exact final-objective certification and
converged MPS dynamics do not prove that K=32768 represents the unrestricted
`4**24` AGP support.
