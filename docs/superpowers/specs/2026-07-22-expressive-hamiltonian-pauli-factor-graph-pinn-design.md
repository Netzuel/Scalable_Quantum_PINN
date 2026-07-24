# Expressive Hamiltonian-Pauli Factor-Graph PINN Design

## Objective

Build a second graph-conditioned sparse-AGP candidate that addresses the
underfitting observed in the first graph decoder while preserving the main
scalability property: coefficient-generating parameters are shared across
qubits and Pauli terms and therefore do not grow with `q` or `K`.

The q15 and q20 candidates are trained independently from random
initialization. No checkpoint, learned coefficient, or schedule is transferred
between systems. Exact ground-state information remains excluded from training
and checkpoint selection.

## Diagnosed Limitations Of Graph v1

The first graph decoder compressed 32,768 coefficient trajectories through a
rank-32 bilinear map and represented Hamiltonian interactions as unsigned
pairwise edge magnitudes. Its order-independent linear term pooling aliased
many distinct Pauli strings. The result was strong coefficient collapse and a
training residual substantially above the retained independent-output PINN.

Graph v2 addresses representation capacity before changing the curriculum or
physical objective.

## Signed Hamiltonian Factor Graph

The Hamiltonian is represented as a bipartite graph:

- one node for every qubit;
- one factor node for every Pauli term in the union of `H_initial` and
  `H_final`;
- Pauli-labelled incidences between factor nodes and their nonidentity qubits.

Factor features retain normalized real, imaginary, and magnitude coefficients
from `H_initial`, `H_final`, and their difference, together with Pauli weight
and symbol fractions. This preserves coefficient sign/phase and higher-body
term identity rather than reducing a term to unsigned pairwise edges.

Alternating shared qubit-to-factor and factor-to-qubit message passing produces
Hamiltonian-conditioned qubit states. No absolute qubit embedding is used, so
simultaneous qubit permutations remain equivariant.

## Expressive Pauli-Term Encoder

Each AGP Pauli incidence combines its qubit state with a shared X/Y/Z symbol
embedding. A nonlinear incidence network is pooled using first and second
moments. A nonlinear term network then combines those moments with fixed-size
structural and commutator fingerprints.

The commutator fingerprints summarize how each candidate Pauli term interacts
with `H_initial`, `H_final`, and their difference. They are derived only from
the training Hamiltonians and contain no ground-state or dynamical-validation
targets.

## Temporal Decoder

The temporal encoder receives normalized time and fixed Fourier features. Its
latent rank is raised from 32 to 128 by default. A shared bilinear decoder maps
the temporal state and every encoded Pauli term to the complete `[B, K]`
coefficient tensor. The coefficient body remains parameter-independent of
`q/K`, while runtime and deterministic graph storage continue to scale with
the evaluated system and support.

The existing learned schedule, global scale, per-term gates, fixed-K support
swap curriculum, temporal refinements, and full-support physical validators
remain unchanged.

## Controlled Experiments

Independent candidates live under:

```text
q15/sweep_test/hamiltonian_pauli_factor_graph/
q20/sweep_test/hamiltonian_pauli_factor_graph/
```

They retain the q-specific `K`, `Q`, curriculum rounds, physical Hamiltonians,
and validation backends used by the retained benchmarks. The output and plot
surface must match the retained pipeline wherever applicable.

## Promotion Gates

Graph v2 replaces no benchmark unless all conditions hold:

1. q15 full-support ground-state fidelity is at least `0.95`;
2. q20 full-support ground-state fidelity is at least `0.95`;
3. q20 fidelity is no more than `0.01` below q15 fidelity;
4. required projected/certification gates are not failed;
5. physical validation uses all `K` learned AGP terms and passes the applicable
   numerical-convergence gates.

Two qubit counts cannot establish asymptotic scaling, so the third condition is
only the local non-degradation gate. A broader qubit grid would still be needed
before claiming absence of systematic fidelity decay in general.
