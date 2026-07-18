# General Full-Support Tensor-Network Validation Design

## Objective

Build a fail-closed physical-validation framework for counterdiabatic dynamics
that remains valid as the qubit count, Pauli support size, interaction density,
locality, and coefficient scale change. The q24 transverse-field spin-HUBO run
is a regression benchmark, not a source of hard-coded assumptions.

The canonical learned-AGP row must use every one of the `K` Pauli labels and
time-dependent coefficients exported by the retained PINN checkpoint. Ranked
term removal is an ablation and cannot produce a canonical physical claim.

## Non-Goal

No tensor-network representation is efficient for every many-body Hamiltonian
and every generated state. Robustness therefore means:

1. characterize the actual operator and state complexity;
2. choose a compatible algorithm from measured properties;
3. quantify every approximation independently; and
4. report `not_feasible` or `not_tested` instead of an uncontrolled number.

It does not mean that every `(q, H, K)` instance must return a fidelity.

## Root Cause Of The Superseded q24 Failure

The superseded learned path factorized the direct-CD coefficient matrix in time,
compresses each static temporal mode independently, and block-sums the mode
MPOs during evolution. The q24 checkpoint has temporal rank 13. At MPO bond 64,
the separately compressed modes lose substantial operator weight and their
dynamic block sum reaches bond 859. The measured full-operator action error is
about `0.941`, so the resulting one-step energy and fidelity are diagnostic
artifacts rather than physical results.

The failure is not caused by selecting only a subset of learned labels: all
32,768 labels enter the source. It is caused by an uncontrolled representation
of their combined instantaneous action.

## Canonical Invariants

For every learned checkpoint and every canonical resolution:

- `source_terms == checkpoint_terms == K` before exact duplicate combination;
- no coefficient threshold, top-term selector, or support pruning is active;
- a stable hash covers the ordered labels and sampled direct-CD coefficients;
- temporal reconstruction error is measured when temporal factorization is used;
- exact and represented term counts are exported at every time probe;
- all operator and dynamics gates fail closed;
- no final energy or fidelity is published as canonical unless every required
  gate passes.

## Hamiltonian And Checkpoint Characterization

Before evolution, profile `H_initial`, `H_final`, nested `l=1`, and the learned
direct-CD operator over deterministic time probes. Record:

- qubit count and term counts;
- Pauli-weight and interaction-range distributions;
- interaction-hypergraph connected components and chain cut widths;
- coefficient maximum, RMS, minimum nonzero magnitude, and dynamic range;
- candidate qubit-order scores;
- exact or bounded MPO ranks at the sampled times;
- estimated construction, contraction, and memory costs.

Global coefficient rescaling is separated from dynamic range. MPO construction
normalizes by a finite per-operator scale and restores that scale after
factorization, so very large or very small finite coefficients do not change
rank decisions merely through floating-point units.

## Backend Router

The router selects by measured complexity, not by qubit count alone:

1. `exact_statevector`: mandatory oracle for `q <= 15`.
2. `joint_time_full_support_mpo_tdvp`: preferred large-q route. Reconstruct all
   `K` direct-CD coefficients at every requested midpoint, combine them with the
   physical Hamiltonians, and factor the complete `time x Pauli` coefficient
   tensor once. Slicing its time core produces each midpoint MPO without
   separately truncating or block-summing temporal modes.
3. `direct_full_support_mpo_tdvp`: a profiling and small-rank fallback that
   independently factorizes the complete instantaneous operator at each time.
4. `adaptive_windowed_full_support_mpo_tdvp`: factor the positioned joint
   time-Pauli tensor over contiguous midpoint windows. A failed window may be
   split, but every accepted child still contains all `K` learned terms and
   must pass the coefficient, action, and workspace gates.
5. `temporal_mode_mpo_tdvp`: retained as a diagnostic comparison, never chosen
   automatically after a direct-path action gate failure.
6. `unsupported_topology`: returned when chain MPO/MPS complexity exceeds the
   resource envelope. A future tree-TN backend may consume this classification,
   but the router must not relabel an uncontrolled MPS calculation as valid.

No route may silently reduce `K`.

## Direct-Time Full-Support MPO

For midpoint `t`, construct the complete instantaneous Pauli coefficient map:

```text
H_CD(t) = [1 - lambda(t)] H_initial + lambda(t) H_final
          + dot(lambda)(t) sum(j=1..K) a_j(t) P_j.
```

Duplicate Pauli labels are combined with stable component-wise summation.
Arithmetic zeros may disappear only after combination and must be recorded.
The original `K` source labels and coefficient hash remain part of provenance.

The sparse Pauli-coordinate tensor is converted directly to a quantized tensor
train/MPO with a workspace-bounded SVD. With zero cutoff and a sufficient bond
cap, the representation receives an exact-identity certificate. Otherwise it
is an approximation and must pass full-source action probes.

## Joint Time-Pauli Tensor Train

Repeated direct factorization is unsuitable when a full-support midpoint MPO
has a large bond. The canonical amortized representation therefore forms the
complete coefficient tensor only on the requested TDVP midpoint grid:

```text
C[r, p_1, ..., p_q] = coefficient of P(p_1,...,p_q) in H_CD(t_r).
```

The tensor is stored sparsely at the source. The finite time axis may be placed
at any measured position among the Pauli axes, and its position is part of the
operator/cache identity. A time slice contracts the selected time row into its
neighboring Pauli core and returns one ordinary finite MPO. Construction starts
with the configured contiguous midpoint window; if that window fails, it may be
split recursively without reducing `K`. In the limiting one-midpoint window,
the method is a direct full-support instantaneous factorization with the same
certification contract.

Pauli labels use arbitrary-precision encodings; the method must not impose a
32-qubit limit. Workspace preflight is based on measured local unfoldings and
fails before allocation when a requested rank is infeasible.

The operator gate combines two complementary checks. First, the accumulated
TT-SVD discarded norm provides a conservative coefficient-space error bound
for every time slice. Second, deterministic exact sparse full-K actions are
compared with sliced MPO actions at representative times. Neither check may be
replaced by source-term truncation.

## Operator Certification

Each protocol and sampled time receives independent gates for:

- source completeness and hash consistency;
- finite coefficient normalization and reconstruction;
- exact-identity evidence or measured action error;
- operator bond and workspace limits;
- Hermiticity;
- either a conservative absolute representation-error certificate or
  consistency under at least two operator resolutions.

Product-state probes use exact sparse Pauli action. Small-q probes also compare
dense matrices. Random-MPS probes are required only when their exact contraction
fits the configured work cap; an unavailable random probe is not converted into
a pass. During dynamics, the current MPS is added as a state-dependent probe
whenever exact full-K action can be evaluated within the cap.

## Dynamics Certification

Two-site TDVP is the default MPS integrator because its bond can grow. Canonical
results require explicit, independent pairs that refine:

- time step;
- MPS maximum bond and cutoff;
- MPO maximum bond and cutoff when no absolute operator-error certificate is
  available; and
- qubit order as a separate sensitivity test when multiple orders are
  competitive.

Record physical norm drift, TDVP truncation error, peak state bond, operator
bonds, runtimes, final energy, target-state fidelity, and configured local
observables. The timestep pair fixes MPS/MPO settings; the state pair fixes the
timestep and MPO settings. For `q <= 15`, all three protocols must also agree
with exact statevector dynamics.

## General Regression Matrix

Automated tests cover multiple qubit counts and operator regimes:

- sparse nearest-neighbor Hamiltonians;
- sparse long-range Hamiltonians;
- dense but low-rank/factorizable Pauli sums;
- non-compressible high-cut-width Pauli sums;
- global coefficient rescalings over several orders of magnitude; and
- mixed coefficient dynamic ranges including exact zeros.

Tests verify both successful routing and deliberate fail-closed behavior. The
q24 HUBO checkpoint is then run as an integration benchmark with all 32,768
learned labels.

## Acceptance Criteria

- Canonical learned validation consumes every checkpoint term.
- q<=15 MPO/TDVP results agree with exact statevector results within configured
  energy and fidelity tolerances for every regression regime.
- Global coefficient rescaling does not change structural routing or normalized
  operator error beyond floating-point tolerance.
- At least two resolutions converge before a physical result can pass.
- q24 produces either a converged, full-K comparison or a precise certified
  obstruction. Uncontrolled one-step values are never presented as performance.
- Generated result tables and plots show only canonical passed results; failed
  diagnostics remain visibly labeled and isolated.
