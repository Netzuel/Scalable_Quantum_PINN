# Stable Unseen Diagnostics And Compressed-MPO Evaluation Design

## Objective

Make two validation surfaces reliable for arbitrary sparse-AGP studies:

1. keep unseen diagnostics defined throughout a feedback curriculum without
   manufacturing a quotient when the AGP=0 reference is zero; and
2. evaluate the final PINN counterdiabatic dynamics with a scalable,
   evaluation-only tensor-network backend.

The PINN architecture, loss, support curriculum, and learned coefficients are
not changed by this work.

## Stable Unseen Diagnostics

### Fixed probe partitions

At curriculum initialization, generate residual candidates beyond the labels
required by the feedback budget. Preserve the existing immutable
`probe_gate`, `probe_watch`, and `probe_test` pools used by curriculum
certification. Allocate an additional fixed unseen-diagnostic pool, disjoint
from those certification probes and from a separate feedback-candidate pool.
Partition the unseen-diagnostic pool into:

- `probe_unseen_active`: labels with AGP=0 reference RMS above the configured
  numerical threshold;
- `probe_unseen_null`: labels with AGP=0 reference RMS at or below that
  threshold.

Only `feedback_candidates` labels are eligible to enter the training residual
basis.

The two unseen-diagnostic partitions and all three certification probes are
immutable and never eligible for feedback promotion. Their construction,
seed, sizes, labels, reference norms, and selection threshold are persisted
before round 1. The candidate generator automatically expands until the
configured feedback and probe budgets are met or records an explicit
insufficiency status.

### Metrics

Every round reports:

```text
active_unseen_relative =
    ||R(A)||^2_probe_active / ||R(A=0)||^2_probe_active

null_unseen_absolute_per_term =
    ||R(A)||^2_probe_null / |probe_unseen_null|

null_unseen_scaled =
    null_unseen_absolute_per_term /
    (||R(A=0)||^2_probe_active / |probe_unseen_active|)
```

The active quotient retains the existing physical meaning. The null metric
measures AGP-induced leakage into directions absent from the no-AGP residual.
No epsilon-clamped zero denominator is reported as a relative residual.

The moving holdout unseen metric remains available for curriculum diagnosis,
but plots distinguish it from the fixed probes. If its denominator becomes
zero, the plot shows the absolute null leakage instead of an unexplained gap.
Only a Hamiltonian path with an identically zero fixed reference can produce a
`not_tested` active quotient, and that reason must be stated explicitly.

## Compressed-MPO Evaluation Backend

### Scope

This backend runs only after training. Ground energy, ground bitstrings, final
fidelity, and other physical targets never enter the PINN loss or support
selection.

The existing quimb Pauli-product evaluator remains as a small-system reference
and debugging backend. The scalable backend uses TeNPy because it provides
finite-MPS TDVP and time-dependent exponential-MPO evolution for arbitrary
long-range MPO Hamiltonians.

### Full-support policy

All `K` learned Pauli labels and coefficients enter the operator before any
compression. Removing labels by coefficient rank or a top-term threshold is an
ablation and cannot certify the trained model.

Controlled MPO compression is permitted. It approximates the full operator,
not its support definition, and must report its numerical error. No compressed
MPO evolution is described as exact.

### Temporal factorization

Sample the direct counterdiabatic coefficient matrix
`D[t, j] = dot(lambda)(t) C_j(t)` on a configurable dense time grid and compute
an SVD:

```text
D(t, j) ~= sum_r f_r(t) v_r(j).
```

Retain the smallest rank satisfying a configured squared-Frobenius-norm target.
Every static mode `v_r` contains contributions from every learned Pauli label.
Persist singular values, retained rank, retained norm fraction, maximum
coefficient reconstruction error, and endpoint reconstruction error. Enforce
the exact endpoint direct-CD vectors after reconstruction; these are expected
to be zero because the retained schedule has zero endpoint derivatives.

The q24 spin-HUBO checkpoint provides a favorable preflight result: the direct
CD coefficients require 3 temporal modes for 99.9% retained norm, 5 for
99.99%, and 8 for 99.9999%.

### Spatial MPO construction and compression

Build one static Pauli-sum MPO per retained temporal mode, plus the initial and
final Hamiltonian MPOs. Construct the uncompressed operator as an exact sparse
trie/automaton MPO from all input terms; do not construct a dense operator or
prune Pauli labels. Use a deterministic qubit ordering selected by an
interaction-graph preflight. The native order remains an explicit candidate;
an optimized order is accepted only when it lowers measured cut complexity.

Compress each exact sparse MPO with a sweep and a configured bond/cutoff
budget. Record:

- exact input term count and nonzero coefficient count;
- pre- and post-compression MPO bonds;
- the compression sweep's reported truncation error;
- deterministic relative operator-action errors against the uncompressed MPO
  on fixed product and random-MPS probes;
- temporal reconstruction error; and
- build time and peak memory.

If the requested error cannot be achieved below the resource cap, the
resolution is `not_feasible`; the evaluator does not silently prune terms or
relax tolerances.

### Time evolution

At each midpoint, form

```text
H_total(t) = (1-lambda(t)) H_initial + lambda(t) H_final
             + sum_r f_r(t) MPO_r.
```

Evolve the initial MPS with two-site TDVP by default so the state bond can grow.
Time-dependent exponential-MPO evolution is retained as a comparison backend.
The evaluator records norm drift, truncation error, peak MPS bond, MPO bonds,
runtime, final energy, exact-ground-state fidelity when available, and local
observables.

### Numerical certification ladder

A physical result passes only when it is stable across:

1. temporal SVD tolerance;
2. MPO compression cutoff and maximum MPO bond;
3. MPS cutoff and maximum MPS bond; and
4. time-step refinement.

For `q <= 15`, the final ladder result must also agree with exact statevector
evolution at matching full support. Small-q and reduced-support tests compare
the MPO backend with the existing Pauli-product evaluator. For larger `q`, a
failed or incomplete ladder remains `not_tested` physical validation.

## Configuration

The scenario configuration receives separate sections:

```text
feedback.fixed_unseen_probes
tensor_network_validation.mpo_backend
tensor_network_validation.resolutions
```

Defaults remain conservative and reproducible. Backend selection, temporal
tolerance, MPO/MPS bonds, cutoffs, time steps, qubit ordering, random probe
seeds, and resource caps are configuration values rather than script constants.

TeNPy is added only to the optional `tensor-network` dependency group. Training
and residual-only workflows do not require it.

## Artifacts

Each run exports:

- fixed active/null probe metadata and per-round metrics;
- temporal SVD diagnostics;
- qubit-ordering and MPO-compression diagnostics;
- per-resolution evolution metrics and convergence deltas;
- a comparison table for no CD, nested l=1, and learned AGP; and
- an explicit certification status: `pass`, `fail`, `not_tested`, or
  `not_feasible`.

Plots never connect undefined moving-unseen quotients. They label the fixed
active quotient and fixed null leakage separately.

## Acceptance Criteria

- Fixed active/null probe labels never enter the training residual basis.
- Nontrivial paths produce a finite active unseen quotient in every round.
- Zero-reference directions always produce finite absolute and scaled leakage
  metrics without epsilon-defined pseudo-ratios.
- All learned terms enter the MPO builder before compression.
- Temporal and MPO compression errors are independently reported.
- The MPO backend agrees with exact statevector and legacy product-formula
  references on tractable cases.
- q24 evaluation demonstrates a measurable speed improvement over the
  2,502.7-second one-step product-formula baseline before a full ladder starts.
- No physical-fidelity claim is made unless every required numerical gate
  passes.
