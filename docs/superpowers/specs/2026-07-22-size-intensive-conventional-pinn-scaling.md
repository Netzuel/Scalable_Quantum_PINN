# Size-Extensive Active-Support Conventional-PINN Scaling Study

## Question

Test whether the decline of global ground-state fidelity with increasing qubit
count can be alleviated without graph architectures and without cross-system
pretraining. Every Hamiltonian and qubit count uses an independently trained
quadratic PAU PINN. q15 is the existing retained anchor; q20 and q25 are
initialized and trained from scratch with no checkpoint or parameter transfer.

The controlled systems are q15, q20, and q25 instances of the same transverse
driver to diagonal nearest-neighbor Ising problem.

## Physical Motivation

Global many-body fidelity is extensive. If an approximately size-independent
local excitation probability remains after the protocol, total fidelity can
behave approximately as

```text
F(q) ~ exp(-q epsilon_local).
```

Fixed global support, active-gate, residual, curriculum, and runtime budgets
therefore become progressively weaker per qubit. A declining global fidelity
does not by itself identify which budget is responsible, so the first robust
test keeps all controllable budgets approximately intensive.

## Rejected v1

The first candidate scaled `T=q/10` together with linear resource budgets. Its
q15 coarse full-support TDVP fidelity was `0.9363697699`, below the `0.95`
acceptance gate. Changing `T` also changed the physical protocol and therefore
confounded AGP scalability with runtime. v1 is rejected.

## Rejected v2

The fixed-`T`, quadratically scaled v2 q15 checkpoint reached exact full-support
fidelity `0.9265712315`. A top-2048 diagnostic reached `0.9295794505`, excluding
the low-amplitude tail as the main cause. Relative to the retained q15 support,
the fresh support had higher coefficient-weighted Pauli order and only
28278/32768 term overlap. v2 is rejected.

## Rejected v3

The locality-aware v3 q15 checkpoint reached exact full-support fidelity
`0.9187530299` at `T=1`. Explicit normalized-time reparameterization to `T=2`
reached only `0.9211278482`, so physical runtime is not the limiting factor.
The final projected holdout quotient was `0.0012598871`, demonstrating that a
smaller unstratified operator residual can coexist with worse dynamics. v3 is
rejected.

## Rejected v4

v4 replaced the retained absolute projected objective with a
reference-normalized Pauli-order-balanced residual. Its q15 training residual
became small, but exact all-`K` statevector validation reached only
`F=0.8565275292`, `E(T)=-18.38487243`, and energy error `0.86512757`.
The retained q15 all-`K` result is `F=0.9646510261`. v4 is rejected because
equalizing algebraic residual blocks degraded the state-relevant control.

An independent all-`K` q20 duration diagnostic reparameterized the retained
checkpoint from `T=1` to the a priori law `T=20/15`. At 24 TDVP steps and bond
64 it reached `F=0.9421177458`, compared with `0.9377128395` at `T=1`.
All operator windows passed with maximum coefficient error `2.18e-8` and zero
action-error bound. Duration scaling helps slightly but does not by itself meet
the `0.95` gate. More importantly, changing duration changes the physical
protocol and can improve even no-CD evolution in the adiabatic limit. It is
therefore rejected as a mechanism for claiming improved AGP scalability. All
v5 comparisons remain at fixed `T=1`.

## Fixed v5 Scaling Law

The scaling law is declared before any new physical validation:

```text
K(q)        = ceil_to_1024(32768 (q / 15)^2)
K_active(q) = ceil_to_256(2048 (q / 15)^2)
Q(q)        = 2 K(q)
i(q)        = q
DeltaQ(q)   = ceil_to_1024(3072 q / 15)
DeltaK(q)   = ceil_to_256(256 q / 15)
Q_probe(q)  = ceil_to_512(4096 q / 15)
T(q)        = 1
```

This gives:

| q | K | K_active | Q | DeltaQ | DeltaK | Q_probe | rounds | T |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 32768 | 2048 | 65536 | 3072 | 256 | 4096 | 15 | 1.0 |
| 20 | 58368 | 3840 | 116736 | 4096 | 512 | 5632 | 20 | 1.0 |
| 25 | 91136 | 5888 | 182272 | 5120 | 512 | 7168 | 25 | 1.0 |

All neural collocation uses `tau=(t-t_initial)/T`. The schedule export records
both `d lambda/d tau` and `d lambda/dt=(1/T)d lambda/d tau`. The common
integration ladder uses 24 and 48 steps at fixed `T=1`. Every canonical physical
evaluation deploys all learned `K` terms.

v5 restores the retained absolute projected loss. It scales both total and
effective AGP capacity: `K`, `Q`, and target gate mass grow quadratically, while
per-round residual additions, support swaps, and frozen probes grow linearly so
their cumulative fractions remain approximately constant over `i=q` rounds.
Physical duration is held fixed to isolate the learned-control quality from
ordinary adiabatic slowing. No exact energy, bitstring, fidelity, or observable
enters training or checkpoint selection. Every subprocess starts with
`PYTHONHASHSEED=0`.

## Controlled Variables

The following remain common:

- `H_initial = -sum_i X_i`;
- diagonal open-chain Ising `H_final` with the same field/coupling construction;
- independent-output quadratic PINN, width 96, four hidden layers, PAU;
- fixed-K support-swap curriculum and residual-only training;
- jointly trained schedule, global CD scale, and soft support gates;
- no checkpoint transfer between q values;
- seed 11 and the retained optimization schedule.

Exact final energy, exact bitstrings, fidelity, and physical observables are not
used during training or checkpoint selection.

## Physical Evaluation

Only `learned_sparse_agp` dynamics are required for this study. No-CD and
nested-commutator rows are intentionally omitted. q15 still seeks the required
exact-statevector check where computationally controlled; all three systems
also use full-support tensor-network evolution so the size trend is evaluated
with one common backend. q20 and q25 use tensor networks canonically.

Each tensor-network result must pass independent timestep and state/bond
convergence with fixed `K`. Numerical-zero filtering is allowed only under the
project rulebook; coefficient-ranked pruning is forbidden.

## Acceptance

The scaling law succeeds only if:

```text
F_q15 >= 0.95
F_q20 >= 0.95
F_q25 >= 0.95
F_q20 >= F_q15 - 0.01
F_q25 >= F_q20 - 0.01
```

Passing three points is evidence against degradation over q15-q25, not proof of
asymptotic size independence. Any failed or unavailable numerical convergence
gate remains `fail` or `not tested` even when the raw fidelity exceeds 0.95.
