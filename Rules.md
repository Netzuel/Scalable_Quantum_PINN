# Scalable Quantum PINN Project Rules

This file is the canonical scientific and operational rulebook for this
repository. Every model or contributor must read it before changing the
training methodology, running a benchmark, evaluating an AGP, or interpreting
results.

These rules apply for every qubit count, Hamiltonian family, neural
architecture, support size `K`, residual budget `Q`, and curriculum length `i`.
Experiment-specific configuration may make a rule stricter, but may not weaken
it silently.

## Authority And Required Reading

Before training or evaluation, read these files in order:

1. `AGENTS.md` for repository working instructions.
2. `Rules.md` for the mandatory project contract.
3. `AGP_CERTIFICATION_CRITERIA.md` for certification gates and claim levels.
4. `docs/CURRENT_SPARSE_AGP_METHODOLOGY.md` for the retained implementation.
5. The selected study's `config.json`, `README.md`, and existing result summary.

When documents overlap, use the rule that produces the more conservative
scientific claim. An unperformed check is always `not tested`, never `pass`.

## Scientific Objective And Claim Boundary

The project learns a useful sparse or full-basis approximation to the
adiabatic gauge potential for

```text
H_AD(lambda) = (1 - lambda) H_initial + lambda H_final,
tau = (t - t_initial) / T in [0, 1],
A_lambda(tau) = sum_{P in S_AGP} C_P(tau) P,
d lambda / dt = (1 / T) d lambda / d tau,
H_CD(tau) = H_AD(lambda(tau)) + (1 / T)(d lambda / d tau) A_lambda(tau).
```

The neural input and all schedule collocation grids use normalized time `tau`.
Physical dynamics may use a duration `T`, but every derivative entering the
physical counterdiabatic Hamiltonian must include the chain-rule factor `1/T`.
Exports must distinguish `d_lambda_d_tau` from `d_lambda_dt`; a checkpoint
trained for one `T` cannot be silently evaluated at another `T`.

### Fixed Physical Duration For CD Benchmark Claims

The canonical counterdiabatic benchmark duration is fixed at `T=1` for every
qubit count, Hamiltonian instance, architecture, support size, and candidate
methodology. Comparisons and promotion decisions must evaluate all methods at
this same finite physical duration.

- Do not optimize, sweep, enlarge, or post-hoc reparameterize `T` to improve a
  reported fidelity or energy error.
- A larger `T` approaches the ordinary adiabatic limit and can improve no-CD
  evolution without improving the learned AGP. It is therefore not evidence of
  better counterdiabatic control or better scaling with system size.
- Duration-scaling experiments must not be rerun as part of methodology search
  or benchmark promotion. Existing duration diagnostics are historical
  ablations only and remain non-retained.
- Normalized time remains mandatory: `tau=(t-t_initial)/T` and
  `d/dt=(1/T)d/dtau`. With the canonical `T=1`, the factor is numerically one
  but must remain explicit in code and exports.
- A separate physical study may define a non-unit duration only when duration
  itself is the declared scientific variable. Such a study is outside the
  canonical CD benchmark, must compare every method at the same predeclared
  `T`, and cannot replace or promote against the `T=1` result.

The training residual is evaluated in Pauli-coordinate space:

```text
R(A) = [i dH_AD/dlambda - [A_lambda, H_AD], H_AD].
```

For a selected support and residual basis, a small residual is a projected
exactness statement. It is not proof that the learned operator equals the
unrestricted AGP over the complete `4**q` Pauli basis.

The purpose is not to claim a perfect many-body AGP. The purpose is to produce
an AGP that is physically useful, numerically reliable, robust under controlled
tests, and described at the correct certification level.

## Basis And Scalability Rules

- For `q <= 8`, use all `4**q` Pauli strings unless a study is explicitly and
  clearly labelled as a sparse ablation.
- For `q > 8`, never materialize the full `4**q` basis. Use a configured,
  bounded AGP support `K` and residual pool `Q`.
- `K` is the number of trainable AGP output terms. `Q` is a residual-analysis
  budget; increasing `Q` must not silently increase the neural output size.
- Track the trained AGP support, residual training basis, holdout basis, unseen
  basis, and fixed probes separately.
- Prefer locality-, geometry-, symmetry-, commutator-, and residual-informed
  support construction over random term generation.
- Exploratory support swaps must keep the configured active support size fixed
  unless the experiment is explicitly a `K` sweep.
- There is no universal `K_min=f(q)`. Support adequacy depends on the complete
  Hamiltonian path, locality, spectrum, schedule, tolerance, and training
  procedure.

## Training Rules

- Training must remain self-supervised through the AGP residual and declared
  regularizers.
- The current retained benchmark objective is
  `L_total = L_projected + 0.1 L_action + L_regularization`, where
  `L_action` is the squared sparse-Pauli norm of
  `i dH_AD/dlambda - [A_lambda,H_AD]` divided by the squared norm of
  `i dH_AD/dlambda` with a numerical floor. Future benchmark candidates must
  compare against this normalized variational-action v6 baseline.
- Exact final energy, exact ground-state fidelity, exact observables, and other
  benchmark ground truth must not enter training or hyperparameter selection
  unless the experiment is explicitly labelled as supervised.
- Learned schedule parameters, global CD scale, support gates, and activation
  parameters must be optimized inside the declared training curriculum when
  they are part of the retained model.
- Record `q`, `K`, requested and realized `Q`, curriculum rounds `i`, seed,
  architecture, activation, optimizer, time grid, schedule, and checkpoint
  lineage.
- A low training residual alone is never sufficient. Holdout, unseen, fixed
  probe, coefficient-regularity, and physical tests must be classified using
  `AGP_CERTIFICATION_CRITERIA.md`.
- Ground-truth physical metrics may be inspected only after the candidate and
  its training choices have been fixed.

## Mandatory Physical Evaluation Decision Tree

Every retained PINN-based counterdiabatic result must undergo physical
evaluation whenever a defensible numerical route exists.

### Step 1: Establish The Ground Reference

First determine whether the ground energy and ground-state manifold of
`H_final` can be computed exactly.

Use an exact reference when any suitable route is feasible, including:

- direct diagonalization for sufficiently small Hilbert spaces;
- sparse exact eigensolvers when the matrix and required eigenvectors fit;
- exhaustive computational-basis enumeration for small diagonal problems;
- exact classical optimization, dynamic programming, transfer matrices, or a
  proven closed form when `H_final` is diagonal and structurally tractable;
- an exact solver such as Gurobi or CPLEX for a compatible classical objective,
  subject to solver availability and a verified optimality certificate.

The feasibility decision depends on all of the following, not only `q`:

```text
operator form of H_final,
locality and interaction graph,
diagonal versus non-diagonal structure,
sparsity,
symmetries,
ground-state degeneracy,
available memory and solver complexity.
```

For a diagonal `H_final`, an exact energy and exact minimizing bitstring may be
available at qubit counts far beyond dense-statevector limits. This does not
make the full time-dependent quantum evolution exact.

If an exact ground reference is unavailable, use a separately converged
approximate ground-state method, such as DMRG or another tensor-network ground
solver, when appropriate. Label that target as approximate and report its own
convergence evidence. Never call an approximate tensor-network target exact.

### Step 2: Choose The Dynamical Evaluator

Use the following project-wide backend threshold for dynamical validation:

```text
q <= 15: exact statevector evolution
q > 15:  tensor-network evolution
```

This threshold fixes the validation backend, not the ground-reference solver.
The exact `H_final` ground energy and ground-state manifold must still be sought
at every `q` using the Hamiltonian structure. For example, a large diagonal
problem may have an exact classical ground reference even though its dynamics
are evaluated with tensor networks.

For `q <= 15`:

- evolve `no_cd`, the declared nested-commutator baseline, and the PINN AGP
  under matched physical and numerical settings;
- compare against the exact `H_final` ground energy and ground manifold;
- use the complete learned AGP support unless the run is explicitly an
  ablation;
- verify time-step or integrator convergence.

For `q > 15`:

- use an MPS, MPO, or other appropriate tensor-network representation;
- route from measured operator rank, cut width, workspace, and state complexity,
  not from `q`, term density, or coefficient count alone;
- prefer the full-support joint time-Pauli MPO/TDVP path when it passes its
  coefficient and action gates;
- use the exact `H_final` ground reference if it is independently available;
- otherwise use a separately converged approximate ground reference and label
  it accordingly;
- run the numerical convergence ladder defined below.

If the backend required by this threshold cannot provide a controlled result,
mark
physical validation as `not tested` and explain the computational obstruction.
Residual diagnostics may still be reported, but they do not become physical
validation by substitution.

`H_initial` also matters: its ground state or initial preparation must be
defined and verified. Tensor-network feasibility additionally depends on the
entanglement generated by the full path and on the spatial range of the Pauli
terms, not only on the form of `H_final`.

## Full Learned-Support Rule

The canonical physical evaluation of a trained PINN AGP must deploy every AGP
term learned in the retained checkpoint.

If training produced

```text
S_AGP = {P_1, ..., P_K},
```

then the default evaluated operator is

```text
A_lambda(t) = sum_{j=1}^K C_j(t) P_j.
```

This rule is especially mandatory for tensor-network evaluation. Training a
`K`-term AGP and then evaluating only the largest `K_deploy < K` coefficients
does not constitute canonical validation of the trained model.

The following restrictions apply:

- Do not rank by coefficient magnitude and retain only a top subset for the
  primary PINN result.
- Do not use retained coefficient norm as proof that omitted terms are
  dynamically irrelevant.
- A numerical coefficient threshold may remove only exact zeros or values at a
  declared numerical-zero tolerance justified by arithmetic precision and
  convergence. It must not act as coefficient-ranked pruning. Report the
  threshold and the number of terms removed.
- Tensor-network integrators may combine terms that share an identical
  occupied-qubit support into one local Hamiltonian exponential, provided every
  learned coefficient is included, the grouping policy is recorded and fixed
  across the convergence ladder, and timestep convergence is demonstrated.
- A joint time-Pauli tensor train may move the finite time axis along the Pauli
  chain to reduce rank. Slicing must reconstruct every learned coefficient at
  every requested midpoint; changing the time-axis position must change the
  cache/implementation identity.
- A failed multi-time window may be split into smaller contiguous windows, but
  no split may reduce `K`. Every accepted window must independently pass source
  completeness, coefficient-error, action-error, and workspace gates.
- Reduced-support runs are allowed only as explicitly labelled pruning,
  sensitivity, cost, or support-size ablations.
- A reduced-support ablation must never replace the full learned-support row in
  the primary comparison table.
- If full learned-support tensor-network evolution cannot be completed, report
  the full-support physical gate as `not tested`. A converged top-term
  deployment remains an ablation, not a substitute pass.
- Existing reduced-support artifacts do not become full-model validation
  retroactively. Re-evaluate them with all learned terms before promotion.

This rule concerns the support learned by the PINN. It does not imply that the
learned `K` terms are sufficient relative to the omitted `4**q-K` Pauli
strings. That separate question remains governed by residual probes, support
swaps, `K` sweeps, seed stability, and the certification criteria.

## Matched-Protocol Rule

The comparison between `no_cd`, nested commutators, and the PINN method must
hold fixed, wherever applicable:

```text
H_initial and H_final,
total evolution time,
initial state,
schedule constraints,
time integrator and resolution,
observable definitions,
ground-reference convention,
and numerical convergence tolerances.
```

Any method-specific schedule or learned control must be disclosed. Do not give
one method a finer integrator or more favorable target convention without a
reported convergence reason.

## Exact And Tensor-Network Convergence

An evolution result is not accepted merely because the program finished.

For exact or sparse-statevector evolution, report and test:

- integration method and time-step convergence;
- state norm;
- Hamiltonian and support identity;
- agreement with known small cases when available.

For tensor-network evolution, report and test:

- time-step or Trotter-order convergence;
- bond-dimension convergence;
- truncation cutoff convergence;
- maximum and peak bond;
- state norm and, when available, discarded weight;
- gate count, runtime, and active learned-term count;
- final energy, fidelity, and observable changes across the numerical ladder;
- small-`q` agreement with exact statevector evolution using the same full
  learned support and protocol.

The operator and state approximations are separate. A represented MPO must
pass a declared coefficient-space bound and a full-source action bound before
it is evolved. A cancellation-limited probe may pass when its finite,
conservative action-error upper bound is below the configured threshold; an
unbounded or nonfinite interval is `not tested`.

At least two numerical resolutions are required for a tensor-network pass.
The full learned support must remain fixed across the convergence ladder.
Changing support size and numerical resolution simultaneously does not isolate
either source of error.

Timestep and MPS convergence must be assessed on independent pairs. The
timestep pair keeps MPS and MPO settings fixed; the state pair keeps timestep
and MPO settings fixed while increasing bond capacity and/or tightening the
cutoff. A pair that changes both axes is `not comparable`, even if its final
metrics happen to agree.

Tensor-network geometry must be justified. MPS is not automatically reliable
for highly nonlocal interactions or volume-law entanglement. A low peak bond is
useful evidence only together with explicit convergence checks.

## Required Physical Metrics

Report the following whenever they are defined:

- exact or approximate target ground energy `E0`;
- final-state energy for every protocol;
- absolute and normalized energy error;
- ground-state fidelity;
- excitation probability or excitation density;
- relevant local and correlation observables;
- ratios relative to `AGP=0` or `no_cd`;
- numerical resource and convergence diagnostics.

For a non-degenerate ground state,

```text
F0 = |<psi_0 | psi(T)>|^2.
```

For a degenerate ground space, use the projector fidelity

```text
F0 = <psi(T) | Pi_ground | psi(T)>,
```

or the equivalent sum of probabilities over an exact orthonormal ground basis.
Do not select a single convenient ground vector and call that the total
ground-space fidelity.

If only exact ground bitstrings are available for a diagonal target, ground
fidelity is the summed final probability of all exact minimizing bitstrings.

## Residual And Support Certification

Physical performance and AGP-support certification are related but distinct.

- A good physical result does not prove that the sparse AGP support is globally
  sufficient.
- A low projected residual does not guarantee good final dynamics.
- Training, holdout, unseen, and fixed-probe residuals must use clearly
  separated bases.
- A support-size plateau obtained by truncating one trained checkpoint is a
  deployment ablation, not a trained-`K` certification sweep.
- A formal `K` plateau requires separately trained nearby supports under
  matched conditions, as specified in `AGP_CERTIFICATION_CRITERIA.md`.
- Support proposals, pruning lists, and rejected-term rankings are not final
  evidence until retrained or dynamically retested as required.

Every certification gate must be one of:

```text
pass
fail
not tested
```

Do not encode unavailable or unperformed checks as zero, success, or pass.

## Reporting And Provenance

Every retained experiment must record:

- Hamiltonian source and hashes or immutable provenance;
- explicit `H_initial`, `H_final`, and schedule definition;
- `q`, `K`, requested and realized `Q`, and curriculum rounds `i`;
- trained checkpoint and parent lineage;
- neural architecture and activation configuration;
- complete learned support size and evaluated support size;
- exact-ground or approximate-ground solver and its certification status;
- state-evolution backend and numerical ladder;
- all physical metrics and residual gates;
- failures, warnings, unavailable quantities, and claim level.

The main comparison table must distinguish:

```text
exact ground reference,
no_cd,
nested commutator l=1,
PINN AGP with the full learned support.
```

Reduced-support, scale, schedule, or architecture studies belong in separate
ablation tables or figures.

Generated runs, checkpoints, figures, `.pt`, `.h5`, and scratch outputs belong
under configured `runs/` directories and are not committed unless an explicit
repository policy says otherwise. Retained methodology, configuration,
machine-readable summaries, and concise result diagnoses must remain
reproducible from tracked code and documentation.

## Code And Repository Rules

- Keep reusable model architecture in `models.py`.
- Keep sparse Pauli algebra and Hamiltonian utilities in `utils.py`.
- Keep reusable workflows under `scripts/` and experiment configuration under
  `tests/`.
- Do not introduce dense `2**q x 2**q` matrices into reusable large-`q`
  library code.
- Dense diagnostics are allowed only for explicitly small systems and must
  document their qubit limit.
- Use structured parsers for JSON, checkpoints, and tabular data.
- Use the `torch-mps` conda environment for all Python execution.
- Do not silently change a retained benchmark, checkpoint lineage, support,
  Hamiltonian, schedule, seed policy, or validation metric.
- Do not overwrite or reinterpret historical results to satisfy a newer rule;
  label them legacy and rerun the missing check.

## Minimum Validation Commands

After relevant code or configuration changes, run:

```bash
conda run -n torch-mps python -m py_compile models.py utils.py
conda run -n torch-mps python examples/two_qubit_sparse_demo.py
conda run -n torch-mps python -m unittest discover -s tests
```

Also validate every changed JSON file, confirm expected artifacts exist, and
verify that no required training or evaluation process remains active before
claiming completion.

## Final Preflight Checklist

Before accepting or reporting a PINN AGP result, answer all of these questions:

1. Are `H_initial`, `H_final`, `lambda(t)`, the initial state, and total time
   explicit?
2. Was an exact `H_final` ground reference sought using the operator structure,
   rather than rejected from `q` alone?
3. If exact dynamics were feasible, were they used and converged?
4. Was the project backend threshold respected: exact statevector for `q <= 15`
   and controlled tensor networks for `q > 15`?
5. Did the canonical PINN evaluation use all learned AGP terms?
6. Were reduced-support studies labelled only as ablations?
7. Were tensor-network time step, bond, and cutoff independently converged with
   fixed full learned support?
8. Were the exact/no-CD/nested-commutator/PINN protocols compared under matched
   conditions?
9. Were energy, ground-space fidelity, observables, residuals, and numerical
   resources reported?
10. Is every unavailable gate marked `not tested`, with no unsupported claim of
    exactness or global AGP sufficiency?

If any required answer is no, the result must not be promoted to the retained
benchmark until the missing work is completed or the limitation is explicitly
accepted and reported at a lower claim level.
