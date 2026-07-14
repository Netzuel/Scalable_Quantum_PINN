# Q20 Ising Parity Training Design

## Objective

Replace the retained `q20/sweep_test` hydrogen study with a 20-qubit
`TransverseIsingDriverProblem` study that preserves the accepted q15 training
lineage, scales the residual budget to twenty feedback rounds, removes obsolete
q20 hydrogen-specific surfaces, and completes the resulting training workflow.

## Benchmark Contract

The q20 study uses the same Hamiltonian family and interpolation as q15:

```text
H_initial = -sum_i X_i
H_final   = -sum_i h_i Z_i - sum_i J_i Z_i Z_(i+1)
H_AD      = (1 - lambda) H_initial + lambda H_final
```

The retained q15 configuration is the template. Only qubit-dependent and
twenty-round capacity fields change:

- `num_qubits = 20`
- fixed AGP support `K = min(32768, 4**q) = 32768`
- requested holdout residual budget `Q_requested = 81920`
- `feedback_iterations = 20`
- `add_residual_terms_per_iteration = 3072`
- one unseen residual batch remains after the final feedback round

The effective Q may be smaller than 81,920 only when the sparse commutator
generator produces fewer unique residual labels. The run must record requested
and effective Q separately and automatically reduce per-round additions only if
required to preserve all twenty rounds plus the final unseen batch.

## Training Lineage

Training follows q15 without substituting a new method:

1. Generate or reuse the q20 diagonal-Ising Hamiltonian.
2. Train a width-96, four-hidden-layer SiLU baseline at fixed K.
3. Warm-start a fully active PAU feedback model from the compatible baseline.
4. Run twenty fixed-K feedback rounds.
5. Starting at round 2, swap 256 weak AGP strings per round for hard
   residual-derived candidates; K never grows.
6. Continue the final round-20 state through uniform temporal refinement.
7. Continue through adaptive temporal refinement on the fixed final support.

All current q15 optimizer, loss, schedule, calibration, random-seed, and
support-selection values remain unchanged unless a q20 resource constraint is
explicitly represented in configuration and tests.

## Physical Validation

The q20 configuration retains the q15 diagonal-Ising physical-validation
contract with `statevector_qubits = 20`, 96 evolution steps, and a 2,048-term
learned deployment sweep. Its Pauli-action cache must be bounded for the
`2**20` statevector. Physical validation is prepared and documented, but the
requested completion condition is the twenty-round training workflow and its
post-curriculum refinements; a full q20 statevector diagnostic is a separate,
explicitly reported post-training calculation rather than a hidden training
dependency.

## Cleanup Scope

Before the new run:

- remove generated runs under `q20/sweep_test/runs/` that belong to the obsolete
  hydrogen study;
- replace hydrogen-specific q20 configuration and README content;
- remove the q20-only hydrogen energy-validation entrypoint and its tests when
  no retained configuration references it;
- remove hydrogen-specific q20 plot fallback metadata;
- retain reusable sparse molecular-Hamiltonian loaders and indexed Hamiltonian
  data, because they are not q20 scenario artifacts.

No q15, q156, or unrelated process or generated run may be modified or stopped.

## Verification

Preparation is accepted only when:

- tests prove q20 uses `TransverseIsingDriverProblem`, q=20, K=32,768,
  Q_requested=81,920, and twenty rounds;
- parity checks cover q15-to-q20 optimizer, PAU transfer, support swap,
  calibration, schedule, and refinement fields;
- no q20 configuration or documentation references the obsolete hydrogen
  validator or hydrogen energy result;
- a clean preparation run generates the q20 Hamiltonian/baseline inputs;
- repository compilation and unit tests pass in `torch-mps`.

Training is complete only when saved artifacts prove that rounds 1 through 20
finished, temporal refinement finished, adaptive temporal refinement finished,
and the final summary records K, requested/effective Q, residual metrics, and
the certification-gate status without turning an untested gate into a pass.

