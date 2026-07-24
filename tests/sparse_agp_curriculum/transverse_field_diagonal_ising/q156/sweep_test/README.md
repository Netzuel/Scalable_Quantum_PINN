# q156 Sparse AGP Curriculum

This folder records the legacy validated q156 benchmark. It predates the
normalized variational-action v6 methodology and has not yet been retrained
under the current benchmark. It applies the earlier sparse-AGP curriculum to
the 156-qubit driver-to-problem transverse Ising path:

```text
H_initial = - sum_i X_i
H_final   = - sum_i h_i Z_i - sum_i J_i Z_i Z_{i+1}
```

The configured scalable curriculum is:

```text
K = 32768 trainable AGP outputs
Q_max = 81920 requested generated residual holdout terms
i = 20 holdout-feedback iterations
```

The initial support is selected from a bounded nested-commutator Krylov pool.
Each curriculum round adds 3072 hard residual equations and, from round 2,
swaps 256 weak AGP strings for residual-derived candidates while keeping
`K` fixed. The completed run generated 69855 distinct residual strings from
the requested `Q_max=81920`; after round 20, 4319 generated terms remain
outside the 65536-term training residual basis for the final unseen diagnostic.

The model matches the retained architecture but uses the preceding loss: a width-96, four-hidden-layer
quadratic/QRes coefficient network, a SiLU warm start, trainable PAU
activations, jointly learned scale/gates/bounded schedule, and uniform plus
adaptive temporal refinement.

The final diagonal Ising Hamiltonian has the exact ground energy `-209.6` and
unique all-zero computational-basis ground bitstring. This exact reference is
available without dense diagonalization. A full q156 statevector remains
unavailable, but the configured tensor-network validator evolves an MPS and
computes final energy from local contractions plus ground fidelity from the
all-zero product-state amplitude.

Generated artifacts are ignored by git and written only under `runs/`.
Study-local `Images/` and `Models_Data/` scratch folders are not created.

## Prepare Hamiltonian

```bash
conda run -n torch-mps python tests/sparse_agp_curriculum/scripts/build_driver_problem_hamiltonian.py \
  --num-qubits 156 --update-index
```

## Clean

```bash
conda run -n torch-mps python scripts/agp_restart.py \
  --config tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q156/sweep_test/config.json
```

## Train

```bash
conda run --no-capture-output -n torch-mps python scripts/agp_holdout_feedback.py \
  --config tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q156/sweep_test/config.json
```

The training entrypoint automatically trains the missing SiLU baseline, runs
all 20 fixed-K feedback rounds, performs both temporal refinements, and exports
the canonical summaries and figures.

The retained q156 physical deployment resamples the frozen round-20 checkpoint
on 257 time points without further optimization:

```bash
conda run -n torch-mps python scripts/diagnostics/agp_resample_checkpoint.py \
  --config tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q156/sweep_test/config.json \
  --trained-run tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q156/sweep_test/runs/fixed_k_holdout_feedback_trainable_schedule_w96_l4_pau_support_swap_adaptive_temporal_refinement_v1/agp_32768_residual_69855_add_3072_rounds_20/rounds/round_20 \
  --output-dir tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q156/sweep_test/runs/fixed_k_holdout_feedback_trainable_schedule_w96_l4_pau_support_swap_adaptive_temporal_refinement_v1/agp_32768_residual_69855_add_3072_rounds_20/rounds/round_20/deployment_attribution/dense_257_export_v2 \
  --num-points 257 --device cpu
```

See `RESULTS.md` for the retained run metrics and certification status.

## Q-Aware Training Candidate

`config_q_aware_candidate.json` is an isolated, untrained v2 candidate. It uses
reference-normalized residual training, target-normalized gate budgets,
per-qubit resource policies, pre-training immutable probes, and deterministic
locality/spatial stratification for both the residual reservoir and support
swaps. Its output namespace is separate from the retained benchmark.

The earlier v1 q-aware run completed, but it is diagnostic only: its probe
manifest was persisted after baseline training, and a corrected learned-
schedule evaluation of round 19 fails both frozen projected gates. It was not
sent to tensor-network validation and was not promoted.

## Rejected Block-Balanced Candidate

The completed candidate did not produce an eligible champion. Round 20 reached
training and generated-holdout relative residuals of `6.86894e-4` and
`6.84697e-4`, but its frozen-active residual was `1.57992`, above the `1.0`
gate. Temporal and adaptive refinement remained above that gate at `1.48088`
and `1.35409`. A diagnostic-only full-support TDVP run of the deterministic
round-20 endpoint reached `E(T)=-146.48933` and fidelity `1.23696e-9` at 48
steps and bond 64, far worse than the retained benchmark below. The candidate
is therefore rejected and `config.json` remains the canonical methodology. Its
experimental implementation, candidate configuration, generated run, and
temporary plots were removed. Only the Markdown methodology and result record
is retained.

## Tensor-Network Validation

Install the optional pinned backend and run the convergence ladder:

```bash
conda run -n torch-mps python -m pip install 'quimb==1.11.2'
conda run --no-capture-output -n torch-mps python -u \
  tests/sparse_agp_curriculum/scripts/agp_mps_validation.py \
  --config tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q156/sweep_test/config.json
```

The canonical validator keeps all 32,768 learned terms and constructs a
joint-time full-support MPO for two-site TDVP. It separates timestep
convergence (24 versus 48 steps at MPS bond 64) from state convergence (bonds
32 versus 64 at 48 steps). MPO compression, source completeness, sampled
operator action, timestep convergence, and state convergence must all pass.

The retained MPS summary and PDF are written under the validated dense export:

```text
rounds/round_20/deployment_attribution/dense_257_export/mpo_validation/Models_Data/mps_physical_validation_summary.json
rounds/round_20/deployment_attribution/dense_257_export/mpo_validation/Images/physical_method_comparison_table.pdf
```

The fine full-support result reaches `E(T)=-201.851323` and ground fidelity
`0.2591563`, versus `E_0=-209.6`. This is a convergence-gated scalable
dynamical validation, not an exact q156 statevector and not a proof that Pauli
strings outside the trained AGP support are negligible.

## Learned-Support Diagnostic

The retained round-20 checkpoint was additionally evaluated with 2048, 4096,
8192, 16384, and all 32768 learned terms under the same coarse MPS settings.
The 8192-term point was repeated with the fine settings above. The consolidated
machine-readable result is:

```text
mps_support_sweep/Models_Data/support_sweep_summary.json
```

The diagnostic identifies 8192 terms as the smallest tested reduced deployment
that passes both the `5%` within-output support plateau and its coarse/fine MPS
convergence gate. This remains an ablation result only. The separate canonical
ladder now validates all 32,768 learned terms at both timestep resolutions and
both state-bond resolutions. Pauli strings absent from the trained AGP support
remain outside both conclusions.
