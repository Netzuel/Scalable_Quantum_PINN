# q156 Sparse AGP Curriculum

This folder applies the common sparse-AGP benchmark methodology to the
156-qubit driver-to-problem transverse Ising path:

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

The model matches the retained general benchmark: a width-96, four-hidden-layer
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

See `RESULTS.md` for the retained run metrics and certification status.
## Tensor-Network Validation

Install the optional pinned backend and run the convergence ladder:

```bash
conda run -n torch-mps python -m pip install 'quimb==1.11.2'
conda run --no-capture-output -n torch-mps python -u \
  tests/sparse_agp_curriculum/scripts/agp_mps_validation.py \
  --config tests/sparse_agp_curriculum/transverse_field_diagonal_ising/q156/sweep_test/config.json
```

The historical validator configuration uses a symmetric Pauli-product formula
and quimb MPS compression. It compares 24 steps/bond 32/cutoff `1e-9` with 48
steps/bond 64/cutoff `1e-10`, keeping a reduced learned support fixed at 2048
terms. This is a deployment ablation, not canonical validation of the trained
32768-term PINN operator. Under `Rules.md`, canonical tensor-network validation
must keep all 32768 learned terms fixed across the numerical ladder.

The retained MPS summary and PDF are written under the round-20 checkpoint:

```text
mps_validation/Models_Data/mps_physical_validation_summary.json
mps_validation/Images/physical_method_comparison_table.pdf
```

This is a scalable approximate dynamical validation, not an exact q156
statevector and not a proof that the omitted AGP support is negligible.

## Learned-Support Diagnostic

The retained round-20 checkpoint was additionally evaluated with 2048, 4096,
8192, 16384, and all 32768 learned terms under the same coarse MPS settings.
The 8192-term point was repeated with the fine settings above. The consolidated
machine-readable result is:

```text
mps_support_sweep/Models_Data/support_sweep_summary.json
```

The diagnostic identifies 8192 terms as the smallest tested reduced deployment
that passes both the `5%` within-output support plateau and the coarse/fine MPS
convergence gate. This is an ablation result only. The full 32768-term support
has a coarse result but no fine-resolution partner, so canonical physical
validation remains `not tested`. Pauli strings absent from the trained AGP
support remain outside both conclusions.
