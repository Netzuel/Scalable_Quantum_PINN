# Hamiltonian-Conditioned Pauli Graph PINN Implementation Plan

**Goal:** Implement a term-shared graph coefficient architecture, train isolated
q15/q20 candidates from scratch, and compare canonical full-support physical
fidelities with the retained PINN benchmarks.

**Constraints:** Preserve legacy behavior by default; use no cross-system
pretraining or physical ground truth during training; keep all Python execution
inside `torch-mps`; deploy all K learned terms in canonical validation.

## 1. Characterization Tests

- Add tests proving graph-model parameter count is invariant to q and K.
- Add tests for output shape, finite gradients, deterministic sparse term
  encoding, and permutation equivariance.
- Add checkpoint and fixed-K support-swap tests proving shared weights survive
  while calibration values remain label-aligned.
- Add legacy-default tests before production changes.

## 2. Graph Architecture

- Add sparse Hamiltonian graph and term-incidence builders to `utils.py`.
- Add the shared graph encoder, term encoder, and temporal readout to `models.py`.
- Keep deterministic structural tensors nonpersistent and avoid dense Hilbert
  matrices or dense `K x q` descriptors.
- Integrate the body into projected training and export models.

## 3. Configuration And Lineage

- Extend configuration parsing, validation, serialization, and checkpoint
  reconstruction with opt-in graph fields.
- Update export/resampling, curriculum reconstruction, support swaps, and
  checkpoint compatibility without changing old configurations.
- Store architecture identity and graph hyperparameters in every candidate
  checkpoint and resolved configuration.

## 4. Export And Plot Parity

- Route graph runs through the existing coefficient/support plotting pipeline.
- Preserve physical-validation JSON/CSV schemas and method names.
- Generate the same HCD summary and physical comparison PDFs where applicable.
- Test plot annotation discovery against independent candidate paths.

## 5. Verification Before Training

- Run focused model/config/checkpoint/export tests.
- Compile modified modules, run the full unit suite, run the sparse demo, and
  check `git diff --check`.
- Confirm candidate roots are independent and retained q15/q20 artifacts are
  byte-count/file-count stable before starting training.

## 6. q15 Candidate

- Derive an isolated graph configuration from the retained q15 configuration.
- Train the complete curriculum from scratch in the graph candidate folder.
- Require projected/frozen-probe gates before physical validation.
- Run exact full-support statevector validation and regenerate all applicable
  result tables and PDFs.

## 7. q20 Candidate

- Derive an isolated graph configuration from the retained q20 configuration.
- Train the complete curriculum from scratch in the graph candidate folder.
- Require projected/frozen-probe gates before physical validation.
- Run convergence-gated full-support tensor-network validation and regenerate
  all applicable result tables and PDFs.

## 8. Comparison And Closeout

- Read old and new metrics from canonical saved summaries, not console output.
- Report retained versus graph ground-state fidelity and energy error for q15
  and q20, with certification status and convergence caveats.
- Keep candidate outputs independent; promote only when the documented gates
  and physical metrics justify it.
