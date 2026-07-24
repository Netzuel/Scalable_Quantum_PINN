# Hamiltonian-Conditioned Pauli Graph PINN Design

## Objective

Replace the projected sparse PINN's `time -> K coefficients` output head with a
Hamiltonian-conditioned, term-shared graph decoder. Every Hamiltonian and qubit
count is trained independently from a random initialization. Exact energies,
ground states, and physical-validation observables remain excluded from
training and model selection.

The first controlled experiments are the retained q15 and q20 transverse-field
to diagonal-Ising scenarios. They run in independent candidate directories and
must not overwrite, warm-start from, or mutate the retained PINN benchmarks.

## Why The Coefficient-Network Parameter Count Does Not Scale With K Or q

The existing final layer contains one independently parameterized output row
per AGP term, so its parameter count grows with `K`. The graph architecture
still evaluates all `K` coefficients, but treats the terms as a batch:

```text
node states = shared_graph_encoder(H_initial, H_final)
term state_j = shared_term_encoder(P_j, node states)
time state_t = shared_time_encoder(t)
a_j(t) = shared_readout(time state_t, term state_j)
```

The graph, term, and temporal weights are reused for every qubit and Pauli term.
Consequently, the number of trainable parameters is fixed by configured hidden
widths, message-passing depth, and latent rank rather than by `q` or `K`.
Runtime, intermediate activations, and deterministic graph data still grow with
the number of graph edges and evaluated terms. This is parameter scalability,
not constant-cost execution.

The complete projected model also retains the existing trainable per-term
calibration gates. Those gates contribute `O(K)` scalar optimization variables,
so strict `q/K` independence applies to the coefficient-generating graph body,
not to every auxiliary parameter in the full training wrapper.

## Architecture

### Hamiltonian graph

Each qubit is a node. Fixed-width node features summarize the real, imaginary,
and magnitude contributions of local X, Y, and Z factors in `H_initial`,
`H_final`, and their difference. Multi-qubit Hamiltonian terms induce sparse
weighted edges. A shared residual message-passing stack encodes local
Hamiltonian context without absolute-qubit output parameters.

### Pauli-term encoder

Each candidate AGP Pauli string is represented by sparse nonidentity
incidences `(term, qubit, symbol)`. The encoder combines a shared X/Y/Z symbol
embedding with the corresponding graph node state and pools over the term's
support. Fixed-dimensional invariant scalars describe Pauli weight and symbol
fractions. No dense `K x q` descriptor is constructed.

### Shared temporal readout

A configurable temporal network maps normalized time to a latent vector. A
shared term network maps every pooled term descriptor to a latent vector. Their
scaled inner product, plus a shared term bias, produces the complete `[B, K]`
coefficient tensor. Existing learned schedule, global CD scale, and per-term
calibration gates remain applicable.

The public `model.body(tau) -> [batch, K]` contract is preserved so projected
losses, coefficient exports, and physical validators require minimal changes.

## Support-Swap And Checkpoint Semantics

Hamiltonian graph and sparse term descriptors are deterministic nonpersistent
buffers reconstructed from the saved Hamiltonian/configuration and active
support. They are not trainable checkpoint rows.

When fixed-K support swaps replace terms:

- shared graph, temporal, and term-decoder weights are retained unchanged;
- descriptors are rebuilt for the new support;
- calibration values are remapped by Pauli label as before;
- no random output row is introduced for a newly discovered term.

Legacy dense-output checkpoints and configurations retain their current loading
behavior. The graph architecture is opt-in through configuration.

## Configuration Contract

The model configuration gains an architecture selector and graph settings:

```json
{
  "coefficient_architecture": "hamiltonian_pauli_graph",
  "graph_node_width": 64,
  "graph_message_layers": 3,
  "graph_latent_rank": 32,
  "graph_term_chunk_size": 4096
}
```

Defaults select the existing independent-output architecture and preserve all
retained runs. Candidate q15/q20 configurations change only architecture fields,
experiment identity, and output roots.

## Training And Certification

Both graph candidates run the complete retained sparse-AGP curriculum from
scratch for their own Hamiltonian. The same support budgets, residual budgets,
round count, frozen probes, support-swap policy, schedule learning, calibration,
and temporal refinements are retained.

Physical evaluation follows the project rulebook:

- q15 seeks exact statevector evolution for the canonical rulebook claim; when
  full-`K` statevector evolution is infeasible, full-support tensor-network
  evolution is reported only as a diagnostic and the exact gate is `not tested`;
- q20 uses convergence-gated tensor-network evolution;
- canonical evaluation deploys every learned AGP term;
- top-term truncations remain labeled ablations only.

The new and retained benchmarks are compared on the same no-CD, nested-l1, and
learned-AGP columns. Ground energy/error, ground-state fidelity, observable
errors, convergence evidence, and certification state are exported wherever
the retained pipeline provides them.

## Artifact Parity

Each candidate is stored below an independent graph-method folder inside its
q15 or q20 test. It must produce the closest applicable equivalent of the
retained PINN surface:

- resolved configuration and lineage metadata;
- checkpoints and complete coefficient exports;
- curriculum and residual histories;
- support/coefficient plots and `hcd_connection_summary.pdf`;
- physical-validation JSON/CSV data;
- `physical_method_comparison_table.pdf` and related comparison plots;
- certification summaries and a concise result diagnosis.

Missing artifacts must be explicitly marked inapplicable or `not tested`; they
must not silently disappear.

## Acceptance

The graph method is reported as an independent candidate regardless of outcome.
It replaces no retained benchmark unless projected gates pass, canonical
full-support physical validation passes, and ground-state fidelity improves
without an unacceptable energy-error regression.

## Deferred Research Direction

A curriculum across increasing qubit counts, using self-supervised graph
weights from smaller systems as initialization, may be studied later. It is
explicitly excluded here: q15 and q20 are initialized and trained independently,
and no cross-system checkpoint transfer is permitted.
