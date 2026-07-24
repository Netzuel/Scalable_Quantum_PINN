# Transverse-Field to Diagonal-Ising Benchmarks

This scenario is the tractable physical playground used to evaluate the sparse
AGP curriculum across system sizes. It uses

```text
H_initial = -sum_i X_i
H_final   = -sum_i h_i Z_i - sum_i J_i Z_i Z_{i+1}
H_AD(lambda) = (1 - lambda) H_initial + lambda H_final.
```

The final Hamiltonian is diagonal in the computational basis, so its exact
ground energy and ground bitstrings are available from the structure-aware
solver under `scripts/numerical_solver/`. This makes the scenario useful for
physical validation without making those exact targets part of the PINN loss.

## Studies

- `q15/sweep_test/size_intensive_pinn/`: retained v6 curriculum with exact
  statevector validation.
- `q20/sweep_test/size_intensive_pinn/`: retained v6 curriculum with canonical
  full-support MPS validation.
- `q25/sweep_test/size_intensive_pinn/`: retained v6 scaling benchmark with
  canonical full-support MPS validation.
- `q156/sweep_test/`: legacy large-system benchmark with canonical full-support
  tensor-network validation; not yet retrained under v6.

Each `sweep_test/` keeps its `config.json`, local documentation, and ignored
`runs/` tree together. Reusable training code remains in the repository-level
`scripts/` directory; benchmark-family validation entrypoints remain in
`tests/sparse_agp_curriculum/scripts/`.

## Tensor-Network Validation Examples

All three retained TN rows use the joint-time MPO/TDVP backend. The canonical
PINN deployment always includes all `K=32768` learned AGP terms.

| q | Exact `E_0` | PINN `E(T)` | Energy error | PINN ground fidelity | TN status |
|---:|---:|---:|---:|---:|---|
| 15 | -19.25 | -19.1097006 | 0.1402994 | 0.9646510 | timestep/bond/MPO pass; full-support exact check not tested |
| 20 | -26.0 | -25.6478383 | 0.3521617 | 0.9377128 | certified pass |
| 156 | -209.6 | -201.3901459 | 8.2098541 | 0.2394617 | certified pass |

At q15, an additional matching-support 2,048-term ablation compares TN and
exact statevector propagation directly. The learned-protocol differences are
`9.15015e-5` in final energy and `1.49043e-5` in fidelity, validating the
propagator without mislabeling that reduced-support check as canonical. At q20
and q156, exact time evolution is unavailable under the project threshold, so
the physical rows are accepted only after independent timestep, state-bond,
MPO-action, and learned-source-completeness gates pass.

## Current Normalized Variational-Action Benchmark

The `size_intensive_pinn/` studies define the retained projected loss plus a
reference-normalized variational-action term with fixed weight `0.1`.
All systems were trained independently from scratch at `T=1`, and physical
validation deployed every learned term.

| q | K | Energy error | Ground fidelity | Validation |
|---:|---:|---:|---:|---|
| 15 | 32,768 | 0.1339216 | 0.9768832 | exact statevector |
| 20 | 58,368 | 0.1607203 | 0.9764755 | certified TN |
| 25 | 91,136 | 0.2998297 | 0.9547459 | certified TN |

Each size passes the `0.95` fidelity threshold. The q20-to-q25 fidelity drop is
`0.0217296`, above the original `0.01` smoothness diagnostic. The methodology
is retained because it improves the preceding q15 and q20 benchmarks and keeps
q25 above `0.95`; the drop remains an explicit scaling limitation.
