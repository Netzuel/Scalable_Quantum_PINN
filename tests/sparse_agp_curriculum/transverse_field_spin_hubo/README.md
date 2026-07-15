# Transverse-Field To Spin-HUBO Benchmarks

This scenario keeps the transverse-field driver and replaces the open-chain
Ising objective with a diagonal spin polynomial:

```text
H_initial = -sum_i X_i
H_final   = sum_S c_S product_(i in S) Z_i
```

The source coefficients use `s_i in {-1,+1}`. They map directly to `Z_i`
eigenvalues, so source spin `+1` corresponds to computational bit `0` and
source spin `-1` corresponds to bit `1`.

## Studies

- `run_002_hamiltonian_341/q24/sweep_test/`: first retained nonlocal
  spin-HUBO curriculum benchmark, selected from the source run_001--run_007
  pool while excluding run_008 and run_009.

The shared curriculum remains under the repository-level `scripts/` folder.
Spin-HUBO conversion and tensor-network validation helpers remain under
`tests/sparse_agp_curriculum/scripts/`.
