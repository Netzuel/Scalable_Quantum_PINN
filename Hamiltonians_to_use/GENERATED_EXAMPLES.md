# Generated Hamiltonian Examples

These examples were generated directly as Qiskit Nature/PySCF sparse Pauli
operators with:

```bash
conda run -n torch-mps python tools/generate_qiskit_pauli_hamiltonian.py chemistry ...
```

The stored endpoint convention is:

- `initial`: computational-basis diagonal projection of the final Hamiltonian,
  built symbolically by retaining only `I/Z` Pauli strings.
- `final`: full Qiskit `SparsePauliOp` problem Hamiltonian.

All examples use H2 at a bond distance of 1.0 Angstrom, Jordan-Wigner mapping,
and include the nuclear-repulsion constant as an identity shift.

| Pair id | Basis | Active electrons | Active spatial orbitals | Qubits | Initial terms | Final terms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Hidrogen_8_qubits_1_0` | `6-31g` | full problem | full problem | 8 | 37 | 185 |
| `Hidrogen_10_qubits_1_0` | `cc-pvdz` | 2 | 5 | 10 | 56 | 252 |
| `Hidrogen_12_qubits_1_0` | `cc-pvdz` | 2 | 6 | 12 | 79 | 327 |
| `Hidrogen_20_qubits_1_0` | `cc-pvdz` | full problem | full problem | 20 | 211 | 2951 |

These are Hamiltonian-input examples only. A full-Pauli AGP PINN with `4**q`
outputs is not practical for the largest entries; these files are meant to
support sparse/support-selected AGP experiments.

## Analytic Sparse Spin-Model Examples

The generator also supports large-qubit analytic sparse Pauli inputs for smoke
tests, without constructing dense Hilbert-space matrices and without running a
large chemistry backend:

```bash
conda run -n torch-mps python tools/generate_qiskit_pauli_hamiltonian.py transverse-ising ...
```

| Pair id | Model | Qubits | Initial terms | Final terms |
| --- | --- | ---: | ---: | ---: |
| `TransverseIsing_156_qubits_1_0` | open-chain transverse-field Ising with weak deterministic inhomogeneity | 156 | 155 | 311 |

The q=156 example is intended to exercise the projected sparse AGP machinery at
the target qubit count. It is not a chemistry Hamiltonian and it is not a
full-basis AGP experiment.
