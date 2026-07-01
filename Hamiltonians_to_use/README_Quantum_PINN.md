# Hamiltonian Inputs

This folder contains the compact Hamiltonian inputs used by the retained
training scripts.

## Files

- `Hamiltonians.h5`: HDF5 container with initial Hartree-Fock-like Hamiltonians
  (`*_H_hf_*`) and final problem Hamiltonians (`*_H_prob_*`) for the molecule
  and bond-distance cases used by the scripts.
- `Generate_h5s.ipynb`: notebook used to generate the HDF5 file.
- `h2_6-31g.pickle`: auxiliary molecular data used by the historical
  generation workflow.

## HDF5 Key Pattern

The key pattern is:

```text
<system>_<n>_qubits_H_hf_<distance>
<system>_<n>_qubits_H_prob_<distance>
```

Examples:

```text
Hidrogen_2_qubits_H_hf_1_0      shape (4, 4)
Hidrogen_2_qubits_H_prob_1_0    shape (4, 4)
Li_4_qubits_H_hf_1_0            shape (16, 16)
Li_4_qubits_H_prob_1_0          shape (16, 16)
Hidrogen_6_qubits_H_hf_1_0      shape (64, 64)
Hidrogen_6_qubits_H_prob_1_0    shape (64, 64)
```

The local file contains 42 datasets covering H2/two-qubit, Li/four-qubit, and
hydrogen-family/six-qubit cases at distances from `0.5` to `3.5` where
available.

## Usage

Training scripts read this file through `h5py`, convert selected datasets to
PyTorch tensors, and feed them into `models.QPINN_complex_Pauli` as `H_hf` and
`H_prob`. Keep dataset names stable; scripts build keys with string
concatenation.
