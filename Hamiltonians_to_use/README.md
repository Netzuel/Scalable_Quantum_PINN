# Hamiltonian Inputs

This folder contains Hamiltonians copied from the original `Quantum_PINN`
repository and adapted Pauli-coordinate representations for this scalable code
line.

- `Hamiltonians.h5`: copied dense source data from `Quantum_PINN`. It is kept
  only as provenance/input material for conversion.
- `pauli_decompositions/index.json`: canonical sparse Pauli-coordinate
  Hamiltonian index used by training scripts.
- `pauli_decompositions/<system>/<q>_qubits/distance_<r>.json`: one
  self-contained endpoint pair per system, qubit count, and distance. Each file
  stores `initial` (`H_hf`, `tau=0`) and `final` (`H_prob`, `tau=1`) as
  Pauli coefficients.
- `Hamiltonians_pauli.json`: legacy aggregate sparse Pauli-coefficient file,
  kept for backward-compatible loading.
- `h2_6-31g.pickle`: auxiliary source artifact copied from `Quantum_PINN`.
- `README_Quantum_PINN.md`: original source-folder note.
- `PROVENANCE.md`: notes on how the dense source file was generated in the old
  repository.
- `GENERATED_EXAMPLES.md`: Qiskit Nature/PySCF examples generated directly as
  sparse Pauli-coordinate Hamiltonian pairs for larger qubit counts.

The coefficient convention is:

```text
H = sum_P C_P P
C_P = Tr(P H) / 2**q
P in {I, X, Y, Z}^{tensor q}
```

Regenerate the symbolic files with:

```bash
conda run -n torch-mps python tools/adapt_quantum_pinn_hamiltonians.py
```

For new Qiskit Nature chemistry Hamiltonians, generate sparse Pauli-coordinate
JSON directly instead of going through dense matrices:

```bash
conda run -n torch-mps python tools/generate_qiskit_pauli_hamiltonian.py chemistry --help
```

The reusable PINN/loss implementation does not construct dense Hilbert-space
matrices; dense matrices are only touched by the one-time adapter above.
