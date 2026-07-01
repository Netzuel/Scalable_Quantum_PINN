# Hamiltonian Provenance

The dense source Hamiltonians were copied from the original `Quantum_PINN`
repository. They are retained only as provenance and conversion inputs.

## Original Generation Path

The original source folder contains:

- `Generate_h5s.ipynb`
- `Hamiltonians.h5`
- `h2_6-31g.pickle`

Inspection of `Generate_h5s.ipynb` shows:

- The 2-qubit `Hidrogen` matrices are hard-coded in the notebook. The notebook
  comments this section as `n_qubits = 2 (stgo3)`, which appears to be the
  historical spelling used there.
- The 4-qubit `Li` matrices are also hard-coded in the notebook.
- The 6-qubit `Hidrogen` matrices are loaded from `h2_6-31g.pickle` using
  `pandas.read_pickle`. The pickle entries contain `Molecule: h2`,
  `Basis: 6-31g`, `Distance`, `H_hf`, `H_fin`, and `Commutator`.
- The notebook writes all endpoint matrices to `Hamiltonians.h5` as float32
  HDF5 datasets.

The HDF5 keys have this pattern:

```text
<system>_<n>_qubits_H_hf_<distance>
<system>_<n>_qubits_H_prob_<distance>
```

`H_hf` is the initial endpoint used at `tau=0`; `H_prob` is the final endpoint
used at `tau=1`.

## Pauli Decomposition Used Here

The scalable code does not train from dense matrices. Each dense endpoint was
converted once into Pauli coordinates:

```text
H = sum_P C_P P
C_P = Tr(P H) / 2**q
P in {I, X, Y, Z}^{tensor q}
```

Only coefficients with `abs(C_P) > 1e-8` are stored. This keeps the
Hamiltonians sparse in Pauli-coordinate form while preserving the exact
endpoint operator up to the configured drop tolerance.

The canonical training input is:

```text
Hamiltonians_to_use/pauli_decompositions/index.json
```

The dense `Hamiltonians.h5` and `h2_6-31g.pickle` files are provenance/source
material for regeneration only.

## Generating New Hamiltonians

For new chemistry cases, prefer generating the qubit Hamiltonian directly as a
Qiskit `SparsePauliOp` and exporting its Pauli coefficients, instead of
building a dense `2**q x 2**q` matrix and decomposing it afterward. The helper
script is:

```bash
conda run -n torch-mps python tools/generate_qiskit_pauli_hamiltonian.py chemistry \
  --system Hidrogen \
  --distance 1.0 \
  --atom "H 0 0 0; H 0 0 1.0" \
  --basis sto3g \
  --mapper parity \
  --taper \
  --target-qubits 2 \
  --include-nuclear-repulsion \
  --update-index
```

This path requires the optional chemistry dependencies:

```bash
conda run -n torch-mps python -m pip install -e ".[chemistry]"
```

The requested qubit count `q` is not independent for molecular Hamiltonians.
It is determined by the chosen active space, fermion-to-qubit mapper, and any
symmetry tapering. The generator treats `--target-qubits` as a validation check
and raises if the selected physical/mapping setup produces a different number
of qubits.

The generated endpoint pair follows the same protocol as the retained
`Quantum_PINN` data:

- `final`: the full problem Hamiltonian in Pauli coordinates.
- `initial`: the computational-basis diagonal projection of `final`, obtained
  symbolically by retaining only Pauli strings made from `I` and `Z`.

This preserves the scalable rule that training consumes Pauli-coordinate JSON,
not dense Hilbert-space matrices.
