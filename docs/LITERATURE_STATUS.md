# Literature Status

A focused literature check in June 2026 did not find a general method that
extracts exact AGP terms for generic large-qubit interacting systems in a
polynomially scalable way.

The nearby literature has moved in several useful directions:

- exact or near-exact AGP construction for special algebraic, symmetric, or
  integrable systems;
- Krylov-space and Lanczos formulations;
- regularized/local AGP approximations with performance guarantees;
- sparse Pauli and tensor-network methods;
- adaptive or weighted nested-commutator expansions.

The important conclusion for this repository is that the original scalability
caveat still appears valid for generic systems: exact AGPs can have
exponentially many nonlocal Pauli terms, so even writing the answer can be
exponential. This repository therefore treats scalable AGP learning as a
sparse/projected/local problem unless a special model structure proves
otherwise.

Useful papers and preprints to revisit while developing this project:

- Lawrence et al., "A numerical approach for calculating exact non-adiabatic
  terms in quantum dynamics", SciPost Phys. 18, 014 (2025).
- Takahashi and del Campo, "Shortcuts to Adiabaticity in Krylov Space",
  Phys. Rev. X 14, 011032 (2024).
- Morawetz and Polkovnikov, "Universal Counterdiabatic Driving in Krylov
  Space", PRX Quantum 6, 040320 (2025).
- Finzgar et al., "Counterdiabatic Driving with Performance Guarantees",
  Phys. Rev. Lett. 135, 180602 (2025).
- Hatomura, "Universal Digitized Counterdiabatic Driving", arXiv:2601.15972.
- Tang, Chen, and Wei, "Weighted Nested Commutators for Scalable
  Counterdiabatic State Preparation", arXiv:2603.25625.
- Cipolla and Durastante, "Pauli-Sparse regularised Counterdiabatic Shortcuts
  for Linear-Ramp QAOA", arXiv:2606.28536.

