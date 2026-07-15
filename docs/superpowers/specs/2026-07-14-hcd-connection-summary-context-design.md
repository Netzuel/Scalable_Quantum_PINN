# HCD Connection Summary Context Design

## Goal

Make every `hcd_connection_summary.pdf` identify the Hamiltonian path and show
the exact final-Hamiltonian ground energy beside the final energy obtained from
the PINN-driven evolution. Regenerate the retained q15 and q20 figures.

## Data provenance

- Read the physical system and qubit count from the nearest run/study config.
- Read energies only from a compatible physical-validation summary. A study
  config may provide an independently validated exact ground energy when no
  dynamical summary exists.
- Never turn a missing q20 dynamical calculation into a numeric PINN energy.
  The footer must say `not computed` until a compatible artifact exists.
- q15 uses the saved statevector summary. q20 uses its H2/cc-pVDZ,
  Jordan-Wigner Hamiltonian metadata and an independently validated FCI ground
  energy.

## Figure layout

Keep the two existing data panels unchanged. Add a centered four-line context
band below them:

1. the source-specific `H_initial` expression;
2. the source-specific `H_final` expression;
3. `E_0(H_final)`;
4. `E_PINN(T)`.

Use STIX/Matplotlib mathtext, increase the figure height and bottom margin, and
keep all lines inside the PDF canvas without overlap.

For the transverse-Ising driver/problem family, show the analytic sums. For
q20 hydrogen, show the stored diagonal-projection initial operator and the
sparse Pauli expansion of the molecular final operator.

## Code boundaries

- `scripts/agp_plot_annotations.py` owns config/summary discovery and formatted
  context lines.
- `scripts/projected_sparse_training_common.py` owns figure dimensions and
  draws the returned lines.
- `tests/sparse_agp_curriculum/scripts/agp_regenerate_hcd_summaries.py` remains
  the shared regeneration entrypoint for both studies.

## Validation

Add focused tests for both physical systems, missing dynamical data, and the
exact/PINN energy labels. Run them in `torch-mps`, regenerate q15 and q20, then
render the canonical adaptive-refinement PDFs to PNG and inspect their spacing
and text.
