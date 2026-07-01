from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from projected_sparse_training_common import (  # noqa: E402
    ProjectedSparseLossWeights,
    build_projected_support,
    make_projected_model,
    select_device,
)
from training_script import model_config_from_payload  # noqa: E402
from utils import load_pauli_hamiltonian_pair  # noqa: E402


RUN_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = RUN_DIR / "config.json"


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_body_weights(model: torch.nn.Module, checkpoint: dict[str, object]) -> None:
    state = checkpoint["model_state_dict"]
    body_state = {
        key.removeprefix("body."): value
        for key, value in state.items()
        if key.startswith("body.")
    }
    model.body.load_state_dict(body_state)


def norm_sq_subset(values: torch.Tensor, indices: list[int]) -> torch.Tensor:
    if not indices:
        return torch.zeros((), dtype=values.real.dtype, device=values.device)
    index = torch.tensor(indices, dtype=torch.long, device=values.device)
    subset = values.index_select(-1, index)
    return torch.mean(torch.sum(torch.abs(subset) ** 2, dim=-1).real)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained q=20 AGP on a larger residual holdout basis.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trained-run", type=Path, default=RUN_DIR / "runs" / "agp_1536")
    parser.add_argument("--residual-top-k", type=int, default=8192)
    parser.add_argument("--intermediate-top-k", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = load_json(args.config)
    config = model_config_from_payload(payload)
    training = payload.get("training", {})
    parameters = training.get("parameters", {}) if isinstance(training, dict) else {}
    support_sweep = payload.get("support_sweep", {})
    intermediate_top_k = (
        int(args.intermediate_top_k)
        if args.intermediate_top_k is not None
        else int(support_sweep.get("intermediate_top_k", 2048))
        if isinstance(support_sweep, dict)
        else 2048
    )
    num_points = int(parameters.get("num_points", 16))

    checkpoint_path = args.trained_run / "Models_Data" / "training_checkpoint.pt"
    train_metadata_path = args.trained_run / "Models_Data" / "support_metadata.json"
    train_history_path = args.trained_run / "Models_Data" / "loss_history.json"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    trained_agp_labels = list(checkpoint["agp_labels"])
    trained_residual_labels = set(checkpoint["residual_labels"])
    train_metadata = load_json(train_metadata_path)
    train_history = load_json(train_history_path)

    hamiltonian_path = Path(config.hamiltonian_source)
    if not hamiltonian_path.is_absolute():
        hamiltonian_path = ROOT / hamiltonian_path
    h0, h1 = load_pauli_hamiltonian_pair(
        hamiltonian_path,
        system=config.system,
        n_qubits=config.n_qubits,
        distance=config.distance,
    )
    support = build_projected_support(
        h0,
        h1,
        agp_top_k=len(trained_agp_labels),
        intermediate_top_k=intermediate_top_k,
        residual_top_k=args.residual_top_k,
        agp_labels=trained_agp_labels,
        stage=0,
    )

    device = select_device(args.device)
    model = make_projected_model(h0, h1, support, config, device)
    load_body_weights(model, checkpoint)
    model.eval()

    tau = torch.linspace(0.0, 1.0, num_points, device=device).view(-1, 1)
    t = config.t_initial + config.physical_time * tau
    weights = ProjectedSparseLossWeights(residual=1.0, agp_l2=0.0)
    with torch.no_grad():
        _, diagnostics = model.loss(t, weights=weights)
        residual = model.euler_lagrange_residual(t)
        reference = model.euler_lagrange_reference_residual(t)

    residual_labels = list(model.residual_labels)
    unseen_indices = [idx for idx, label in enumerate(residual_labels) if label not in trained_residual_labels]
    unseen_residual = norm_sq_subset(residual, unseen_indices)
    unseen_reference = norm_sq_subset(reference, unseen_indices)
    eps = torch.finfo(unseen_residual.dtype).eps
    unseen_relative = unseen_residual / torch.clamp(unseen_reference, min=eps)
    full_basis_size = 4 ** config.n_qubits

    result = {
        "trained_run": str(args.trained_run.relative_to(RUN_DIR) if args.trained_run.is_relative_to(RUN_DIR) else args.trained_run),
        "n_qubits": config.n_qubits,
        "agp_terms": len(model.agp_labels),
        "agp_fraction_of_full_basis": len(model.agp_labels) / full_basis_size,
        "train_residual_terms": int(train_metadata["final_residual_terms"]),
        "holdout_residual_terms": len(model.residual_labels),
        "unseen_residual_terms": len(unseen_indices),
        "intermediate_terms": len(model.intermediate_labels),
        "hamiltonian_terms": len(model.hamiltonian_labels),
        "first_commutator_nnz": model.first_commutator.nnz,
        "second_commutator_nnz": model.second_commutator.nnz,
        "training_final_relative_residual": float(train_history[-1]["relative_residual"]),
        "training_best_relative_residual": float(min(train_history, key=lambda row: row["total"])["relative_residual"]),
        "holdout_total_residual": float(diagnostics["residual"].detach().cpu().item()),
        "holdout_reference_residual": float(diagnostics["reference_residual"].detach().cpu().item()),
        "holdout_relative_residual": float(diagnostics["relative_residual"].detach().cpu().item()),
        "unseen_residual": float(unseen_residual.detach().cpu().item()),
        "unseen_reference_residual": float(unseen_reference.detach().cpu().item()),
        "unseen_relative_residual": float(unseen_relative.detach().cpu().item()),
        "residual_basis_note": (
            "The model weights are unchanged. Only the projected residual basis is enlarged; "
            "unseen metrics are computed on residual labels absent from the original training projection."
        ),
    }
    output = args.output or (RUN_DIR / "Models_Data" / f"holdout_residual_agp_{len(model.agp_labels)}_residual_{len(model.residual_labels)}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
