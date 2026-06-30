"""PINN models for sparse Pauli-coordinate AGP discovery.

The key design choice is that the network emits coefficients only for a chosen
counterdiabatic ansatz support. The Euler-Lagrange residual is evaluated in a
fixed sparse Pauli algebra, not through dense matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn

from utils import (
    PauliAlgebra,
    SparsePauliOperator,
    build_commutator_closure,
    sort_pauli_labels,
)


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation {name!r}.")


class MLP(nn.Module):
    """Small fully connected network used by the sparse AGP PINN."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_width: int = 64,
        hidden_layers: int = 4,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width_in = in_features
        for _ in range(hidden_layers):
            layers.append(nn.Linear(width_in, hidden_width))
            layers.append(_activation(activation))
            width_in = hidden_width
        layers.append(nn.Linear(width_in, out_features))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass(frozen=True)
class LossWeights:
    """Weights for the sparse AGP PINN objective."""

    residual: float = 1.0
    boundary: float = 100.0
    velocity: float = 1e-3
    agp_l2: float = 1e-6


class ScalableAGPPINN(nn.Module):
    """Physics-informed sparse Pauli model for the adiabatic gauge potential.

    Parameters
    ----------
    h_initial, h_final:
        Sparse Pauli Hamiltonians for the initial and final interpolation.
    agp_labels:
        Pauli strings that the network is allowed to output for ``A_lambda``.
        This is the primary scalability control.
    max_closure_weight:
        Optional locality cap for the residual algebra. Keeping this finite
        turns the loss into a projected/local residual; leaving it ``None``
        computes the full closure implied by the selected support, subject to
        ``max_closure_terms``.
    """

    def __init__(
        self,
        h_initial: SparsePauliOperator,
        h_final: SparsePauliOperator,
        agp_labels: Sequence[str],
        *,
        hidden_width: int = 64,
        hidden_layers: int = 4,
        activation: str = "tanh",
        t_min: float = 0.0,
        t_max: float = 1.0,
        closure_rounds: int = 2,
        max_closure_weight: int | None = None,
        max_closure_terms: int = 20000,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()
        if h_initial.n_qubits != h_final.n_qubits:
            raise ValueError("Initial and final Hamiltonians must use the same qubit count.")
        if t_max <= t_min:
            raise ValueError("t_max must be greater than t_min.")
        self.n_qubits = h_initial.n_qubits
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.agp_labels = sort_pauli_labels(agp_labels)
        seed_labels = set(h_initial.labels) | set(h_final.labels) | set(self.agp_labels)
        basis_labels = build_commutator_closure(
            seed_labels,
            max_rounds=closure_rounds,
            max_weight=max_closure_weight,
            max_terms=max_closure_terms,
        )
        self.algebra = PauliAlgebra(basis_labels)
        agp_indices = [self.algebra.index[label] for label in self.agp_labels]
        self.register_buffer("agp_indices", torch.tensor(agp_indices, dtype=torch.long))
        self.register_buffer("h_initial_vec", self.algebra.vector_from_operator(h_initial, dtype=dtype))
        self.register_buffer("h_final_vec", self.algebra.vector_from_operator(h_final, dtype=dtype))
        self.body = MLP(
            1,
            1 + len(self.agp_labels),
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
            activation=activation,
        )

    @property
    def basis_labels(self) -> list[str]:
        return list(self.algebra.basis_labels)

    def _time_column(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        if t.ndim != 2 or t.shape[-1] != 1:
            raise ValueError("Time input must have shape (batch,) or (batch, 1).")
        return t

    def _normalized_time(self, t: torch.Tensor) -> torch.Tensor:
        return (t - self.t_min) / (self.t_max - self.t_min)

    def forward(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        t = self._time_column(t)
        tau = self._normalized_time(t)
        raw = self.body(tau)
        correction = torch.tanh(raw[..., :1])
        # Exact endpoint constraints: lambda(0)=0 and lambda(1)=1.
        lam = tau + tau * (1.0 - tau) * correction
        agp_coefficients = raw[..., 1:]
        return {
            "lambda": lam,
            "agp_coefficients": agp_coefficients,
        }

    def embed_agp(self, agp_coefficients: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(
            agp_coefficients.shape[:-1] + (self.algebra.size,),
            dtype=self.h_initial_vec.dtype,
            device=agp_coefficients.device,
        )
        out[..., self.agp_indices.to(agp_coefficients.device)] = agp_coefficients.to(out.dtype)
        return out

    def sparse_operators(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        prediction = self.forward(t)
        lam = prediction["lambda"].to(self.h_initial_vec.dtype)
        h0 = self.h_initial_vec.to(t.device)
        h1 = self.h_final_vec.to(t.device)
        h_ad = (1.0 - lam) * h0 + lam * h1
        d_h_d_lambda = h1 - h0
        agp = self.embed_agp(prediction["agp_coefficients"])
        return {
            "lambda": prediction["lambda"],
            "agp": agp,
            "h_ad": h_ad,
            "d_h_d_lambda": d_h_d_lambda.expand_as(h_ad),
            "agp_coefficients": prediction["agp_coefficients"],
        }

    def euler_lagrange_residual(self, t: torch.Tensor) -> torch.Tensor:
        operators = self.sparse_operators(t)
        h_ad = operators["h_ad"]
        agp = operators["agp"]
        d_h_d_lambda = operators["d_h_d_lambda"]
        generator = 1.0j * d_h_d_lambda - self.algebra.commutator(agp, h_ad)
        return self.algebra.commutator(generator, h_ad)

    def counterdiabatic_hamiltonian(self, t: torch.Tensor) -> torch.Tensor:
        """Return sparse coefficients for ``H_AD + dot(lambda) A``."""

        t = self._time_column(t)
        if not t.requires_grad:
            t = t.clone().detach().requires_grad_(True)
        operators = self.sparse_operators(t)
        lam = operators["lambda"]
        d_lambda_dt = torch.autograd.grad(lam.sum(), t, create_graph=True)[0].to(operators["agp"].dtype)
        return operators["h_ad"] + d_lambda_dt * operators["agp"]

    def loss(
        self,
        t_collocation: torch.Tensor,
        *,
        weights: LossWeights | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the sparse physics-informed loss."""

        weights = weights or LossWeights()
        t_collocation = self._time_column(t_collocation)
        if not t_collocation.requires_grad:
            t_collocation = t_collocation.clone().detach().requires_grad_(True)

        prediction = self.forward(t_collocation)
        residual = self.euler_lagrange_residual(t_collocation)
        residual_loss = self.algebra.norm_sq(residual)

        d_lambda_dt = torch.autograd.grad(
            prediction["lambda"].sum(),
            t_collocation,
            create_graph=True,
        )[0]
        velocity_loss = torch.mean(d_lambda_dt.pow(2))
        agp_l2_loss = torch.mean(prediction["agp_coefficients"].pow(2))

        endpoints = torch.tensor(
            [[self.t_min], [self.t_max]],
            dtype=t_collocation.dtype,
            device=t_collocation.device,
        )
        endpoint_prediction = self.forward(endpoints)["lambda"]
        target = torch.tensor([[0.0], [1.0]], dtype=t_collocation.dtype, device=t_collocation.device)
        boundary_loss = torch.mean((endpoint_prediction - target).pow(2))

        total = (
            weights.residual * residual_loss
            + weights.boundary * boundary_loss
            + weights.velocity * velocity_loss
            + weights.agp_l2 * agp_l2_loss
        )
        diagnostics = {
            "total": total.detach(),
            "residual": residual_loss.detach(),
            "boundary": boundary_loss.detach(),
            "velocity": velocity_loss.detach(),
            "agp_l2": agp_l2_loss.detach(),
            "basis_size": torch.tensor(float(self.algebra.size), device=t_collocation.device),
            "agp_terms": torch.tensor(float(len(self.agp_labels)), device=t_collocation.device),
        }
        return total, diagnostics


def make_sparse_agp_pinn(
    h_initial_terms: Mapping[str, complex],
    h_final_terms: Mapping[str, complex],
    agp_labels: Sequence[str],
    **kwargs,
) -> ScalableAGPPINN:
    """Convenience constructor from plain Pauli-term dictionaries."""

    h_initial = SparsePauliOperator(h_initial_terms)
    h_final = SparsePauliOperator(h_final_terms, n_qubits=h_initial.n_qubits)
    return ScalableAGPPINN(h_initial, h_final, agp_labels, **kwargs)

