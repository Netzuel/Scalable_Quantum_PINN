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
    ProjectedCommutator,
    SparseRightCommutator,
    SparsePauliOperator,
    all_pauli_labels,
    build_commutator_closure,
    fixed_sinusoidal_schedule,
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


class QuadraticLayer(nn.Module):
    """QRes-style layer with a linear path plus a small quadratic branch."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.quad_left = nn.Linear(in_features, out_features)
        self.quad_right = nn.Linear(in_features, out_features)

        nn.init.xavier_normal_(self.linear.weight, gain=1.0)
        nn.init.constant_(self.linear.bias, 0.0)
        nn.init.normal_(self.quad_left.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.quad_right.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.quad_left.bias, 0.0)
        nn.init.constant_(self.quad_right.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.quad_left(x) * self.quad_right(x)


class QuadraticMLP(nn.Module):
    """QRes-style MLP used for the full-Pauli coefficient PINN."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_width: int = 56,
        hidden_layers: int = 3,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [QuadraticLayer(in_features, hidden_width)]
            + [QuadraticLayer(hidden_width, hidden_width) for _ in range(hidden_layers)]
            + [QuadraticLayer(hidden_width, out_features)]
        )
        self.hidden_activations = nn.ModuleList([_activation(activation) for _ in range(len(self.layers) - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, activation in zip(self.layers[:-1], self.hidden_activations):
            x = activation(layer(x))
        return self.layers[-1](x)


def _make_body(
    in_features: int,
    out_features: int,
    *,
    hidden_width: int,
    hidden_layers: int,
    activation: str,
    layer_type: str,
) -> nn.Module:
    layer_type = layer_type.lower()
    if layer_type in {"quadratic", "qres"}:
        return QuadraticMLP(
            in_features,
            out_features,
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
            activation=activation,
        )
    if layer_type == "linear":
        return MLP(
            in_features,
            out_features,
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
            activation=activation,
        )
    raise ValueError(f"Unsupported layer_type {layer_type!r}.")


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
        layer_type: str = "linear",
        t_min: float = 0.0,
        t_max: float = 1.0,
        closure_rounds: int = 2,
        max_closure_weight: int | None = None,
        max_closure_terms: int = 20000,
        fixed_schedule: bool = False,
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
        self.fixed_schedule = bool(fixed_schedule)
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
        self.layer_type = layer_type
        out_features = len(self.agp_labels) if self.fixed_schedule else 1 + len(self.agp_labels)
        self.body = _make_body(
            1,
            out_features,
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
            activation=activation,
            layer_type=layer_type,
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
        if self.fixed_schedule:
            lam, d_lambda_dt = fixed_sinusoidal_schedule(t, t_min=self.t_min, t_max=self.t_max)
            agp_coefficients = raw
        else:
            correction = torch.tanh(raw[..., :1])
            # Exact endpoint constraints: lambda(0)=0 and lambda(1)=1.
            lam = tau + tau * (1.0 - tau) * correction
            d_lambda_dt = None
            agp_coefficients = raw[..., 1:]
        return {
            "lambda": lam,
            "d_lambda_dt": d_lambda_dt,
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
            "d_lambda_dt": prediction["d_lambda_dt"],
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

    def euler_lagrange_reference_residual(self, t: torch.Tensor) -> torch.Tensor:
        """Residual for the same projection with ``A_lambda=0``."""

        operators = self.sparse_operators(t)
        generator = 1.0j * operators["d_h_d_lambda"]
        return self.algebra.commutator(generator, operators["h_ad"])

    def counterdiabatic_hamiltonian(self, t: torch.Tensor) -> torch.Tensor:
        """Return sparse coefficients for ``H_AD + dot(lambda) A``."""

        t = self._time_column(t)
        if not t.requires_grad:
            t = t.clone().detach().requires_grad_(True)
        operators = self.sparse_operators(t)
        d_lambda_dt = operators["d_lambda_dt"]
        if d_lambda_dt is None:
            lam = operators["lambda"]
            d_lambda_dt = torch.autograd.grad(lam.sum(), t, create_graph=True)[0]
        d_lambda_dt = d_lambda_dt.to(operators["agp"].dtype)
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
        reference_residual = self.euler_lagrange_reference_residual(t_collocation)
        reference_loss = self.algebra.norm_sq(reference_residual)
        eps = torch.finfo(residual_loss.dtype).eps
        relative_residual = residual_loss / torch.clamp(reference_loss, min=eps)
        residual_per_term = residual_loss / max(self.algebra.size, 1)
        reference_residual_per_term = reference_loss / max(self.algebra.size, 1)

        if self.fixed_schedule:
            velocity_loss = torch.zeros((), dtype=t_collocation.dtype, device=t_collocation.device)
            boundary_loss = torch.zeros((), dtype=t_collocation.dtype, device=t_collocation.device)
        else:
            d_lambda_dt = torch.autograd.grad(
                prediction["lambda"].sum(),
                t_collocation,
                create_graph=True,
            )[0]
            velocity_loss = torch.mean(d_lambda_dt.pow(2))

            endpoints = torch.tensor(
                [[self.t_min], [self.t_max]],
                dtype=t_collocation.dtype,
                device=t_collocation.device,
            )
            endpoint_prediction = self.forward(endpoints)["lambda"]
            target = torch.tensor([[0.0], [1.0]], dtype=t_collocation.dtype, device=t_collocation.device)
            boundary_loss = torch.mean((endpoint_prediction - target).pow(2))
        agp_l2_loss = torch.mean(prediction["agp_coefficients"].pow(2))

        total = (
            weights.residual * residual_loss
            + weights.boundary * boundary_loss
            + weights.velocity * velocity_loss
            + weights.agp_l2 * agp_l2_loss
        )
        diagnostics = {
            "total": total.detach(),
            "residual": residual_loss.detach(),
            "reference_residual": reference_loss.detach(),
            "relative_residual": relative_residual.detach(),
            "residual_per_term": residual_per_term.detach(),
            "reference_residual_per_term": reference_residual_per_term.detach(),
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


@dataclass(frozen=True)
class ProjectedSparseLossWeights:
    """Weights for fixed-schedule projected sparse AGP experiments."""

    residual: float = 1.0
    agp_l2: float = 1e-8


class ProjectedSparseAGPPINN(nn.Module):
    """Sparse AGP PINN with explicit projected commutator bases.

    This model is intended for large-qubit Hamiltonians where even automatic
    commutator closure is too large. It keeps the full sparse Hamiltonian
    support, but the AGP support, intermediate generator support, and residual
    support are explicit research choices.
    """

    def __init__(
        self,
        h_initial: SparsePauliOperator,
        h_final: SparsePauliOperator,
        agp_labels: Sequence[str],
        intermediate_labels: Sequence[str],
        residual_labels: Sequence[str],
        *,
        hidden_width: int = 56,
        hidden_layers: int = 3,
        activation: str = "silu",
        layer_type: str = "quadratic",
        t_min: float = 0.0,
        t_max: float = 1.0,
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
        self.hamiltonian_labels = sort_pauli_labels(set(h_initial.labels) | set(h_final.labels))
        self.agp_labels = sort_pauli_labels(agp_labels)
        self.intermediate_labels = sort_pauli_labels(set(intermediate_labels) | set(self.hamiltonian_labels))
        self.residual_labels = sort_pauli_labels(residual_labels)
        self.intermediate_index = {label: idx for idx, label in enumerate(self.intermediate_labels)}

        missing_agp = sorted(set(self.agp_labels) - set(self.intermediate_labels))
        if missing_agp:
            raise ValueError(f"AGP labels must be present in intermediate_labels; missing {missing_agp[:5]}.")
        missing_delta = sorted(set(self.hamiltonian_labels) - set(self.intermediate_labels))
        if missing_delta:
            raise ValueError(f"Hamiltonian labels must be present in intermediate_labels; missing {missing_delta[:5]}.")

        self.register_buffer(
            "h_initial_sparse",
            torch.tensor([h_initial.coefficient(label) for label in self.hamiltonian_labels], dtype=dtype),
        )
        self.register_buffer(
            "h_final_sparse",
            torch.tensor([h_final.coefficient(label) for label in self.hamiltonian_labels], dtype=dtype),
        )
        delta_intermediate = torch.zeros(len(self.intermediate_labels), dtype=dtype)
        for label in self.hamiltonian_labels:
            delta_intermediate[self.intermediate_index[label]] = h_final.coefficient(label) - h_initial.coefficient(label)
        self.register_buffer("h_delta_intermediate", delta_intermediate)
        self.first_commutator = ProjectedCommutator(
            self.agp_labels,
            self.hamiltonian_labels,
            self.intermediate_labels,
        )
        self.second_commutator = ProjectedCommutator(
            self.intermediate_labels,
            self.hamiltonian_labels,
            self.residual_labels,
        )
        self.layer_type = layer_type
        self.body = _make_body(
            1,
            len(self.agp_labels),
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
            activation=activation,
            layer_type=layer_type,
        )

    @property
    def output_terms(self) -> int:
        return len(self.agp_labels)

    def _time_column(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        if t.ndim != 2 or t.shape[-1] != 1:
            raise ValueError("Time input must have shape (batch,) or (batch, 1).")
        return t

    def _normalized_time(self, t: torch.Tensor) -> torch.Tensor:
        return (t - self.t_min) / (self.t_max - self.t_min)

    def schedule(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self._time_column(t)
        return fixed_sinusoidal_schedule(t, t_min=self.t_min, t_max=self.t_max)

    def forward(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        t = self._time_column(t)
        tau = self._normalized_time(t)
        lam, d_lambda_dt = self.schedule(t)
        return {
            "lambda": lam,
            "d_lambda_dt": d_lambda_dt,
            "agp_coefficients": self.body(tau),
        }

    def sparse_operators(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        prediction = self.forward(t)
        lam = prediction["lambda"].to(self.h_initial_sparse.dtype)
        h0 = self.h_initial_sparse.to(t.device)
        h1 = self.h_final_sparse.to(t.device)
        h_ad_sparse = (1.0 - lam) * h0 + lam * h1
        return {
            "lambda": prediction["lambda"],
            "d_lambda_dt": prediction["d_lambda_dt"],
            "agp_coefficients": prediction["agp_coefficients"],
            "h_ad_sparse": h_ad_sparse,
            "d_h_d_lambda": self.h_delta_intermediate.to(t.device).expand(
                prediction["agp_coefficients"].shape[:-1] + self.h_delta_intermediate.shape
            ),
        }

    def euler_lagrange_residual(self, t: torch.Tensor) -> torch.Tensor:
        operators = self.sparse_operators(t)
        commutator_1 = self.first_commutator.commutator(
            operators["agp_coefficients"],
            operators["h_ad_sparse"],
        )
        generator = 1.0j * operators["d_h_d_lambda"] - commutator_1
        return self.second_commutator.commutator(generator, operators["h_ad_sparse"])

    def euler_lagrange_reference_residual(self, t: torch.Tensor) -> torch.Tensor:
        """Projected residual for the same Hamiltonian path with ``A_lambda=0``."""

        operators = self.sparse_operators(t)
        generator = 1.0j * operators["d_h_d_lambda"]
        return self.second_commutator.commutator(generator, operators["h_ad_sparse"])

    def loss(
        self,
        t_collocation: torch.Tensor,
        *,
        weights: ProjectedSparseLossWeights | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        weights = weights or ProjectedSparseLossWeights()
        t_collocation = self._time_column(t_collocation)
        prediction = self.forward(t_collocation)
        residual = self.euler_lagrange_residual(t_collocation)
        residual_loss = PauliAlgebra.norm_sq(residual)
        reference_residual = self.euler_lagrange_reference_residual(t_collocation)
        reference_loss = PauliAlgebra.norm_sq(reference_residual)
        eps = torch.finfo(residual_loss.dtype).eps
        relative_residual = residual_loss / torch.clamp(reference_loss, min=eps)
        residual_per_term = residual_loss / max(len(self.residual_labels), 1)
        reference_residual_per_term = reference_loss / max(len(self.residual_labels), 1)
        agp_l2_loss = torch.mean(prediction["agp_coefficients"].pow(2))
        total = weights.residual * residual_loss
        if weights.agp_l2 != 0.0:
            total = total + weights.agp_l2 * agp_l2_loss
        diagnostics = {
            "total": total.detach(),
            "residual": residual_loss.detach(),
            "reference_residual": reference_loss.detach(),
            "relative_residual": relative_residual.detach(),
            "residual_per_term": residual_per_term.detach(),
            "reference_residual_per_term": reference_residual_per_term.detach(),
            "agp_l2": agp_l2_loss.detach(),
            "agp_terms": torch.tensor(float(len(self.agp_labels)), device=t_collocation.device),
            "hamiltonian_terms": torch.tensor(float(len(self.hamiltonian_labels)), device=t_collocation.device),
            "intermediate_terms": torch.tensor(float(len(self.intermediate_labels)), device=t_collocation.device),
            "residual_terms": torch.tensor(float(len(self.residual_labels)), device=t_collocation.device),
            "first_commutator_nnz": torch.tensor(float(self.first_commutator.nnz), device=t_collocation.device),
            "second_commutator_nnz": torch.tensor(float(self.second_commutator.nnz), device=t_collocation.device),
        }
        return total, diagnostics


@dataclass(frozen=True)
class FullPauliLossWeights:
    """Weights for the fixed-schedule full-Pauli AGP objective."""

    residual: float = 1.0
    agp_l2: float = 0.0


class FullPauliAGPPINN(nn.Module):
    """PINN that emits all ``4**q`` Pauli coefficients for the AGP.

    The schedule is fixed to ``lambda(t)=sin^2(pi tau / 2)`` on the configured
    interval, with ``tau=(t-t_min)/(t_max-t_min)``. The reported
    ``d_lambda_dt`` includes the chain-rule factor. The Euler-Lagrange residual
    is evaluated by symbolic Pauli commutators against the sparse Hamiltonian
    support, without dense matrices.
    """

    def __init__(
        self,
        h_initial: SparsePauliOperator,
        h_final: SparsePauliOperator,
        *,
        hidden_width: int = 56,
        hidden_layers: int = 3,
        activation: str = "silu",
        layer_type: str = "quadratic",
        t_min: float = 0.0,
        t_max: float = 1.0,
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
        self.pauli_labels = all_pauli_labels(self.n_qubits)
        self.agp_labels = list(self.pauli_labels)
        h_support = set(h_initial.labels) | set(h_final.labels)
        self.hamiltonian_labels = [label for label in self.pauli_labels if label in h_support]
        self.full_index = {label: idx for idx, label in enumerate(self.pauli_labels)}
        hamiltonian_indices = [self.full_index[label] for label in self.hamiltonian_labels]
        self.register_buffer("hamiltonian_indices", torch.tensor(hamiltonian_indices, dtype=torch.long))
        self.register_buffer(
            "h_initial_sparse",
            torch.tensor([h_initial.coefficient(label) for label in self.hamiltonian_labels], dtype=dtype),
        )
        self.register_buffer(
            "h_final_sparse",
            torch.tensor([h_final.coefficient(label) for label in self.hamiltonian_labels], dtype=dtype),
        )
        h_delta_full = torch.zeros(len(self.pauli_labels), dtype=dtype)
        for label in self.hamiltonian_labels:
            h_delta_full[self.full_index[label]] = h_final.coefficient(label) - h_initial.coefficient(label)
        self.register_buffer("h_delta_full", h_delta_full)
        self.right_commutator = SparseRightCommutator(self.pauli_labels, self.hamiltonian_labels)
        self.layer_type = layer_type
        self.body = _make_body(
            1,
            len(self.pauli_labels),
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
            activation=activation,
            layer_type=layer_type,
        )

    @property
    def basis_labels(self) -> list[str]:
        return list(self.pauli_labels)

    @property
    def output_terms(self) -> int:
        return len(self.pauli_labels)

    def _time_column(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        if t.ndim != 2 or t.shape[-1] != 1:
            raise ValueError("Time input must have shape (batch,) or (batch, 1).")
        return t

    def _normalized_time(self, t: torch.Tensor) -> torch.Tensor:
        return (t - self.t_min) / (self.t_max - self.t_min)

    def schedule(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self._time_column(t)
        return fixed_sinusoidal_schedule(t, t_min=self.t_min, t_max=self.t_max)

    def forward(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        t = self._time_column(t)
        tau = self._normalized_time(t)
        lam, d_lambda_dt = self.schedule(t)
        return {
            "lambda": lam,
            "d_lambda_dt": d_lambda_dt,
            "agp_coefficients": self.body(tau),
        }

    def embed_hamiltonian(self, hamiltonian_coefficients: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(
            hamiltonian_coefficients.shape[:-1] + (len(self.pauli_labels),),
            dtype=self.h_delta_full.dtype,
            device=hamiltonian_coefficients.device,
        )
        out[..., self.hamiltonian_indices.to(hamiltonian_coefficients.device)] = hamiltonian_coefficients.to(out.dtype)
        return out

    def sparse_operators(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        prediction = self.forward(t)
        lam = prediction["lambda"].to(self.h_initial_sparse.dtype)
        h0 = self.h_initial_sparse.to(t.device)
        h1 = self.h_final_sparse.to(t.device)
        h_ad_sparse = (1.0 - lam) * h0 + lam * h1
        return {
            "lambda": prediction["lambda"],
            "d_lambda_dt": prediction["d_lambda_dt"],
            "agp": prediction["agp_coefficients"].to(self.h_initial_sparse.dtype),
            "h_ad_sparse": h_ad_sparse,
            "h_ad": self.embed_hamiltonian(h_ad_sparse),
            "d_h_d_lambda": self.h_delta_full.to(t.device).expand(
                prediction["agp_coefficients"].shape[:-1] + self.h_delta_full.shape
            ),
            "agp_coefficients": prediction["agp_coefficients"],
        }

    def euler_lagrange_residual(self, t: torch.Tensor) -> torch.Tensor:
        operators = self.sparse_operators(t)
        commutator_1 = self.right_commutator.commutator(operators["agp"], operators["h_ad_sparse"])
        generator = 1.0j * operators["d_h_d_lambda"] - commutator_1
        return self.right_commutator.commutator(generator, operators["h_ad_sparse"])

    def euler_lagrange_reference_residual(self, t: torch.Tensor) -> torch.Tensor:
        """Residual for the same Hamiltonian path with ``A_lambda=0``."""

        operators = self.sparse_operators(t)
        generator = 1.0j * operators["d_h_d_lambda"]
        return self.right_commutator.commutator(generator, operators["h_ad_sparse"])

    def counterdiabatic_hamiltonian(self, t: torch.Tensor) -> torch.Tensor:
        operators = self.sparse_operators(t)
        d_lambda_dt = operators["d_lambda_dt"].to(operators["agp"].dtype)
        return operators["h_ad"] + d_lambda_dt * operators["agp"]

    def loss(
        self,
        t_collocation: torch.Tensor,
        *,
        weights: FullPauliLossWeights | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the symbolic Euler-Lagrange action loss.

        No endpoint loss is needed here: the schedule is fixed with
        ``lambda(t_min)=0``, ``lambda(t_max)=1`` and
        ``d lambda / dt = 0`` at both endpoints, so the counterdiabatic
        Hamiltonian exactly reduces to the initial/final Hamiltonians in
        Pauli-coordinate space.
        """

        weights = weights or FullPauliLossWeights()
        t_collocation = self._time_column(t_collocation)
        prediction = self.forward(t_collocation)
        residual = self.euler_lagrange_residual(t_collocation)
        action_loss = PauliAlgebra.norm_sq(residual)
        reference_residual = self.euler_lagrange_reference_residual(t_collocation)
        reference_loss = PauliAlgebra.norm_sq(reference_residual)
        eps = torch.finfo(action_loss.dtype).eps
        relative_residual = action_loss / torch.clamp(reference_loss, min=eps)
        residual_per_term = action_loss / max(len(self.pauli_labels), 1)
        reference_residual_per_term = reference_loss / max(len(self.pauli_labels), 1)
        agp_l2_loss = torch.mean(prediction["agp_coefficients"].pow(2))
        total = weights.residual * action_loss
        if weights.agp_l2 != 0.0:
            total = total + weights.agp_l2 * agp_l2_loss
        diagnostics = {
            "total": total.detach(),
            "action": action_loss.detach(),
            "residual": action_loss.detach(),
            "reference_residual": reference_loss.detach(),
            "relative_residual": relative_residual.detach(),
            "residual_per_term": residual_per_term.detach(),
            "reference_residual_per_term": reference_residual_per_term.detach(),
            "agp_l2": agp_l2_loss.detach(),
            "basis_size": torch.tensor(float(len(self.pauli_labels)), device=t_collocation.device),
            "agp_terms": torch.tensor(float(len(self.agp_labels)), device=t_collocation.device),
            "hamiltonian_terms": torch.tensor(float(len(self.hamiltonian_labels)), device=t_collocation.device),
        }
        return total, diagnostics


def make_full_pauli_agp_pinn(
    h_initial_terms: Mapping[str, complex],
    h_final_terms: Mapping[str, complex],
    **kwargs,
) -> FullPauliAGPPINN:
    """Convenience constructor for the full ``4**q`` coefficient model."""

    h_initial = SparsePauliOperator(h_initial_terms)
    h_final = SparsePauliOperator(h_final_terms, n_qubits=h_initial.n_qubits)
    return FullPauliAGPPINN(h_initial, h_final, **kwargs)
