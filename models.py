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
    hamiltonian_pauli_factor_graph_data,
    hamiltonian_pauli_graph_data,
    sort_pauli_labels,
)


class TrainableSiLU(nn.Module):
    """SiLU/Swish activation with a learned inverse-temperature slope."""

    def __init__(self, initial_beta: float = 1.0) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(initial_beta), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = self.beta.to(dtype=x.dtype, device=x.device)
        return x * torch.sigmoid(beta * x)


class PadeActivation(nn.Module):
    """Trainable Padé activation unit with a stable positive denominator.

    The parameterization follows the usual PAU shape,
    ``P_m(x) / (1 + |Q_n(x)|)``. The numerator is initialized to a modest
    SiLU-like polynomial so the activation starts near the retained benchmark
    nonlinearity while the denominator can adapt during training.
    """

    def __init__(self, numerator_order: int = 5, denominator_order: int = 4) -> None:
        super().__init__()
        if numerator_order < 1:
            raise ValueError("numerator_order must be at least 1.")
        if denominator_order < 1:
            raise ValueError("denominator_order must be at least 1.")
        numerator = torch.zeros(numerator_order + 1, dtype=torch.float32)
        silu_like = [0.07055594, 0.5, 0.17009126, 0.0, -0.00315486, 0.0]
        numerator[: min(len(numerator), len(silu_like))] = torch.tensor(
            silu_like[: min(len(numerator), len(silu_like))],
            dtype=torch.float32,
        )
        denominator = torch.zeros(denominator_order, dtype=torch.float32)
        denominator[0] = 1e-2
        if denominator_order > 2:
            denominator[2] = 1e-3
        self.numerator = nn.Parameter(numerator)
        self.denominator = nn.Parameter(denominator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numerator = torch.zeros_like(x) + self.numerator[-1].to(dtype=x.dtype, device=x.device)
        for coefficient in reversed(self.numerator[:-1]):
            numerator = numerator * x + coefficient.to(dtype=x.dtype, device=x.device)

        denominator_poly = torch.zeros_like(x)
        power = x
        for coefficient in self.denominator:
            denominator_poly = denominator_poly + coefficient.to(dtype=x.dtype, device=x.device) * power
            power = power * x
        return numerator / (1.0 + torch.abs(denominator_poly))

    def reset_to_silu_rational_fit(self) -> None:
        """Reset a catastrophic SiLU-to-PAU transfer to a bounded rational fit."""

        numerator = [
            0.0093122767,
            0.4999999720,
            0.2354375574,
            0.0404303232,
            0.0028942993,
            0.0000719501,
        ]
        denominator = [-5.5385029e-8, 0.0808606583, -8.4226952e-10, 0.0001439002]
        with torch.no_grad():
            self.numerator.zero_()
            self.denominator.zero_()
            self.numerator[: min(len(self.numerator), len(numerator))].copy_(
                torch.tensor(
                    numerator[: min(len(self.numerator), len(numerator))],
                    dtype=self.numerator.dtype,
                    device=self.numerator.device,
                )
            )
            self.denominator[: min(len(self.denominator), len(denominator))].copy_(
                torch.tensor(
                    denominator[: min(len(self.denominator), len(denominator))],
                    dtype=self.denominator.dtype,
                    device=self.denominator.device,
                )
            )


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name in {"trainable_silu", "adaptive_silu", "swish_trainable"}:
        return TrainableSiLU()
    if name in {"pau", "pade", "pade_activation"}:
        return PadeActivation()
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


class HamiltonianGraphMessageLayer(nn.Module):
    """Shared edge-gated message-passing layer for a qubit graph."""

    def __init__(self, width: int, activation: str) -> None:
        super().__init__()
        self.self_linear = nn.Linear(width, width)
        self.message_linear = nn.Linear(width, width, bias=False)
        self.edge_gate = nn.Linear(3, width)
        self.activation = _activation(activation)

    def forward(
        self,
        node_states: torch.Tensor,
        edge_sources: torch.Tensor,
        edge_targets: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        aggregate = torch.zeros_like(node_states)
        if edge_sources.numel():
            messages = self.message_linear(node_states.index_select(0, edge_sources))
            messages = messages * torch.sigmoid(self.edge_gate(edge_features))
            aggregate.index_add_(0, edge_targets, messages)
            degree = torch.zeros(node_states.shape[0], dtype=node_states.dtype, device=node_states.device)
            degree.index_add_(0, edge_targets, torch.ones_like(edge_targets, dtype=node_states.dtype))
            aggregate = aggregate / degree.clamp_min(1.0).sqrt().unsqueeze(-1)
        return self.activation(node_states + self.self_linear(node_states) + aggregate)


class HamiltonianPauliGraphCoefficientNetwork(nn.Module):
    """Term-shared graph network producing every active AGP coefficient."""

    def __init__(
        self,
        h_initial: SparsePauliOperator,
        h_final: SparsePauliOperator,
        agp_labels: Sequence[str],
        *,
        hidden_width: int = 56,
        hidden_layers: int = 3,
        activation: str = "silu",
        layer_type: str = "quadratic",
        node_width: int = 32,
        message_layers: int = 2,
        latent_rank: int = 32,
        term_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        if node_width < 1 or message_layers < 0 or latent_rank < 1 or term_chunk_size < 1:
            raise ValueError("Graph widths/rank/chunk size must be positive and message_layers non-negative.")
        self.n_qubits = int(h_initial.n_qubits)
        self.agp_labels = sort_pauli_labels(agp_labels)
        self.output_terms = len(self.agp_labels)
        self.latent_rank = int(latent_rank)
        self.term_chunk_size = int(term_chunk_size)
        graph = hamiltonian_pauli_graph_data(h_initial, h_final, self.agp_labels)
        for name, value in (
            ("node_features", graph.node_features),
            ("edge_sources", graph.edge_sources),
            ("edge_targets", graph.edge_targets),
            ("edge_features", graph.edge_features),
            ("term_indices", graph.term_indices),
            ("term_nodes", graph.term_nodes),
            ("term_symbols", graph.term_symbols),
            ("term_scalars", graph.term_scalars),
        ):
            self.register_buffer(name, value, persistent=False)

        self.node_input = nn.Linear(graph.node_features.shape[1], int(node_width))
        self.node_activation = _activation(activation)
        self.message_layers = nn.ModuleList(
            [HamiltonianGraphMessageLayer(int(node_width), activation) for _ in range(int(message_layers))]
        )
        self.symbol_embedding = nn.Embedding(3, int(node_width))
        self.incidence_activation = _activation(activation)
        self.term_readout = nn.Linear(int(node_width) + graph.term_scalars.shape[1], int(latent_rank) + 1)
        self.time_encoder = _make_body(
            1,
            int(latent_rank),
            hidden_width=int(hidden_width),
            hidden_layers=int(hidden_layers),
            activation=activation,
            layer_type=layer_type,
        )

    def _term_latent_and_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        node_states = self.node_activation(self.node_input(self.node_features))
        for layer in self.message_layers:
            node_states = layer(node_states, self.edge_sources, self.edge_targets, self.edge_features)

        pooled = torch.zeros(
            (self.output_terms, node_states.shape[1]),
            dtype=node_states.dtype,
            device=node_states.device,
        )
        if self.term_indices.numel():
            incidence = node_states.index_select(0, self.term_nodes)
            incidence = self.incidence_activation(incidence + self.symbol_embedding(self.term_symbols))
            pooled.index_add_(0, self.term_indices, incidence)
            counts = torch.zeros(self.output_terms, dtype=node_states.dtype, device=node_states.device)
            counts.index_add_(0, self.term_indices, torch.ones_like(self.term_indices, dtype=node_states.dtype))
            pooled = pooled / counts.clamp_min(1.0).sqrt().unsqueeze(-1)
        descriptors = torch.cat((pooled, self.term_scalars.to(node_states.dtype)), dim=-1)
        chunks = [
            self.term_readout(descriptors[start : start + self.term_chunk_size])
            for start in range(0, self.output_terms, self.term_chunk_size)
        ]
        encoded = torch.cat(chunks, dim=0)
        return encoded[:, : self.latent_rank], encoded[:, self.latent_rank]

    def forward(self, normalized_time: torch.Tensor) -> torch.Tensor:
        time_latent = self.time_encoder(normalized_time)
        term_latent, term_bias = self._term_latent_and_bias()
        scale = float(self.latent_rank) ** -0.5
        return scale * time_latent @ term_latent.transpose(0, 1) + term_bias.unsqueeze(0)


class HamiltonianFactorGraphMessageLayer(nn.Module):
    """Alternating shared messages on signed Hamiltonian factor incidences."""

    def __init__(self, width: int, activation: str) -> None:
        super().__init__()
        self.q_to_f_symbol = nn.Embedding(3, width)
        self.f_to_q_symbol = nn.Embedding(3, width)
        self.q_to_f = nn.Sequential(
            nn.Linear(2 * width, width),
            _activation(activation),
            nn.Linear(width, width),
        )
        self.factor_update = nn.Sequential(
            nn.Linear(2 * width, width),
            _activation(activation),
            nn.Linear(width, width),
        )
        self.f_to_q = nn.Sequential(
            nn.Linear(2 * width, width),
            _activation(activation),
            nn.Linear(width, width),
        )
        self.qubit_update = nn.Sequential(
            nn.Linear(2 * width, width),
            _activation(activation),
            nn.Linear(width, width),
        )
        self.factor_norm = nn.LayerNorm(width)
        self.qubit_norm = nn.LayerNorm(width)

    @staticmethod
    def _normalized_index_sum(
        values: torch.Tensor,
        indices: torch.Tensor,
        output_size: int,
    ) -> torch.Tensor:
        output = torch.zeros(
            (output_size, values.shape[-1]), dtype=values.dtype, device=values.device
        )
        if not indices.numel():
            return output
        output.index_add_(0, indices, values)
        counts = torch.zeros(output_size, dtype=values.dtype, device=values.device)
        counts.index_add_(0, indices, torch.ones_like(indices, dtype=values.dtype))
        return output / counts.clamp_min(1.0).sqrt().unsqueeze(-1)

    def forward(
        self,
        qubit_states: torch.Tensor,
        factor_states: torch.Tensor,
        factor_indices: torch.Tensor,
        factor_qubits: torch.Tensor,
        factor_symbols: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if factor_indices.numel():
            q_to_f_input = torch.cat(
                (
                    qubit_states.index_select(0, factor_qubits),
                    self.q_to_f_symbol(factor_symbols),
                ),
                dim=-1,
            )
            q_to_f_messages = self.q_to_f(q_to_f_input)
            factor_aggregate = self._normalized_index_sum(
                q_to_f_messages, factor_indices, factor_states.shape[0]
            )
        else:
            factor_aggregate = torch.zeros_like(factor_states)
        factor_states = self.factor_norm(
            factor_states + self.factor_update(torch.cat((factor_states, factor_aggregate), dim=-1))
        )

        if factor_indices.numel():
            f_to_q_input = torch.cat(
                (
                    factor_states.index_select(0, factor_indices),
                    self.f_to_q_symbol(factor_symbols),
                ),
                dim=-1,
            )
            f_to_q_messages = self.f_to_q(f_to_q_input)
            qubit_aggregate = self._normalized_index_sum(
                f_to_q_messages, factor_qubits, qubit_states.shape[0]
            )
        else:
            qubit_aggregate = torch.zeros_like(qubit_states)
        qubit_states = self.qubit_norm(
            qubit_states + self.qubit_update(torch.cat((qubit_states, qubit_aggregate), dim=-1))
        )
        return qubit_states, factor_states


class HamiltonianPauliFactorGraphCoefficientNetwork(nn.Module):
    """Signed factor-graph decoder with nonlinear moment-based term pooling."""

    def __init__(
        self,
        h_initial: SparsePauliOperator,
        h_final: SparsePauliOperator,
        agp_labels: Sequence[str],
        *,
        hidden_width: int = 96,
        hidden_layers: int = 4,
        activation: str = "pau",
        layer_type: str = "quadratic",
        node_width: int = 96,
        message_layers: int = 4,
        term_width: int = 192,
        latent_rank: int = 128,
        time_fourier_order: int = 4,
        term_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        if min(node_width, term_width, latent_rank, term_chunk_size) < 1:
            raise ValueError("Factor-graph widths, rank, and chunk size must be positive.")
        if message_layers < 0 or time_fourier_order < 0:
            raise ValueError("Message-layer and Fourier orders must be non-negative.")
        self.n_qubits = int(h_initial.n_qubits)
        self.agp_labels = sort_pauli_labels(agp_labels)
        self.output_terms = len(self.agp_labels)
        self.latent_rank = int(latent_rank)
        self.time_fourier_order = int(time_fourier_order)
        self.term_chunk_size = int(term_chunk_size)
        graph = hamiltonian_pauli_factor_graph_data(h_initial, h_final, self.agp_labels)
        for name, value in (
            ("qubit_features", graph.qubit_features),
            ("factor_features", graph.factor_features),
            ("factor_indices", graph.factor_indices),
            ("factor_qubits", graph.factor_qubits),
            ("factor_symbols", graph.factor_symbols),
            ("term_indices", graph.term_indices),
            ("term_qubits", graph.term_qubits),
            ("term_symbols", graph.term_symbols),
            ("term_scalars", graph.term_scalars),
        ):
            self.register_buffer(name, value, persistent=False)

        self.qubit_input = nn.Linear(graph.qubit_features.shape[1], int(node_width))
        self.factor_input = nn.Linear(graph.factor_features.shape[1], int(node_width))
        self.input_activation = _activation(activation)
        self.message_layers = nn.ModuleList(
            [
                HamiltonianFactorGraphMessageLayer(int(node_width), activation)
                for _ in range(int(message_layers))
            ]
        )
        self.term_symbol = nn.Embedding(3, int(node_width))
        self.incidence_encoder = nn.Sequential(
            nn.Linear(2 * int(node_width), int(term_width)),
            _activation(activation),
            nn.Linear(int(term_width), int(term_width)),
            _activation(activation),
        )
        descriptor_width = 2 * int(term_width) + int(graph.term_scalars.shape[1])
        self.term_encoder = nn.Sequential(
            nn.Linear(descriptor_width, int(term_width)),
            _activation(activation),
            nn.Linear(int(term_width), int(term_width)),
            _activation(activation),
            nn.LayerNorm(int(term_width)),
        )
        self.term_readout = nn.Linear(int(term_width), int(latent_rank) + 1)
        time_features = 1 + 2 * int(time_fourier_order)
        self.time_encoder = _make_body(
            time_features,
            int(latent_rank),
            hidden_width=int(hidden_width),
            hidden_layers=int(hidden_layers),
            activation=activation,
            layer_type=layer_type,
        )

    def _time_features(self, normalized_time: torch.Tensor) -> torch.Tensor:
        features = [normalized_time]
        for order in range(1, self.time_fourier_order + 1):
            angle = float(order) * torch.pi * normalized_time
            features.extend((torch.sin(angle), torch.cos(angle)))
        return torch.cat(features, dim=-1)

    def _term_latent_and_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        qubit_states = self.input_activation(self.qubit_input(self.qubit_features))
        factor_states = self.input_activation(self.factor_input(self.factor_features))
        for layer in self.message_layers:
            qubit_states, factor_states = layer(
                qubit_states,
                factor_states,
                self.factor_indices,
                self.factor_qubits,
                self.factor_symbols,
            )

        mean = torch.zeros(
            (self.output_terms, self.incidence_encoder[0].out_features),
            dtype=qubit_states.dtype,
            device=qubit_states.device,
        )
        second_moment = torch.zeros_like(mean)
        counts = torch.zeros(self.output_terms, dtype=qubit_states.dtype, device=qubit_states.device)
        if self.term_indices.numel():
            incidence_input = torch.cat(
                (
                    qubit_states.index_select(0, self.term_qubits),
                    self.term_symbol(self.term_symbols),
                ),
                dim=-1,
            )
            incidence = self.incidence_encoder(incidence_input)
            mean.index_add_(0, self.term_indices, incidence)
            second_moment.index_add_(0, self.term_indices, incidence.square())
            counts.index_add_(
                0, self.term_indices, torch.ones_like(self.term_indices, dtype=qubit_states.dtype)
            )
            divisor = counts.clamp_min(1.0).unsqueeze(-1)
            mean = mean / divisor
            second_moment = second_moment / divisor
        standard_deviation = (second_moment - mean.square()).clamp_min(0.0).add(1e-8).sqrt()
        descriptors = torch.cat(
            (mean, standard_deviation, self.term_scalars.to(qubit_states.dtype)), dim=-1
        )
        chunks = [
            self.term_readout(self.term_encoder(descriptors[start : start + self.term_chunk_size]))
            for start in range(0, self.output_terms, self.term_chunk_size)
        ]
        encoded = torch.cat(chunks, dim=0)
        return encoded[:, : self.latent_rank], encoded[:, self.latent_rank]

    def forward(self, normalized_time: torch.Tensor) -> torch.Tensor:
        time_latent = self.time_encoder(self._time_features(normalized_time))
        term_latent, term_bias = self._term_latent_and_bias()
        scale = float(self.latent_rank) ** -0.5
        return scale * time_latent @ term_latent.transpose(0, 1) + term_bias.unsqueeze(0)


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
    residual_objective: str = "absolute"
    variational_action: float = 0.0
    agp_l2: float = 1e-8
    residual_block_normalization: str = "none"
    agp_smoothness: float = 0.0
    agp_curvature: float = 0.0
    schedule_monotonic: float = 0.0
    schedule_correction_l2: float = 0.0
    calibration_budget: float = 0.0
    calibration_budget_normalization: str = "support"
    calibration_binary: float = 0.0
    calibration_scale_l2: float = 0.0


def projected_residual_objective(
    residual_loss: torch.Tensor,
    reference_loss: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    """Resolve the optimized projected residual without changing diagnostics."""

    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "absolute":
        return residual_loss
    if normalized_mode == "reference_normalized":
        eps = torch.finfo(residual_loss.dtype).eps
        denominator = reference_loss.detach().clamp_min(eps)
        return residual_loss / denominator
    raise ValueError(
        "Unsupported projected residual objective "
        f"{mode!r}; expected 'absolute' or 'reference_normalized'."
    )


def calibration_budget_penalty(
    active_gate_sum: torch.Tensor,
    *,
    target_active_terms: int | float,
    support_terms: int,
    mode: str,
) -> torch.Tensor:
    """Return a dimensionless soft-gate count penalty."""

    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "support":
        denominator = max(int(support_terms), 1)
    elif normalized_mode == "target":
        denominator = max(float(target_active_terms), 1.0)
    else:
        raise ValueError(
            "Unsupported calibration budget normalization "
            f"{mode!r}; expected 'support' or 'target'."
        )
    return ((active_gate_sum - float(target_active_terms)) / denominator) ** 2


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
        coefficient_architecture: str = "independent_outputs",
        graph_node_width: int = 32,
        graph_message_layers: int = 2,
        graph_latent_rank: int = 32,
        graph_term_width: int = 192,
        graph_time_fourier_order: int = 4,
        graph_term_chunk_size: int = 4096,
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
        residual_order_counts: dict[int, int] = {}
        for label in self.residual_labels:
            weight = sum(symbol != "I" for symbol in label)
            residual_order_counts[weight] = residual_order_counts.get(weight, 0) + 1
        residual_block_weights = []
        block_count = max(len(residual_order_counts), 1)
        for label in self.residual_labels:
            weight = sum(symbol != "I" for symbol in label)
            residual_block_weights.append(1.0 / (block_count * max(residual_order_counts.get(weight, 1), 1)))
        self.register_buffer("residual_block_weights", torch.tensor(residual_block_weights, dtype=torch.float32))

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
        self.coefficient_architecture = str(coefficient_architecture).strip().lower()
        if self.coefficient_architecture == "independent_outputs":
            self.body = _make_body(
                1,
                len(self.agp_labels),
                hidden_width=hidden_width,
                hidden_layers=hidden_layers,
                activation=activation,
                layer_type=layer_type,
            )
        elif self.coefficient_architecture == "hamiltonian_pauli_graph":
            self.body = HamiltonianPauliGraphCoefficientNetwork(
                h_initial,
                h_final,
                self.agp_labels,
                hidden_width=hidden_width,
                hidden_layers=hidden_layers,
                activation=activation,
                layer_type=layer_type,
                node_width=graph_node_width,
                message_layers=graph_message_layers,
                latent_rank=graph_latent_rank,
                term_chunk_size=graph_term_chunk_size,
            )
        elif self.coefficient_architecture == "hamiltonian_pauli_factor_graph":
            self.body = HamiltonianPauliFactorGraphCoefficientNetwork(
                h_initial,
                h_final,
                self.agp_labels,
                hidden_width=hidden_width,
                hidden_layers=hidden_layers,
                activation=activation,
                layer_type=layer_type,
                node_width=graph_node_width,
                message_layers=graph_message_layers,
                term_width=graph_term_width,
                latent_rank=graph_latent_rank,
                time_fourier_order=graph_time_fourier_order,
                term_chunk_size=graph_term_chunk_size,
            )
        else:
            raise ValueError(
                "Unsupported coefficient_architecture "
                f"{coefficient_architecture!r}; expected 'independent_outputs' or "
                "a supported Hamiltonian-Pauli graph architecture."
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

    def has_trainable_schedule(self) -> bool:
        return hasattr(self, "schedule_body")

    def enable_trainable_schedule(
        self,
        *,
        hidden_width: int = 16,
        hidden_layers: int = 1,
        activation: str = "tanh",
        base: str = "sinusoidal_sin2",
        correction_amplitude: float = 2.4,
    ) -> None:
        self.schedule_base = str(base)
        self.schedule_correction_amplitude = float(correction_amplitude)
        self.schedule_body = MLP(
            1,
            1,
            hidden_width=int(hidden_width),
            hidden_layers=int(hidden_layers),
            activation=str(activation),
        )
        final_linear = [module for module in self.schedule_body.network if isinstance(module, nn.Linear)][-1]
        nn.init.constant_(final_linear.weight, 0.0)
        nn.init.constant_(final_linear.bias, 0.0)
        self.schedule_body.to(next(self.parameters()).device)

    def _base_schedule_tau(self, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base = str(getattr(self, "schedule_base", "sinusoidal_sin2")).lower()
        if base in {"sinusoidal_sin2", "sin2", "fixed_sinusoidal"}:
            return fixed_sinusoidal_schedule(tau, t_min=0.0, t_max=1.0)
        if base in {"smoothstep", "cubic_smoothstep"}:
            lam = 3.0 * tau.pow(2) - 2.0 * tau.pow(3)
            d_lambda_d_tau = 6.0 * tau - 6.0 * tau.pow(2)
            start_mask = tau <= 0.0
            end_mask = tau >= 1.0
            lam = torch.where(start_mask, torch.zeros_like(lam), lam)
            lam = torch.where(end_mask, torch.ones_like(lam), lam)
            d_lambda_d_tau = torch.where(
                start_mask | end_mask,
                torch.zeros_like(d_lambda_d_tau),
                d_lambda_d_tau,
            )
            return lam, d_lambda_d_tau
        raise ValueError(f"Unsupported schedule base {base!r}.")

    def schedule_tau(self, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return lambda and d lambda / d tau on normalized time tau in [0, 1]."""

        tau = self._time_column(tau)
        if not self.has_trainable_schedule():
            return fixed_sinusoidal_schedule(tau, t_min=0.0, t_max=1.0)
        with torch.enable_grad():
            tau_for_grad = tau
            if not tau_for_grad.requires_grad:
                tau_for_grad = tau_for_grad.detach().clone().requires_grad_(True)
            base_lam, _ = self._base_schedule_tau(tau_for_grad)
            envelope = tau_for_grad.pow(2) * (1.0 - tau_for_grad).pow(2)
            raw_schedule = self.schedule_body(tau_for_grad)
            correction = float(getattr(self, "schedule_correction_amplitude", 2.4)) * envelope * torch.tanh(raw_schedule)
            lam = base_lam + correction
            d_lambda_d_tau = torch.autograd.grad(
                lam.sum(), tau_for_grad, create_graph=True
            )[0]
            start_mask = tau_for_grad <= 0.0
            end_mask = tau_for_grad >= 1.0
            lam = torch.where(start_mask, torch.zeros_like(lam), lam)
            lam = torch.where(end_mask, torch.ones_like(lam), lam)
            d_lambda_d_tau = torch.where(
                start_mask | end_mask,
                torch.zeros_like(d_lambda_d_tau),
                d_lambda_d_tau,
            )
            self._last_schedule_prediction = {
                "base_lambda": base_lam,
                "correction": correction,
                "raw": raw_schedule,
            }
            return lam, d_lambda_d_tau

    def schedule(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self._time_column(t)
        tau = self._normalized_time(t)
        lam, d_lambda_d_tau = self.schedule_tau(tau)
        return lam, d_lambda_d_tau / (self.t_max - self.t_min)

    def forward(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        t = self._time_column(t)
        tau = self._normalized_time(t)
        lam, d_lambda_d_tau = self.schedule_tau(tau)
        d_lambda_dt = d_lambda_d_tau / (self.t_max - self.t_min)
        raw_agp_coefficients = self.body(tau)
        output = {
            "lambda": lam,
            "d_lambda_d_tau": d_lambda_d_tau,
            "d_lambda_dt": d_lambda_dt,
            "raw_agp_coefficients": raw_agp_coefficients,
            "agp_coefficients": self.apply_agp_calibration(raw_agp_coefficients),
        }
        if self.has_trainable_schedule():
            schedule_prediction = getattr(self, "_last_schedule_prediction", {})
            output["schedule_base_lambda"] = schedule_prediction.get("base_lambda", torch.zeros_like(lam))
            output["schedule_correction"] = schedule_prediction.get("correction", torch.zeros_like(lam))
            output["schedule_raw"] = schedule_prediction.get("raw", torch.zeros_like(lam))
        return output

    def has_agp_calibration(self) -> bool:
        return hasattr(self, "agp_log_gamma") and hasattr(self, "agp_gate_logits")

    def agp_calibration_gamma(self) -> torch.Tensor:
        if not self.has_agp_calibration():
            return torch.ones((), dtype=torch.float32, device=self.h_initial_sparse.device)
        return torch.exp(self.agp_log_gamma)

    def agp_calibration_gates(self) -> torch.Tensor:
        if not self.has_agp_calibration():
            return torch.ones(len(self.agp_labels), dtype=torch.float32, device=self.h_initial_sparse.device)
        temperature = float(getattr(self, "agp_gate_temperature", 1.0))
        return torch.sigmoid(self.agp_gate_logits / temperature)

    def apply_agp_calibration(self, raw_agp_coefficients: torch.Tensor) -> torch.Tensor:
        if not self.has_agp_calibration():
            return raw_agp_coefficients
        gamma = self.agp_calibration_gamma().to(raw_agp_coefficients.device)
        gates = self.agp_calibration_gates().to(raw_agp_coefficients.device)
        return gamma * raw_agp_coefficients * gates

    def sparse_operators(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        prediction = self.forward(t)
        return self._sparse_operators_from_prediction(t, prediction)

    def _sparse_operators_from_prediction(
        self,
        t: torch.Tensor,
        prediction: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Build sparse operator coordinates from an existing network pass."""

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
        operators = self._sparse_operators_from_prediction(t_collocation, prediction)
        commutator_1 = self.first_commutator.commutator(
            operators["agp_coefficients"], operators["h_ad_sparse"]
        )
        generator = 1.0j * operators["d_h_d_lambda"] - commutator_1
        residual = self.second_commutator.commutator(generator, operators["h_ad_sparse"])
        reference_generator = 1.0j * operators["d_h_d_lambda"]
        reference_residual = self.second_commutator.commutator(
            reference_generator, operators["h_ad_sparse"]
        )
        variational_action_loss = PauliAlgebra.norm_sq(generator)
        reference_variational_action_loss = PauliAlgebra.norm_sq(reference_generator)
        action_eps = torch.finfo(variational_action_loss.dtype).eps
        relative_variational_action_loss = variational_action_loss / (
            reference_variational_action_loss.detach().clamp_min(action_eps)
        )
        if str(weights.residual_block_normalization).lower() in {"pauli_order", "order", "by_order"}:
            block_weights = self.residual_block_weights.to(device=residual.device, dtype=residual.real.dtype)
            residual_loss = torch.mean(torch.sum(torch.abs(residual) ** 2 * block_weights, dim=-1).real)
            reference_loss = torch.mean(torch.sum(torch.abs(reference_residual) ** 2 * block_weights, dim=-1).real)
        else:
            residual_loss = PauliAlgebra.norm_sq(residual)
            reference_loss = PauliAlgebra.norm_sq(reference_residual)
        eps = torch.finfo(residual_loss.dtype).eps
        relative_residual = residual_loss / torch.clamp(reference_loss, min=eps)
        optimized_residual = projected_residual_objective(
            residual_loss,
            reference_loss,
            mode=weights.residual_objective,
        )
        residual_per_term = residual_loss / max(len(self.residual_labels), 1)
        reference_residual_per_term = reference_loss / max(len(self.residual_labels), 1)
        agp_l2_loss = torch.mean(prediction["agp_coefficients"].pow(2))
        agp_smoothness_loss = torch.zeros((), dtype=agp_l2_loss.dtype, device=agp_l2_loss.device)
        agp_curvature_loss = torch.zeros((), dtype=agp_l2_loss.dtype, device=agp_l2_loss.device)
        if (weights.agp_smoothness != 0.0 or weights.agp_curvature != 0.0) and t_collocation.shape[0] > 1:
            tau = self._normalized_time(t_collocation).squeeze(-1)
            order = torch.argsort(tau)
            sorted_tau = tau.index_select(0, order)
            sorted_coefficients = prediction["agp_coefficients"].index_select(0, order)
            dtau = torch.diff(sorted_tau).clamp_min(torch.finfo(sorted_tau.dtype).eps)
            first_diff = torch.diff(sorted_coefficients, dim=0) / dtau[:, None]
            agp_smoothness_loss = torch.mean(first_diff.pow(2))
            if t_collocation.shape[0] > 2:
                mid_dtau = ((dtau[1:] + dtau[:-1]) / 2.0).clamp_min(torch.finfo(sorted_tau.dtype).eps)
                second_diff = torch.diff(first_diff, dim=0) / mid_dtau[:, None]
                agp_curvature_loss = torch.mean(second_diff.pow(2))
        schedule_monotonic_loss = torch.zeros((), dtype=agp_l2_loss.dtype, device=agp_l2_loss.device)
        schedule_correction_l2_loss = torch.zeros((), dtype=agp_l2_loss.dtype, device=agp_l2_loss.device)
        if self.has_trainable_schedule():
            schedule_monotonic_loss = torch.mean(torch.relu(-prediction["d_lambda_dt"]).pow(2))
            schedule_correction_l2_loss = torch.mean(prediction["schedule_correction"].pow(2))
        calibration_budget_loss = torch.zeros((), dtype=agp_l2_loss.dtype, device=agp_l2_loss.device)
        calibration_binary_loss = torch.zeros((), dtype=agp_l2_loss.dtype, device=agp_l2_loss.device)
        calibration_scale_l2_loss = torch.zeros((), dtype=agp_l2_loss.dtype, device=agp_l2_loss.device)
        calibration_gamma = torch.ones((), dtype=agp_l2_loss.dtype, device=agp_l2_loss.device)
        calibration_active_gate_sum = torch.tensor(
            float(len(self.agp_labels)),
            dtype=agp_l2_loss.dtype,
            device=agp_l2_loss.device,
        )
        if self.has_agp_calibration():
            gates = self.agp_calibration_gates().to(device=agp_l2_loss.device, dtype=agp_l2_loss.dtype)
            calibration_gamma = self.agp_calibration_gamma().to(device=agp_l2_loss.device, dtype=agp_l2_loss.dtype)
            target_active_terms = float(getattr(self, "agp_target_active_terms", len(self.agp_labels)))
            calibration_active_gate_sum = torch.sum(gates)
            calibration_budget_loss = calibration_budget_penalty(
                calibration_active_gate_sum,
                target_active_terms=target_active_terms,
                support_terms=len(self.agp_labels),
                mode=weights.calibration_budget_normalization,
            )
            calibration_binary_loss = torch.mean(gates * (1.0 - gates))
            calibration_scale_l2_loss = (calibration_gamma - 1.0) ** 2
        total = weights.residual * optimized_residual
        if weights.variational_action != 0.0:
            total = total + weights.variational_action * relative_variational_action_loss
        if weights.agp_l2 != 0.0:
            total = total + weights.agp_l2 * agp_l2_loss
        if weights.agp_smoothness != 0.0:
            total = total + weights.agp_smoothness * agp_smoothness_loss
        if weights.agp_curvature != 0.0:
            total = total + weights.agp_curvature * agp_curvature_loss
        if weights.schedule_monotonic != 0.0:
            total = total + weights.schedule_monotonic * schedule_monotonic_loss
        if weights.schedule_correction_l2 != 0.0:
            total = total + weights.schedule_correction_l2 * schedule_correction_l2_loss
        if weights.calibration_budget != 0.0:
            total = total + weights.calibration_budget * calibration_budget_loss
        if weights.calibration_binary != 0.0:
            total = total + weights.calibration_binary * calibration_binary_loss
        if weights.calibration_scale_l2 != 0.0:
            total = total + weights.calibration_scale_l2 * calibration_scale_l2_loss
        diagnostics = {
            "total": total.detach(),
            "residual": residual_loss.detach(),
            "optimized_residual": optimized_residual.detach(),
            "reference_residual": reference_loss.detach(),
            "relative_residual": relative_residual.detach(),
            "variational_action": variational_action_loss.detach(),
            "reference_variational_action": reference_variational_action_loss.detach(),
            "relative_variational_action": relative_variational_action_loss.detach(),
            "residual_per_term": residual_per_term.detach(),
            "reference_residual_per_term": reference_residual_per_term.detach(),
            "agp_l2": agp_l2_loss.detach(),
            "agp_smoothness": agp_smoothness_loss.detach(),
            "agp_curvature": agp_curvature_loss.detach(),
            "schedule_monotonic": schedule_monotonic_loss.detach(),
            "schedule_correction_l2": schedule_correction_l2_loss.detach(),
            "calibration_gamma": calibration_gamma.detach(),
            "calibration_active_gate_sum": calibration_active_gate_sum.detach(),
            "calibration_budget": calibration_budget_loss.detach(),
            "calibration_binary": calibration_binary_loss.detach(),
            "calibration_scale_l2": calibration_scale_l2_loss.detach(),
            "agp_terms": torch.tensor(float(len(self.agp_labels)), device=t_collocation.device),
            "hamiltonian_terms": torch.tensor(float(len(self.hamiltonian_labels)), device=t_collocation.device),
            "intermediate_terms": torch.tensor(float(len(self.intermediate_labels)), device=t_collocation.device),
            "residual_terms": torch.tensor(float(len(self.residual_labels)), device=t_collocation.device),
            "first_commutator_nnz": torch.tensor(float(self.first_commutator.nnz), device=t_collocation.device),
            "second_commutator_nnz": torch.tensor(float(self.second_commutator.nnz), device=t_collocation.device),
        }
        return total, diagnostics


class ProjectedSparseAGPExportModel(ProjectedSparseAGPPINN):
    """Forward-only projected AGP model without commutator construction."""

    def __init__(
        self,
        n_qubits: int,
        agp_labels: Sequence[str],
        *,
        hidden_width: int = 56,
        hidden_layers: int = 3,
        activation: str = "silu",
        layer_type: str = "quadratic",
        coefficient_architecture: str = "independent_outputs",
        h_initial: SparsePauliOperator | None = None,
        h_final: SparsePauliOperator | None = None,
        graph_node_width: int = 32,
        graph_message_layers: int = 2,
        graph_latent_rank: int = 32,
        graph_term_width: int = 192,
        graph_time_fourier_order: int = 4,
        graph_term_chunk_size: int = 4096,
        t_min: float = 0.0,
        t_max: float = 1.0,
    ) -> None:
        nn.Module.__init__(self)
        if int(n_qubits) < 1:
            raise ValueError("n_qubits must be positive.")
        if t_max <= t_min:
            raise ValueError("t_max must be greater than t_min.")
        self.n_qubits = int(n_qubits)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.agp_labels = sort_pauli_labels(agp_labels)
        if not self.agp_labels:
            raise ValueError("agp_labels must be non-empty.")
        self.layer_type = str(layer_type)
        self.coefficient_architecture = str(coefficient_architecture).strip().lower()
        if self.coefficient_architecture == "independent_outputs":
            self.body = _make_body(
                1,
                len(self.agp_labels),
                hidden_width=int(hidden_width),
                hidden_layers=int(hidden_layers),
                activation=str(activation),
                layer_type=str(layer_type),
            )
        elif self.coefficient_architecture == "hamiltonian_pauli_graph":
            if h_initial is None or h_final is None:
                raise ValueError("Graph checkpoint export requires h_initial and h_final.")
            self.body = HamiltonianPauliGraphCoefficientNetwork(
                h_initial,
                h_final,
                self.agp_labels,
                hidden_width=int(hidden_width),
                hidden_layers=int(hidden_layers),
                activation=str(activation),
                layer_type=str(layer_type),
                node_width=int(graph_node_width),
                message_layers=int(graph_message_layers),
                latent_rank=int(graph_latent_rank),
                term_chunk_size=int(graph_term_chunk_size),
            )
        elif self.coefficient_architecture == "hamiltonian_pauli_factor_graph":
            if h_initial is None or h_final is None:
                raise ValueError("Factor-graph checkpoint export requires h_initial and h_final.")
            self.body = HamiltonianPauliFactorGraphCoefficientNetwork(
                h_initial,
                h_final,
                self.agp_labels,
                hidden_width=int(hidden_width),
                hidden_layers=int(hidden_layers),
                activation=str(activation),
                layer_type=str(layer_type),
                node_width=int(graph_node_width),
                message_layers=int(graph_message_layers),
                term_width=int(graph_term_width),
                latent_rank=int(graph_latent_rank),
                time_fourier_order=int(graph_time_fourier_order),
                term_chunk_size=int(graph_term_chunk_size),
            )
        else:
            raise ValueError(f"Unsupported coefficient_architecture {coefficient_architecture!r}.")
        self.register_buffer("h_initial_sparse", torch.empty(0, dtype=torch.complex64))

    def sparse_operators(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        raise RuntimeError("ProjectedSparseAGPExportModel supports forward coefficient export only.")


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
