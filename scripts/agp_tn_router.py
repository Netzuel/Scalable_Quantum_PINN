"""Fail-closed routing for full-support tensor-network validation."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TensorNetworkProblemProfile:
    n_qubits: int
    learned_terms: int
    estimated_exact_mpo_bond: int
    estimated_workspace_bytes: int
    max_cut_terms: int
    term_density: float
    coefficient_dynamic_range: float

    def __post_init__(self) -> None:
        for name in (
            "n_qubits",
            "learned_terms",
            "estimated_exact_mpo_bond",
            "estimated_workspace_bytes",
            "max_cut_terms",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not math.isfinite(self.term_density) or not 0.0 <= self.term_density <= 1.0:
            raise ValueError("term_density must be finite and in [0, 1].")
        if (
            not math.isfinite(self.coefficient_dynamic_range)
            or self.coefficient_dynamic_range < 1.0
        ):
            raise ValueError("coefficient_dynamic_range must be finite and at least one.")


@dataclass(frozen=True)
class TensorNetworkPolicy:
    exact_statevector_qubits: int = 15
    exact_mpo_bond_cap: int = 256
    compressed_mpo_source_bond_cap: int = 2048
    workspace_cap_bytes: int = 8 * 1024**3

    def __post_init__(self) -> None:
        if self.exact_statevector_qubits < 1:
            raise ValueError("exact_statevector_qubits must be positive.")
        if self.exact_mpo_bond_cap < 1:
            raise ValueError("exact_mpo_bond_cap must be positive.")
        if self.compressed_mpo_source_bond_cap < self.exact_mpo_bond_cap:
            raise ValueError(
                "compressed_mpo_source_bond_cap must not be below exact_mpo_bond_cap."
            )
        if self.workspace_cap_bytes < 1:
            raise ValueError("workspace_cap_bytes must be positive.")


@dataclass(frozen=True)
class TensorNetworkRoute:
    name: str
    status: str
    reason: str
    full_support_required: bool = True
    requires_operator_approximation: bool = False
    requires_action_gate: bool = False
    adaptive_time_windows: bool = False


def route_tensor_network_backend(
    profile: TensorNetworkProblemProfile,
    policy: TensorNetworkPolicy,
) -> TensorNetworkRoute:
    """Select a backend from measured complexity while preserving full support."""

    if profile.n_qubits <= policy.exact_statevector_qubits:
        return TensorNetworkRoute(
            name="exact_statevector",
            status="available",
            reason="System is within the configured exact statevector validation limit.",
        )
    if profile.estimated_workspace_bytes > policy.workspace_cap_bytes:
        return TensorNetworkRoute(
            name="unsupported_topology",
            status="not_feasible",
            reason="Measured full-support construction workspace exceeds the configured cap.",
        )
    if profile.estimated_exact_mpo_bond <= policy.exact_mpo_bond_cap:
        return TensorNetworkRoute(
            name="direct_full_support_mpo_tdvp",
            status="available",
            reason="The measured exact instantaneous MPO rank fits the exact bond cap.",
        )
    if profile.estimated_exact_mpo_bond <= policy.compressed_mpo_source_bond_cap:
        return TensorNetworkRoute(
            name="adaptive_windowed_full_support_mpo_tdvp",
            status="candidate",
            reason=(
                "The measured minimum-window source rank fits construction limits; "
                "joint time-Pauli windows may be split but every window must retain all "
                "learned terms and pass coefficient and action gates."
            ),
            requires_operator_approximation=True,
            requires_action_gate=True,
            adaptive_time_windows=True,
        )
    return TensorNetworkRoute(
        name="unsupported_topology",
        status="not_feasible",
        reason=(
            "Measured chain-MPO rank exceeds the configured full-support source cap; "
            "no controlled chain tensor-network route is available."
        ),
    )
