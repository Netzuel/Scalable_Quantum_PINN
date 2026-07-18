from __future__ import annotations

import unittest

from scripts.agp_tn_router import (
    TensorNetworkPolicy,
    TensorNetworkProblemProfile,
    route_tensor_network_backend,
)


class TensorNetworkRouterTests(unittest.TestCase):
    def policy(self) -> TensorNetworkPolicy:
        return TensorNetworkPolicy(
            exact_statevector_qubits=15,
            exact_mpo_bond_cap=256,
            compressed_mpo_source_bond_cap=2048,
            workspace_cap_bytes=8 * 1024**3,
        )

    def profile(
        self,
        *,
        q: int,
        k: int,
        rank: int,
        workspace: int = 1024**3,
        term_density: float = 0.5,
    ) -> TensorNetworkProblemProfile:
        return TensorNetworkProblemProfile(
            n_qubits=q,
            learned_terms=k,
            estimated_exact_mpo_bond=rank,
            estimated_workspace_bytes=workspace,
            max_cut_terms=max(rank, 1),
            term_density=term_density,
            coefficient_dynamic_range=1.0e6,
        )

    def test_small_system_routes_to_exact_statevector_oracle(self) -> None:
        route = route_tensor_network_backend(
            self.profile(q=12, k=4096, rank=400), self.policy()
        )

        self.assertEqual(route.name, "exact_statevector")
        self.assertTrue(route.full_support_required)

    def test_large_system_with_bounded_exact_rank_uses_direct_full_support_mpo(self) -> None:
        route = route_tensor_network_backend(
            self.profile(q=24, k=32768, rank=211), self.policy()
        )

        self.assertEqual(route.name, "direct_full_support_mpo_tdvp")
        self.assertFalse(route.requires_operator_approximation)

    def test_large_system_with_larger_but_bounded_rank_uses_adaptive_full_support_windows(self) -> None:
        route = route_tensor_network_backend(
            self.profile(q=40, k=65536, rank=900), self.policy()
        )

        self.assertEqual(route.name, "adaptive_windowed_full_support_mpo_tdvp")
        self.assertTrue(route.requires_operator_approximation)
        self.assertTrue(route.requires_action_gate)
        self.assertTrue(route.adaptive_time_windows)

    def test_excessive_rank_or_workspace_fails_closed(self) -> None:
        rank_failure = route_tensor_network_backend(
            self.profile(q=24, k=32768, rank=4096), self.policy()
        )
        workspace_failure = route_tensor_network_backend(
            self.profile(q=24, k=32768, rank=128, workspace=16 * 1024**3),
            self.policy(),
        )

        self.assertEqual(rank_failure.name, "unsupported_topology")
        self.assertEqual(workspace_failure.name, "unsupported_topology")
        self.assertEqual(rank_failure.status, "not_feasible")
        self.assertEqual(workspace_failure.status, "not_feasible")

    def test_density_does_not_override_measured_rank(self) -> None:
        sparse = route_tensor_network_backend(
            self.profile(q=30, k=1200, rank=120, term_density=1.0e-9), self.policy()
        )
        dense = route_tensor_network_backend(
            self.profile(q=30, k=100000, rank=120, term_density=0.9), self.policy()
        )

        self.assertEqual(sparse.name, "direct_full_support_mpo_tdvp")
        self.assertEqual(dense.name, sparse.name)


if __name__ == "__main__":
    unittest.main()
