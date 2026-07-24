"""Sparse Pauli algebra utilities for scalable AGP PINNs.

The original Quantum_PINN repository learned dense matrices for the
counterdiabatic operator. This module keeps the same physics target but works
with sparse Pauli-coordinate operators, so memory and output size scale with
the selected ansatz support rather than with the full Hilbert-space matrix.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

try:  # Optional at import time; required only when SOAP is selected.
    import pytorch_optimizer
except ImportError:  # pragma: no cover - depends on the local environment.
    pytorch_optimizer = None

PAULI_ALPHABET = ("I", "X", "Y", "Z")
FULL_PAULI_EXACT_MAX_QUBITS = 8

_MUL_TABLE: dict[tuple[str, str], tuple[complex, str]] = {
    ("I", "I"): (1.0 + 0.0j, "I"),
    ("I", "X"): (1.0 + 0.0j, "X"),
    ("I", "Y"): (1.0 + 0.0j, "Y"),
    ("I", "Z"): (1.0 + 0.0j, "Z"),
    ("X", "I"): (1.0 + 0.0j, "X"),
    ("Y", "I"): (1.0 + 0.0j, "Y"),
    ("Z", "I"): (1.0 + 0.0j, "Z"),
    ("X", "X"): (1.0 + 0.0j, "I"),
    ("Y", "Y"): (1.0 + 0.0j, "I"),
    ("Z", "Z"): (1.0 + 0.0j, "I"),
    ("X", "Y"): (0.0 + 1.0j, "Z"),
    ("Y", "Z"): (0.0 + 1.0j, "X"),
    ("Z", "X"): (0.0 + 1.0j, "Y"),
    ("Y", "X"): (0.0 - 1.0j, "Z"),
    ("Z", "Y"): (0.0 - 1.0j, "X"),
    ("X", "Z"): (0.0 - 1.0j, "Y"),
}

_PAULI_CODE = {"I": 0, "X": 1, "Y": 2, "Z": 3}
_PAULI_SYMBOL = "IXYZ"
_PRODUCT_CODE = (
    (0, 1, 2, 3),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 2, 1, 0),
)
_PRODUCT_PHASE = (
    (1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j),
    (1.0 + 0.0j, 1.0 + 0.0j, 0.0 + 1.0j, 0.0 - 1.0j),
    (1.0 + 0.0j, 0.0 - 1.0j, 1.0 + 0.0j, 0.0 + 1.0j),
    (1.0 + 0.0j, 0.0 + 1.0j, 0.0 - 1.0j, 1.0 + 0.0j),
)


if pytorch_optimizer is not None:

    class SafeMPSSOAP(pytorch_optimizer.SOAP):
        """SOAP variant with CPU fallback for MPS eigendecomposition/QR steps."""

        @staticmethod
        def get_orthogonal_matrix(mat: torch.Tensor) -> list[torch.Tensor]:
            matrices: list[torch.Tensor] = []
            for m in mat:
                if len(m) == 0:
                    matrices.append([])
                    continue

                if m.device.type == "mps":
                    m_cpu = m.detach().to(device="cpu", dtype=torch.float32)
                    eye = torch.eye(m_cpu.shape[0], device="cpu", dtype=m_cpu.dtype)
                    _, q_cpu = torch.linalg.eigh(m_cpu + 1e-30 * eye)
                    q = torch.flip(q_cpu, dims=[1]).to(device=m.device, dtype=m.dtype)
                else:
                    try:
                        eye = torch.eye(m.shape[0], device=m.device, dtype=m.dtype)
                        _, q = torch.linalg.eigh(m + 1e-30 * eye)
                    except Exception:  # pragma: no cover - backend-specific fallback.
                        eye = torch.eye(m.shape[0], device=m.device, dtype=torch.float64)
                        _, q = torch.linalg.eigh(m.to(torch.float64) + 1e-30 * eye)
                        q = q.to(m.dtype)
                    q = torch.flip(q, dims=[1])

                matrices.append(q)

            return matrices

        def get_orthogonal_matrix_qr(
            self,
            state,
            max_precondition_dim: int = 10000,
            merge_dims: bool = False,
        ):
            if not any(len(m) != 0 and m.device.type == "mps" for m in state["GG"]):
                return super().get_orthogonal_matrix_qr(state, max_precondition_dim, merge_dims)

            original_shape = state["exp_avg_sq"].shape
            permuted_shape = original_shape
            if self.data_format == "channels_last" and len(original_shape) == 4:
                permuted_shape = state["exp_avg_sq"].permute(0, 3, 1, 2).shape

            exp_avg_sq = state["exp_avg_sq"]
            if merge_dims:
                from pytorch_optimizer.optimizer.soap import merge_small_dims

                exp_avg_sq = exp_avg_sq.reshape(merge_small_dims(exp_avg_sq.size(), max_precondition_dim))

            matrices = []
            for ind, (m, o) in enumerate(zip(state["GG"], state["Q"])):
                if len(m) == 0:
                    matrices.append([])
                    continue

                m_cpu = m.detach().to(device="cpu", dtype=torch.float32)
                o_cpu = o.detach().to(device="cpu", dtype=torch.float32)
                est_eig = torch.diag(o_cpu.T @ m_cpu @ o_cpu)
                sort_idx_cpu = torch.argsort(est_eig, descending=True)
                sort_idx = sort_idx_cpu.to(exp_avg_sq.device)
                exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)

                power_iter = m_cpu @ o_cpu[:, sort_idx_cpu]
                q_cpu, _ = torch.linalg.qr(power_iter)
                matrices.append(q_cpu.to(device=m.device, dtype=m.dtype))

            if merge_dims:
                if self.data_format == "channels_last" and len(original_shape) == 4:
                    exp_avg_sq = exp_avg_sq.reshape(permuted_shape).permute(0, 2, 3, 1)
                else:
                    exp_avg_sq = exp_avg_sq.reshape(original_shape)

            state["exp_avg_sq"] = exp_avg_sq
            return matrices

else:

    class SafeMPSSOAP(torch.optim.Optimizer):
        """Placeholder that reports the missing optional SOAP dependency."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Install pytorch-optimizer to use SOAP or SafeMPSSOAP.")


SafeMPSSoap = SafeMPSSOAP


def validate_pauli_label(label: str, n_qubits: int | None = None) -> str:
    """Validate and normalize a Pauli-string label such as ``IXYZ``."""

    if not isinstance(label, str) or not label:
        raise ValueError("Pauli labels must be non-empty strings.")
    label = label.upper()
    invalid = sorted({symbol for symbol in label if symbol not in _PAULI_CODE})
    if invalid:
        raise ValueError(f"Invalid Pauli symbols {invalid} in label {label!r}.")
    if n_qubits is not None and len(label) != n_qubits:
        raise ValueError(f"Expected {n_qubits} qubits, got label {label!r}.")
    return label


def pauli_weight(label: str) -> int:
    """Return the number of non-identity factors in a Pauli label."""

    return sum(symbol != "I" for symbol in validate_pauli_label(label))


def infer_n_qubits(labels: Iterable[str]) -> int:
    """Infer a common qubit count from labels."""

    labels = [validate_pauli_label(label) for label in labels]
    if not labels:
        raise ValueError("Cannot infer qubit count from an empty label set.")
    lengths = {len(label) for label in labels}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent Pauli-label lengths: {sorted(lengths)}")
    return lengths.pop()


def multiply_pauli_labels(left: str, right: str) -> tuple[complex, str]:
    """Multiply two Pauli strings and return ``(phase, label)``."""

    left = validate_pauli_label(left)
    right = validate_pauli_label(right, len(left))
    phase = 1.0 + 0.0j
    out: list[str] = []
    for l_symbol, r_symbol in zip(left, right):
        local_phase, local_label = _MUL_TABLE[(l_symbol, r_symbol)]
        phase *= local_phase
        out.append(local_label)
    return phase, "".join(out)


def commutator_pauli_labels(left: str, right: str) -> tuple[complex, str] | None:
    """Return the Pauli string and phase for ``[left, right]``.

    ``None`` means the two strings commute.
    """

    phase_lr, label_lr = multiply_pauli_labels(left, right)
    phase_rl, label_rl = multiply_pauli_labels(right, left)
    if label_lr != label_rl:
        raise RuntimeError("Pauli multiplication produced inconsistent labels.")
    phase = phase_lr - phase_rl
    if abs(phase) < 1e-14:
        return None
    return phase, label_lr


def _commutator_pauli_labels_unchecked(left: str, right: str) -> tuple[complex, str] | None:
    """Fast commutator for labels already validated to the same length."""

    return _commutator_pauli_codes_unchecked(_encode_pauli_label(left), _encode_pauli_label(right))


def _encode_pauli_label(label: str) -> tuple[int, ...]:
    return tuple(_PAULI_CODE[symbol] for symbol in label)


def _commutator_pauli_codes_unchecked(
    left_codes: tuple[int, ...],
    right_codes: tuple[int, ...],
) -> tuple[complex, str] | None:
    phase = 1.0 + 0.0j
    anticommute_parity = 0
    out_codes: list[int] = []
    for left_code, right_code in zip(left_codes, right_codes):
        if left_code != 0 and right_code != 0 and left_code != right_code:
            anticommute_parity ^= 1
        phase *= _PRODUCT_PHASE[left_code][right_code]
        out_codes.append(_PRODUCT_CODE[left_code][right_code])
    if anticommute_parity == 0:
        return None
    return 2.0 * phase, "".join(_PAULI_SYMBOL[code] for code in out_codes)


def sort_pauli_labels(labels: Iterable[str]) -> list[str]:
    """Sort labels by locality first, then lexicographically."""

    normalized: set[str] = set()
    for label in labels:
        if not isinstance(label, str) or not label:
            raise ValueError("Pauli labels must be non-empty strings.")
        upper = label.upper()
        invalid = sorted({symbol for symbol in upper if symbol not in _PAULI_CODE})
        if invalid:
            raise ValueError(f"Invalid Pauli symbols {invalid} in label {upper!r}.")
        normalized.add(upper)
    return sorted(normalized, key=lambda x: (sum(symbol != "I" for symbol in x), x))


def all_pauli_labels(n_qubits: int) -> list[str]:
    """Generate the full ``4**n_qubits`` Pauli-product basis."""

    if n_qubits < 1:
        raise ValueError("Use at least one qubit.")
    return ["".join(symbols) for symbols in product(PAULI_ALPHABET, repeat=n_qubits)]


def pauli_training_regime(n_qubits: int) -> str:
    """Return the default AGP coefficient regime for a qubit count."""

    if n_qubits < 1:
        raise ValueError("Use at least one qubit.")
    if n_qubits <= FULL_PAULI_EXACT_MAX_QUBITS:
        return "full_pauli_exact"
    return "adaptive_projected_sparse"


def fixed_sinusoidal_schedule(
    t: torch.Tensor,
    *,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``lambda(t)=sin^2(pi tau / 2)`` and ``d lambda / dt``.

    Here ``tau = (t - t_min) / T`` and ``T = t_max - t_min``. The derivative
    includes the chain-rule factor ``1 / T``.
    """

    if t_max <= t_min:
        raise ValueError("t_max must be greater than t_min.")
    tau = (t - t_min) / (t_max - t_min)
    pi = torch.as_tensor(torch.pi, dtype=t.dtype, device=t.device)
    lam = torch.sin(0.5 * pi * tau).pow(2)
    d_lambda_dt = 0.5 * pi * torch.sin(pi * tau) / (t_max - t_min)
    start_mask = tau <= 0.0
    end_mask = tau >= 1.0
    lam = torch.where(start_mask, torch.zeros_like(lam), lam)
    lam = torch.where(end_mask, torch.ones_like(lam), lam)
    d_lambda_dt = torch.where(start_mask | end_mask, torch.zeros_like(d_lambda_dt), d_lambda_dt)
    return lam, d_lambda_dt


def format_distance_token(distance: str | float) -> str:
    """Normalize a distance value to the historical HDF5 key token."""

    if isinstance(distance, str):
        return distance.replace(".", "_")
    return str(float(distance)).replace(".", "_")


def _decode_json_complex(value: object) -> complex:
    if isinstance(value, (int, float)):
        return complex(float(value), 0.0)
    if isinstance(value, list) and len(value) == 2:
        return complex(float(value[0]), float(value[1]))
    raise ValueError(f"Cannot decode complex coefficient from {value!r}.")


def _decode_json_terms(terms: Mapping[str, object]) -> dict[str, complex]:
    return {label: _decode_json_complex(coeff) for label, coeff in terms.items()}


def _load_pauli_pair_payload(
    payload: Mapping[str, object],
    *,
    system: str,
    n_qubits: int,
    distance: str,
) -> tuple["SparsePauliOperator", "SparsePauliOperator"]:
    if str(payload.get("system")) != system:
        raise KeyError(f"Requested system {system!r}, but pair file contains {payload.get('system')!r}.")
    if int(payload.get("n_qubits", -1)) != n_qubits:
        raise KeyError(f"Requested {n_qubits} qubits, but pair file contains {payload.get('n_qubits')!r}.")
    if str(payload.get("distance")) != distance:
        raise KeyError(f"Requested distance {distance!r}, but pair file contains {payload.get('distance')!r}.")
    hamiltonians = payload.get("hamiltonians")
    if not isinstance(hamiltonians, Mapping):
        raise KeyError("Pauli pair payload does not contain a 'hamiltonians' mapping.")
    initial = hamiltonians.get("initial")
    final = hamiltonians.get("final")
    if not isinstance(initial, Mapping) or not isinstance(final, Mapping):
        raise KeyError("Pauli pair payload must contain 'initial' and 'final' Hamiltonians.")
    initial_terms = initial.get("terms")
    final_terms = final.get("terms")
    if not isinstance(initial_terms, Mapping) or not isinstance(final_terms, Mapping):
        raise KeyError("Pauli pair endpoints must contain term mappings.")
    return (
        SparsePauliOperator(_decode_json_terms(initial_terms), n_qubits=n_qubits),
        SparsePauliOperator(_decode_json_terms(final_terms), n_qubits=n_qubits),
    )


def load_pauli_hamiltonian_pair(
    path: str | Path,
    *,
    system: str,
    n_qubits: int,
    distance: str | float,
) -> tuple[SparsePauliOperator, SparsePauliOperator]:
    """Load adapted Pauli-coordinate Hamiltonians.

    Supported inputs are the organized ``pauli_decompositions/index.json``
    format, a direct ``pauli_hamiltonian_pair_v1`` JSON file, or the older
    aggregate ``Hamiltonians_pauli.json`` format.
    """

    path = Path(path)
    if path.is_dir():
        path = path / "index.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    distance_token = format_distance_token(distance)
    payload_format = payload.get("format")
    if payload_format == "pauli_hamiltonian_index_v1":
        pair_id = f"{system}_{n_qubits}_qubits_{distance_token}"
        pairs = payload["pairs"]
        if pair_id not in pairs:
            available = ", ".join(sorted(pairs)[:5])
            raise KeyError(f"Missing Hamiltonian pair {pair_id!r}. First available pairs: {available}")
        pair_file = path.parent / pairs[pair_id]["file"]
        with pair_file.open("r", encoding="utf-8") as handle:
            pair_payload = json.load(handle)
        return _load_pauli_pair_payload(
            pair_payload,
            system=system,
            n_qubits=n_qubits,
            distance=distance_token,
        )
    if payload_format == "pauli_hamiltonian_pair_v1":
        return _load_pauli_pair_payload(
            payload,
            system=system,
            n_qubits=n_qubits,
            distance=distance_token,
        )

    datasets = payload["datasets"]
    prefix = f"{system}_{n_qubits}_qubits"
    h0_key = f"{prefix}_H_hf_{distance_token}"
    h1_key = f"{prefix}_H_prob_{distance_token}"
    missing = [key for key in (h0_key, h1_key) if key not in datasets]
    if missing:
        available = ", ".join(sorted(datasets)[:5])
        raise KeyError(f"Missing Hamiltonian key(s) {missing}. First available keys: {available}")
    h0_terms = _decode_json_terms(datasets[h0_key]["terms"])
    h1_terms = _decode_json_terms(datasets[h1_key]["terms"])
    return (
        SparsePauliOperator(h0_terms, n_qubits=n_qubits),
        SparsePauliOperator(h1_terms, n_qubits=n_qubits),
    )


@dataclass(frozen=True)
class SparsePauliOperator:
    """Sparse Pauli-coordinate operator.

    The coefficients may be complex. Hermitian Hamiltonians normally use real
    coefficients; commutators naturally generate imaginary coefficients.
    """

    terms: Mapping[str, complex]
    n_qubits: int | None = None
    drop_tol: float = 1e-14

    def __post_init__(self) -> None:
        labels = list(self.terms)
        n_qubits = self.n_qubits if self.n_qubits is not None else infer_n_qubits(labels)
        clean_terms: dict[str, complex] = {}
        for label, coeff in self.terms.items():
            label = validate_pauli_label(label, n_qubits)
            coeff = complex(coeff)
            if abs(coeff) > self.drop_tol:
                clean_terms[label] = coeff
        object.__setattr__(self, "terms", clean_terms)
        object.__setattr__(self, "n_qubits", n_qubits)

    @classmethod
    def zero(cls, n_qubits: int) -> "SparsePauliOperator":
        return cls({}, n_qubits=n_qubits)

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, complex]]) -> "SparsePauliOperator":
        accumulator: defaultdict[str, complex] = defaultdict(complex)
        labels: list[str] = []
        for label, coeff in pairs:
            label = validate_pauli_label(label)
            labels.append(label)
            accumulator[label] += complex(coeff)
        return cls(dict(accumulator), n_qubits=infer_n_qubits(labels))

    @property
    def labels(self) -> list[str]:
        return sort_pauli_labels(self.terms)

    def coefficient(self, label: str) -> complex:
        return self.terms.get(validate_pauli_label(label, self.n_qubits), 0.0 + 0.0j)

    def scale(self, scalar: complex) -> "SparsePauliOperator":
        return SparsePauliOperator(
            {label: scalar * coeff for label, coeff in self.terms.items()},
            n_qubits=self.n_qubits,
        )

    def add(self, other: "SparsePauliOperator", scale: complex = 1.0) -> "SparsePauliOperator":
        if self.n_qubits != other.n_qubits:
            raise ValueError("Cannot add operators with different qubit counts.")
        terms: defaultdict[str, complex] = defaultdict(complex)
        terms.update(self.terms)
        for label, coeff in other.terms.items():
            terms[label] += scale * coeff
        return SparsePauliOperator(dict(terms), n_qubits=self.n_qubits)

    def __add__(self, other: "SparsePauliOperator") -> "SparsePauliOperator":
        return self.add(other)

    def __sub__(self, other: "SparsePauliOperator") -> "SparsePauliOperator":
        return self.add(other, scale=-1.0)

    def commutator(self, other: "SparsePauliOperator") -> "SparsePauliOperator":
        if self.n_qubits != other.n_qubits:
            raise ValueError("Cannot commute operators with different qubit counts.")
        terms: defaultdict[str, complex] = defaultdict(complex)
        for left_label, left_coeff in self.terms.items():
            for right_label, right_coeff in other.terms.items():
                item = commutator_pauli_labels(left_label, right_label)
                if item is None:
                    continue
                phase, out_label = item
                terms[out_label] += left_coeff * right_coeff * phase
        return SparsePauliOperator(dict(terms), n_qubits=self.n_qubits)

    def l2_norm_sq(self) -> float:
        return float(sum(abs(coeff) ** 2 for coeff in self.terms.values()))

    def to_vector(
        self,
        basis_labels: Sequence[str],
        *,
        dtype: torch.dtype = torch.complex64,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        basis_labels = [validate_pauli_label(label, self.n_qubits) for label in basis_labels]
        vector = torch.zeros(len(basis_labels), dtype=dtype, device=device)
        for idx, label in enumerate(basis_labels):
            vector[idx] = self.coefficient(label)
        return vector


@dataclass(frozen=True)
class HamiltonianPauliGraphData:
    """Fixed-size graph features and sparse AGP-term incidences.

    The tensors are deterministic functions of the Hamiltonian pair and active
    Pauli support. Models register them as nonpersistent buffers so checkpoints
    contain only reusable trainable weights.
    """

    node_features: torch.Tensor
    edge_sources: torch.Tensor
    edge_targets: torch.Tensor
    edge_features: torch.Tensor
    term_indices: torch.Tensor
    term_nodes: torch.Tensor
    term_symbols: torch.Tensor
    term_scalars: torch.Tensor


@dataclass(frozen=True)
class HamiltonianPauliFactorGraphData:
    """Signed Hamiltonian factor graph and expressive AGP-term descriptors."""

    qubit_features: torch.Tensor
    factor_features: torch.Tensor
    factor_indices: torch.Tensor
    factor_qubits: torch.Tensor
    factor_symbols: torch.Tensor
    term_indices: torch.Tensor
    term_qubits: torch.Tensor
    term_symbols: torch.Tensor
    term_scalars: torch.Tensor


def hamiltonian_pauli_graph_data(
    h_initial: SparsePauliOperator,
    h_final: SparsePauliOperator,
    agp_labels: Sequence[str],
) -> HamiltonianPauliGraphData:
    """Encode a sparse Hamiltonian pair and Pauli support as graph tensors."""

    if h_initial.n_qubits != h_final.n_qubits:
        raise ValueError("Initial and final Hamiltonians must use the same qubit count.")
    q = int(h_initial.n_qubits)
    labels = [validate_pauli_label(label, q) for label in agp_labels]
    if not labels:
        raise ValueError("agp_labels must be non-empty.")

    operators = (h_initial, h_final, h_final - h_initial)
    node_features = torch.zeros((q, 27), dtype=torch.float32)
    edge_maps: list[defaultdict[tuple[int, int], float]] = [defaultdict(float) for _ in operators]
    for operator_index, operator in enumerate(operators):
        scale = max((abs(coefficient) for coefficient in operator.terms.values()), default=1.0)
        scale = max(float(scale), 1e-12)
        for label, coefficient in operator.terms.items():
            normalized = complex(coefficient) / scale
            support = [site for site, symbol in enumerate(label) if symbol != "I"]
            for site in support:
                symbol_index = _PAULI_CODE[label[site]] - 1
                offset = operator_index * 9 + symbol_index * 3
                node_features[site, offset] += float(normalized.real)
                node_features[site, offset + 1] += float(normalized.imag)
                node_features[site, offset + 2] += float(abs(normalized))
            pair_normalization = max(len(support) * (len(support) - 1) // 2, 1)
            edge_weight = float(abs(normalized)) / pair_normalization
            for left, right in combinations(support, 2):
                edge_maps[operator_index][(min(left, right), max(left, right))] += edge_weight

    undirected_edges = sorted(set().union(*(set(edge_map) for edge_map in edge_maps)))
    edge_sources: list[int] = []
    edge_targets: list[int] = []
    edge_features: list[list[float]] = []
    undirected_features: dict[tuple[int, int], tuple[float, float, float]] = {}
    for edge in undirected_edges:
        features = tuple(float(edge_map.get(edge, 0.0)) for edge_map in edge_maps)
        undirected_features[edge] = features
        for source, target in (edge, (edge[1], edge[0])):
            edge_sources.append(source)
            edge_targets.append(target)
            edge_features.append(list(features))

    incidence_terms: list[int] = []
    incidence_nodes: list[int] = []
    incidence_symbols: list[int] = []
    term_scalars = torch.zeros((len(labels), 7), dtype=torch.float32)
    for term_index, label in enumerate(labels):
        support = [site for site, symbol in enumerate(label) if symbol != "I"]
        weight = len(support)
        counts = [label.count(symbol) for symbol in "XYZ"]
        term_scalars[term_index, 0] = float(weight) / max(q, 1)
        if weight:
            term_scalars[term_index, 1:4] = torch.tensor(
                [float(count) / weight for count in counts], dtype=torch.float32
            )
        for site in support:
            incidence_terms.append(term_index)
            incidence_nodes.append(site)
            incidence_symbols.append(_PAULI_CODE[label[site]] - 1)
        induced = torch.zeros(3, dtype=torch.float32)
        induced_pairs = 0
        for left, right in combinations(support, 2):
            features = undirected_features.get((min(left, right), max(left, right)))
            if features is None:
                continue
            induced += torch.tensor(features, dtype=torch.float32)
            induced_pairs += 1
        if induced_pairs:
            term_scalars[term_index, 4:7] = induced / induced_pairs

    edge_feature_tensor = (
        torch.tensor(edge_features, dtype=torch.float32)
        if edge_features
        else torch.empty((0, 3), dtype=torch.float32)
    )
    return HamiltonianPauliGraphData(
        node_features=node_features,
        edge_sources=torch.tensor(edge_sources, dtype=torch.long),
        edge_targets=torch.tensor(edge_targets, dtype=torch.long),
        edge_features=edge_feature_tensor,
        term_indices=torch.tensor(incidence_terms, dtype=torch.long),
        term_nodes=torch.tensor(incidence_nodes, dtype=torch.long),
        term_symbols=torch.tensor(incidence_symbols, dtype=torch.long),
        term_scalars=term_scalars,
    )


@lru_cache(maxsize=500_000)
def _pauli_commutator_fingerprint(
    label: str,
    operator_signature: tuple[tuple[str, complex], ...],
) -> tuple[float, ...]:
    """Return fixed-width signed features for one Pauli/Hamiltonian commutator."""

    q = len(label)
    if not operator_signature:
        return (0.0,) * 8
    accumulated: defaultdict[str, complex] = defaultdict(complex)
    individual_abs: list[float] = []
    weighted_locality = 0.0
    signed_sum = 0.0 + 0.0j
    for h_label, h_coefficient in operator_signature:
        item = _commutator_pauli_labels_unchecked(label, h_label)
        if item is None:
            continue
        phase, output_label = item
        value = complex(phase) * complex(h_coefficient)
        accumulated[output_label] += value
        magnitude = abs(value)
        individual_abs.append(magnitude)
        signed_sum += value
        weighted_locality += magnitude * sum(symbol != "I" for symbol in output_label)
    if not individual_abs:
        return (0.0,) * 8
    individual_l1 = sum(individual_abs)
    individual_l2 = sum(value * value for value in individual_abs) ** 0.5
    output_abs = [abs(value) for value in accumulated.values()]
    output_l1 = sum(output_abs)
    output_l2 = sum(value * value for value in output_abs) ** 0.5
    scale = max(individual_l1, 1e-12)
    return (
        len(individual_abs) / max(len(operator_signature), 1),
        float(signed_sum.real) / scale,
        float(signed_sum.imag) / scale,
        output_l2 / max(individual_l2, 1e-12),
        output_l1 / scale,
        weighted_locality / (scale * max(q, 1)),
        len(accumulated) / max(len(individual_abs), 1),
        max(output_abs, default=0.0) / scale,
    )


def hamiltonian_pauli_factor_graph_data(
    h_initial: SparsePauliOperator,
    h_final: SparsePauliOperator,
    agp_labels: Sequence[str],
) -> HamiltonianPauliFactorGraphData:
    """Encode a signed Pauli factor graph without dense ``K x q`` tensors."""

    if h_initial.n_qubits != h_final.n_qubits:
        raise ValueError("Initial and final Hamiltonians must use the same qubit count.")
    q = int(h_initial.n_qubits)
    labels = [validate_pauli_label(label, q) for label in agp_labels]
    if not labels:
        raise ValueError("agp_labels must be non-empty.")

    h_delta = h_final - h_initial
    operators = (h_initial, h_final, h_delta)
    scales = [
        max((abs(coefficient) for coefficient in operator.terms.values()), default=1.0)
        for operator in operators
    ]
    scales = [max(float(scale), 1e-12) for scale in scales]

    qubit_features = torch.zeros((q, 27), dtype=torch.float32)
    for operator_index, (operator, scale) in enumerate(zip(operators, scales)):
        for label, coefficient in operator.terms.items():
            normalized = complex(coefficient) / scale
            for site, symbol in enumerate(label):
                if symbol == "I":
                    continue
                symbol_index = _PAULI_CODE[symbol] - 1
                offset = operator_index * 9 + symbol_index * 3
                qubit_features[site, offset] += float(normalized.real)
                qubit_features[site, offset + 1] += float(normalized.imag)
                qubit_features[site, offset + 2] += float(abs(normalized))

    factor_labels = sort_pauli_labels(set(h_initial.labels) | set(h_final.labels))
    factor_features = torch.zeros((len(factor_labels), 13), dtype=torch.float32)
    factor_indices: list[int] = []
    factor_qubits: list[int] = []
    factor_symbols: list[int] = []
    for factor_index, label in enumerate(factor_labels):
        for operator_index, (operator, scale) in enumerate(zip(operators, scales)):
            normalized = complex(operator.coefficient(label)) / scale
            offset = operator_index * 3
            factor_features[factor_index, offset] = float(normalized.real)
            factor_features[factor_index, offset + 1] = float(normalized.imag)
            factor_features[factor_index, offset + 2] = float(abs(normalized))
        support = [(site, symbol) for site, symbol in enumerate(label) if symbol != "I"]
        factor_features[factor_index, 9] = len(support) / max(q, 1)
        if support:
            factor_features[factor_index, 10:13] = torch.tensor(
                [label.count(symbol) / len(support) for symbol in "XYZ"], dtype=torch.float32
            )
        for site, symbol in support:
            factor_indices.append(factor_index)
            factor_qubits.append(site)
            factor_symbols.append(_PAULI_CODE[symbol] - 1)

    signatures = tuple(
        tuple(sorted((label, complex(coefficient)) for label, coefficient in operator.terms.items()))
        for operator in operators
    )
    term_indices: list[int] = []
    term_qubits: list[int] = []
    term_symbols: list[int] = []
    term_scalars = torch.zeros((len(labels), 31), dtype=torch.float32)
    for term_index, label in enumerate(labels):
        support = [(site, symbol) for site, symbol in enumerate(label) if symbol != "I"]
        weight = len(support)
        term_scalars[term_index, 0] = weight / max(q, 1)
        if weight:
            term_scalars[term_index, 1:4] = torch.tensor(
                [label.count(symbol) / weight for symbol in "XYZ"], dtype=torch.float32
            )
            support_sites = {site for site, _ in support}
            touched = 0
            contained = 0
            overlap_sum = 0.0
            for factor_label in factor_labels:
                factor_sites = {
                    site for site, symbol in enumerate(factor_label) if symbol != "I"
                }
                overlap = len(support_sites & factor_sites)
                if overlap:
                    touched += 1
                    overlap_sum += overlap / max(len(factor_sites), 1)
                if factor_sites and factor_sites <= support_sites:
                    contained += 1
            factor_count = max(len(factor_labels), 1)
            term_scalars[term_index, 4] = touched / factor_count
            term_scalars[term_index, 5] = contained / factor_count
            term_scalars[term_index, 6] = overlap_sum / max(touched, 1)
        for site, symbol in support:
            term_indices.append(term_index)
            term_qubits.append(site)
            term_symbols.append(_PAULI_CODE[symbol] - 1)
        fingerprint = tuple(
            value
            for signature in signatures
            for value in _pauli_commutator_fingerprint(label, signature)
        )
        term_scalars[term_index, 7:] = torch.tensor(fingerprint, dtype=torch.float32)

    return HamiltonianPauliFactorGraphData(
        qubit_features=qubit_features,
        factor_features=factor_features,
        factor_indices=torch.tensor(factor_indices, dtype=torch.long),
        factor_qubits=torch.tensor(factor_qubits, dtype=torch.long),
        factor_symbols=torch.tensor(factor_symbols, dtype=torch.long),
        term_indices=torch.tensor(term_indices, dtype=torch.long),
        term_qubits=torch.tensor(term_qubits, dtype=torch.long),
        term_symbols=torch.tensor(term_symbols, dtype=torch.long),
        term_scalars=term_scalars,
    )


def interpolate_operator(
    initial: SparsePauliOperator,
    final: SparsePauliOperator,
    lam: float,
) -> SparsePauliOperator:
    """Return ``(1 - lam) initial + lam final`` in sparse Pauli form."""

    return initial.scale(1.0 - lam).add(final, scale=lam)


def exact_agp_residual_operator(
    initial: SparsePauliOperator,
    final: SparsePauliOperator,
    agp: SparsePauliOperator,
    lam: float,
) -> SparsePauliOperator:
    """Euler-Lagrange residual ``[i dH/dlambda - [A,H], H]``.

    This is exact inside the supplied sparse Pauli support. It does not build
    dense Hilbert-space matrices.
    """

    h_ad = interpolate_operator(initial, final, lam)
    d_h = final - initial
    generator = d_h.scale(1.0j) - agp.commutator(h_ad)
    return generator.commutator(h_ad)


def all_local_pauli_labels(
    n_qubits: int,
    max_weight: int,
    *,
    alphabet: Sequence[str] = ("X", "Y", "Z"),
    include_identity: bool = False,
) -> list[str]:
    """Generate all Pauli strings up to a chosen locality/weight."""

    if max_weight < 0:
        raise ValueError("max_weight must be non-negative.")
    labels: set[str] = {"I" * n_qubits} if include_identity else set()
    for weight in range(1, max_weight + 1):
        for sites in combinations(range(n_qubits), weight):
            for local_symbols in product(alphabet, repeat=weight):
                chars = ["I"] * n_qubits
                for site, symbol in zip(sites, local_symbols):
                    chars[site] = symbol
                labels.add("".join(chars))
    return sort_pauli_labels(labels)


def transverse_field_ising_problem(
    n_qubits: int,
    *,
    field: float = 1.0,
    coupling: float = 1.0,
    periodic: bool = False,
) -> tuple[SparsePauliOperator, SparsePauliOperator]:
    """Return a common scalable benchmark ``H0=-sum X_i``, ``H1=-sum Z_i Z_j``."""

    if n_qubits < 2:
        raise ValueError("Use at least two qubits for an Ising-chain problem.")
    h0_terms: dict[str, complex] = {}
    h1_terms: dict[str, complex] = {}
    for site in range(n_qubits):
        label = ["I"] * n_qubits
        label[site] = "X"
        h0_terms["".join(label)] = -float(field)
    edges = [(site, site + 1) for site in range(n_qubits - 1)]
    if periodic and n_qubits > 2:
        edges.append((n_qubits - 1, 0))
    for left, right in edges:
        label = ["I"] * n_qubits
        label[left] = "Z"
        label[right] = "Z"
        h1_terms["".join(label)] = -float(coupling)
    return SparsePauliOperator(h0_terms, n_qubits), SparsePauliOperator(h1_terms, n_qubits)


def build_commutator_closure(
    seed_labels: Iterable[str],
    *,
    max_rounds: int = 2,
    max_weight: int | None = None,
    max_terms: int = 20000,
) -> list[str]:
    """Expand labels under pairwise commutators for a fixed number of rounds."""

    labels = set(sort_pauli_labels(seed_labels))
    n_qubits = infer_n_qubits(labels)
    for _ in range(max_rounds):
        additions: set[str] = set()
        current = list(labels)
        for left in current:
            for right in current:
                item = commutator_pauli_labels(left, right)
                if item is None:
                    continue
                _, out_label = item
                if max_weight is not None and pauli_weight(out_label) > max_weight:
                    continue
                validate_pauli_label(out_label, n_qubits)
                additions.add(out_label)
                if len(labels) + len(additions) > max_terms:
                    raise RuntimeError(
                        "Commutator closure exceeded max_terms. Reduce the ansatz, "
                        "lower max_weight, or raise max_terms deliberately."
                    )
        old_size = len(labels)
        labels.update(additions)
        if len(labels) == old_size:
            break
    return sort_pauli_labels(labels)


class PauliAlgebra:
    """Differentiable Pauli algebra over a fixed sparse operator basis."""

    def __init__(self, basis_labels: Sequence[str]):
        self.basis_labels = sort_pauli_labels(basis_labels)
        self.n_qubits = infer_n_qubits(self.basis_labels)
        self.index = {label: idx for idx, label in enumerate(self.basis_labels)}
        out_idx: list[int] = []
        left_idx: list[int] = []
        right_idx: list[int] = []
        coeffs: list[complex] = []
        for left_label, left_position in self.index.items():
            for right_label, right_position in self.index.items():
                item = commutator_pauli_labels(left_label, right_label)
                if item is None:
                    continue
                phase, out_label = item
                if out_label not in self.index:
                    continue
                out_idx.append(self.index[out_label])
                left_idx.append(left_position)
                right_idx.append(right_position)
                coeffs.append(phase)
        self._out_idx = tuple(out_idx)
        self._left_idx = tuple(left_idx)
        self._right_idx = tuple(right_idx)
        self._coeffs = tuple(coeffs)

    @property
    def size(self) -> int:
        return len(self.basis_labels)

    def vector_from_operator(
        self,
        operator: SparsePauliOperator,
        *,
        dtype: torch.dtype = torch.complex64,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        return operator.to_vector(self.basis_labels, dtype=dtype, device=device)

    def embed_subset(
        self,
        subset_labels: Sequence[str],
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        """Embed coefficients on ``subset_labels`` into the full algebra basis."""

        indices = [self.index[validate_pauli_label(label, self.n_qubits)] for label in subset_labels]
        out = torch.zeros(
            coefficients.shape[:-1] + (self.size,),
            dtype=torch.complex64 if not coefficients.is_complex() else coefficients.dtype,
            device=coefficients.device,
        )
        out[..., torch.tensor(indices, device=coefficients.device)] = coefficients.to(out.dtype)
        return out

    def commutator(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Compute ``[left, right]`` for coefficient tensors on this basis."""

        left, right = torch.broadcast_tensors(left, right)
        dtype = torch.promote_types(left.dtype, right.dtype)
        if not torch.empty((), dtype=dtype).is_complex():
            dtype = torch.complex64
        left = left.to(dtype)
        right = right.to(dtype)
        result = torch.zeros_like(left, dtype=dtype)
        if not self._coeffs:
            return result
        device = left.device
        out_idx = torch.tensor(self._out_idx, dtype=torch.long, device=device)
        left_idx = torch.tensor(self._left_idx, dtype=torch.long, device=device)
        right_idx = torch.tensor(self._right_idx, dtype=torch.long, device=device)
        coeffs = torch.tensor(self._coeffs, dtype=dtype, device=device)
        source = coeffs * left[..., left_idx] * right[..., right_idx]
        result.index_add_(-1, out_idx, source)
        return result

    @staticmethod
    def norm_sq(coefficients: torch.Tensor) -> torch.Tensor:
        """Mean squared sparse-operator norm over the last axis."""

        return torch.mean(torch.sum(torch.abs(coefficients) ** 2, dim=-1).real)


class SparseRightCommutator:
    """Symbolic commutator ``[A, B]`` with full-basis ``A`` and sparse ``B``.

    This avoids precomputing all full-basis pairwise commutators. It is intended
    for the full-AGP setting where ``A`` has ``4**q`` coefficients but the
    Hamiltonian support is sparse.
    """

    def __init__(self, basis_labels: Sequence[str], right_labels: Sequence[str]):
        self.basis_labels = [validate_pauli_label(label) for label in basis_labels]
        self.right_labels = [validate_pauli_label(label, len(self.basis_labels[0])) for label in right_labels]
        self.n_qubits = infer_n_qubits(self.basis_labels)
        self.index = {label: idx for idx, label in enumerate(self.basis_labels)}
        if len(self.index) != len(self.basis_labels):
            raise ValueError("basis_labels must be unique.")
        left_idx: list[int] = []
        right_slot: list[int] = []
        out_idx: list[int] = []
        coeffs: list[complex] = []
        for slot, right_label in enumerate(self.right_labels):
            for left_position, left_label in enumerate(self.basis_labels):
                item = commutator_pauli_labels(left_label, right_label)
                if item is None:
                    continue
                phase, out_label = item
                if out_label not in self.index:
                    raise ValueError(f"Output label {out_label} is not in the commutator basis.")
                left_idx.append(left_position)
                right_slot.append(slot)
                out_idx.append(self.index[out_label])
                coeffs.append(phase)
        self._left_idx = tuple(left_idx)
        self._right_slot = tuple(right_slot)
        self._out_idx = tuple(out_idx)
        self._coeffs = tuple(coeffs)

    @property
    def basis_size(self) -> int:
        return len(self.basis_labels)

    @property
    def right_size(self) -> int:
        return len(self.right_labels)

    def commutator(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Compute ``[left, right]`` in the full basis."""

        if left.shape[-1] != self.basis_size:
            raise ValueError(f"left last axis must be {self.basis_size}, got {left.shape[-1]}.")
        if right.shape[-1] != self.right_size:
            raise ValueError(f"right last axis must be {self.right_size}, got {right.shape[-1]}.")
        prefix = torch.broadcast_shapes(left.shape[:-1], right.shape[:-1])
        left = left.expand(prefix + (self.basis_size,))
        right = right.expand(prefix + (self.right_size,))
        dtype = torch.promote_types(left.dtype, right.dtype)
        if not torch.empty((), dtype=dtype).is_complex():
            dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        left = left.to(dtype)
        right = right.to(dtype)
        result = torch.zeros(prefix + (self.basis_size,), dtype=dtype, device=left.device)
        if not self._coeffs:
            return result
        device = left.device
        chunk_size = 250_000
        for start in range(0, len(self._coeffs), chunk_size):
            stop = min(start + chunk_size, len(self._coeffs))
            left_idx = torch.tensor(self._left_idx[start:stop], dtype=torch.long, device=device)
            right_slot = torch.tensor(self._right_slot[start:stop], dtype=torch.long, device=device)
            out_idx = torch.tensor(self._out_idx[start:stop], dtype=torch.long, device=device)
            coeffs = torch.tensor(self._coeffs[start:stop], dtype=dtype, device=device)
            source = coeffs * left[..., left_idx] * right[..., right_slot]
            result.index_add_(-1, out_idx, source)
        return result


class ProjectedCommutator:
    """Symbolic commutator projected between explicit sparse label sets.

    It computes ``[left, right]`` where ``left`` lives on ``left_labels``,
    ``right`` lives on ``right_labels``, and only outputs whose Pauli label is
    present in ``output_labels`` are retained. This is useful for large-qubit
    projected residuals where full commutator closure is intentionally avoided.
    """

    def __init__(
        self,
        left_labels: Sequence[str],
        right_labels: Sequence[str],
        output_labels: Sequence[str],
    ) -> None:
        self.left_labels = [validate_pauli_label(label) for label in left_labels]
        self.right_labels = [validate_pauli_label(label, len(self.left_labels[0])) for label in right_labels]
        self.output_labels = [validate_pauli_label(label, len(self.left_labels[0])) for label in output_labels]
        self.n_qubits = infer_n_qubits(self.left_labels + self.right_labels + self.output_labels)
        self.output_index = {label: idx for idx, label in enumerate(self.output_labels)}
        if len(self.output_index) != len(self.output_labels):
            raise ValueError("output_labels must be unique.")

        left_idx: list[int] = []
        right_idx: list[int] = []
        out_idx: list[int] = []
        coeffs: list[complex] = []
        left_codes = [_encode_pauli_label(label) for label in self.left_labels]
        right_codes = [_encode_pauli_label(label) for label in self.right_labels]
        for left_position, left_code in enumerate(left_codes):
            for right_position, right_code in enumerate(right_codes):
                item = _commutator_pauli_codes_unchecked(left_code, right_code)
                if item is None:
                    continue
                phase, out_label = item
                output_position = self.output_index.get(out_label)
                if output_position is None:
                    continue
                left_idx.append(left_position)
                right_idx.append(right_position)
                out_idx.append(output_position)
                coeffs.append(phase)

        self._left_idx = tuple(left_idx)
        self._right_idx = tuple(right_idx)
        self._out_idx = tuple(out_idx)
        self._coeffs = tuple(coeffs)

    @property
    def left_size(self) -> int:
        return len(self.left_labels)

    @property
    def right_size(self) -> int:
        return len(self.right_labels)

    @property
    def output_size(self) -> int:
        return len(self.output_labels)

    @property
    def nnz(self) -> int:
        return len(self._coeffs)

    def commutator(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Compute the projected commutator."""

        if left.shape[-1] != self.left_size:
            raise ValueError(f"left last axis must be {self.left_size}, got {left.shape[-1]}.")
        if right.shape[-1] != self.right_size:
            raise ValueError(f"right last axis must be {self.right_size}, got {right.shape[-1]}.")
        prefix = torch.broadcast_shapes(left.shape[:-1], right.shape[:-1])
        left = left.expand(prefix + (self.left_size,))
        right = right.expand(prefix + (self.right_size,))
        dtype = torch.promote_types(left.dtype, right.dtype)
        if not torch.empty((), dtype=dtype).is_complex():
            dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        left = left.to(dtype)
        right = right.to(dtype)
        result = torch.zeros(prefix + (self.output_size,), dtype=dtype, device=left.device)
        if not self._coeffs:
            return result
        device = left.device
        chunk_size = 250_000
        for start in range(0, len(self._coeffs), chunk_size):
            stop = min(start + chunk_size, len(self._coeffs))
            left_idx = torch.tensor(self._left_idx[start:stop], dtype=torch.long, device=device)
            right_idx = torch.tensor(self._right_idx[start:stop], dtype=torch.long, device=device)
            out_idx = torch.tensor(self._out_idx[start:stop], dtype=torch.long, device=device)
            coeffs = torch.tensor(self._coeffs[start:stop], dtype=dtype, device=device)
            source = coeffs * left[..., left_idx] * right[..., right_idx]
            result.index_add_(-1, out_idx, source)
        return result
