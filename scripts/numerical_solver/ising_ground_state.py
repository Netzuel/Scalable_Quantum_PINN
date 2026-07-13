from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DEFAULT_ATOL = 1e-10


@dataclass(frozen=True)
class DiagonalIsingProblem:
    """Sparse diagonal Ising Hamiltonian in Pauli-label order.

    The energy is ``constant + sum_i fields[i] z_i + sum_ij J_ij z_i z_j``
    with ``z_i=+1`` for bit ``0`` and ``z_i=-1`` for bit ``1``.
    """

    num_qubits: int
    constant: float
    fields: tuple[float, ...]
    couplings: tuple[tuple[int, int, float], ...]

    def __post_init__(self) -> None:
        if self.num_qubits < 2:
            raise ValueError("Use at least two qubits.")
        if len(self.fields) != self.num_qubits:
            raise ValueError("The field vector length must equal num_qubits.")
        seen: set[tuple[int, int]] = set()
        for left, right, _ in self.couplings:
            if not 0 <= left < right < self.num_qubits:
                raise ValueError(f"Invalid coupling edge {(left, right)}.")
            edge = (left, right)
            if edge in seen:
                raise ValueError(f"Duplicate coupling edge {edge}.")
            seen.add(edge)
            if right != left + 1 and edge != (0, self.num_qubits - 1):
                raise ValueError("Exact dynamic programming supports only path or cycle couplings.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QuboProblem:
    """QUBO energy ``constant + linear*x + quadratic*x*x``."""

    num_variables: int
    constant: float
    linear: tuple[float, ...]
    quadratic: tuple[tuple[int, int, float], ...]

    def energy(self, bitstring: str) -> float:
        bits = _validate_bitstring(bitstring, self.num_variables)
        value = self.constant + sum(coeff * bit for coeff, bit in zip(self.linear, bits))
        value += sum(coeff * bits[left] * bits[right] for left, right, coeff in self.quadratic)
        return float(value)


@dataclass(frozen=True)
class GroundStateSolution:
    method: str
    ground_energy: float
    ground_bitstrings: tuple[str, ...]
    ground_state_degeneracy: int
    ground_bitstrings_truncated: bool
    first_excited_energy: float | None
    spectral_gap: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _EnergyLevel:
    energy: float
    count: int
    paths: tuple[tuple[int, ...], ...]


def _validate_bitstring(bitstring: str, length: int) -> tuple[int, ...]:
    if len(bitstring) != length or set(bitstring) - {"0", "1"}:
        raise ValueError(f"Expected a {length}-character binary bitstring.")
    return tuple(int(bit) for bit in bitstring)


def _coefficient(value: object, *, label: str) -> float:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(f"Coefficient for {label!r} must contain [real, imag].")
        real, imag = float(value[0]), float(value[1])
    else:
        real, imag = float(value), 0.0
    if abs(imag) > DEFAULT_ATOL:
        raise ValueError(f"Coefficient for {label!r} is not real.")
    return real


def load_final_ising_problem(path: str | Path) -> DiagonalIsingProblem:
    """Load a diagonal path/cycle Ising ``H_final`` from a pair JSON file."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    num_qubits = int(payload["n_qubits"])
    terms = payload["hamiltonians"]["final"]["terms"]
    if not isinstance(terms, Mapping):
        raise TypeError("The final Hamiltonian terms must be a mapping.")

    constant = 0.0
    fields = [0.0] * num_qubits
    coupling_map: dict[tuple[int, int], float] = {}
    for raw_label, raw_value in terms.items():
        label = str(raw_label).upper()
        if len(label) != num_qubits:
            raise ValueError(f"Pauli label {label!r} has the wrong length.")
        if set(label) - {"I", "Z"}:
            raise ValueError("The final Hamiltonian is not diagonal in the computational basis.")
        coefficient = _coefficient(raw_value, label=label)
        sites = tuple(index for index, symbol in enumerate(label) if symbol == "Z")
        if not sites:
            constant += coefficient
        elif len(sites) == 1:
            fields[sites[0]] += coefficient
        elif len(sites) == 2:
            edge = (sites[0], sites[1])
            coupling_map[edge] = coupling_map.get(edge, 0.0) + coefficient
        else:
            raise ValueError("The final Hamiltonian contains interactions above Ising order two.")

    couplings = tuple((left, right, value) for (left, right), value in sorted(coupling_map.items()))
    return DiagonalIsingProblem(num_qubits, float(constant), tuple(fields), couplings)


def _scaled_coefficient(value: float, gradient: float, index: int, count: int) -> float:
    if count <= 1 or gradient == 0.0:
        return float(value)
    center = 0.5 * (count - 1)
    normalized = (index - center) / max(center, 1.0)
    return float(value) * (1.0 + float(gradient) * normalized)


def build_driver_problem_ising(
    *,
    num_qubits: int,
    z_field: float = 0.35,
    zz_coupling: float = 1.0,
    field_gradient: float = 0.15,
    coupling_gradient: float = 0.10,
    periodic: bool = False,
) -> DiagonalIsingProblem:
    """Construct the current analytic ``H_final`` without a Hilbert-space matrix."""

    if num_qubits < 2:
        raise ValueError("Use at least two qubits.")
    fields = tuple(
        -_scaled_coefficient(z_field, field_gradient, site, num_qubits) for site in range(num_qubits)
    )
    edges = [(site, site + 1) for site in range(num_qubits - 1)]
    if periodic:
        edges.append((0, num_qubits - 1))
    couplings = tuple(
        (
            left,
            right,
            -_scaled_coefficient(zz_coupling, coupling_gradient, edge_index, len(edges)),
        )
        for edge_index, (left, right) in enumerate(edges)
    )
    return DiagonalIsingProblem(num_qubits, 0.0, fields, couplings)


def energy_of_bitstring(problem: DiagonalIsingProblem, bitstring: str) -> float:
    bits = _validate_bitstring(bitstring, problem.num_qubits)
    spins = tuple(1 - 2 * bit for bit in bits)
    energy = problem.constant + sum(field * spin for field, spin in zip(problem.fields, spins))
    energy += sum(value * spins[left] * spins[right] for left, right, value in problem.couplings)
    return float(energy)


def to_qubo(problem: DiagonalIsingProblem) -> QuboProblem:
    """Convert ``z_i=1-2*x_i`` exactly into a binary quadratic objective."""

    constant = problem.constant + sum(problem.fields)
    linear = [-2.0 * field for field in problem.fields]
    quadratic: list[tuple[int, int, float]] = []
    for left, right, coupling in problem.couplings:
        constant += coupling
        linear[left] -= 2.0 * coupling
        linear[right] -= 2.0 * coupling
        quadratic.append((left, right, 4.0 * coupling))
    return QuboProblem(problem.num_qubits, float(constant), tuple(linear), tuple(quadratic))


def _merge_levels(
    levels: Iterable[_EnergyLevel],
    *,
    max_levels: int,
    max_bitstrings: int,
    atol: float,
) -> tuple[_EnergyLevel, ...]:
    merged: list[_EnergyLevel] = []
    for level in sorted(levels, key=lambda item: item.energy):
        match = next((item for item in merged if math.isclose(item.energy, level.energy, abs_tol=atol, rel_tol=0.0)), None)
        if match is None:
            if len(merged) >= max_levels:
                continue
            merged.append(level)
            continue
        index = merged.index(match)
        paths = tuple(dict.fromkeys(match.paths + level.paths))[:max_bitstrings]
        merged[index] = _EnergyLevel(match.energy, match.count + level.count, paths)
    return tuple(merged)


def _solve_fixed_first_spin(
    problem: DiagonalIsingProblem,
    *,
    first_spin: int,
    adjacent: Sequence[float],
    closing: float | None,
    max_bitstrings: int,
    atol: float,
) -> tuple[_EnergyLevel, ...]:
    states: dict[int, tuple[_EnergyLevel, ...]] = {
        first_spin: (_EnergyLevel(problem.constant + problem.fields[0] * first_spin, 1, ((first_spin,),)),)
    }
    for site in range(1, problem.num_qubits):
        next_states: dict[int, tuple[_EnergyLevel, ...]] = {}
        for spin in (-1, 1):
            candidates: list[_EnergyLevel] = []
            for previous_spin, levels in states.items():
                increment = problem.fields[site] * spin + adjacent[site - 1] * previous_spin * spin
                for level in levels:
                    candidates.append(
                        _EnergyLevel(
                            level.energy + increment,
                            level.count,
                            tuple(path + (spin,) for path in level.paths),
                        )
                    )
            next_states[spin] = _merge_levels(
                candidates,
                max_levels=2,
                max_bitstrings=max_bitstrings,
                atol=atol,
            )
        states = next_states

    candidates = []
    for last_spin, levels in states.items():
        closing_increment = 0.0 if closing is None else closing * first_spin * last_spin
        candidates.extend(
            _EnergyLevel(level.energy + closing_increment, level.count, level.paths) for level in levels
        )
    return _merge_levels(candidates, max_levels=2, max_bitstrings=max_bitstrings, atol=atol)


def solve_dynamic_programming(
    problem: DiagonalIsingProblem,
    *,
    max_bitstrings: int = 256,
    atol: float = DEFAULT_ATOL,
) -> GroundStateSolution:
    """Solve a diagonal path or cycle Ising model exactly in linear time."""

    edge_map = {(left, right): value for left, right, value in problem.couplings}
    adjacent = [edge_map.get((site, site + 1), 0.0) for site in range(problem.num_qubits - 1)]
    closing = edge_map.get((0, problem.num_qubits - 1)) if problem.num_qubits > 2 else None
    levels = _merge_levels(
        (
            level
            for first_spin in (-1, 1)
            for level in _solve_fixed_first_spin(
                problem,
                first_spin=first_spin,
                adjacent=adjacent,
                closing=closing,
                max_bitstrings=max_bitstrings,
                atol=atol,
            )
        ),
        max_levels=2,
        max_bitstrings=max_bitstrings,
        atol=atol,
    )
    ground = levels[0]
    first_excited = levels[1].energy if len(levels) > 1 else None
    bitstrings = tuple("".join("0" if spin == 1 else "1" for spin in path) for path in ground.paths)
    return GroundStateSolution(
        method="exact_path_dynamic_programming",
        ground_energy=float(ground.energy),
        ground_bitstrings=bitstrings,
        ground_state_degeneracy=int(ground.count),
        ground_bitstrings_truncated=ground.count > len(bitstrings),
        first_excited_energy=None if first_excited is None else float(first_excited),
        spectral_gap=None if first_excited is None else float(first_excited - ground.energy),
    )


def solve_brute_force(
    problem: DiagonalIsingProblem,
    *,
    max_bitstrings: int = 256,
    atol: float = DEFAULT_ATOL,
) -> GroundStateSolution:
    """Exhaustively enumerate all bitstrings; intended only for small q."""

    if problem.num_qubits > 24:
        raise ValueError("Brute-force enumeration is restricted to at most 24 qubits.")
    best_energy = math.inf
    second_energy = math.inf
    ground_count = 0
    ground_bitstrings: list[str] = []
    for state in range(1 << problem.num_qubits):
        bitstring = format(state, f"0{problem.num_qubits}b")
        energy = energy_of_bitstring(problem, bitstring)
        if energy < best_energy - atol:
            if math.isfinite(best_energy):
                second_energy = min(second_energy, best_energy)
            best_energy = energy
            ground_count = 1
            ground_bitstrings = [bitstring]
        elif math.isclose(energy, best_energy, abs_tol=atol, rel_tol=0.0):
            ground_count += 1
            if len(ground_bitstrings) < max_bitstrings:
                ground_bitstrings.append(bitstring)
        elif energy < second_energy - atol:
            second_energy = energy

    first_excited = second_energy if math.isfinite(second_energy) else None
    return GroundStateSolution(
        method="exact_exhaustive_enumeration",
        ground_energy=float(best_energy),
        ground_bitstrings=tuple(ground_bitstrings),
        ground_state_degeneracy=ground_count,
        ground_bitstrings_truncated=ground_count > len(ground_bitstrings),
        first_excited_energy=None if first_excited is None else float(first_excited),
        spectral_gap=None if first_excited is None else float(first_excited - best_energy),
    )


def ferromagnetic_closed_form(
    problem: DiagonalIsingProblem,
    *,
    atol: float = DEFAULT_ATOL,
) -> GroundStateSolution | None:
    """Return the termwise exact solution when every term favors one aligned state."""

    if any(coupling > atol for _, _, coupling in problem.couplings):
        return None
    if all(field < -atol for field in problem.fields):
        spins = (1,) * problem.num_qubits
        bitstring = "0" * problem.num_qubits
    elif all(field > atol for field in problem.fields):
        spins = (-1,) * problem.num_qubits
        bitstring = "1" * problem.num_qubits
    else:
        return None
    energy = problem.constant + sum(field * spin for field, spin in zip(problem.fields, spins))
    energy += sum(value * spins[left] * spins[right] for left, right, value in problem.couplings)
    return GroundStateSolution(
        method="termwise_ferromagnetic_closed_form",
        ground_energy=float(energy),
        ground_bitstrings=(bitstring,),
        ground_state_degeneracy=1,
        ground_bitstrings_truncated=False,
        first_excited_energy=None,
        spectral_gap=None,
    )
