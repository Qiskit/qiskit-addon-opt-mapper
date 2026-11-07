# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An application for low autocorrelation binary sequences (LABS)."""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from itertools import combinations
from numbers import Integral, Real

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qiskit_addon_opt_mapper.problems.optimization_problem import CoeffLike, OptimizationProblem
from qiskit_addon_opt_mapper.translators import to_ising

from .optimization_application import OptimizationApplication


def _positive_integer(value: int, name: str) -> int:
    """Validate a sequence length without accepting Boolean values."""
    if isinstance(value, bool | np.bool_) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")
    return int(value)


def _spin_sequence(sequence: Iterable[int]) -> list[int]:
    """Validate a nonempty spin sequence and promote NumPy integers."""
    spins = list(sequence)
    if not spins:
        raise ValueError("A LABS sequence must not be empty.")
    for spin in spins:
        if isinstance(spin, bool | np.bool_) or not isinstance(spin, Real) or spin not in (-1, 1):
            raise ValueError("A LABS sequence must contain only real spin values -1 and 1.")
    return [int(spin) for spin in spins]


class LABS(OptimizationApplication):
    r"""Optimization application for low autocorrelation binary sequences.

    For :math:`N` spins :math:`s_i\in\{-1,+1\}`, minimize the aperiodic sidelobe
    energy

    .. math::

        E(s)=\sum_{k=1}^{N-1}C_k(s)^2,\qquad
        C_k(s)=\sum_{i=0}^{N-k-1}s_i s_{i+k}.

    The model uses spin variables and a constant, quadratic, and quartic
    objective. It introduces no constraints or auxiliary variables. Classical
    energy evaluation takes :math:`O(N^2)` time; model construction takes
    :math:`O(N^3)` time and space in the worst case.

    Solver results for this model are already spins. Before interpreting a
    binary sample, convert it with ``1 - 2 * bits``. When using a converted
    model, use the converter's ``interpret`` method first.

    References:
        T. Packebusch and S. Mertens, "Low autocorrelation binary sequences",
        J. Phys. A 49, 165001 (2016), https://arxiv.org/abs/1512.02475.
        R. Shaydulin et al., "Evidence of scaling advantage for the quantum
        approximate optimization algorithm on a classically intractable
        problem", Science Advances 10, eadm6761 (2024),
        https://arxiv.org/abs/2308.02342.
    """

    def __init__(self, num_bits: int) -> None:
        """Create a LABS instance.

        Args:
            num_bits: Positive sequence length. NumPy integers are supported.

        Raises:
            TypeError: If the length is not an integer or is a Boolean.
            ValueError: If the length is less than one.
        """
        self._num_bits = _positive_integer(num_bits, "num_bits")

    @property
    def num_bits(self) -> int:
        """Return the number of spins in the sequence."""
        return self._num_bits

    def to_optimization_problem(self) -> OptimizationProblem:
        r"""Build an unconstrained spin minimization problem.

        Each squared correlation contributes ``N - k`` to the constant.
        Products of distinct correlation pairs contribute twice their product.
        Overlapping pairs reduce to quadratic terms using :math:`s_i^2=1`;
        disjoint pairs give quartic terms. Duplicate terms are summed.

        Returns:
            The LABS optimization problem with variables ``s_0`` through
            ``s_{N-1}``, in sequence order.
        """
        quadratic: defaultdict[tuple[int, int], float] = defaultdict(float)
        quartic: defaultdict[tuple[int | str, ...], float] = defaultdict(float)
        for distance in range(1, self.num_bits):
            pairs = [(index, index + distance) for index in range(self.num_bits - distance)]
            for left, right in combinations(pairs, 2):
                # The symmetric difference removes repeated spin factors.
                term = tuple(sorted(set(left) ^ set(right)))
                if len(term) == 2:
                    quadratic[term[0], term[1]] += 2.0
                else:
                    quartic[term] += 2.0

        higher_order: dict[int, CoeffLike] = {4: dict(quartic)} if quartic else {}
        problem = OptimizationProblem("LABS")
        for index in range(self.num_bits):
            problem.spin_var(name=f"s_{index}")
        problem.minimize(
            constant=self.num_bits * (self.num_bits - 1) / 2,
            quadratic=dict(quadratic),
            higher_order=higher_order,
        )
        return problem

    def interpret(self, result: np.ndarray) -> list[int]:
        """Interpret a spin result in sequence order.

        Args:
            result: One-dimensional NumPy array of spins, with one value per
                model variable. Values must be exactly -1 or +1.

        Returns:
            The spin sequence as Python integers.

        Raises:
            TypeError: If the result is not a NumPy array.
            ValueError: If its shape, length, or spin values are invalid.
        """
        values = self._result_to_x(result)
        if values.ndim != 1 or values.size != self.num_bits:
            raise ValueError(f"Expected a one-dimensional result of length {self.num_bits}.")
        return _spin_sequence(values)

    def energy(self, sequence: Iterable[int]) -> int:
        """Evaluate the sidelobe energy of a spin sequence for this instance.

        Args:
            sequence: Spins in sequence order, of length ``num_bits``.

        Returns:
            The nonnegative sidelobe energy.

        Raises:
            TypeError: If the sequence is not iterable.
            ValueError: If its length or spin values are invalid.
        """
        spins = _spin_sequence(sequence)
        if len(spins) != self.num_bits:
            raise ValueError(f"Expected a sequence of length {self.num_bits}.")
        return evaluate_labs(spins)

    def merit_factor(self, sequence: Iterable[int]) -> float:
        """Evaluate the merit factor ``N**2 / (2 * energy)``.

        Args:
            sequence: Spins in sequence order, of length ``num_bits``.

        Returns:
            The merit factor, or positive infinity for the zero-energy
            single-spin case.

        Raises:
            TypeError: If the sequence is not iterable.
            ValueError: If its length or spin values are invalid.
        """
        energy = self.energy(sequence)
        return float("inf") if energy == 0 else self.num_bits**2 / (2 * energy)

    def cost_hamiltonian(self) -> SparsePauliOp:
        """Return the full cost Hamiltonian, including the constant energy.

        Unlike :func:`~qiskit_addon_opt_mapper.translators.to_ising`, which
        returns an operator and a separate offset, this convenience method
        includes the offset as an identity term. Variable ``s_i`` maps to
        qubit ``i``, with binary 0 representing spin +1.

        Returns:
            The full diagonal LABS energy operator on ``num_bits`` qubits.
        """
        operator, offset = to_ising(self.to_optimization_problem())
        identity = SparsePauliOp("I" * self.num_bits, coeffs=[offset])
        return (operator + identity).simplify(atol=0)


def evaluate_labs(sequence: Iterable[int]) -> int:
    """Evaluate the LABS energy directly from the aperiodic correlations.

    Args:
        sequence: Nonempty iterable of spins -1 and +1. NumPy scalar values
            are converted to Python integers before arithmetic.

    Returns:
        The sidelobe energy, computed in quadratic time.

    Raises:
        TypeError: If the sequence is not iterable.
        ValueError: If it is empty or contains invalid spins.

    Examples:
        >>> evaluate_labs([1, 1, -1])
        1
    """
    spins = _spin_sequence(sequence)
    return sum(
        sum(spins[index] * spins[index + distance] for index in range(len(spins) - distance)) ** 2
        for distance in range(1, len(spins))
    )


def labs_statistics(num_spins: int) -> dict[int, int]:
    r"""Enumerate the exact energy histogram of all :math:`2^N` sequences.

    This function takes :math:`O(N^2 2^N)` time and is intended only for small
    instances. Calling it is explicit; constructing a LABS model never
    enumerates the solution space.

    Args:
        num_spins: Positive sequence length.

    Returns:
        A dictionary mapping energies to sequence counts, sorted by energy.

    Raises:
        TypeError: If the length is not an integer or is a Boolean.
        ValueError: If the length is less than one.

    Examples:
        >>> labs_statistics(3)
        {1: 4, 5: 4}
    """
    width = _positive_integer(num_spins, "num_spins")
    histogram: defaultdict[int, int] = defaultdict(int)
    for state in range(1 << width):
        spins = [1 - 2 * ((state >> index) & 1) for index in range(width)]
        histogram[evaluate_labs(spins)] += 1
    return dict(sorted(histogram.items()))


def energy_distribution(
    counts: Mapping[str | int, int], num_spins: int | None = None
) -> dict[int, int]:
    """Aggregate measurement counts by LABS energy.

    Qiskit strings are displayed most-significant bit first; integer keys
    encode qubit zero in the least-significant bit. Both are decoded in
    logical qubit order using ``spin = 1 - 2 * bit``. Counts must contain only
    the LABS register, with any routing permutation already corrected.

    Args:
        counts: Bit strings or nonnegative integer states mapped to
            nonnegative integer sample counts. Spaces in strings are removed.
        num_spins: Positive width. Required for integer keys, even in mixed
            mappings. If omitted for string keys, infer it from the first key
            and require all remaining keys to have the same width.

    Returns:
        Energy-to-count dictionary sorted by energy. Empty input, or input
        with only zero counts, returns an empty dictionary.

    Raises:
        TypeError: If a width, key, or count has an unsupported type.
        ValueError: If a width, bit string, state, or count is invalid.

    Examples:
        >>> energy_distribution({"000": 2, "001": 5})
        {1: 5, 5: 2}
    """
    if num_spins is not None:
        num_spins = _positive_integer(num_spins, "num_spins")
    width = num_spins
    distribution: defaultdict[int, int] = defaultdict(int)
    for state, count in counts.items():
        if isinstance(count, bool | np.bool_) or not isinstance(count, Integral):
            raise TypeError("Sample counts must be integers, not Booleans.")
        if count < 0:
            raise ValueError("Sample counts must be nonnegative.")
        if isinstance(state, str):
            bits = state.replace(" ", "")
            if not bits or set(bits) - {"0", "1"}:
                raise ValueError(f"Invalid bit string: {state!r}.")
            if width is None:
                width = len(bits)
            if len(bits) != width:
                raise ValueError(f"Expected bit strings of width {width}.")
            spins = [1 - 2 * int(bit) for bit in reversed(bits)]
        elif isinstance(state, Integral) and not isinstance(state, bool | np.bool_):
            if num_spins is None:
                raise ValueError("num_spins is required for integer count keys.")
            # The explicit width was validated above; keep leading zero bits.
            state = int(state)
            if not 0 <= state < 1 << num_spins:
                raise ValueError("Integer state does not fit the specified width.")
            spins = [1 - 2 * ((state >> index) & 1) for index in range(num_spins)]
        else:
            raise TypeError("Count keys must be bit strings or integers, not Booleans.")
        if count:
            distribution[evaluate_labs(spins)] += int(count)
    return dict(sorted(distribution.items()))


# Compatibility with the original research script's function spelling.
evaluate_LABS = evaluate_labs
