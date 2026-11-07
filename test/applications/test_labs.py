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

"""Tests for the LABS application and classical evaluation helpers."""

from itertools import product

import numpy as np
from qiskit_addon_opt_mapper.applications import LABS
from qiskit_addon_opt_mapper.applications.labs import (
    energy_distribution,
    evaluate_LABS,
    evaluate_labs,
    labs_statistics,
)
from qiskit_addon_opt_mapper.converters import SpinToBinary
from qiskit_addon_opt_mapper.problems import OptimizationObjective, VarType
from qiskit_addon_opt_mapper.translators import to_ising

from ..optimization_test_case import OptimizationTestCase


def _direct_energy(spins):
    """Independent definition of the aperiodic sidelobe energy."""
    energy = 0
    for lag in range(1, len(spins)):
        correlation = 0
        for first, second in zip(spins[:-lag], spins[lag:], strict=True):
            correlation += first * second
        energy += correlation * correlation
    return energy


class TestLABS(OptimizationTestCase):
    """Verify the model, interpretation, and mapping against direct correlations."""

    def test_constructor_and_property(self):
        for length in (1, 5, np.int32(3), np.int64(4)):
            with self.subTest(length=length):
                application = LABS(length)
                self.assertEqual(application.num_bits, int(length))
                self.assertIsInstance(application.num_bits, int)
                with self.assertRaises(AttributeError):
                    application.num_bits = 7

    def test_invalid_lengths(self):
        for length in (True, False, np.bool_(True), 3.0, "3", None):
            for factory in (LABS, labs_statistics):
                with self.subTest(length=length, factory=factory), self.assertRaises(TypeError):
                    factory(length)
        for length in (0, -1, np.int64(-2)):
            for factory in (LABS, labs_statistics):
                with self.subTest(length=length, factory=factory), self.assertRaises(ValueError):
                    factory(length)

    def test_model_structure(self):
        for length in range(1, 9):
            problem = LABS(length).to_optimization_problem()
            with self.subTest(length=length):
                self.assertEqual(problem.name, "LABS")
                self.assertEqual(problem.get_num_vars(), length)
                self.assertEqual(
                    [var.name for var in problem.variables],
                    [f"s_{index}" for index in range(length)],
                )
                self.assertTrue(all(var.vartype == VarType.SPIN for var in problem.variables))
                self.assertEqual(problem.objective.sense, OptimizationObjective.Sense.MINIMIZE)
                self.assertEqual(problem.objective.constant, length * (length - 1) / 2)
                self.assertEqual(problem.objective.linear.to_dict(), {})
                self.assertEqual(problem.linear_constraints, [])
                self.assertEqual(problem.quadratic_constraints, [])
                self.assertEqual(problem.higher_order_constraints, [])

    def test_explicit_coefficients(self):
        single = LABS(1).to_optimization_problem().objective
        self.assertEqual(single.quadratic.to_dict(), {})
        self.assertEqual(single.higher_order, {})
        three = LABS(3).to_optimization_problem().objective
        self.assertEqual(three.quadratic.to_dict(), {(0, 2): 2})
        self.assertEqual(three.higher_order, {})
        four = LABS(4).to_optimization_problem().objective
        self.assertEqual(four.quadratic.to_dict(), {(0, 2): 2, (1, 3): 2})
        self.assertEqual(four.higher_order[4].to_dict(), {(0, 1, 2, 3): 4})

    def test_exhaustive_objective_and_helpers(self):
        # Published minima for N=3..8: Packebusch & Mertens (2016), Table 1.
        minima = (0, 1, 1, 2, 2, 7, 3, 8)
        for length, expected_minimum in enumerate(minima, start=1):
            application = LABS(length)
            problem = application.to_optimization_problem()
            observed_minimum = float("inf")
            for spins in product((-1, 1), repeat=length):
                expected = _direct_energy(spins)
                with self.subTest(length=length, spins=spins):
                    self.assertEqual(problem.objective.evaluate(list(spins)), expected)
                    self.assertEqual(evaluate_labs(spins), expected)
                    self.assertEqual(application.energy(spins), expected)
                observed_minimum = min(observed_minimum, expected)
            self.assertEqual(observed_minimum, expected_minimum)

    def test_ising_and_full_hamiltonian(self):
        for length in range(1, 9):
            application = LABS(length)
            operator, offset = to_ising(application.to_optimization_problem())
            full = application.cost_hamiltonian()
            self.assertEqual(operator.num_qubits, length)
            self.assertEqual(full.num_qubits, length)
            self.assertEqual(offset, length * (length - 1) / 2)
            diagonal = np.diag(operator.to_matrix()).real
            full_diagonal = np.diag(full.to_matrix()).real
            for state in range(1 << length):
                spins = [1 - 2 * ((state >> index) & 1) for index in range(length)]
                with self.subTest(length=length, state=state):
                    self.assertEqual(diagonal[state] + offset, _direct_energy(spins))
                    self.assertEqual(full_diagonal[state], _direct_energy(spins))

    def test_binary_converter(self):
        for length in range(1, 9):
            application = LABS(length)
            converter = SpinToBinary()
            binary_problem = converter.convert(application.to_optimization_problem())
            self.assertTrue(all(var.vartype == VarType.BINARY for var in binary_problem.variables))
            for bits in product((0, 1), repeat=length):
                spins = [1 - 2 * bit for bit in bits]
                with self.subTest(length=length, bits=bits):
                    self.assertEqual(
                        binary_problem.objective.evaluate(list(bits)), _direct_energy(spins)
                    )
                    self.assertEqual(application.interpret(converter.interpret(list(bits))), spins)

    def test_interpret(self):
        application = LABS(3)
        for result in (np.array([1, -1, 1]), np.array([1.0, -1.0, 1.0])):
            self.assertEqual(application.interpret(result), [1, -1, 1])
            self.assertTrue(all(type(spin) is int for spin in application.interpret(result)))
        for result in ([1, -1, 1], (1, -1, 1), None):
            with self.subTest(result=result), self.assertRaises(TypeError):
                application.interpret(result)
        for result in (
            np.array(1),
            np.array([[1, -1, 1]]),
            np.array([1, -1]),
            np.array([]),
            np.array([1, 0, 1]),
            np.array([1, 0.5, 1]),
        ):
            with self.subTest(result=result), self.assertRaises(ValueError):
                application.interpret(result)

    def test_energy_validation(self):
        application = LABS(3)
        with self.assertRaises(ValueError):
            application.energy([1, -1])
        for invalid in (
            [],
            [0],
            [2],
            [True],
            [np.bool_(True)],
            [float("nan")],
            [float("inf")],
            [1j],
            ["1"],
            [[1, -1]],
            np.ones((2, 2)),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                evaluate_labs(invalid)
        with self.assertRaises(TypeError):
            evaluate_labs(1)
        with self.assertRaises(ValueError):
            application.merit_factor([0, 1, 1])

    def test_python_integer_arithmetic(self):
        spins = np.ones(20, dtype=np.int8)
        energy = evaluate_labs(spins)
        self.assertEqual(energy, 2470)
        self.assertIsInstance(energy, int)
        self.assertEqual(evaluate_labs(iter([1, 1, -1])), 1)
        self.assertIs(evaluate_LABS, evaluate_labs)

    def test_merit_factor(self):
        self.assertEqual(LABS(3).merit_factor([1, 1, -1]), 4.5)
        self.assertEqual(LABS(1).energy([-1]), 0)
        self.assertEqual(LABS(1).merit_factor([-1]), float("inf"))

    def test_symmetries(self):
        for length in range(1, 7):
            for spins in product((-1, 1), repeat=length):
                energy = evaluate_labs(spins)
                self.assertEqual(evaluate_labs(spins[::-1]), energy)
                self.assertEqual(evaluate_labs([-spin for spin in spins]), energy)
                self.assertEqual(
                    evaluate_labs([spin * (-1) ** index for index, spin in enumerate(spins)]),
                    energy,
                )

    def test_statistics(self):
        self.assertEqual(labs_statistics(1), {0: 2})
        self.assertEqual(labs_statistics(np.int64(3)), {1: 4, 5: 4})
        for length in range(1, 9):
            expected = {}
            for spins in product((-1, 1), repeat=length):
                energy = _direct_energy(spins)
                expected[energy] = expected.get(energy, 0) + 1
            histogram = labs_statistics(length)
            self.assertEqual(histogram, dict(sorted(expected.items())))
            self.assertEqual(list(histogram), sorted(histogram))
            self.assertEqual(sum(histogram.values()), 1 << length)

    def test_energy_distribution(self):
        strings = {"000": 2, "001": 5, "011": 3, "1 11": 4}
        integers = {0: 2, 1: 5, 3: 3, np.int64(7): np.int64(4)}
        expected = {1: 8, 5: 6}
        self.assertEqual(energy_distribution(strings), expected)
        self.assertEqual(energy_distribution(strings, 3), expected)
        self.assertEqual(energy_distribution(integers, np.int64(3)), expected)
        self.assertEqual(energy_distribution({"000": 2, 1: 5}, 3), {1: 5, 5: 2})
        self.assertEqual(energy_distribution({}), {})
        self.assertEqual(energy_distribution({}, 3), {})
        self.assertEqual(energy_distribution({"000": 0}), {})
        self.assertEqual(energy_distribution({0: 2}, 1), {0: 2})

    def test_distribution_validation(self):
        for counts, width in (
            ({"": 1}, None),
            ({"   ": 1}, None),
            ({"010x": 1}, None),
            ({"0": 1, "00": 2}, None),
            ({"000": 1}, 2),
            ({0: 1}, None),
            ({"000": 1, 1: 1}, None),
            ({-1: 1}, 3),
            ({8: 1}, 3),
            ({"000": -1}, None),
            ({}, 0),
        ):
            with self.subTest(counts=counts, width=width), self.assertRaises(ValueError):
                energy_distribution(counts, width)
        for counts, width in (
            ({True: 1}, 3),
            ({np.bool_(True): 1}, 3),
            ({1.5: 1}, 3),
            ({b"001": 1}, None),
            ({"001": 1.5}, None),
            ({"001": True}, None),
            ({"001": np.bool_(True)}, None),
            ({}, True),
            ({}, "3"),
        ):
            with self.subTest(counts=counts, width=width), self.assertRaises(TypeError):
                energy_distribution(counts, width)
