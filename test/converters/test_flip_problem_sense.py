# This code is a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Flip problem sense tests."""

import unittest

from qiskit_addon_opt_mapper import OptimizationProblem
from qiskit_addon_opt_mapper.converters import MaximizeToMinimize, MinimizeToMaximize

from ..optimization_test_case import OptimizationTestCase


class TestFlipProblemSense(OptimizationTestCase):
    """Tests various flips of problem sense."""

    def test_maximize_to_minimize(self):
        """Test maximization to minimization conversion."""
        op_max = OptimizationProblem()
        op_min = OptimizationProblem()
        for i in range(2):
            op_max.binary_var(name=f"x{i}")
            op_min.binary_var(name=f"x{i}")
        op_max.integer_var(name="x2", lowerbound=-3, upperbound=3)
        op_min.integer_var(name="x2", lowerbound=-3, upperbound=3)
        op_max.maximize(constant=3, linear={"x0": 1}, quadratic={("x1", "x2"): 2})
        op_min.minimize(constant=3, linear={"x0": 1}, quadratic={("x1", "x2"): 2})

        # check conversion of maximization problem
        conv = MaximizeToMinimize()
        op_conv = conv.convert(op_max)
        self.assertEqual(op_conv.objective.sense, op_conv.objective.Sense.MINIMIZE)
        x = [0, 1, 2]
        fval_min = op_conv.objective.evaluate(conv.interpret(x))
        self.assertAlmostEqual(fval_min, -7)
        self.assertAlmostEqual(op_max.objective.evaluate(x), -fval_min)

        # check conversion of minimization problem
        op_conv = conv.convert(op_min)
        self.assertEqual(op_conv.objective.sense, op_min.objective.sense)
        fval_min = op_conv.objective.evaluate(conv.interpret(x))
        self.assertAlmostEqual(op_min.objective.evaluate(x), fval_min)

    def test_minimize_to_maximize(self):
        """Test minimization to maximization conversion."""
        op_max = OptimizationProblem()
        op_min = OptimizationProblem()
        for i in range(2):
            op_max.binary_var(name=f"x{i}")
            op_min.binary_var(name=f"x{i}")
        op_max.integer_var(name="x2", lowerbound=-3, upperbound=3)
        op_min.integer_var(name="x2", lowerbound=-3, upperbound=3)
        op_max.maximize(constant=3, linear={"x0": 1}, quadratic={("x1", "x2"): 2})
        op_min.minimize(constant=3, linear={"x0": 1}, quadratic={("x1", "x2"): 2})

        # check conversion of maximization problem
        conv = MinimizeToMaximize()
        op_conv = conv.convert(op_min)
        self.assertEqual(op_conv.objective.sense, op_conv.objective.Sense.MAXIMIZE)
        x = [0, 1, 2]
        fval_max = op_conv.objective.evaluate(conv.interpret(x))
        self.assertAlmostEqual(fval_max, -7)
        self.assertAlmostEqual(op_max.objective.evaluate(x), -fval_max)

        # check conversion of maximization problem
        op_conv = conv.convert(op_max)
        self.assertEqual(op_conv.objective.sense, op_max.objective.sense)
        fval_max = op_conv.objective.evaluate(conv.interpret(x))
        self.assertAlmostEqual(op_min.objective.evaluate(x), fval_max)

    def test_higher_order_maximize_to_minimize(self):
        """Test conversion of higher order objective."""
        op = OptimizationProblem()
        for i in range(3):
            op.binary_var(name=f"x{i}")
        op.maximize(
            constant=1,
            linear={"x0": 1, "x1": 1},
            quadratic={("x0", "x1"): 2},
            higher_order={3: {("x0", "x1", "x2"): 3}},
        )

        conv = MaximizeToMinimize()
        op_conv = conv.convert(op)
        self.assertEqual(op_conv.objective.sense, op_conv.objective.Sense.MINIMIZE)
        x = [1, 1, 1]
        fval_min = op_conv.objective.evaluate(conv.interpret(x))
        self.assertAlmostEqual(fval_min, -8)
        self.assertAlmostEqual(op.objective.evaluate(x), -fval_min)

    def test_higher_order_minimize_to_maximize(self):
        """Test conversion of higher order objective."""
        op = OptimizationProblem()
        for i in range(3):
            op.binary_var(name=f"x{i}")
        op.minimize(
            constant=1,
            linear={"x0": 1, "x1": 1},
            quadratic={("x0", "x1"): 2},
            higher_order={3: {("x0", "x1", "x2"): 3}},
        )

        conv = MinimizeToMaximize()
        op_conv = conv.convert(op)
        self.assertEqual(op_conv.objective.sense, op_conv.objective.Sense.MAXIMIZE)
        x = [1, 1, 1]
        fval_max = op_conv.objective.evaluate(conv.interpret(x))
        self.assertAlmostEqual(fval_max, -8)
        self.assertAlmostEqual(op.objective.evaluate(x), -fval_max)


if __name__ == "__main__":
    unittest.main()
