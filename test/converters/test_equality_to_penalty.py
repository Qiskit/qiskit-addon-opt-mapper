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

"""Test EqualityToPenalty Converters"""

import unittest

import numpy as np
from qiskit_addon_opt_mapper import OptimizationError, OptimizationProblem
from qiskit_addon_opt_mapper.converters import (
    EqualityToPenalty,
)
from qiskit_addon_opt_mapper.problems import Constraint

from ..optimization_test_case import OptimizationTestCase


class TestEqualityToPenaltyConverter(OptimizationTestCase):
    """Test EqualityToPenalty Converters"""

    def test_penalize_sense(self):
        """Test PenalizeLinearEqualityConstraints with senses"""
        op = OptimizationProblem()
        for i in range(3):
            op.binary_var(name=f"x{i}")
        # Linear constraints
        linear_constraint = {"x0": 1, "x1": 1}
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, "x0x1")
        linear_constraint = {"x1": 1, "x2": -1}
        op.linear_constraint(linear_constraint, Constraint.Sense.LE, 2, "x1x2")
        linear_constraint = {"x0": 1, "x2": 3}
        op.linear_constraint(linear_constraint, Constraint.Sense.GE, 2, "x0x2")
        self.assertEqual(op.get_num_linear_constraints(), 3)
        conv = EqualityToPenalty()
        with self.assertRaises(OptimizationError):
            conv.convert(op)

    def test_penalize_binary(self):
        """Test PenalizeLinearEqualityConstraints with binary variables"""
        op = OptimizationProblem()
        for i in range(3):
            op.binary_var(name=f"x{i}")
        # Linear constraints
        linear_constraint = {"x0": 1, "x1": 1}
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, "x0x1")
        linear_constraint = {"x1": 1, "x2": -1}
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 2, "x1x2")
        linear_constraint = {"x0": 1, "x2": 3}
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 2, "x0x2")
        self.assertEqual(op.get_num_linear_constraints(), 3)
        conv = EqualityToPenalty()
        op2 = conv.convert(op)
        self.assertEqual(op2.get_num_linear_constraints(), 0)

        new_x = conv.interpret(np.arange(3))
        np.testing.assert_array_almost_equal(new_x, np.arange(3))

    def test_penalize_integer(self):
        """Test PenalizeLinearEqualityConstraints with integer variables"""
        op = OptimizationProblem()
        for i in range(3):
            op.integer_var(name=f"x{i}", lowerbound=-3, upperbound=3)
        # Linear constraints
        linear_constraint = {"x0": 1, "x1": 1}
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, "x0x1")
        linear_constraint = {"x1": 1, "x2": -1}
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 2, "x1x2")
        linear_constraint = {"x0": 1, "x2": -1}
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, "x0x2")
        op.minimize(constant=3, linear={"x0": 1}, quadratic={("x1", "x2"): 2})
        self.assertEqual(op.get_num_linear_constraints(), 3)
        conv = EqualityToPenalty()
        op2 = conv.convert(op)
        self.assertEqual(op2.get_num_linear_constraints(), 0)

        new_x = conv.interpret([0, 1, -1])
        np.testing.assert_array_almost_equal(new_x, [0, 1, -1])

    def test_auto_penalty(self):
        """Test auto penalty function"""
        op = OptimizationProblem()
        op.binary_var("x")
        op.binary_var("y")
        op.binary_var("z")
        op.minimize(
            constant=3,
            linear={"x": 1},
            quadratic={("x", "y"): 2},
            higher_order={3: {("x", "y", "z"): 3}},
        )
        op.linear_constraint(linear={"x": 1, "y": 1, "z": 1}, sense="EQ", rhs=2, name="xyz_eq")
        lineq2penalty_auto = EqualityToPenalty()
        qubo_auto = lineq2penalty_auto.convert(op)
        self.assertEqual(lineq2penalty_auto.penalty, 7)
        self.assertEqual(qubo_auto.objective.linear.to_dict(), {0: -27, 1: -28, 2: -28})
        self.assertEqual(
            qubo_auto.objective.quadratic.to_dict(),
            {
                (0, 0): 7.0,
                (0, 1): 16,
                (0, 2): 14.0,
                (1, 1): 7.0,
                (1, 2): 14.0,
                (2, 2): 7.0,
            },
        )
        self.assertEqual(qubo_auto.objective.higher_order[3].to_dict(), {(0, 1, 2): 3})

    def test_auto_penalty_warning(self):
        """Test warnings of auto penalty function"""
        op = OptimizationProblem()
        op.binary_var("x")
        op.binary_var("y")
        op.binary_var("z")
        op.minimize(linear={"x": 1, "y": 2})
        op.linear_constraint(linear={"x": 0.5, "y": 0.5, "z": 0.5}, sense="EQ", rhs=1, name="xyz")
        with self.assertLogs("qiskit_addon_opt_mapper", level="WARNING") as log:
            lineq2penalty = EqualityToPenalty()
            _ = lineq2penalty.convert(op)
        warning = (
            "WARNING:qiskit_addon_opt_mapper.converters.equality_to_penalty:"
            "Warning: Using 100000.000000 for the penalty coefficient because a float "
            "coefficient exists in constraints. \nThe value could be too small. If so, "
            "set the penalty coefficient manually."
        )
        self.assertIn(warning, log.output)

    def test_penalty_recalculation_when_reusing(self):
        """Test the penalty retrieval and recalculation of EqualityToPenalty"""
        op = OptimizationProblem()
        op.binary_var("x")
        op.binary_var("y")
        op.binary_var("z")
        op.minimize(constant=3, linear={"x": 1}, quadratic={("x", "y"): 2})
        op.linear_constraint(linear={"x": 1, "y": 1, "z": 1}, sense="EQ", rhs=2, name="xyz_eq")
        # First, create a converter with no penalty
        lineq2penalty = EqualityToPenalty()
        self.assertIsNone(lineq2penalty.penalty)
        # Then converter must calculate the penalty for the problem (should be 4.0)
        lineq2penalty.convert(op)
        self.assertEqual(4, lineq2penalty.penalty)
        # Re-use the converter with a newly defined penalty
        lineq2penalty.penalty = 3
        lineq2penalty.convert(op)
        self.assertEqual(3, lineq2penalty.penalty)
        # Re-use the converter letting the penalty be calculated again
        lineq2penalty.penalty = None
        lineq2penalty.convert(op)
        self.assertEqual(4, lineq2penalty.penalty)

    def test_penalty_recalculation_when_reusing2(self):
        """Test the penalty retrieval and recalculation of EqualityToPenalty 2"""
        op = OptimizationProblem()
        op.binary_var("x")
        op.binary_var("y")
        op.binary_var("z")
        op.minimize(constant=3, linear={"x": 1}, quadratic={("x", "y"): 2})
        op.linear_constraint(linear={"x": 1, "y": 1, "z": 1}, sense="EQ", rhs=2, name="xyz_eq")
        # First, create a converter with no penalty
        lineq2penalty = EqualityToPenalty()
        self.assertIsNone(lineq2penalty.penalty)
        # Then converter must calculate the penalty for the problem (should be 4.0)
        lineq2penalty.convert(op)
        self.assertEqual(4, lineq2penalty.penalty)
        # Re-use the converter for a new problem
        op2 = OptimizationProblem()
        op2.binary_var("x")
        op2.minimize(linear={"x": 10})
        op2.linear_constraint({"x": 1}, "==", 0)
        lineq2penalty.convert(op2)
        self.assertEqual(11, lineq2penalty.penalty)

    def test_higher_order_constraint(self):
        """Test PenalizeLinearEqualityConstraints with higher order constraints"""
        op = OptimizationProblem()
        for i in range(4):
            op.binary_var(name=f"x{i}")
        # Linear constraints
        op.higher_order_constraint(
            higher_order={3: {(0, 1, 2): 1}, 4: {(0, 1, 2, 3): 1}},
            sense="==",
            rhs=5,
            name="c0",
        )
        conv = EqualityToPenalty(penalty=10)
        op2 = conv.convert(op)
        self.assertEqual(op2.get_num_higher_order_constraints(), 0)
        self.assertEqual(op2.objective.higher_order[3].to_dict(), {(0, 1, 2): -100})
        self.assertEqual(op2.objective.higher_order[4].to_dict(), {(0, 1, 2, 3): -100})
        self.assertEqual(op2.objective.higher_order[6].to_dict(), {(0, 0, 1, 1, 2, 2): 10})
        self.assertEqual(op2.objective.higher_order[7].to_dict(), {(0, 0, 1, 1, 2, 2, 3): 20})
        self.assertEqual(op2.objective.higher_order[8].to_dict(), {(0, 0, 1, 1, 2, 2, 3, 3): 10})
        self.assertEqual(op2.objective.constant, 250)


if __name__ == "__main__":
    unittest.main()
