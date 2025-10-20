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

"""Test InequalityToEquality Converters"""

import unittest

import numpy as np
from qiskit_addon_opt_mapper import OptimizationProblem
from qiskit_addon_opt_mapper.converters import (
    InequalityToEquality,
)
from qiskit_addon_opt_mapper.problems import Constraint, Variable

from ..optimization_test_case import OptimizationTestCase


class TestInequalityToEqualityConverter(OptimizationTestCase):
    """Test InequalityToEquality Converters"""

    def test_inequality_binary(self):
        """Test InequalityToEqualityConverter with binary variables"""
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
        # Quadratic constraints
        quadratic = {("x0", "x1"): 1, ("x1", "x2"): 2}
        op.quadratic_constraint({}, quadratic, Constraint.Sense.LE, 3, "x0x1_x1x2LE")
        quadratic = {("x0", "x1"): 3, ("x1", "x2"): 4}
        op.quadratic_constraint({}, quadratic, Constraint.Sense.GE, 3, "x0x1_x1x2GE")
        # Convert inequality constraints into equality constraints
        conv = InequalityToEquality()
        op2 = conv.convert(op)
        self.assertListEqual(
            [v.name for v in op2.variables],
            [
                "x0",
                "x1",
                "x2",
                "x1x2@int_slack",
                "x0x2@int_slack",
                "x0x1_x1x2LE@int_slack",
                "x0x1_x1x2GE@int_slack",
            ],
        )
        # Check names and objective senses
        self.assertEqual(op.name, op2.name)
        self.assertEqual(op.objective.sense, op2.objective.sense)
        # For linear constraints
        lst = [
            op2.linear_constraints[0].linear.to_dict()[0],
            op2.linear_constraints[0].linear.to_dict()[1],
        ]
        self.assertListEqual(lst, [1, 1])
        self.assertEqual(op2.linear_constraints[0].sense, Constraint.Sense.EQ)
        lst = [
            op2.linear_constraints[1].linear.to_dict()[1],
            op2.linear_constraints[1].linear.to_dict()[2],
            op2.linear_constraints[1].linear.to_dict()[3],
        ]
        self.assertListEqual(lst, [1, -1, 1])
        lst = [op2.variables[3].lowerbound, op2.variables[3].upperbound]
        self.assertListEqual(lst, [0, 3])
        self.assertEqual(op2.linear_constraints[1].sense, Constraint.Sense.EQ)
        lst = [
            op2.linear_constraints[2].linear.to_dict()[0],
            op2.linear_constraints[2].linear.to_dict()[2],
            op2.linear_constraints[2].linear.to_dict()[4],
        ]
        self.assertListEqual(lst, [1, 3, -1])
        lst = [op2.variables[4].lowerbound, op2.variables[4].upperbound]
        self.assertListEqual(lst, [0, 2])
        self.assertEqual(op2.linear_constraints[2].sense, Constraint.Sense.EQ)
        # For quadratic constraints
        lst = [
            op2.quadratic_constraints[0].quadratic.to_dict()[(0, 1)],
            op2.quadratic_constraints[0].quadratic.to_dict()[(1, 2)],
            op2.quadratic_constraints[0].linear.to_dict()[5],
        ]
        self.assertListEqual(lst, [1, 2, 1])
        lst = [op2.variables[5].lowerbound, op2.variables[5].upperbound]
        self.assertListEqual(lst, [0, 3])
        lst = [
            op2.quadratic_constraints[1].quadratic.to_dict()[(0, 1)],
            op2.quadratic_constraints[1].quadratic.to_dict()[(1, 2)],
            op2.quadratic_constraints[1].linear.to_dict()[6],
        ]
        self.assertListEqual(lst, [3, 4, -1])
        lst = [op2.variables[6].lowerbound, op2.variables[6].upperbound]
        self.assertListEqual(lst, [0, 4])

        new_x = conv.interpret(np.arange(7))
        np.testing.assert_array_almost_equal(new_x, np.arange(3))

    def test_inequality_integer(self):
        """Test InequalityToEqualityConverter with integer variables"""
        op = OptimizationProblem()
        for i in range(3):
            op.integer_var(name=f"x{i}", lowerbound=-3, upperbound=3)
        # Linear constraints
        linear_constraint = {"x0": 1, "x1": 1}
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, "x0x1")
        linear_constraint = {"x1": 1, "x2": -1}
        op.linear_constraint(linear_constraint, Constraint.Sense.LE, 2, "x1x2")
        linear_constraint = {"x0": 1, "x2": 3}
        op.linear_constraint(linear_constraint, Constraint.Sense.GE, 2, "x0x2")
        # Quadratic constraints
        quadratic = {("x0", "x1"): 1, ("x1", "x2"): 2}
        op.quadratic_constraint({}, quadratic, Constraint.Sense.LE, 3, "x0x1_x1x2LE")
        quadratic = {("x0", "x1"): 3, ("x1", "x2"): 4}
        op.quadratic_constraint({}, quadratic, Constraint.Sense.GE, 3, "x0x1_x1x2GE")
        conv = InequalityToEquality()
        op2 = conv.convert(op)
        self.assertListEqual(
            [v.name for v in op2.variables],
            [
                "x0",
                "x1",
                "x2",
                "x1x2@int_slack",
                "x0x2@int_slack",
                "x0x1_x1x2LE@int_slack",
                "x0x1_x1x2GE@int_slack",
            ],
        )
        # For linear constraints
        lst = [
            op2.linear_constraints[0].linear.to_dict()[0],
            op2.linear_constraints[0].linear.to_dict()[1],
        ]
        self.assertListEqual(lst, [1, 1])
        self.assertEqual(op2.linear_constraints[0].sense, Constraint.Sense.EQ)
        lst = [
            op2.linear_constraints[1].linear.to_dict()[1],
            op2.linear_constraints[1].linear.to_dict()[2],
            op2.linear_constraints[1].linear.to_dict()[3],
        ]
        self.assertListEqual(lst, [1, -1, 1])
        lst = [op2.variables[3].lowerbound, op2.variables[3].upperbound]
        self.assertListEqual(lst, [0, 8])
        self.assertEqual(op2.linear_constraints[1].sense, Constraint.Sense.EQ)
        lst = [
            op2.linear_constraints[2].linear.to_dict()[0],
            op2.linear_constraints[2].linear.to_dict()[2],
            op2.linear_constraints[2].linear.to_dict()[4],
        ]
        self.assertListEqual(lst, [1, 3, -1])
        lst = [op2.variables[4].lowerbound, op2.variables[4].upperbound]
        self.assertListEqual(lst, [0, 10])
        self.assertEqual(op2.linear_constraints[2].sense, Constraint.Sense.EQ)
        # For quadratic constraints
        lst = [
            op2.quadratic_constraints[0].quadratic.to_dict()[(0, 1)],
            op2.quadratic_constraints[0].quadratic.to_dict()[(1, 2)],
            op2.quadratic_constraints[0].linear.to_dict()[5],
        ]
        self.assertListEqual(lst, [1, 2, 1])
        lst = [op2.variables[5].lowerbound, op2.variables[5].upperbound]
        self.assertListEqual(lst, [0, 30])
        lst = [
            op2.quadratic_constraints[1].quadratic.to_dict()[(0, 1)],
            op2.quadratic_constraints[1].quadratic.to_dict()[(1, 2)],
            op2.quadratic_constraints[1].linear.to_dict()[6],
        ]
        self.assertListEqual(lst, [3, 4, -1])
        lst = [op2.variables[6].lowerbound, op2.variables[6].upperbound]
        self.assertListEqual(lst, [0, 60])

        new_x = conv.interpret(np.arange(7))
        np.testing.assert_array_almost_equal(new_x, np.arange(3))

    def test_0var_range_inequality(self):
        """Test InequalityToEquality converter when the var_rang of the slack variable is 0"""
        op = OptimizationProblem()
        op.binary_var("x")
        op.binary_var("y")
        op.linear_constraint(linear={"x": 1, "y": 1}, sense="LE", rhs=0, name="xy_leq1")
        op.linear_constraint(linear={"x": 1, "y": 1}, sense="GE", rhs=2, name="xy_geq1")
        op.quadratic_constraint(quadratic={("x", "x"): 1}, sense="LE", rhs=0, name="xy_leq2")
        op.quadratic_constraint(quadratic={("x", "y"): 1}, sense="GE", rhs=1, name="xy_geq2")
        ineq2eq = InequalityToEquality()
        new_op = ineq2eq.convert(op)
        self.assertEqual(new_op.get_num_vars(), 2)
        self.assertTrue(
            all(l_const.sense == Constraint.Sense.EQ for l_const in new_op.linear_constraints)
        )
        self.assertTrue(
            all(q_const.sense == Constraint.Sense.EQ for q_const in new_op.quadratic_constraints)
        )

    def test_inequality_mode_integer(self):
        """Test integer mode of InequalityToEqualityConverter()"""
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
        conv = InequalityToEquality(mode="integer")
        op2 = conv.convert(op)
        lst = [op2.variables[3].vartype, op2.variables[4].vartype]
        self.assertListEqual(lst, [Variable.Type.INTEGER, Variable.Type.INTEGER])

    def test_inequality_mode_continuous(self):
        """Test continuous mode of InequalityToEqualityConverter()"""
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
        conv = InequalityToEquality(mode="continuous")
        op2 = conv.convert(op)
        lst = [op2.variables[3].vartype, op2.variables[4].vartype]
        self.assertListEqual(lst, [Variable.Type.CONTINUOUS, Variable.Type.CONTINUOUS])

    def test_inequality_mode_auto(self):
        """Test auto mode of InequalityToEqualityConverter()"""
        op = OptimizationProblem()
        for i in range(3):
            op.binary_var(name=f"x{i}")
        # Linear constraints
        linear_constraint = {"x0": 1, "x1": 1}
        op.linear_constraint(linear_constraint, Constraint.Sense.EQ, 1, "x0x1")
        linear_constraint = {"x1": 1, "x2": -1}
        op.linear_constraint(linear_constraint, Constraint.Sense.LE, 2, "x1x2")
        linear_constraint = {"x0": 1.1, "x2": 2.2}
        op.linear_constraint(linear_constraint, Constraint.Sense.GE, 3.3, "x0x2")
        conv = InequalityToEquality(mode="auto")
        op2 = conv.convert(op)
        lst = [op2.variables[3].vartype, op2.variables[4].vartype]
        self.assertListEqual(lst, [Variable.Type.INTEGER, Variable.Type.CONTINUOUS])

    def test_inequality_le_ge(self):
        """Test InequalityToEquality for both senses"""
        op = OptimizationProblem()
        op.binary_var(name="x")
        op.minimize(linear={"x": 1})
        op.linear_constraint({"x": 1}, "<=", 1)
        op.linear_constraint({"x": 1}, ">=", 0)
        op_eq = InequalityToEquality().convert(op)
        self.assertEqual(op_eq.get_num_linear_constraints(), 2)
        lin0 = op_eq.get_linear_constraint(0)
        self.assertEqual(lin0.linear.to_dict(use_name=True), {"x": 1.0, "c0@int_slack": 1.0})
        self.assertEqual(lin0.sense, Constraint.Sense.EQ)
        self.assertEqual(lin0.rhs, 1)
        self.assertAlmostEqual(lin0.evaluate([1, 1, 1]), 2)
        lin1 = op_eq.get_linear_constraint(1)
        self.assertEqual(lin1.linear.to_dict(use_name=True), {"x": 1.0, "c1@int_slack": -1.0})
        self.assertEqual(lin1.sense, Constraint.Sense.EQ)
        self.assertEqual(lin1.rhs, 0)
        self.assertAlmostEqual(lin1.evaluate([1, 1, 1]), 0)

    def test_higher_order_inequality(self):
        """Test InequalityToEqualityConverter with higher order constraints"""
        op = OptimizationProblem()
        for i in range(4):
            op.binary_var(name=f"x{i}")
        op.higher_order_constraint(
            higher_order={3: {(0, 1, 2): 10}, 4: {(0, 1, 2, 3): 1}},
            sense="<=",
            rhs=5,
            name="c0",
        )
        op.higher_order_constraint(
            higher_order={3: {(0, 1, 2): 10}, 4: {(0, 1, 2, 3): 1}},
            sense=">=",
            rhs=2,
            name="c1",
        )
        conv = InequalityToEquality()
        eq_op = conv.convert(op)
        self.assertEqual(eq_op.get_variable("c0@int_slack").lowerbound, 0)
        self.assertEqual(eq_op.get_variable("c0@int_slack").upperbound, 5)
        self.assertEqual(eq_op.get_variable("c1@int_slack").lowerbound, 0)
        self.assertEqual(eq_op.get_variable("c1@int_slack").upperbound, 9)

        self.assertEqual(eq_op.get_num_higher_order_constraints(), 2)
        c0 = eq_op.get_higher_order_constraint(0)
        self.assertEqual(c0.sense, Constraint.Sense.EQ)
        self.assertEqual(c0.rhs, 5)
        self.assertDictEqual(
            c0.higher_order[3].to_dict(),
            {
                (0, 1, 2): 10,
            },
        )
        self.assertDictEqual(
            c0.higher_order[4].to_dict(),
            {
                (0, 1, 2, 3): 1,
            },
        )
        self.assertDictEqual(c0.linear.to_dict(use_name=True), {"c0@int_slack": 1})

        c1 = eq_op.get_higher_order_constraint(1)
        self.assertEqual(c1.sense, Constraint.Sense.EQ)
        self.assertEqual(c1.rhs, 2)
        self.assertDictEqual(
            c1.higher_order[3].to_dict(),
            {
                (0, 1, 2): 10,
            },
        )
        self.assertDictEqual(
            c1.higher_order[4].to_dict(),
            {
                (0, 1, 2, 3): 1,
            },
        )
        self.assertDictEqual(c1.linear.to_dict(use_name=True), {"c1@int_slack": -1})


if __name__ == "__main__":
    unittest.main()
