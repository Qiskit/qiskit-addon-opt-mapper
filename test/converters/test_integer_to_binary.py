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

"""Test IntegerToBinary Converters"""

import unittest

import numpy as np
from qiskit_addon_opt_mapper import OptimizationProblem
from qiskit_addon_opt_mapper.converters import (
    IntegerToBinary,
)
from qiskit_addon_opt_mapper.problems import Constraint, Variable

from ..optimization_test_case import OptimizationTestCase


class TestIntegerToBinaryConverter(OptimizationTestCase):
    """Test IntegerToBinary Converters"""

    def test_integer_to_binary(self):
        """Test integer to binary"""
        op = OptimizationProblem()
        for i in range(0, 2):
            op.binary_var(name=f"x{i}")
        op.integer_var(name="x2", lowerbound=0, upperbound=5)
        linear = {}
        for i, x in enumerate(op.variables):
            linear[x.name] = i + 1
        op.maximize(0, linear, {})
        conv = IntegerToBinary()
        op2 = conv.convert(op)
        self.assertEqual(op2.get_num_vars(), 5)
        self.assertListEqual([x.vartype for x in op2.variables], [Variable.Type.BINARY] * 5)
        self.assertListEqual([x.name for x in op2.variables], ["x0", "x1", "x2@0", "x2@1", "x2@2"])
        dct = op2.objective.linear.to_dict()
        self.assertEqual(dct[2], 3)
        self.assertEqual(dct[3], 6)
        self.assertEqual(dct[4], 6)

    def test_integer_to_binary2(self):
        """Test integer to binary variables 2"""
        mod = OptimizationProblem()
        mod.integer_var(name="x", lowerbound=0, upperbound=1)
        mod.integer_var(name="y", lowerbound=0, upperbound=1)
        mod.minimize(1, {"x": 1}, {("x", "y"): 2})
        mod.linear_constraint({"x": 1}, "==", 1)
        mod.quadratic_constraint({"x": 1}, {("x", "y"): 2}, "==", 1)
        mod2 = IntegerToBinary().convert(mod)
        self.assertListEqual(
            [e.name + "@0" for e in mod.variables], [e.name for e in mod2.variables]
        )
        self.assertDictEqual(mod.objective.linear.to_dict(), mod2.objective.linear.to_dict())
        self.assertDictEqual(mod.objective.quadratic.to_dict(), mod2.objective.quadratic.to_dict())
        self.assertEqual(mod.get_num_linear_constraints(), mod2.get_num_linear_constraints())
        for cst, cst2 in zip(mod.linear_constraints, mod2.linear_constraints, strict=False):
            self.assertDictEqual(cst.linear.to_dict(), cst2.linear.to_dict())
        self.assertEqual(mod.get_num_quadratic_constraints(), mod2.get_num_quadratic_constraints())
        for cst, cst2 in zip(mod.quadratic_constraints, mod2.quadratic_constraints, strict=False):
            self.assertDictEqual(cst.linear.to_dict(), cst2.linear.to_dict())
            self.assertDictEqual(cst.quadratic.to_dict(), cst2.quadratic.to_dict())

    def test_integer_to_binary_quadratic(self):
        """Test integer to binary variables with quadratic expressions"""
        mod = OptimizationProblem()
        mod.integer_var(name="x", lowerbound=10, upperbound=13)
        mod.minimize(quadratic={("x", "x"): 1})
        mod2 = IntegerToBinary().convert(mod)
        self.assertListEqual([e.name for e in mod2.variables], ["x@0", "x@1"])
        self.assertEqual(mod.get_num_linear_constraints(), 0)
        self.assertEqual(mod.get_num_quadratic_constraints(), 0)
        self.assertAlmostEqual(mod2.objective.constant, 100)
        self.assertDictEqual(mod2.objective.linear.to_dict(use_name=True), {"x@0": 20, "x@1": 40})
        self.assertDictEqual(
            mod2.objective.quadratic.to_dict(use_name=True),
            {("x@0", "x@0"): 1, ("x@1", "x@1"): 4, ("x@0", "x@1"): 4},
        )

    def test_integer_to_binary_zero_range_variable(self):
        """Test integer to binary variables with zero range variables"""

        with self.subTest("zero range variable in a linear expression of the objective"):
            mod = OptimizationProblem()
            mod.integer_var(name="x", lowerbound=10, upperbound=10)
            mod.minimize(linear={"x": 1})
            mod2 = IntegerToBinary().convert(mod)
            self.assertListEqual([e.name for e in mod2.variables], ["x@0"])
            self.assertEqual(mod.get_num_linear_constraints(), 0)
            self.assertEqual(mod.get_num_quadratic_constraints(), 0)
            self.assertAlmostEqual(mod2.objective.constant, 10)
            self.assertDictEqual(mod2.objective.linear.to_dict(), {})
            self.assertDictEqual(mod2.objective.quadratic.to_dict(), {})

        with self.subTest("zero range variable in a quadratic expression of the objective"):
            mod = OptimizationProblem()
            mod.integer_var(name="x", lowerbound=10, upperbound=10)
            mod.minimize(quadratic={("x", "x"): 1})
            mod2 = IntegerToBinary().convert(mod)
            self.assertListEqual([e.name for e in mod2.variables], ["x@0"])
            self.assertEqual(mod.get_num_linear_constraints(), 0)
            self.assertEqual(mod.get_num_quadratic_constraints(), 0)
            self.assertAlmostEqual(mod2.objective.constant, 100)
            self.assertDictEqual(mod2.objective.linear.to_dict(), {})
            self.assertDictEqual(mod2.objective.quadratic.to_dict(), {})

        with self.subTest("zero range variable in a linear constraint"):
            mod = OptimizationProblem()
            mod.integer_var(name="x", lowerbound=10, upperbound=10)
            mod.binary_var(name="y")
            mod.linear_constraint({"x": 1, "y": 1}, "<=", 100)
            mod2 = IntegerToBinary().convert(mod)
            self.assertListEqual([e.name for e in mod2.variables], ["x@0", "y"])
            self.assertEqual(mod.get_num_linear_constraints(), 1)
            self.assertEqual(mod.get_num_quadratic_constraints(), 0)
            self.assertAlmostEqual(mod2.objective.constant, 0)
            self.assertDictEqual(mod2.objective.linear.to_dict(), {})
            self.assertDictEqual(mod2.objective.quadratic.to_dict(), {})
            cst = mod2.get_linear_constraint(0)
            self.assertDictEqual(cst.linear.to_dict(use_name=True), {"y": 1})
            self.assertEqual(cst.sense, Constraint.Sense.LE)
            self.assertAlmostEqual(cst.rhs, 90)
            self.assertEqual(cst.name, "c0")

        with self.subTest("zero range variable in a quadratic constraint"):
            mod = OptimizationProblem()
            mod.integer_var(name="x", lowerbound=10, upperbound=10)
            mod.binary_var(name="y")
            mod.quadratic_constraint({"x": 1}, {("x", "x"): 2, ("x", "y"): 3}, ">=", 100)
            mod2 = IntegerToBinary().convert(mod)
            self.assertListEqual([e.name for e in mod2.variables], ["x@0", "y"])
            self.assertEqual(mod.get_num_linear_constraints(), 0)
            self.assertEqual(mod.get_num_quadratic_constraints(), 1)
            self.assertAlmostEqual(mod2.objective.constant, 0)
            self.assertDictEqual(mod2.objective.linear.to_dict(), {})
            self.assertDictEqual(mod2.objective.quadratic.to_dict(), {})
            cst = mod2.get_quadratic_constraint(0)
            self.assertDictEqual(cst.linear.to_dict(use_name=True), {"y": 30})
            self.assertEqual(cst.sense, Constraint.Sense.GE)
            self.assertAlmostEqual(cst.rhs, -110)
            self.assertEqual(cst.name, "q0")

    def test_binary_to_integer(self):
        """Test binary to integer"""
        op = OptimizationProblem()
        for i in range(0, 2):
            op.binary_var(name=f"x{i}")
        op.integer_var(name="x2", lowerbound=0, upperbound=5)
        linear = {"x0": 1, "x1": 2, "x2": 1}
        op.maximize(0, linear, {})
        linear = {}
        for x in op.variables:
            linear[x.name] = 1
        op.linear_constraint(linear, Constraint.Sense.EQ, 6, "x0x1x2")
        conv = IntegerToBinary()
        _ = conv.convert(op)
        new_x = conv.interpret([0, 1, 1, 1, 1])
        np.testing.assert_array_almost_equal(new_x, [0, 1, 5])

    def test_higher_order_integer_to_binary(self):
        op = OptimizationProblem("test")
        op.integer_var(name="x0", lowerbound=10, upperbound=17)
        op.binary_var(name="x1")
        op.binary_var(name="x2")
        op.binary_var(name="x3")
        op.higher_order_constraint(
            higher_order={3: {(0, 1, 2): 2}, 4: {(0, 1, 2, 3): 3}},
            sense="==",
            rhs=5,
            name="c0",
        )
        converter = IntegerToBinary()
        op2 = converter.convert(op)
        self.assertEqual(op2.get_num_vars(), 6)
        self.assertListEqual([x.vartype for x in op2.variables], [Variable.Type.BINARY] * 6)
        self.assertListEqual(
            [x.name for x in op2.variables],
            ["x0@0", "x0@1", "x0@2", "x1", "x2", "x3"],
        )
        expected_cubic = {
            (0, 3, 4): 2.0,
            (1, 3, 4): 4.0,
            (2, 3, 4): 8.0,
            (3, 4, 5): 30.0,
        }
        self.assertDictEqual(
            op2.higher_order_constraints[0].higher_order[3].to_dict(), expected_cubic
        )
        expected_quartic = {
            (0, 3, 4, 5): 3.0,
            (1, 3, 4, 5): 6.0,
            (2, 3, 4, 5): 12.0,
        }
        self.assertDictEqual(
            op2.higher_order_constraints[0].higher_order[4].to_dict(), expected_quartic
        )
        self.assertEqual(op2.higher_order_constraints[0].sense, Constraint.Sense.EQ)
        self.assertAlmostEqual(op2.higher_order_constraints[0].rhs, 5)


if __name__ == "__main__":
    unittest.main()
