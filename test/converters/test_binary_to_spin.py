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

"""Test BinaryToSpin Converters"""

import unittest

from qiskit_addon_opt_mapper import OptimizationProblem
from qiskit_addon_opt_mapper.converters import (
    BinaryToSpin,
)
from qiskit_addon_opt_mapper.problems import Constraint, Variable

from ..optimization_test_case import OptimizationTestCase


class TestBinaryToSpinConverter(OptimizationTestCase):
    """Test BinaryToSpin Converters"""

    def test_binary_to_spin_obj(self):
        """Test binary to spin"""
        op = OptimizationProblem()
        op.binary_var_list(3, name="x")
        op.minimize(
            linear={"x0": 1},
            quadratic={("x0", "x1"): 1},
            higher_order={3: {("x0", "x1", "x2"): 1}},
        )
        conv = BinaryToSpin()
        op2 = conv.convert(op)
        self.assertEqual(op2.get_num_vars(), 3)
        self.assertListEqual([x.vartype for x in op2.variables], [Variable.Type.SPIN] * 3)
        self.assertListEqual([x.name for x in op2.variables], ["x0@spin", "x1@spin", "x2@spin"])
        self.assertAlmostEqual(op2.objective.constant, 0.875)
        self.assertDictEqual(
            op2.objective.linear.to_dict(use_name=True),
            {"x0@spin": -0.875, "x1@spin": -0.375, "x2@spin": -0.125},
        )
        self.assertDictEqual(
            op2.objective.quadratic.to_dict(use_name=True),
            {
                ("x0@spin", "x1@spin"): 0.375,
                ("x0@spin", "x2@spin"): 0.125,
                ("x1@spin", "x2@spin"): 0.125,
            },
        )
        self.assertDictEqual(
            op2.objective.higher_order[3].to_dict(use_name=True),
            {("x0@spin", "x1@spin", "x2@spin"): -0.125},
        )

    def test_binary_to_spin_constr(self):
        """Test binary to spin constraints"""
        op = OptimizationProblem()
        op.binary_var_list(3, name="x")
        op.linear_constraint(linear={"x0": 1, "x1": 1}, sense="==", rhs=1, name="c0")
        op.quadratic_constraint(
            linear={"x0": 1, "x1": 1},
            quadratic={("x0", "x1"): 1},
            sense="<=",
            rhs=1,
            name="c1",
        )
        op.higher_order_constraint(
            linear={"x0": 1, "x1": 1},
            quadratic={("x0", "x1"): 1},
            higher_order={3: {("x0", "x1", "x2"): 1}},
            sense=">=",
            rhs=1,
            name="c2",
        )
        conv = BinaryToSpin()
        op2 = conv.convert(op)
        self.assertEqual(op2.get_num_vars(), 3)
        self.assertListEqual([x.vartype for x in op2.variables], [Variable.Type.SPIN] * 3)
        self.assertListEqual([x.name for x in op2.variables], ["x0@spin", "x1@spin", "x2@spin"])
        c0 = op2.linear_constraints[0]
        self.assertEqual(c0.name, "c0")
        self.assertEqual(c0.sense, Constraint.Sense.EQ)
        self.assertAlmostEqual(c0.rhs, 0)
        self.assertDictEqual(c0.linear.to_dict(use_name=True), {"x0@spin": -0.5, "x1@spin": -0.5})
        c1 = op2.quadratic_constraints[0]
        self.assertEqual(c1.name, "c1")
        self.assertEqual(c1.sense, Constraint.Sense.LE)
        self.assertAlmostEqual(c1.rhs, -0.25)
        self.assertDictEqual(c1.linear.to_dict(use_name=True), {"x0@spin": -0.75, "x1@spin": -0.75})
        self.assertDictEqual(c1.quadratic.to_dict(use_name=True), {("x0@spin", "x1@spin"): 0.25})
        c2 = op2.higher_order_constraints[0]
        self.assertEqual(c2.name, "c2")
        self.assertEqual(c2.sense, Constraint.Sense.GE)
        self.assertAlmostEqual(c2.rhs, -0.375)
        self.assertDictEqual(
            c2.linear.to_dict(use_name=True),
            {"x0@spin": -0.875, "x1@spin": -0.875, "x2@spin": -0.125},
        )
        self.assertDictEqual(
            c2.quadratic.to_dict(use_name=True),
            {
                ("x0@spin", "x1@spin"): 0.375,
                ("x0@spin", "x2@spin"): 0.125,
                ("x1@spin", "x2@spin"): 0.125,
            },
        )
        self.assertDictEqual(
            c2.higher_order[3].to_dict(use_name=True),
            {("x0@spin", "x1@spin", "x2@spin"): -0.125},
        )


if __name__ == "__main__":
    unittest.main()
