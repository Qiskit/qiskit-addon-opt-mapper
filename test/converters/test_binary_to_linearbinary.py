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

"""Test BinaryToLinearBinary Converters"""

import unittest

from qiskit_addon_opt_mapper import OptimizationProblem
from qiskit_addon_opt_mapper.converters import (
    BinaryToLinearBinary,
)
from qiskit_addon_opt_mapper.problems import Variable
from qiskit_addon_opt_mapper.problems.constraint import ConstraintSense

from ..optimization_test_case import OptimizationTestCase


class TestBinaryToLinearBinaryConverter(OptimizationTestCase):
    """Test BinaryToLinearBinary Converters"""

    def check_constraint(self, op2, constr_id: int, lhs: dict, rhs: float, cs: ConstraintSense):
        self.assertDictEqual(
            op2.get_linear_constraint(constr_id).linear.to_dict(use_name=True),
            lhs,
        )
        self.assertEqual(op2.get_linear_constraint(constr_id).rhs, rhs)
        self.assertEqual(op2.get_linear_constraint(constr_id).sense, cs)

    def test_binary_to_linear_binary_obj(self):
        """Test binary to linear_binary"""
        op = OptimizationProblem()
        op.binary_var_list(3, name="x")
        op.minimize(
            constant=12,
            linear={"x1": 2},
            quadratic={("x0", "x2"): 3},
            higher_order={3: {("x0", "x1", "x2"): 4}},
        )
        conv = BinaryToLinearBinary()
        op2 = conv.convert(op)

        self.assertEqual(op2.get_num_vars(), 5)

        self.assertListEqual([x.vartype for x in op2.variables], [Variable.Type.BINARY] * 5)
        self.assertListEqual(
            [x.name for x in op2.variables], ["x0", "x1", "x2", "x0ANDx2", "x0ANDx1ANDx2"]
        )
        self.assertAlmostEqual(op2.objective.constant, 12)
        self.assertDictEqual(
            op2.objective.linear.to_dict(use_name=True),
            {"x1": 2, "x0ANDx2": 3, "x0ANDx1ANDx2": 4},
        )
        self.assertDictEqual(
            op2.objective.quadratic.to_dict(use_name=True),
            {},
        )
        self.assertDictEqual(op2.objective.higher_order, {})

        constraints = [
            ({"x0ANDx2": 1, "x0": -1}, 0, ConstraintSense.LE),
            ({"x0ANDx2": 1, "x2": -1}, 0, ConstraintSense.LE),
            ({"x0ANDx2": 1, "x0": -1, "x2": -1}, -1, ConstraintSense.GE),
            ({"x0ANDx1ANDx2": 1, "x0": -1}, 0, ConstraintSense.LE),
            ({"x0ANDx1ANDx2": 1, "x1": -1}, 0, ConstraintSense.LE),
            ({"x0ANDx1ANDx2": 1, "x2": -1}, 0, ConstraintSense.LE),
            ({"x0ANDx1ANDx2": 1, "x0": -1, "x1": -1, "x2": -1}, -2, ConstraintSense.GE),
        ]

        self.assertEqual(op2.get_num_linear_constraints(), len(constraints))
        self.assertEqual(op2.get_num_quadratic_constraints(), 0)
        self.assertEqual(op2.get_num_higher_order_constraints(), 0)

        for idx, constr in enumerate(constraints):
            self.check_constraint(op2, idx, *constr)

    def test_binary_to_linear_binary_constr(self):
        """Test binary to linear_binary"""
        op = OptimizationProblem()
        op.binary_var_list(3, name="x")
        op.minimize(
            constant=12,
            linear={"x1": 2},
            quadratic={("x0", "x2"): 3},
            higher_order={3: {("x0", "x1", "x2"): 4}},
        )
        op.quadratic_constraint(
            linear={"x0": 7},
            quadratic={("x0", 1): 3},
        )
        op.higher_order_constraint(
            higher_order={3: {("x0", 1, 2): 4}},
        )
        conv = BinaryToLinearBinary()
        op2 = conv.convert(op)

        self.assertEqual(op2.get_num_vars(), 6)
        self.assertListEqual([x.vartype for x in op2.variables], [Variable.Type.BINARY] * 6)
        self.assertListEqual(
            [x.name for x in op2.variables],
            ["x0", "x1", "x2", "x0ANDx2", "x0ANDx1ANDx2", "x0ANDx1"],
        )
        self.assertAlmostEqual(op2.objective.constant, 12)
        self.assertDictEqual(
            op2.objective.linear.to_dict(use_name=True),
            {"x1": 2, "x0ANDx2": 3, "x0ANDx1ANDx2": 4},
        )
        self.assertDictEqual(
            op2.objective.quadratic.to_dict(use_name=True),
            {},
        )
        self.assertDictEqual(op2.objective.higher_order, {})

        constraints = [
            ({"x0": 7, "x0ANDx1": 3}, 0, ConstraintSense.LE),
            ({"x0ANDx1ANDx2": 4}, 0, ConstraintSense.LE),
            ({"x0ANDx2": 1, "x0": -1}, 0, ConstraintSense.LE),
            ({"x0ANDx2": 1, "x2": -1}, 0, ConstraintSense.LE),
            ({"x0ANDx2": 1, "x0": -1, "x2": -1}, -1, ConstraintSense.GE),
            ({"x0ANDx1ANDx2": 1, "x0": -1}, 0, ConstraintSense.LE),
            ({"x0ANDx1ANDx2": 1, "x1": -1}, 0, ConstraintSense.LE),
            ({"x0ANDx1ANDx2": 1, "x2": -1}, 0, ConstraintSense.LE),
            ({"x0ANDx1ANDx2": 1, "x0": -1, "x1": -1, "x2": -1}, -2, ConstraintSense.GE),
            ({"x0ANDx1": 1, "x0": -1}, 0, ConstraintSense.LE),
            ({"x0ANDx1": 1, "x1": -1}, 0, ConstraintSense.LE),
            ({"x0ANDx1": 1, "x0": -1, "x1": -1}, -1, ConstraintSense.GE),
        ]

        self.assertEqual(op2.get_num_linear_constraints(), len(constraints))
        self.assertEqual(op2.get_num_quadratic_constraints(), 0)
        self.assertEqual(op2.get_num_higher_order_constraints(), 0)

        for idx, constr in enumerate(constraints):
            self.check_constraint(op2, idx, *constr)


if __name__ == "__main__":
    unittest.main()
