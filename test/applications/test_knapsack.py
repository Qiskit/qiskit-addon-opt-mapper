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
"""Test Knapsack class"""

import numpy as np
from qiskit_addon_opt_mapper import OptimizationProblem
from qiskit_addon_opt_mapper.applications.knapsack import Knapsack
from qiskit_addon_opt_mapper.problems import Constraint, OptimizationObjective, VarType

from ..optimization_test_case import OptimizationTestCase


class TestKnapsack(OptimizationTestCase):
    """Test Knapsack class"""

    def setUp(self):
        """Set up for the tests"""
        super().setUp()
        self.values = [10, 40, 30, 50]
        self.weights = [5, 4, 6, 3]
        self.max_weight = 10
        op = OptimizationProblem()
        for _ in range(4):
            op.binary_var()
        self.result = np.array([0, 1, 0, 1])

    def test_to_optimization_problem(self):
        """Test to_optimization_problem"""
        knapsack = Knapsack(values=self.values, weights=self.weights, max_weight=self.max_weight)
        op = knapsack.to_optimization_problem()
        # Test name
        self.assertEqual(op.name, "Knapsack")
        # Test variables
        self.assertEqual(op.get_num_vars(), 4)
        for var in op.variables:
            self.assertEqual(var.vartype, VarType.BINARY)
        # Test objective
        obj = op.objective
        self.assertEqual(obj.sense, OptimizationObjective.Sense.MAXIMIZE)
        self.assertEqual(obj.constant, 0)
        self.assertDictEqual(obj.linear.to_dict(), {0: 10, 1: 40, 2: 30, 3: 50})
        self.assertEqual(obj.quadratic.to_dict(), {})
        # Test constraint
        lin = op.linear_constraints
        self.assertEqual(len(lin), 1)
        self.assertEqual(lin[0].sense, Constraint.Sense.LE)
        self.assertEqual(lin[0].rhs, self.max_weight)
        self.assertEqual(lin[0].linear.to_dict(), dict(enumerate(self.weights)))

    def test_interpret(self):
        """Test interpret"""
        knapsack = Knapsack(values=self.values, weights=self.weights, max_weight=self.max_weight)
        self.assertEqual(knapsack.interpret(self.result), [1, 3])

    def test_max_weight(self):
        """Test max_weight"""
        knapsack = Knapsack(values=self.values, weights=self.weights, max_weight=self.max_weight)
        knapsack.max_weight = 5
        self.assertEqual(knapsack.max_weight, 5)
