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

"""Test NumberPartition class"""

import numpy as np
from qiskit_addon_opt_mapper import OptimizationProblem
from qiskit_addon_opt_mapper.applications.number_partition import NumberPartition
from qiskit_addon_opt_mapper.problems import Constraint, OptimizationObjective, VarType

from ..optimization_test_case import OptimizationTestCase


class TestNumberPartition(OptimizationTestCase):
    """Test NumberPartition class"""

    def setUp(self):
        """Set up for the test"""
        super().setUp()
        self.num_set = [8, 7, 6, 5, 4]
        op = OptimizationProblem()
        for _ in range(5):
            op.binary_var()
        self.result = np.array([1, 1, 0, 0, 0])

    def test_to_optimization_problem(self):
        """Test to_optimization_problem"""
        number_partition = NumberPartition(self.num_set)
        op = number_partition.to_optimization_problem()
        # Test name
        self.assertEqual(op.name, "Number partitioning")
        # Test variables
        self.assertEqual(op.get_num_vars(), 5)
        for var in op.variables:
            self.assertEqual(var.vartype, VarType.BINARY)
        # Test objective
        obj = op.objective
        self.assertEqual(obj.sense, OptimizationObjective.Sense.MINIMIZE)
        self.assertEqual(obj.constant, 0)
        self.assertDictEqual(obj.linear.to_dict(), {})
        self.assertEqual(obj.quadratic.to_dict(), {})
        # Test constraint
        lin = op.linear_constraints
        self.assertEqual(len(lin), 1)
        self.assertEqual(lin[0].sense, Constraint.Sense.EQ)
        self.assertEqual(lin[0].rhs, -30)
        self.assertEqual(
            lin[0].linear.to_dict(), {i: -2 * num for i, num in enumerate(self.num_set)}
        )

    def test_interpret(self):
        """Test interpret"""
        number_partition = NumberPartition(self.num_set)
        self.assertEqual(number_partition.interpret(self.result), [[6, 5, 4], [8, 7]])
