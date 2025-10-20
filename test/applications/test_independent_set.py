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
"""Test IndependentSet class"""

import networkx as nx
import numpy as np
from qiskit_addon_opt_mapper import OptimizationProblem
from qiskit_addon_opt_mapper.applications.independent_set import IndependentSet
from qiskit_addon_opt_mapper.problems import Constraint, OptimizationObjective, VarType

from ..optimization_test_case import OptimizationTestCase


class TestIndependentSet(OptimizationTestCase):
    """Test IndependentSet class"""

    def setUp(self):
        super().setUp()
        self.graph = nx.gnm_random_graph(5, 4, 3)
        op = OptimizationProblem()
        for _ in range(5):
            op.binary_var()
        self.result = np.array([1, 1, 1, 1, 0])

    def test_to_optimization_problem(self):
        """Test to_optimization_problem"""
        independent_set = IndependentSet(self.graph)
        op = independent_set.to_optimization_problem()
        # Test name
        self.assertEqual(op.name, "Independent set")
        # Test variables
        self.assertEqual(op.get_num_vars(), 5)
        for var in op.variables:
            self.assertEqual(var.vartype, VarType.BINARY)
        # Test objective
        obj = op.objective
        self.assertEqual(obj.sense, OptimizationObjective.Sense.MAXIMIZE)
        self.assertEqual(obj.constant, 0)
        self.assertDictEqual(obj.linear.to_dict(), {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
        # Test constraint
        lin = op.linear_constraints
        self.assertEqual(len(lin), len(self.graph.edges))
        for i, edge in enumerate(self.graph.edges):
            self.assertEqual(lin[i].sense, Constraint.Sense.LE)
            self.assertEqual(lin[i].rhs, 1)
            self.assertEqual(lin[i].linear.to_dict(), {edge[0]: 1, edge[1]: 1})

    def test_interpret(self):
        """Test interpret"""
        independent_set = IndependentSet(self.graph)
        self.assertEqual(independent_set.interpret(self.result), [0, 1, 2, 3])

    def test_node_colors(self):
        """Test node_colors"""
        independent_set = IndependentSet(self.graph)
        self.assertEqual(
            independent_set._node_colors(self.result), ["r", "r", "r", "r", "darkgrey"]
        )
