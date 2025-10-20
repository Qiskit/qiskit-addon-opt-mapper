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

"""Test GraphPartinioning class"""

import networkx as nx
import numpy as np
from qiskit_addon_opt_mapper import OptimizationProblem
from qiskit_addon_opt_mapper.applications.graph_partition import GraphPartition
from qiskit_addon_opt_mapper.problems import Constraint, OptimizationObjective, VarType

from ..optimization_test_case import OptimizationTestCase


class TestGraphPartition(OptimizationTestCase):
    """Test GraphPartitioning class"""

    def setUp(self):
        """Set up for the tests"""
        super().setUp()
        self.graph = nx.gnm_random_graph(4, 4, 123)
        op = OptimizationProblem()
        for _ in range(4):
            op.binary_var()
        self.result = np.array([0, 1, 1, 0])

    def test_to_optimization_problem(self):
        """Test to_optimization_problem"""
        graph_partitioning = GraphPartition(self.graph)
        op = graph_partitioning.to_optimization_problem()
        # Test name
        self.assertEqual(op.name, "Graph partition")
        # Test variables
        self.assertEqual(op.get_num_vars(), 4)
        for var in op.variables:
            self.assertEqual(var.vartype, VarType.BINARY)
        # Test objective
        obj = op.objective
        self.assertEqual(obj.sense, OptimizationObjective.Sense.MINIMIZE)
        self.assertEqual(obj.constant, 0)
        self.assertDictEqual(obj.linear.to_dict(), {0: 3.0, 2: 2.0, 3: 1.0, 1: 2.0})
        self.assertDictEqual(
            obj.quadratic.to_dict(),
            {(0, 1): -2.0, (0, 2): -2.0, (1, 2): -2.0, (0, 3): -2.0},
        )
        # Test constraint
        lin = op.linear_constraints
        self.assertEqual(len(lin), 1)
        self.assertEqual(lin[0].sense, Constraint.Sense.EQ)
        self.assertEqual(lin[0].rhs, 2)
        self.assertEqual(lin[0].linear.to_dict(), {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0})

    def test_interpret(self):
        """Test interpret"""
        graph_partitioning = GraphPartition(self.graph)
        self.assertEqual(graph_partitioning.interpret(self.result), [[0, 3], [1, 2]])

    def test_node_colors(self):
        """Test _node_colors"""
        graph_partitioning = GraphPartition(self.graph)
        self.assertEqual(graph_partitioning._node_colors(self.result), ["b", "r", "r", "b"])
