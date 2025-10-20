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

import unittest

import networkx as nx
import numpy as np
import qiskit_addon_opt_mapper.optionals as _optionals
from qiskit_addon_opt_mapper import OptimizationProblem
from qiskit_addon_opt_mapper.applications.max_cut import Maxcut
from qiskit_addon_opt_mapper.problems import OptimizationObjective, VarType

from ..optimization_test_case import OptimizationTestCase


class TestMaxcut(OptimizationTestCase):
    """Test Maxcut class"""

    def setUp(self):
        super().setUp()
        self.graph = nx.gnm_random_graph(4, 6, 123)
        op = OptimizationProblem()
        for _ in range(4):
            op.binary_var()
        self.result = np.array([1, 1, 0, 0])

    def test_to_optimization_problem(self):
        """Test to_optimization_problem"""
        maxcut = Maxcut(self.graph)
        op = maxcut.to_optimization_problem()
        # Test name
        self.assertEqual(op.name, "Max-cut")
        # Test variables
        self.assertEqual(op.get_num_vars(), 4)
        for var in op.variables:
            self.assertEqual(var.vartype, VarType.BINARY)
        # Test objective
        obj = op.objective
        self.assertEqual(obj.sense, OptimizationObjective.Sense.MAXIMIZE)
        self.assertEqual(obj.constant, 0)
        self.assertDictEqual(obj.linear.to_dict(), {0: 3.0, 1: 3.0, 2: 3.0, 3: 3.0})
        self.assertDictEqual(
            obj.quadratic.to_dict(),
            {
                (0, 1): -2.0,
                (0, 2): -2.0,
                (1, 2): -2.0,
                (0, 3): -2.0,
                (1, 3): -2.0,
                (2, 3): -2.0,
            },
        )
        # Test constraint
        lin = op.linear_constraints
        self.assertEqual(len(lin), 0)

    def test_interpret(self):
        """Test interpret"""
        maxcut = Maxcut(self.graph)
        self.assertEqual(maxcut.interpret(self.result), [[2, 3], [0, 1]])

    def test_node_color(self):
        """Test _node_color"""
        maxcut = Maxcut(self.graph)
        self.assertEqual(maxcut._node_color(self.result), ["b", "b", "r", "r"])

    @unittest.skipIf(_optionals.HAS_MATPLOTLIB, "Matplotlib is available.")
    def test_draw_without_maxplotlin(self):
        """Test whether draw raises an error if matplotlib is not installed"""
        maxcut = Maxcut(self.graph)
        from qiskit.exceptions import MissingOptionalLibraryError

        with self.assertRaises(MissingOptionalLibraryError):
            maxcut.draw()


if __name__ == "__main__":
    unittest.main()
