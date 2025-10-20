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

"""Test Gurobi Solver"""

import unittest

import numpy as np
import qiskit_addon_opt_mapper.optionals as _optionals
from ddt import data, ddt
from docplex.mp.model import Model
from docplex.mp.model_reader import ModelReader
from qiskit_addon_opt_mapper.solvers import GurobiSolver
from qiskit_addon_opt_mapper.translators import from_docplex_mp

from ..optimization_test_case import OptimizationTestCase


@ddt
class TestGurobiOptimizer(OptimizationTestCase):
    """Gurobi Optimizer Tests."""

    @data(
        ("op_ip1.lp", [0, 2], 6),
        ("op_mip1.lp", [0, 1, 1], 5.5),
        ("op_lp1.lp", [0.25, 1.75], 5.8750),
    )
    @unittest.skipIf(not _optionals.HAS_GUROBIPY, "Gurobi not available.")
    @unittest.skipIf(not _optionals.HAS_CPLEX, "CPLEX not available.")
    def test_gurobi_optimizer(self, config):
        """Gurobi Optimizer Test"""
        # unpack configuration
        gurobi_optimizer = GurobiSolver(disp=False)
        filename, x, fval = config

        # load optimization problem
        lp_file = self.get_resource_path(filename, "solvers/resources")
        model = Model()
        model = ModelReader.read(lp_file)
        problem = from_docplex_mp(model)

        # solve problem with gurobi
        result = gurobi_optimizer.solve(problem)

        # analyze results
        self.assertAlmostEqual(result.fval, fval)
        np.testing.assert_array_almost_equal(result.x, x)


if __name__ == "__main__":
    unittest.main()
