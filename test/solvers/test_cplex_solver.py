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

"""Test Cplex Solver"""

import unittest

import numpy as np
import qiskit_addon_opt_mapper.optionals as _optionals
from ddt import data, ddt
from docplex.mp.model import Model
from docplex.mp.model_reader import ModelReader
from qiskit_addon_opt_mapper.solvers import CplexSolver, SolverResultStatus
from qiskit_addon_opt_mapper.translators import from_docplex_mp

from ..optimization_test_case import OptimizationTestCase


@ddt
class TestCplexSolver(OptimizationTestCase):
    """CPLEX Solver Tests."""

    @data(
        ("op_ip1.lp", [0, 2], 6),
        ("op_mip1.lp", [0, 1, 1], 5.5),
        ("op_lp1.lp", [0.25, 1.75], 5.8750),
    )
    @unittest.skipIf(not _optionals.HAS_CPLEX, "CPLEX not available.")
    def test_cplex_optimizer(self, config):
        """CPLEX Optimizer Test"""
        cplex_optimizer = CplexSolver(disp=False, cplex_parameters={"threads": 1, "randomseed": 1})
        # unpack configuration
        filename, x, fval = config

        # load optimization problem with docplex
        lp_file = self.get_resource_path(filename, "solvers/resources")
        model = Model()
        model = ModelReader.read(lp_file)
        problem = from_docplex_mp(model)

        # solve problem with cplex
        result = cplex_optimizer.solve(problem)

        # analyze results
        self.assertAlmostEqual(result.fval, fval)
        for i in range(problem.get_num_vars()):
            self.assertAlmostEqual(result.x[i], x[i])

    @data(
        ("op_ip1.lp", [0, 2], 6),
        ("op_mip1.lp", [0, 1, 1], 5.5),
        ("op_lp1.lp", [0.25, 1.75], 5.8750),
    )
    @unittest.skipIf(not _optionals.HAS_CPLEX, "CPLEX not available.")
    def test_cplex_optimizer_no_solution(self, config):
        """CPLEX Optimizer Test if no solution is found"""
        cplex_optimizer = CplexSolver(disp=False, cplex_parameters={"dettimelimit": 0})
        # unpack configuration
        filename, _, _ = config

        # load optimization problem with docplex
        lp_file = self.get_resource_path(filename, "solvers/resources")
        model = ModelReader.read(lp_file)
        problem = from_docplex_mp(model)

        # solve problem with cplex
        with self.assertWarns(UserWarning):
            result = cplex_optimizer.solve(problem)
        np.testing.assert_array_almost_equal(result.x, np.zeros(problem.get_num_vars()))
        self.assertEqual(result.status, SolverResultStatus.FAILURE)
        self.assertEqual(result.raw_results, None)


if __name__ == "__main__":
    unittest.main()
