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

"""Solvers module.

Classical Solvers (:mod:`qiskit_addon_opt_mapper.solvers`).
===============================================================

.. currentmodule:: qiskit_addon_opt_mapper.solvers

Base class for solvers and results
---------------------------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   OptimizationSolver
   SolverResult

Classical Solvers
---------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   CplexSolver
   GurobiSolver
   ScipyMilpSolver


"""

from .cplex_solver import CplexSolver
from .gurobi_solver import GurobiSolver
from .scipy_milp_solver import ScipyMilpSolver
from .solver import OptimizationSolver, SolverResult, SolverResultStatus

__all__ = [
    "CplexSolver",
    "GurobiSolver",
    "OptimizationSolver",
    "ScipyMilpSolver",
    "SolverResult",
    "SolverResultStatus",
]
