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


"""Optimization problem model elements.

Optimization problems (:mod:`qiskit_addon_opt_mapper.problems`).
===================================================================

.. currentmodule:: qiskit_addon_opt_mapper.problems

Optimization problem
----------------------
Structures for defining an optimization problem.

Note:
    The following classes are not intended to be instantiated directly.
    Objects of these types are available within an instantiated
    :class:`~qiskit_addon_opt_mapper.OptimizationProblem`.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Constraint
   LinearExpression
   LinearConstraint
   QuadraticExpression
   QuadraticConstraint
   OptimizationProblemElement
   Variable
   HigherOrderExpression
   HigherOrderConstraint
   HigherOrderExpression
   OptimizationObjective
"""

from .constraint import Constraint
from .higher_order_constraint import HigherOrderConstraint
from .higher_order_expression import HigherOrderExpression
from .linear_constraint import LinearConstraint
from .linear_expression import LinearExpression
from .optimization_objective import ObjSense, OptimizationObjective
from .optimization_problem import OptimizationProblem
from .optimization_problem_element import OptimizationProblemElement
from .quadratic_constraint import QuadraticConstraint
from .quadratic_expression import QuadraticExpression
from .variable import Variable, VarType

__all__ = [
    "Constraint",
    "HigherOrderConstraint",
    "HigherOrderExpression",
    "LinearConstraint",
    "LinearExpression",
    "ObjSense",
    "OptimizationObjective",
    "OptimizationProblem",
    "OptimizationProblemElement",
    "QuadraticConstraint",
    "QuadraticExpression",
    "VarType",
    "Variable",
]
