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

"""Converters module for Optimization add-on.

Optimization converters (:mod:`qiskit_addon_opt_mapper.converters`).
======================================================================

.. currentmodule:: qiskit_addon_opt_mapper.converters

This is a set of converters having `convert` functionality to go between different representations
of a given :class:`~qiskit_addon_opt_mapper.problems.QuadraticProgram` and to `interpret` a given
result for the problem, based on the original problem before conversion, to return an appropriate
:class:`~qiskit_addon_opt_mapper.solvers.SolverResult`.

Base class for converters
------------------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   OptimizationProblemConverter

Converters
---------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BinaryToLinearBinary
   BinaryToSpin
   InequalityToEquality
   IntegerToBinary
   LinearInequalityToPenalty
   MaximizeToMinimize
   MinimizeToMaximize
   OptimizationProblemToHubo
   OptimizationProblemToQubo
   SpinToBinary

"""

from .binary_to_linear_binary import BinaryToLinearBinary
from .binary_to_spin import BinaryToSpin
from .equality_to_penalty import EqualityToPenalty
from .flip_problem_sense import MaximizeToMinimize, MinimizeToMaximize
from .inequality_to_equality import InequalityToEquality
from .integer_to_binary import IntegerToBinary
from .linear_inequality_to_penalty import LinearInequalityToPenalty
from .optimization_problem_converter import OptimizationProblemConverter
from .optimization_problem_to_hubo import OptimizationProblemToHubo
from .optimization_problem_to_qubo import OptimizationProblemToQubo
from .spin_to_binary import SpinToBinary

__all__ = [
    "BinaryToLinearBinary",
    "BinaryToSpin",
    "EqualityToPenalty",
    "InequalityToEquality",
    "IntegerToBinary",
    "LinearInequalityToPenalty",
    "MaximizeToMinimize",
    "MinimizeToMaximize",
    "OptimizationProblemConverter",
    "OptimizationProblemToHubo",
    "OptimizationProblemToQubo",
    "SpinToBinary",
]
