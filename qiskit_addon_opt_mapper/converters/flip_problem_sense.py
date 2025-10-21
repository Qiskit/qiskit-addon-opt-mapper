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

"""Converters to flip problem sense, e.g. maximization to minimization and vice versa."""

import copy

import numpy as np

from ..exceptions import OptimizationError
from ..problems.optimization_objective import ObjSense
from ..problems.optimization_problem import OptimizationProblem
from .optimization_problem_converter import OptimizationProblemConverter


class _FlipProblemSense(OptimizationProblemConverter):
    """Flip the sense of a problem.

    e.g. converts from maximization to minimization and
    vice versa, regardless of the current sense.
    """

    def __init__(self) -> None:
        self._src_num_vars: int | None = None

    def convert(self, problem: OptimizationProblem) -> OptimizationProblem:
        """Flip the sense of a problem.

        Args:
            problem: The problem to be flipped.


        Returns:
            A converted problem, that has the flipped sense.
        """
        # copy original number of variables as reference.
        self._src_num_vars = problem.get_num_vars()
        desired_sense = self._get_desired_sense(problem)

        # flip the problem sense
        if problem.objective.sense != desired_sense:
            desired_problem = copy.deepcopy(problem)
            desired_problem.objective.sense = desired_sense
            desired_problem.objective.constant = (-1) * problem.objective.constant
            desired_problem.objective.linear = (-1) * problem.objective.linear.coefficients
            desired_problem.objective.quadratic = (-1) * problem.objective.quadratic.coefficients
            desired_problem.objective.higher_order = {
                degree: (-1) * ho.to_array()
                for degree, ho in problem.objective.higher_order.items()
            }

        else:
            desired_problem = problem

        return desired_problem

    def _get_desired_sense(self, problem: OptimizationProblem) -> ObjSense:
        """Computes a desired sense of the problem. By default, flip the sense.

        Args:
            problem: a problem to check


        Returns:
            A desired sense, if the problem was a minimization problem, then the sense is
            maximization and vice versa.
        """
        if problem.objective.sense == ObjSense.MAXIMIZE:
            return ObjSense.MINIMIZE
        return ObjSense.MAXIMIZE

    def interpret(self, x: np.ndarray | list[float]) -> np.ndarray:
        """Convert the result of the converted problem back to that of the original problem.

        Note: This implementation does not modify the result, but the method is required because
        the base class defines `interpret` as an abstract method.


        Args:
            x: The result of the converted problem or the given result in case of FAILURE.


        Returns:
            The result of the original problem.

        Raises:
            OptimizationError: if the number of variables in the result differs from
                                     that of the original problem.
        """
        if len(x) != self._src_num_vars:
            raise OptimizationError(
                f"The number of variables in the passed result differs from "
                f"that of the original problem, should be {self._src_num_vars}, but got {len(x)}."
            )
        return np.asarray(x)


class MaximizeToMinimize(_FlipProblemSense):
    """Convert a maximization problem to a minimization problem only if it is a maximization problem.

    Otherwise problem's sense is unchanged.
    """

    def _get_desired_sense(self, problem: OptimizationProblem) -> ObjSense:
        return ObjSense.MINIMIZE


class MinimizeToMaximize(_FlipProblemSense):
    """Convert a minimization problem to a maximization problem only if it is a minimization problem.

    Otherwise problem's sense is unchanged.
    """

    def _get_desired_sense(self, problem: OptimizationProblem) -> ObjSense:
        return ObjSense.MAXIMIZE
