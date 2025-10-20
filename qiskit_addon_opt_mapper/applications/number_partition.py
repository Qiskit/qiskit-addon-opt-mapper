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
"""An application class for the number partitioning."""

import numpy as np
from docplex.mp.model import Model

from qiskit_addon_opt_mapper.problems.optimization_problem import OptimizationProblem
from qiskit_addon_opt_mapper.translators import from_docplex_mp

from .optimization_application import OptimizationApplication


class NumberPartition(OptimizationApplication):
    """Optimization application for the "number partition" [1] problem.

    References:
        [1]: "Partition problem",
        https://en.wikipedia.org/wiki/Partition_problem
    """

    def __init__(self, number_set: list[int]) -> None:
        """Init method.

        Args:
            number_set: A list of integers
        """
        self._number_set = number_set

    def to_optimization_problem(self) -> OptimizationProblem:
        """Represent as an optimization problem.

        Convert a number partitioning problem instance into a
        :class:`~qiskit_addon_opt_mapper.problems.OptimizationProblem`


        Returns:
            The :class:`~qiskit_addon_opt_mapper.problems.OptimizationProblem` created
            from the number partitioning problem instance.
        """
        mdl = Model(name="Number partitioning")
        x = {i: mdl.binary_var(name=f"x_{i}") for i in range(len(self._number_set))}
        mdl.add_constraint(
            mdl.sum(num * (-2 * x[i] + 1) for i, num in enumerate(self._number_set)) == 0
        )
        op = from_docplex_mp(mdl)
        return op

    def interpret(self, result: np.ndarray) -> list[list[int]]:
        """Interpret a result as a list of subsets.

        Args:
            result: The calculated result of the problem


        Returns:
            A list of subsets whose sum is the half of the total.
        """
        x = self._result_to_x(result)
        num_subsets = [[], []]  # type: list[list[int]]
        for i, value in enumerate(x):
            if value == 0:
                num_subsets[0].append(self._number_set[i])
            else:
                num_subsets[1].append(self._number_set[i])
        return num_subsets
