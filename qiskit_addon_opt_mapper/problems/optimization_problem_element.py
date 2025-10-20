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

"""Interface for all objects that have a parent OptimizationProblem."""

# We import `problems` module not `OptimizationProblem` class
# to resolve the circular import issue of sphinx.
# See https://github.com/agronholm/sphinx-autodoc-typehints#dealing-with-circular-imports

from .. import problems  # pylint: disable=unused-import, cyclic-import


class OptimizationProblemElement:
    """Interface class for all objects that have a parent OptimizationProblem."""

    def __init__(self, optimization_problem: "problems.OptimizationProblem") -> None:
        """Initialize object with parent OptimizationProblem.

        Args:
            optimization_problem: The parent OptimizationProblem.

        Raises:
            TypeError: OptimizationProblem instance expected.
        """
        # pylint: disable=cyclic-import
        from .optimization_problem import OptimizationProblem

        if not isinstance(optimization_problem, OptimizationProblem):
            raise TypeError("OptimizationProblem instance expected")

        self._optimization_problem = optimization_problem

    @property
    def optimization_problem(self) -> "problems.OptimizationProblem":
        """Returns the parent OptimizationProblem.

        Returns:
            The parent OptimizationProblem.
        """
        return self._optimization_problem

    @optimization_problem.setter
    def optimization_problem(self, optimization_problem: "problems.OptimizationProblem") -> None:
        """Sets the parent OptimizationProblem.

        Args:
            optimization_problem: The parent OptimizationProblem.

        Raises:
            TypeError: OptimizationProblem instance expected.
        """
        # pylint: disable=cyclic-import
        from .optimization_problem import OptimizationProblem

        if not isinstance(optimization_problem, OptimizationProblem):
            raise TypeError("OptimizationProblem instance expected")

        self._optimization_problem = optimization_problem
