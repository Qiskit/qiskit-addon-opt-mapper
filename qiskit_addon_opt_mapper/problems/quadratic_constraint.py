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

"""Quadratic Constraint."""

from typing import Any

from numpy import ndarray
from scipy.sparse import spmatrix

from .constraint import Constraint, ConstraintSense
from .linear_expression import LinearExpression
from .quadratic_expression import QuadraticExpression


class QuadraticConstraint(Constraint):
    """Representation of a quadratic constraint."""

    # Note: added, duplicating in effect that in Constraint, to avoid issues with Sphinx
    Sense = ConstraintSense

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        optimization_problem: Any,
        name: str,
        linear: ndarray | spmatrix | list[float] | dict[str | int, float],
        quadratic: (
            ndarray | spmatrix | list[list[float]] | dict[tuple[str | int, str | int], float]
        ),
        sense: ConstraintSense,
        rhs: float,
    ) -> None:
        """Constructs a quadratic constraint, consisting of a linear and a quadratic term.

        Args:
            optimization_problem: The parent optimization problem.
            name: The name of the constraint.
            linear: The coefficients specifying the linear part of the constraint.
            quadratic: The coefficients specifying the linear part of the constraint.
            sense: The sense of the constraint.
            rhs: The right-hand-side of the constraint.
        """
        super().__init__(optimization_problem, name, sense, rhs)
        self._linear = LinearExpression(optimization_problem, linear)
        self._quadratic = QuadraticExpression(optimization_problem, quadratic)

    @property
    def linear(self) -> LinearExpression:
        """Returns the linear expression corresponding to the left-hand-side of the constraint.

        Returns:
            The left-hand-side linear expression.
        """
        return self._linear

    @linear.setter
    def linear(
        self,
        linear: ndarray | spmatrix | list[float] | dict[str | int, float],
    ) -> None:
        """Sets the linear expression corresponding to the left-hand-side of the constraint.

        The coefficients can either be given by an array, a (sparse) 1d matrix, a list or a
        dictionary.

        Args:
            linear: The linear coefficients of the left-hand-side.
        """
        self._linear = LinearExpression(self.optimization_problem, linear)

    @property
    def quadratic(self) -> QuadraticExpression:
        """Returns the quadratic expression corresponding to the left-hand-side of the constraint.

        Returns:
            The left-hand-side quadratic expression.
        """
        return self._quadratic

    @quadratic.setter
    def quadratic(
        self,
        quadratic: (
            ndarray | spmatrix | list[list[float]] | dict[tuple[str | int, str | int], float]
        ),
    ) -> None:
        """Sets the quadratic expression corresponding to the left-hand-side of the constraint.

        The coefficients can either be given by an array, a (sparse) matrix, a list or a
        dictionary.

        Args:
            quadratic: The quadratic coefficients of the left-hand-side.
        """
        self._quadratic = QuadraticExpression(self.optimization_problem, quadratic)

    def evaluate(self, x: ndarray | list | dict[str | int, float]) -> float:
        """Evaluate the left-hand-side of the constraint.

        Args:
            x: The values of the variables to be evaluated.


        Returns:
            The left-hand-side of the constraint given the variable values.
        """
        return self.linear.evaluate(x) + self.quadratic.evaluate(x)

    def __repr__(self):
        """Repr. for QuadraticConstraint."""
        # pylint: disable=cyclic-import
        from ..translators.prettyprint import DEFAULT_TRUNCATE, expr2str

        lhs = expr2str(linear=self.linear, quadratic=self.quadratic, truncate=DEFAULT_TRUNCATE)
        return f"<{self.__class__.__name__}: {lhs} {self.sense.label} {self.rhs} '{self.name}'>"

    def __str__(self):
        """Str. for QuadraticConstraint."""
        # pylint: disable=cyclic-import
        from ..translators.prettyprint import expr2str

        lhs = expr2str(linear=self.linear, quadratic=self.quadratic)
        return f"{lhs} {self.sense.label} {self.rhs} '{self.name}'"
