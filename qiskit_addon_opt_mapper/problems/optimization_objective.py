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

"""Optimization Objective."""

from enum import Enum
from typing import Any, cast

from numpy import ndarray
from scipy.sparse import spmatrix

from ..exceptions import OptimizationError
from .higher_order_expression import HigherOrderExpression
from .linear_constraint import LinearExpression
from .optimization_problem_element import OptimizationProblemElement
from .quadratic_expression import QuadraticExpression

CoeffLike = ndarray | dict[tuple[int | str, ...], float] | list  # nested list as dense tensor


class ObjSense(Enum):
    """Objective Sense Type."""

    MINIMIZE = 1
    MAXIMIZE = -1


class OptimizationObjective(OptimizationProblemElement):
    """Optimization objective element.

    Follows:
        constant + linear(x) + x^T Q x + sum_{k>=3} H_k(x).
    """

    Sense = ObjSense

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        optimization_problem: Any,
        constant: float = 0.0,
        linear: ndarray | spmatrix | list[float] | dict[int | str, float] | None = None,
        quadratic: (
            ndarray | spmatrix | list[list[float]] | dict[tuple[int | str, int | str], float] | None
        ) = None,
        higher_order: dict[int, CoeffLike] | None = None,
        sense: ObjSense = ObjSense.MINIMIZE,
    ) -> None:
        """Construct an objective with linear, quadratic, and optional higher-order parts.

        Args:
            optimization_problem: The optimization problem this objective belongs to.
            constant: The constant part of the objective function.
            linear: The coefficients for the linear part of the objective function.
            quadratic: The coefficients for the quadratic part of the objective function.
            higher_order: A single higher-order expression or a dictionary of {order: coeffs}
                for multiple orders (k>=3).
            sense: The sense of the objective function (e.g., MINIMIZE, MAXIMIZE).
        """
        super().__init__(optimization_problem)
        self._constant = float(constant)

        if linear is None:
            linear = {}
        self._linear = LinearExpression(optimization_problem, linear)

        if quadratic is None:
            quadratic = {}
        self._quadratic = QuadraticExpression(optimization_problem, quadratic)

        # Store multiple higher-order expressions keyed by order (k>=3)
        if higher_order is None:
            self._higher_order: dict[int, HigherOrderExpression] = {}
        else:
            self.higher_order = higher_order

        self._sense = sense

    @property
    def constant(self) -> float:
        """Returns the constant part of the objective function.

        Returns:
            The constant part of the objective function.
        """
        return self._constant

    @constant.setter
    def constant(self, constant: float) -> None:
        """Sets the constant part of the objective function.

        Args:
            constant: The constant part of the objective function.
        """
        self._constant = float(constant)

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
        linear: ndarray | spmatrix | list[float] | dict[int | str, float],
    ) -> None:
        """Sets the linear expression corresponding to the left-hand-side of the constraint.

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
            ndarray | spmatrix | list[list[float]] | dict[tuple[int | str, int | str], float]
        ),
    ) -> None:
        """Sets the quadratic expression corresponding to the left-hand-side of the constraint.

        Args:
            quadratic: The quadratic coefficients of the left-hand-side.
        """
        self._quadratic = QuadraticExpression(self.optimization_problem, quadratic)

    @property
    def higher_order(self) -> dict[int, HigherOrderExpression]:
        """Return a shallow copy of {order: HigherOrderExpression}.

        Returns:
            A dictionary mapping order (k>=3) to HigherOrderExpression.
        """
        return dict(self._higher_order)

    @higher_order.setter
    def higher_order(
        self,
        higher_order: dict[int, CoeffLike],
    ) -> None:
        """Sets the higher-order expressions.

        Args:
            higher_order: A dictionary of
                {order: HigherOrderExpression} for multiple orders.
        """
        self._higher_order = {}

        for k, coeffs in higher_order.items():
            self._higher_order[k] = HigherOrderExpression(self.optimization_problem, coeffs)

    @property
    def sense(self) -> ObjSense:
        """Returns the sense of the objective function.

        Returns:
            The sense of the objective function (e.g., MINIMIZE, MAXIMIZE).
        """
        return self._sense

    @sense.setter
    def sense(self, sense: ObjSense) -> None:
        """Sets the sense of the objective function.

        Args:
            sense: The sense of the objective function.
        """
        self._sense = sense

    def evaluate(self, x: ndarray | list | dict[int | str, float]) -> float:
        """Evaluate objective value at x.

        Args:
            x: The values of the variables to be evaluated.

        Returns:
            The objective value given the variable values.
        """
        n = self.optimization_problem.get_num_vars()
        if self.linear.coefficients.shape != (
            1,
            n,
        ) or self.quadratic.coefficients.shape != (n, n):
            raise OptimizationError(
                "The shape of the objective function does not match the number of variables. "
                "Define the objective after defining all variables"
            )
        val = self.constant + self.linear.evaluate(x) + self.quadratic.evaluate(x)
        for expr in self._higher_order.values():
            val += expr.evaluate(x)
        return float(val)

    def evaluate_gradient(self, x: ndarray | list | dict[int | str, float]) -> ndarray:
        """Evaluate gradient of the objective at x.

        Args:
            x: The values of the variables to be evaluated.

        Returns:
            The gradient of the objective function given the variable values.
        """
        n = self.optimization_problem.get_num_vars()
        if self.linear.coefficients.shape != (
            1,
            n,
        ) or self.quadratic.coefficients.shape != (n, n):
            raise OptimizationError(
                "The shape of the objective function does not match the number of variables. "
                "Define the objective after defining all variables"
            )
        g = self.linear.evaluate_gradient(x) + self.quadratic.evaluate_gradient(x)
        for expr in self._higher_order.values():
            g = g + expr.evaluate_gradient(x)
        return cast(ndarray, g)

    def __repr__(self):
        """Repr. for OptimizationObjective."""
        # pylint: disable=cyclic-import
        from ..translators.prettyprint import DEFAULT_TRUNCATE, expr2str

        expr_str = expr2str(
            self.constant,
            self.linear,
            self.quadratic,
            self._higher_order,  # multiple orders
            truncate=DEFAULT_TRUNCATE,
        )
        return f"<{self.__class__.__name__}: {self._sense.name.lower()} {expr_str}>"

    def __str__(self):
        """Str. for OptimizationObjective."""
        # pylint: disable=cyclic-import
        from ..translators.prettyprint import expr2str

        expr_str = expr2str(
            self.constant,
            self.linear,
            self.quadratic,
            self._higher_order,  # multiple orders
        )
        return f"{self._sense.name.lower()} {expr_str}"
