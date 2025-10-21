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
"""Higher-order Constraint with linear, quadratic, and higher-order terms."""

from typing import Any

from numpy import ndarray
from scipy.sparse import spmatrix

from .constraint import Constraint, ConstraintSense
from .higher_order_expression import HigherOrderExpression
from .linear_expression import LinearExpression
from .quadratic_expression import QuadraticExpression

CoeffLike = ndarray | dict[tuple[str | int, ...], float] | list


class HigherOrderConstraint(Constraint):
    """Constraint in higher order form.

    e.g. ``linear(x) + x^T Q x + sum_{k>=3}  sum_{|t|=k} C_k[t] * prod_{i in t} x[i]`` ``sense`` ``rhs``
    where ``sense`` is one of the ConstraintSense values (e.g., LE, <=) and ``rhs`` is a float.

    Supports both a single higher-order term (order+coeffs) and multiple via
    higher_orders={k: coeffs}.
    """

    Sense = ConstraintSense  # duplicated for Sphinx compatibility

    def __init__(
        self,
        optimization_problem: Any,
        name: str,
        # linear/quadratic
        linear: ndarray | spmatrix | list[float] | dict[str | int, float] | None = None,
        quadratic: (
            ndarray | spmatrix | list[list[float]] | dict[tuple[int | str, int | str], float] | None
        ) = None,
        # higher-order
        higher_order: dict[int, CoeffLike] | None = None,
        sense: ConstraintSense = ConstraintSense.LE,
        rhs: float = 0.0,
    ) -> None:
        """Construct a higher-order constraint with linear, quadratic, and optional higher-order parts.

        Args:
            optimization_problem: The optimization problem this constraint belongs to.
            name: The name of the constraint.
            linear: Coefficients for the linear part.
            quadratic: Coefficients for the quadratic part.
            higher_order: A single higher-order expression or a dictionary of {order: coeffs} for multiple orders (k≥3).
            sense: The sense of the constraint (e.g., LE, <=).
            rhs: The right-hand-side value of the constraint.
        """
        super().__init__(optimization_problem, name, sense, rhs)

        self._linear = LinearExpression(optimization_problem, {} if linear is None else linear)
        self._quadratic = QuadraticExpression(
            optimization_problem, {} if quadratic is None else quadratic
        )

        # Store multiple higher-order expressions keyed by order (k>=3)
        if higher_order is None:
            self._higher_order: dict[int, HigherOrderExpression] = {}
        else:
            self.higher_order = higher_order

    # --- properties ---
    @property
    def linear(self) -> LinearExpression:
        """Returns the linear expression corresponding to the left-hand-side of the constraint.

        Returns:
            The left-hand-side linear expression.
        """
        return self._linear

    @linear.setter
    def linear(self, linear):
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
    def quadratic(self, quadratic):
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
            higher_order: A dictionary of {order: HigherOrderExpression} for multiple orders.
        """
        self._higher_order = {}

        for k, coeffs in higher_order.items():
            self._higher_order[k] = HigherOrderExpression(self.optimization_problem, coeffs)

    def evaluate(self, x: ndarray | list | dict[str | int, float]) -> float:
        """Evaluate the left-hand-side of the constraint.

        Args:
            x: The values of the variables to be evaluated.

        Returns:
            The left-hand-side of the constraint given the variable values.
        """
        val = self.linear.evaluate(x) + self.quadratic.evaluate(x)
        for expr in self._higher_order.values():
            val += expr.evaluate(x)
        return float(val)

    def __repr__(self):
        """Repr for higher order constraint."""
        # pylint: disable=cyclic-import
        from ..translators.prettyprint import DEFAULT_TRUNCATE, expr2str

        lhs = expr2str(
            linear=self.linear,
            quadratic=self.quadratic,
            higher_order=self._higher_order,
            truncate=DEFAULT_TRUNCATE,
        )
        return f"<{self.__class__.__name__}: {lhs} {self.sense.label} {self.rhs} '{self.name}'>"

    def __str__(self):
        """Str for higher order constraint."""
        # pylint: disable=cyclic-import
        from ..translators.prettyprint import expr2str

        lhs = expr2str(
            linear=self.linear,
            quadratic=self.quadratic,
            higher_order=self._higher_order,
        )
        return f"{lhs} {self.sense.label} {self.rhs} '{self.name}'"
