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

"""Quadratic expression interface."""

from typing import Any, cast

import numpy as np
from numpy import ndarray
from scipy.sparse import dok_matrix, spmatrix, tril, triu

from ..exceptions import OptimizationError
from ..infinity import INFINITY
from .linear_expression import ExpressionBounds
from .optimization_problem_element import OptimizationProblemElement


class QuadraticExpression(OptimizationProblemElement):
    """Representation of a quadratic expression by its coefficients."""

    def __init__(
        self,
        optimization_problem: Any,
        coefficients: (
            ndarray | spmatrix | list[list[float]] | dict[tuple[str | int, str | int], float]
        ),
    ) -> None:
        """Creates a new quadratic expression.

        The quadratic expression can be defined via an array, a list, a sparse matrix, or a
        dictionary that uses variable names or indices as keys and stores the values internally as a
        dok_matrix. We stores values in a compressed way, i.e., values at symmetric positions are
        summed up in the upper triangle. For example, {(0, 1): 1, (1, 0): 2} -> {(0, 1): 3}.

        Args:
            optimization_problem: The parent OptimizationProblem.
            coefficients: The (sparse) representation of the coefficients.

        """
        super().__init__(optimization_problem)
        self.coefficients = coefficients

    def __getitem__(self, key: tuple[str | int, str | int]) -> float:
        """Returns the coefficient where i, j can be a variable names or indices.

        Args:
            key: The tuple of indices or names of the variables corresponding to the coefficient.


        Returns:
            The coefficient corresponding to the addressed variables.
        """
        i, j = key
        if isinstance(i, str):
            i = self.optimization_problem.variables_index[i]
        if isinstance(j, str):
            j = self.optimization_problem.variables_index[j]
        return float(self.coefficients[min(i, j), max(i, j)])

    def __setitem__(self, key: tuple[str | int, str | int], value: float) -> None:
        """Sets the coefficient where i, j can be a variable names or indices.

        Args:
            key: The tuple of indices or names of the variables corresponding to the coefficient.
            value: The coefficient corresponding to the addressed variables.
        """
        i, j = key
        if isinstance(i, str):
            i = self.optimization_problem.variables_index[i]
        if isinstance(j, str):
            j = self.optimization_problem.variables_index[j]
        self.coefficients[min(i, j), max(i, j)] = value

    def _coeffs_to_dok_matrix(
        self,
        coefficients: (
            ndarray | spmatrix | list[list[float]] | dict[tuple[str | int, str | int], float]
        ),
    ) -> dok_matrix:
        """Maps given coefficients to a dok_matrix.

        Args:
            coefficients: The coefficients to be mapped.


        Returns:
            The given coefficients as a dok_matrix

        Raises:
            OptimizationError: if coefficients are given in unsupported format.
        """
        if isinstance(coefficients, list):
            n = self.optimization_problem.get_num_vars()
            coeffs = dok_matrix((n, n))
            if len(coefficients) != n:
                raise OptimizationError(
                    f"The coefficient list for the quadratic expression must be a list of {n} "
                    f"lists, each of length {n}, matching the number of variables."
                )
            for i, row in enumerate(coefficients):
                if len(row) != n:
                    raise OptimizationError(
                        f"Each inner list for the quadratic expression must be of length {n}"
                        f"matching the number of variables."
                    )
                for j, value in enumerate(row):
                    coeffs[i, j] = value
            coefficients = coeffs
        elif isinstance(coefficients, ndarray):
            n = self.optimization_problem.get_num_vars()
            if coefficients.ndim != 2 or coefficients.shape != (n, n):
                raise OptimizationError(
                    f"The coefficient numpy array for the quadratic expression must be a (n, n) "
                    f"matrix with shape ({n}, {n}) matching the number of variables."
                )
            coefficients = dok_matrix(coefficients)
        elif isinstance(coefficients, spmatrix):
            coefficients = dok_matrix(coefficients)
        elif isinstance(coefficients, dict):
            n = self.optimization_problem.get_num_vars()
            coeffs = dok_matrix((n, n))
            for (i, j), value in coefficients.items():  # type: ignore
                i_idx = self.optimization_problem.variables_index[i] if isinstance(i, str) else i
                j_idx = self.optimization_problem.variables_index[j] if isinstance(j, str) else j
                coeffs[i_idx, j_idx] = value
            coefficients = coeffs
        else:
            raise OptimizationError(f"Unsupported format for coefficients: {coefficients}")
        return self._triangle_matrix(coefficients)

    @staticmethod
    def _triangle_matrix(mat: dok_matrix) -> dok_matrix:
        lower = tril(mat, -1, format="dok")
        # `todok` is necessary because subtraction results in other format
        return (mat + lower.transpose() - lower).todok()

    @staticmethod
    def _symmetric_matrix(mat: dok_matrix) -> dok_matrix:
        upper = triu(mat, 1, format="dok") / 2
        # `todok` is necessary because subtraction results in other format
        return (mat + upper.transpose() - upper).todok()

    @property
    def coefficients(self) -> dok_matrix:
        """Returns the coefficients of the quadratic expression.

        Returns:
            The coefficients of the quadratic expression.
        """
        return self._coefficients

    @coefficients.setter
    def coefficients(
        self,
        coefficients: (
            ndarray | spmatrix | list[list[float]] | dict[tuple[str | int, str | int], float]
        ),
    ) -> None:
        """Sets the coefficients of the quadratic expression.

        Args:
            coefficients: The coefficients of the quadratic expression.
        """
        self._coefficients = self._coeffs_to_dok_matrix(coefficients)

    def to_array(self, symmetric: bool = False) -> ndarray:
        """Returns the coefficients of the quadratic expression as array.

        Args:
            symmetric: Determines whether the output is in a symmetric form or not.


        Returns:
            An array with the coefficients corresponding to the quadratic expression.
        """
        coeffs = self._symmetric_matrix(self._coefficients) if symmetric else self._coefficients
        return cast(ndarray, coeffs.toarray())

    def to_dict(
        self, symmetric: bool = False, use_name: bool = False
    ) -> dict[tuple[int, int] | tuple[str, str], float]:
        """Returns the coefficients of the quadratic expression as dictionary.

        Either using tuples of variable names or indices as keys.

        Args:
            symmetric: Determines whether the output is in a symmetric form or not.
            use_name: Determines whether to use index or names to refer to variables.


        Returns:
            An dictionary with the coefficients corresponding to the quadratic expression.
        """
        coeffs = self._symmetric_matrix(self._coefficients) if symmetric else self._coefficients
        if use_name:
            return {
                (
                    self.optimization_problem.variables[i].name,
                    self.optimization_problem.variables[j].name,
                ): v
                for (i, j), v in coeffs.items()
            }
        return {(int(i), int(j)): v for (i, j), v in coeffs.items()}

    def evaluate(self, x: ndarray | list | dict[str | int, float]) -> float:
        """Evaluate the quadratic expression for given variables: x * Q * x.

        Args:
            x: The values of the variables to be evaluated.


        Returns:
            The value of the quadratic expression given the variable values.
        """
        x = self._cast_as_array(x)

        # compute x * Q * x for the quadratic expression
        val = x @ self.coefficients @ x

        # return the result
        return float(val)

    def evaluate_gradient(self, x: ndarray | list | dict[str | int, float]) -> ndarray:
        """Evaluate the gradient of the quadratic expression for given variables.

        Args:
            x: The values of the variables to be evaluated.


        Returns:
            The value of the gradient quadratic expression given the variable values.
        """
        x = self._cast_as_array(x)

        # compute (Q' + Q) * x for the quadratic expression
        val = (self.coefficients.transpose() + self.coefficients) @ x

        # return the result
        return cast(ndarray, val)

    def _cast_as_array(self, x: ndarray | list | dict[str | int, float]) -> dok_matrix | np.ndarray:
        """Converts input to an array if it is a dictionary or list."""
        if isinstance(x, dict):
            x_aux = np.zeros(self.optimization_problem.get_num_vars())
            for i, v in x.items():
                if isinstance(i, str):
                    i = self.optimization_problem.variables_index[i]
                x_aux[i] = v
            x = x_aux
        if isinstance(x, list):
            x = np.array(x)
        return x

    @property
    def bounds(self) -> ExpressionBounds:
        """Returns the lower bound and the upper bound of the quadratic expression.

        Returns:
            The lower bound and the upper bound of the quadratic expression

        Raises:
            OptimizationError: if the quadratic expression contains any unbounded variable
        """
        l_b = u_b = 0.0
        for (ind1, ind2), coeff in self.to_dict().items():
            x = self.optimization_problem.get_variable(ind1)
            if x.lowerbound == -INFINITY or x.upperbound == INFINITY:
                raise OptimizationError(
                    f"Quadratic expression contains an unbounded variable: {x.name}"
                )
            y = self.optimization_problem.get_variable(ind2)
            if y.lowerbound == -INFINITY or y.upperbound == INFINITY:
                raise OptimizationError(
                    f"Quadratic expression contains an unbounded variable: {y.name}"
                )
            lst = []
            if ind1 == ind2:
                if x.lowerbound * x.upperbound <= 0.0:
                    # lower bound and upper bound have different signs
                    lst.append(0.0)
                lst.extend([x.lowerbound**2, x.upperbound**2])
            else:
                lst.extend(
                    [
                        x.lowerbound * y.lowerbound,
                        x.lowerbound * y.upperbound,
                        x.upperbound * y.lowerbound,
                        x.upperbound * y.upperbound,
                    ]
                )
            lst2 = [coeff * val for val in lst]
            l_b += min(lst2)
            u_b += max(lst2)
        return ExpressionBounds(lowerbound=l_b, upperbound=u_b)

    def __repr__(self):
        """Repr. for QuadraticExpression."""
        # pylint: disable=cyclic-import
        from ..translators.prettyprint import DEFAULT_TRUNCATE, expr2str

        return f"<{self.__class__.__name__}: {expr2str(quadratic=self, truncate=DEFAULT_TRUNCATE)}>"

    def __str__(self):
        """Repr. for QuadraticExpression."""
        # pylint: disable=cyclic-import
        from ..translators.prettyprint import expr2str

        return f"{expr2str(quadratic=self)}"
