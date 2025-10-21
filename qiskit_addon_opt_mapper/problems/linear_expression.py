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

"""Linear expression interface."""

from dataclasses import dataclass
from typing import Any, cast

from numpy import array, ndarray
from scipy.sparse import dok_matrix, spmatrix

from ..exceptions import OptimizationError
from ..infinity import INFINITY
from .optimization_problem_element import OptimizationProblemElement


@dataclass
class ExpressionBounds:
    """Lower bound and upper bound of a linear expression or a quadratic expression."""

    lowerbound: float
    """Lower bound"""

    upperbound: float
    """Upper bound"""


class LinearExpression(OptimizationProblemElement):
    """Representation of a linear expression by its coefficients."""

    def __init__(
        self,
        optimization_problem: Any,
        coefficients: ndarray | spmatrix | list[float] | dict[int | str, float],
    ) -> None:
        """Creates a new linear expression.

        The linear expression can be defined via an array, a list, a sparse matrix, or a dictionary
        that uses variable names or indices as keys and stores the values internally as a
        dok_matrix.

        Args:
            optimization_problem: The parent OptimizationProblem.
            coefficients: The (sparse) representation of the coefficients.

        """
        super().__init__(optimization_problem)
        self.coefficients = coefficients

    def __getitem__(self, i: int | str) -> float:
        """Returns the i-th coefficient where i can be a variable name or index.

        Args:
            i: the index or name of the variable corresponding to the coefficient.


        Returns:
            The coefficient corresponding to the addressed variable.
        """
        if isinstance(i, str):
            i = self.optimization_problem.variables_index[i]
        return float(self.coefficients[0, i])

    def __setitem__(self, i: int | str, value: float) -> None:
        """Set item for LinearExpression."""
        if isinstance(i, str):
            i = self.optimization_problem.variables_index[i]
        self._coefficients[0, i] = value

    def _coeffs_to_dok_matrix(
        self, coefficients: ndarray | spmatrix | list | dict[int | str, float]
    ) -> dok_matrix:
        """Maps given 1d-coefficients to a dok_matrix.

        Args:
            coefficients: The 1d-coefficients to be mapped.


        Returns:
            The given 1d-coefficients as a dok_matrix

        Raises:
            OptimizationError: if coefficients are given in unsupported format.
        """
        if isinstance(coefficients, list):
            # convert list to numpy array first
            # then check the shape and convert to dok_matrix
            coefficients = array(coefficients)
            if (
                coefficients.ndim != 1
                or coefficients.shape[0] != self.optimization_problem.get_num_vars()
            ):
                raise OptimizationError(
                    "The coefficient list for the linear expression must be one-dimensional and "
                    "length must match the number of variables."
                )
            coefficients = dok_matrix([coefficients])

        elif isinstance(coefficients, ndarray):
            if (
                coefficients.ndim != 1
                or coefficients.shape[0] != self.optimization_problem.get_num_vars()
            ):
                raise OptimizationError(
                    "The coefficient numpy array for the linear expression must be a (1, n) row "
                    "vector with length matching the number of variables."
                )
            coefficients = dok_matrix([coefficients])

        elif isinstance(coefficients, spmatrix):
            coefficients = dok_matrix(coefficients)
        elif isinstance(coefficients, dict):
            coeffs = dok_matrix((1, self.optimization_problem.get_num_vars()))
            for index, value in coefficients.items():
                if isinstance(index, str):
                    index = self.optimization_problem.variables_index[index]
                coeffs[0, index] = value
            coefficients = coeffs
        else:
            raise OptimizationError(
                f"Unsupported format for coefficients: {type(coefficients)}. Supported formats are:"
                "list, numpy array, scipy sparse matrix, dictionary."
            )
        return coefficients

    @property
    def coefficients(self) -> dok_matrix:
        """Returns the coefficients of the linear expression.

        Returns:
            The coefficients of the linear expression.
        """
        return self._coefficients

    @coefficients.setter
    def coefficients(
        self,
        coefficients: ndarray | spmatrix | list[float] | dict[int | str, float],
    ) -> None:
        """Sets the coefficients of the linear expression.

        Args:
            coefficients: The coefficients of the linear expression.
        """
        self._coefficients = self._coeffs_to_dok_matrix(coefficients)

    def to_array(self) -> ndarray:
        """Returns the coefficients of the linear expression as array.

        Returns:
            An array with the coefficients corresponding to the linear expression.
        """
        return cast(ndarray, self._coefficients.toarray()[0])

    def to_dict(self, use_name: bool = False) -> dict[int | str, float]:
        """Returns the coefficients of the linear expression as dictionary.

        Either using variable names or indices as keys.

        Args:
            use_name: Determines whether to use index or names to refer to variables.


        Returns:
            An dictionary with the coefficients corresponding to the linear expression.
        """
        if use_name:
            return {
                self.optimization_problem.variables[k].name: float(v)
                for (_, k), v in self._coefficients.items()
            }
        return {k: float(v) for (_, k), v in self._coefficients.items()}

    def evaluate(self, x: ndarray | list | dict[int | str, float]) -> float:
        """Evaluate the linear expression for given variables.

        Args:
            x: The values of the variables to be evaluated.


        Returns:
            The value of the linear expression given the variable values.
        """
        # cast input to dok_matrix if it is a dictionary
        x = self._coeffs_to_dok_matrix(x)

        # compute the dot-product of the input and the linear coefficients
        val = (x @ self.coefficients.transpose())[0, 0]

        # return the result
        return float(val)

    # pylint: disable=unused-argument
    def evaluate_gradient(self, x: ndarray | list | dict[int | str, float]) -> ndarray:
        """Evaluate the gradient of the linear expression for given variables.

        Args:
            x: The values of the variables to be evaluated.


        Returns:
            The value of the gradient of the linear expression given the variable values.
        """
        # extract the coefficients as array and return it
        return self.to_array()

    @property
    def bounds(self) -> ExpressionBounds:
        """Returns the lower bound and the upper bound of the linear expression.

        Returns:
            The lower bound and the upper bound of the linear expression

        Raises:
            OptimizationError: if the linear expression contains any unbounded variable

        """
        l_b = u_b = 0.0
        for ind, coeff in self.to_dict().items():
            x = self.optimization_problem.get_variable(ind)
            if x.lowerbound == -INFINITY or x.upperbound == INFINITY:
                raise OptimizationError(
                    f"Linear expression contains an unbounded variable: {x.name}"
                )
            lst = [coeff * x.lowerbound, coeff * x.upperbound]
            l_b += min(lst)
            u_b += max(lst)
        return ExpressionBounds(lowerbound=l_b, upperbound=u_b)

    def __repr__(self):
        """Repr. for LinearExpression."""
        # pylint: disable=cyclic-import
        from ..translators.prettyprint import DEFAULT_TRUNCATE, expr2str

        return f"<{self.__class__.__name__}: {expr2str(linear=self, truncate=DEFAULT_TRUNCATE)}>"

    def __str__(self):
        """Str. for LinearExpression."""
        # pylint: disable=cyclic-import
        from ..translators.prettyprint import expr2str

        return f"{expr2str(linear=self)}"
