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

"""A converter from OptimizationProblem to HUBO form."""

from typing import cast

import numpy as np

from ..converters.flip_problem_sense import MaximizeToMinimize
from ..converters.inequality_to_equality import InequalityToEquality
from ..converters.integer_to_binary import IntegerToBinary
from ..converters.linear_inequality_to_penalty import LinearInequalityToPenalty
from ..converters.spin_to_binary import SpinToBinary
from ..exceptions import OptimizationError
from ..problems.optimization_problem import OptimizationProblem
from .equality_to_penalty import EqualityToPenalty
from .optimization_problem_converter import OptimizationProblemConverter


class OptimizationProblemToHubo(OptimizationProblemConverter):
    """Convert an optimization problem into a HUBO form.

    HUBO stands for "higher-order unconstrained binary optimization".
    The conversion is achieved by converting variables to binary and eliminating constraints.
    The resulting problem has no constraints and a higher-order polynomial objective function.
    This combines several converters: `IntegerToBinary`, `InequalityToPenalty`,
    `EqualityToPenalty`, and `MaximizeToMinimize`, while preserving higher-order terms
    in the objective function. The resulting HUBO problem can be directly mapped to
    an Ising Hamiltonian using the `to_ising()` function.

    **Examples**

    >>> from qiskit_addon_opt_mapper.problems import OptimizationProblem
    >>> from qiskit_addon_opt_mapper.converters import OptimizationProblemToHubo
    >>> problem = OptimizationProblem()
    >>> # define a problem
    >>> conv = OptimizationProblemToHubo()
    >>> problem2 = conv.convert(problem)
    """

    def __init__(self, penalty: float | None = None) -> None:
        """Init method.

        Args:
            penalty: Penalty factor to scale equality constraints that are added to objective.
                If None is passed, a penalty factor will be automatically calculated on every
                conversion.
        """
        self._penalize_eq_constraints = EqualityToPenalty(penalty=penalty)
        self._penalize_lin_ineq_constraints = LinearInequalityToPenalty(penalty=penalty)
        self._converters = [
            self._penalize_lin_ineq_constraints,
            InequalityToEquality(mode="integer"),
            IntegerToBinary(),
            SpinToBinary(),
            self._penalize_eq_constraints,
            MaximizeToMinimize(),
        ]

    def convert(self, problem: OptimizationProblem) -> OptimizationProblem:
        """Convert an optimization problem into a HUBO form.

        The new problem has no constraints and the objective function is higher order polynomial.

        Args:
            problem: An optimization problem to be converted.


        Returns:
            A new optimization problem in the HUBO form.

        Raises:
            OptimizationError: If the input problem is invalid.
        """
        msg = self.get_compatibility_msg(problem)
        if len(msg) > 0:
            raise OptimizationError(f"Incompatible problem: {msg}")

        for conv in self._converters:
            problem = conv.convert(problem)

        return problem

    def interpret(self, x: np.ndarray | list[float]) -> np.ndarray:
        """Convert the result of the converted problem back to that of the original problem.

        Done by applying the `interpret` method of each converter in reverse order.

        Args:
            x: A solution vector of the converted problem.


        Returns:
            A solution vector of the original problem.
        """
        for conv in reversed(self._converters):
            x = conv.interpret(x)

        return cast(np.ndarray, x)

    @staticmethod
    def get_compatibility_msg(problem: OptimizationProblem) -> str:
        """Checks whether the given problem is compatible with HUBO conversion.

        A problem is compatible if it can be converted to a HUBO (Higher-order Unconstrained Binary Optimization).
        If not, this function returns a message explaining the incompatibility.

        The following problems are not compatible:
        - Continuous variables are not supported.
        - Constraints with float coefficients are not supported, because inequality constraints cannot be
        converted to equality constraints using integer slack variables.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            A message describing the incompatibility.
        """
        # initialize message
        msg = ""
        # check whether there are incompatible variable types
        if problem.get_num_continuous_vars() > 0:
            msg += "Continuous variables are not supported. "

        # check whether there are float coefficients in constraints
        compatible_with_integer_slack = True
        for l_constraint in problem.linear_constraints:
            linear = l_constraint.linear.to_dict()
            if any(isinstance(coef, float) and not coef.is_integer() for coef in linear.values()):
                compatible_with_integer_slack = False
        for q_constraint in problem.quadratic_constraints:
            linear = q_constraint.linear.to_dict()
            quadratic = q_constraint.quadratic.to_dict()
            if any(
                isinstance(coef, float) and not coef.is_integer() for coef in quadratic.values()
            ) or any(isinstance(coef, float) and not coef.is_integer() for coef in linear.values()):
                compatible_with_integer_slack = False

        for constraint in problem.higher_order_constraints:
            for expr in constraint.higher_order.values():
                if any(
                    isinstance(coef, float) and not coef.is_integer()
                    for coef in expr.to_dict().values()
                ):
                    compatible_with_integer_slack = False

        if not compatible_with_integer_slack:
            msg += "Can not convert inequality constraints to equality constraint because \
                    float coefficients are in constraints. "

        # if an error occurred, return error message, otherwise, return the empty string
        return msg

    def is_compatible(self, problem: OptimizationProblem) -> bool:
        """Checks whether a given problem can be solved with the optimizer implementing this method.

        Args:
            problem: The optimization problem to check compatibility.


        Returns:
            Returns True if the problem is compatible, False otherwise.
        """
        return len(self.get_compatibility_msg(problem)) == 0

    @property
    def penalty(self) -> float | None:
        """Returns the penalty factor used in conversion.

        Returns:
            The penalty factor used in conversion.
        """
        return self._penalize_eq_constraints.penalty

    @penalty.setter
    def penalty(self, penalty: float | None) -> None:
        """Set a new penalty factor.

        Args:
            penalty: The new penalty factor.
                     If None is passed, a penalty factor will be automatically calculated on every
                     conversion.
        """
        self._penalize_lin_ineq_constraints.penalty = penalty
        self._penalize_eq_constraints.penalty = penalty
