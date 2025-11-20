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

"""Converter that converts a binary problem into a linear binary problem."""

from collections import defaultdict
from collections.abc import Callable
from typing import TypeVar

import numpy as np

from ..exceptions import OptimizationError
from ..problems import HigherOrderExpression, LinearExpression, QuadraticExpression
from ..problems.optimization_problem import OptimizationProblem
from ..problems.variable import Variable
from .optimization_problem_converter import OptimizationProblemConverter

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


class keydefaultdict(defaultdict[_KT, _VT]):
    """A defaultdict whose default_factory gets passed the new entry to be added."""

    default_factory: Callable[[_KT], _VT] | None  # type: ignore

    def __init__(
        self,
        default_factory: Callable[[_KT], _VT] | None,
        /,
        **kwargs: _VT,
    ) -> None:
        """Initialize by providing the default_factory."""
        super().__init__(default_factory, **kwargs)  # type: ignore

    def __missing__(self, key):
        """Handling of missing data. Calls the default_factory."""
        if self.default_factory is None:
            raise KeyError(key)

        ret = self[key] = self.default_factory(key)
        return ret


class BinaryToLinearBinary(OptimizationProblemConverter):
    """Convert all high-degree terms to linear terms, both in objective and constraints. The converter assumes that the problem only contains binary variables.

    The conversion of a term ``x1*x2`` is done through the additional binary variable ``x1ANDx2 = x1*x2`` defined via:
    ```
    x1ANDx2 <= x1
    x1ANDx2 <= x2
    x1ANDx2 >= x1 + x2 - 1
    ```

    Similarly, a term ``x1*x2*x3`` is converted via the additional binary variable ``x1ANDx2ANDx3 = x1*x2*x3`` defined via:
    ```
    x1ANDx2ANDx3 <= x1
    x1ANDx2ANDx3 <= x2
    x1ANDx2ANDx3 <= x3
    x1ANDx2ANDx3 >= x1 + x2 + x3 - 2
    ```

    The definition of the new variables always depends directly on the original variables. For instance, even if ``x1ANDx2`` is included in a problem,
    the definition of ``x1ANDx2ANDx3`` will be based on individual ``x1``, ``x2`` and ``x3``, and will not exploit ``x1ANDx2``. Such an optimization may
    be handled by pre-solvers of commercial solvers, according to the specific needs.

    Note that during the conversion, powers of a single variable are removed: for instance, the expression ``x0^2 + x1*x2^3`` is treated as ``x0 + x1*x2``
    """

    _CONCAT_SEPARATOR = "AND"

    def __init__(self) -> None:
        """Initialize converter."""
        self._dst: OptimizationProblem | None = None
        self._new_vars: keydefaultdict[tuple[str, ...], Variable]

    # ---- public API ----

    def convert(self, problem: OptimizationProblem) -> OptimizationProblem:
        """Convert all high-degree terms (namely, degree 2+) to linear terms."""
        msg = self.get_compatibility_msg(problem)
        if len(msg) > 0:
            raise OptimizationError(f"Incompatible problem: {msg}")

        self._dst = OptimizationProblem(name=problem.name)

        # 1) Copy variables
        for v in problem.variables:
            self._dst.binary_var(v.name)

        # 2) Prepare structure for new vars
        self._new_vars = keydefaultdict(
            lambda x: self._dst.binary_var(self._CONCAT_SEPARATOR.join(x))  # type: ignore
        )

        # 3) Objective function
        obj_lin_dict = self._convert_expr(
            problem.objective.linear, problem.objective.quadratic, problem.objective.higher_order
        )
        if problem.objective.sense == problem.objective.Sense.MINIMIZE:
            self._dst.minimize(problem.objective.constant, obj_lin_dict)  # type: ignore
        else:
            self._dst.maximize(problem.objective.constant, obj_lin_dict)  # type: ignore

        # 4) Constraints
        for lc in problem.linear_constraints:
            self._dst.linear_constraint(lc.linear.to_dict(use_name=True), lc.sense, lc.rhs, lc.name)
        for qc in problem.quadratic_constraints:
            lin_constr_dict = self._convert_expr(qc.linear, qc.quadratic, None)
            self._dst.linear_constraint(lin_constr_dict, qc.sense, qc.rhs, qc.name)
        for hoc in problem.higher_order_constraints:
            lin_constr_dict = self._convert_expr(hoc.linear, hoc.quadratic, hoc.higher_order)
            self._dst.linear_constraint(lin_constr_dict, hoc.sense, hoc.rhs, hoc.name)

        # 5) Additional constraints defining the new variables
        for old_var_names, new_var in self._new_vars.items():
            # z <= x1
            for ovn in old_var_names:
                self._dst.linear_constraint({new_var.name: 1, ovn: -1}, "<=", 0)

            # z >= x1 + x2 - 1
            var_dict = {ovn: -1 for ovn in old_var_names}
            var_dict[new_var.name] = 1
            self._dst.linear_constraint(var_dict, ">=", 1 - len(old_var_names))  # type: ignore

        return self._dst

    def interpret(self, x: np.ndarray | list[float]) -> np.ndarray:
        """Convert a solution of the converted problem back to a solution of the original problem."""
        if self._dst is None:
            raise OptimizationError("You need to convert before interpreting.")
        return np.array(x[: self._dst.get_num_vars() - len(self._new_vars)])

    @staticmethod
    def get_compatibility_msg(problem: OptimizationProblem) -> str:
        """Checks whether the given problem is compatible with the conversion.

        A problem is compatible if all variables are binary.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            A message describing the incompatibility, if any, or an empty string.
        """
        # initialize message
        msg = ""
        # check whether there are incompatible variable types
        if problem.get_num_continuous_vars() > 0:
            msg += "Continuous variables are not supported. "
        if problem.get_num_spin_vars() > 0:
            msg += "Spin variables are not supported. "
        if problem.get_num_integer_vars() > 0:
            msg += "Integer variables are not supported. "

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

    # ---- private methods ----

    def _update_vars_and_lin_dict(
        self, lin_dict: defaultdict[str, float], orig_vars: list[str], coef
    ):
        # remove duplicates while keeping order
        orig_vars_tuple = tuple(dict.fromkeys(orig_vars))

        # add variable to self._new_vars if needed
        new_var_name = (
            self._new_vars[orig_vars_tuple].name if len(orig_vars_tuple) > 1 else orig_vars_tuple[0]
        )

        # update lin_dict
        lin_dict[new_var_name] += coef

    def _convert_expr(
        self,
        linear: LinearExpression,
        quadratic: QuadraticExpression | None,
        ho: dict[int, HigherOrderExpression] | None,
    ) -> defaultdict[str, float]:
        lin_dict: defaultdict[str, float] = defaultdict(float)
        lin_dict.update(linear.to_dict(use_name=True))  # type: ignore

        if quadratic is not None:
            for orig_vars_q, coef in quadratic.to_dict(use_name=True).items():
                self._update_vars_and_lin_dict(lin_dict, orig_vars_q, coef)  # type: ignore
        if ho is not None:
            for hoe in ho.values():
                for orig_vars_h, coef in hoe.to_dict(use_name=True).items():
                    self._update_vars_and_lin_dict(lin_dict, orig_vars_h, coef)  # type: ignore

        return lin_dict
