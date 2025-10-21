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

"""The inequality to equality converter."""

import copy
import math

import numpy as np

from ..exceptions import OptimizationError
from ..problems.constraint import Constraint
from ..problems.higher_order_constraint import HigherOrderConstraint
from ..problems.linear_constraint import LinearConstraint
from ..problems.optimization_objective import OptimizationObjective
from ..problems.optimization_problem import OptimizationProblem
from ..problems.quadratic_constraint import QuadraticConstraint
from ..problems.variable import Variable
from .optimization_problem_converter import OptimizationProblemConverter


class InequalityToEquality(OptimizationProblemConverter):
    """Convert inequality constraints into equality constraints by introducing slack variables.

    Examples:
        >>> from qiskit_addon_opt_mapper.problems import OptimizationProblem
        >>> from qiskit_addon_opt_mapper.converters import InequalityToEquality
        >>> problem = OptimizationProblem()
        >>> # define a problem
        >>> conv = InequalityToEquality()
        >>> problem2 = conv.convert(problem)
    """

    _delimiter = "@"  # users are supposed not to use this character in variable names

    def __init__(self, mode: str = "auto") -> None:
        """Init method.

        Args:
            mode: To choose the type of slack variables. There are 3 options for mode.

                - 'integer': All slack variables will be integer variables.
                - 'continuous': All slack variables will be continuous variables.
                - 'auto': Use integer variables if possible, otherwise use continuous variables.
        """
        self._src: OptimizationProblem | None = None
        self._dst: OptimizationProblem | None = None
        self._mode = mode

    def convert(self, problem: OptimizationProblem) -> OptimizationProblem:
        """Convert a problem with inequality constraints into one with only equality constraints.

        Args:
            problem: The problem to be solved, that may contain inequality constraints.


        Returns:
            The converted problem, that contain only equality constraints.

        Raises:
            OptimizationError: If a variable type is not supported.
            OptimizationError: If an unsupported mode is selected.
            OptimizationError: If an unsupported sense is specified.
        """
        self._src = copy.deepcopy(problem)
        self._dst = OptimizationProblem(name=problem.name)

        # set a converting mode
        mode = self._mode
        if mode not in ["integer", "continuous", "auto"]:
            raise OptimizationError(f"Unsupported mode is selected: {mode}")

        # Copy variables
        assert self._dst is not None
        for x in self._src.variables:
            if x.vartype == Variable.Type.BINARY:
                self._dst._add_variable(
                    name=x.name,
                    vartype=Variable.Type.BINARY,
                    lowerbound=x.lowerbound,
                    upperbound=x.upperbound,
                    internal=True,
                )
            elif x.vartype == Variable.Type.INTEGER:
                self._dst._add_variable(
                    name=x.name,
                    vartype=Variable.Type.INTEGER,
                    lowerbound=x.lowerbound,
                    upperbound=x.upperbound,
                    internal=True,
                )
            elif x.vartype == Variable.Type.CONTINUOUS:
                self._dst._add_variable(
                    name=x.name,
                    vartype=Variable.Type.CONTINUOUS,
                    lowerbound=x.lowerbound,
                    upperbound=x.upperbound,
                    internal=True,
                )
            elif x.vartype == Variable.Type.SPIN:
                self._dst._add_variable(
                    name=x.name,
                    vartype=Variable.Type.SPIN,
                    lowerbound=-1,
                    upperbound=1,
                    internal=True,
                )
            else:
                raise OptimizationError(f"Unsupported variable type {x.vartype}")

        # Add slack variables to linear constraints
        new_linear_constraints = []
        for lin_const in self._src.linear_constraints:
            if lin_const.sense == Constraint.Sense.EQ:
                new_linear_constraints.append(
                    (
                        lin_const.linear.coefficients,
                        lin_const.sense,
                        lin_const.rhs,
                        lin_const.name,
                    )
                )
            elif lin_const.sense in [Constraint.Sense.LE, Constraint.Sense.GE]:
                new_linear_constraints.append(self._add_slack_var_linear_constraint(lin_const))
            else:
                raise OptimizationError(
                    f"Internal error: type of sense in {lin_const.name} is not supported: "
                    f"{lin_const.sense}"
                )

        # Add slack variables to quadratic constraints
        new_quadratic_constraints = []
        for quad_const in self._src.quadratic_constraints:
            if quad_const.sense == Constraint.Sense.EQ:
                new_quadratic_constraints.append(
                    (
                        quad_const.linear.coefficients,
                        quad_const.quadratic.coefficients,
                        quad_const.sense,
                        quad_const.rhs,
                        quad_const.name,
                    )
                )
            elif quad_const.sense in [Constraint.Sense.LE, Constraint.Sense.GE]:
                new_quadratic_constraints.append(
                    self._add_slack_var_quadratic_constraint(quad_const)
                )
            else:
                raise OptimizationError(
                    f"Internal error: type of sense in {quad_const.name} is not supported: "
                    f"{quad_const.sense}"
                )

        # Add slack variables to higher order constraints
        new_higher_order_constraints = []
        for ho_const in self._src.higher_order_constraints:
            if ho_const.sense == Constraint.Sense.EQ:
                new_higher_order_constraints.append(
                    (
                        ho_const.linear.coefficients,
                        ho_const.quadratic.coefficients,
                        {degree: ho.to_dict() for degree, ho in ho_const.higher_order.items()},
                        ho_const.sense,
                        ho_const.rhs,
                        ho_const.name,
                    )
                )
            elif ho_const.sense in [Constraint.Sense.LE, Constraint.Sense.GE]:
                new_higher_order_constraints.append(
                    self._add_slack_var_higher_order_constraint(ho_const)
                )
            else:
                raise OptimizationError(
                    f"Internal error: type of sense in {ho_const.name} is not supported: "
                    f"{ho_const.sense}"
                )

        # Copy the objective function
        constant = self._src.objective.constant
        linear = self._src.objective.linear.to_dict(use_name=True)
        quadratic = self._src.objective.quadratic.to_dict(use_name=True)

        ho = {
            degree: dict(expr.to_dict(use_name=True))  # Ensure it's a plain dict
            for degree, expr in self._src.objective.higher_order.items()
        }

        if self._src.objective.sense == OptimizationObjective.Sense.MINIMIZE:
            self._dst.minimize(constant, linear, quadratic, ho)  # type: ignore
        else:
            self._dst.maximize(constant, linear, quadratic, ho)  # type: ignore

        # Add linear constraints
        for lin_const_args in new_linear_constraints:
            self._dst.linear_constraint(*lin_const_args)

        # Add quadratic constraints
        for quad_const_args in new_quadratic_constraints:
            self._dst.quadratic_constraint(*quad_const_args)

        for ho_const_args in new_higher_order_constraints:
            self._dst.higher_order_constraint(*ho_const_args)  # type: ignore

        return self._dst

    def _add_slack_var_linear_constraint(self, constraint: LinearConstraint):
        linear = constraint.linear
        sense = constraint.sense
        name = constraint.name

        any_float = self._any_float(linear.to_array())
        mode = self._mode
        if mode == "integer":
            if any_float:
                raise OptimizationError(
                    f'"{name}" contains float coefficients. '
                    'We can not use an integer slack variable for "{name}"'
                )
        elif mode == "auto":
            mode = "continuous" if any_float else "integer"

        new_rhs = constraint.rhs
        if mode == "integer":
            # If rhs is float number, round up/down to the nearest integer.
            if sense == Constraint.Sense.LE:
                new_rhs = math.floor(new_rhs)
            if sense == Constraint.Sense.GE:
                new_rhs = math.ceil(new_rhs)

        lin_bounds = linear.bounds
        lhs_lb = lin_bounds.lowerbound
        lhs_ub = lin_bounds.upperbound

        var_ub = 0.0
        sign = 0
        if sense == Constraint.Sense.LE:
            var_ub = new_rhs - lhs_lb
            if var_ub > 0:
                sign = 1
        elif sense == Constraint.Sense.GE:
            var_ub = lhs_ub - new_rhs
            if var_ub > 0:
                sign = -1

        new_linear = linear.to_dict(use_name=True)
        if var_ub > 0:
            # Add a slack variable.
            mode_name = {"integer": "int", "continuous": "continuous"}
            slack_name = f"{name}{self._delimiter}{mode_name[mode]}_slack"
            assert self._dst is not None
            if mode == "integer":
                self._dst._add_variable(
                    name=slack_name,
                    vartype=Variable.Type.INTEGER,
                    lowerbound=0,
                    upperbound=var_ub,
                    internal=True,
                )
            elif mode == "continuous":
                self._dst._add_variable(
                    name=slack_name,
                    vartype=Variable.Type.CONTINUOUS,
                    lowerbound=0,
                    upperbound=var_ub,
                    internal=True,
                )
            new_linear[slack_name] = sign
        return new_linear, "==", new_rhs, name

    def _add_slack_var_quadratic_constraint(self, constraint: QuadraticConstraint):
        quadratic = constraint.quadratic
        linear = constraint.linear
        sense = constraint.sense
        name = constraint.name

        any_float = self._any_float(linear.to_array()) or self._any_float(quadratic.to_array())
        mode = self._mode
        if mode == "integer":
            if any_float:
                raise OptimizationError(
                    f'"{name}" contains float coefficients. '
                    'We can not use an integer slack variable for "{name}"'
                )
        elif mode == "auto":
            mode = "continuous" if any_float else "integer"

        new_rhs = constraint.rhs
        if mode == "integer":
            # If rhs is float number, round up/down to the nearest integer.
            if sense == Constraint.Sense.LE:
                new_rhs = math.floor(new_rhs)
            if sense == Constraint.Sense.GE:
                new_rhs = math.ceil(new_rhs)

        lin_bounds = linear.bounds
        quad_bounds = quadratic.bounds
        lhs_lb = lin_bounds.lowerbound + quad_bounds.lowerbound
        lhs_ub = lin_bounds.upperbound + quad_bounds.upperbound

        var_ub = 0.0
        sign = 0
        if sense == Constraint.Sense.LE:
            var_ub = new_rhs - lhs_lb
            if var_ub > 0:
                sign = 1
        elif sense == Constraint.Sense.GE:
            var_ub = lhs_ub - new_rhs
            if var_ub > 0:
                sign = -1

        new_linear = linear.to_dict(use_name=True)
        if var_ub > 0:
            # Add a slack variable.
            mode_name = {"integer": "int", "continuous": "continuous"}
            slack_name = f"{name}{self._delimiter}{mode_name[mode]}_slack"
            assert self._dst is not None
            if mode == "integer":
                self._dst._add_variable(
                    name=slack_name,
                    vartype=Variable.Type.INTEGER,
                    lowerbound=0,
                    upperbound=var_ub,
                    internal=True,
                )
            elif mode == "continuous":
                self._dst._add_variable(
                    name=slack_name,
                    vartype=Variable.Type.CONTINUOUS,
                    lowerbound=0,
                    upperbound=var_ub,
                    internal=True,
                )
            new_linear[slack_name] = sign
        return new_linear, quadratic.coefficients, "==", new_rhs, name

    def _add_slack_var_higher_order_constraint(self, constraint: HigherOrderConstraint):
        higher_order = constraint.higher_order
        quadratic = constraint.quadratic
        linear = constraint.linear
        sense = constraint.sense
        name = constraint.name

        any_float = (
            self._any_float(linear.to_array())
            or self._any_float(quadratic.to_array())
            or any(self._any_float(ho.to_array()) for ho in higher_order.values())
        )
        mode = self._mode
        if mode == "integer":
            if any_float:
                raise OptimizationError(
                    f'"{name}" contains float coefficients. '
                    'We can not use an integer slack variable for "{name}"'
                )
        elif mode == "auto":
            mode = "continuous" if any_float else "integer"

        new_rhs = constraint.rhs
        if mode == "integer":
            # If rhs is float number, round up/down to the nearest integer.
            if sense == Constraint.Sense.LE:
                new_rhs = math.floor(new_rhs)
            if sense == Constraint.Sense.GE:
                new_rhs = math.ceil(new_rhs)

        lin_bounds = linear.bounds
        quad_bounds = quadratic.bounds
        lhs_lb = lin_bounds.lowerbound + quad_bounds.lowerbound
        lhs_ub = lin_bounds.upperbound + quad_bounds.upperbound
        for expr in higher_order.values():
            ho_bounds = expr.bounds
            lhs_lb += ho_bounds.lowerbound
            lhs_ub += ho_bounds.upperbound

        var_ub = 0.0
        sign = 0
        if sense == Constraint.Sense.LE:
            var_ub = new_rhs - lhs_lb
            if var_ub > 0:
                sign = 1
        elif sense == Constraint.Sense.GE:
            var_ub = lhs_ub - new_rhs
            if var_ub > 0:
                sign = -1

        new_linear = linear.to_dict(use_name=True)
        if var_ub > 0:
            # Add a slack variable.
            mode_name = {"integer": "int", "continuous": "continuous"}
            slack_name = f"{name}{self._delimiter}{mode_name[mode]}_slack"
            assert self._dst is not None
            if mode == "integer":
                self._dst._add_variable(
                    name=slack_name,
                    vartype=Variable.Type.INTEGER,
                    lowerbound=0,
                    upperbound=var_ub,
                    internal=True,
                )
            elif mode == "continuous":
                self._dst._add_variable(
                    name=slack_name,
                    vartype=Variable.Type.CONTINUOUS,
                    lowerbound=0,
                    upperbound=var_ub,
                    internal=True,
                )
            new_linear[slack_name] = sign
        higher_order = {
            degree: expr.to_dict(use_name=True)  # type: ignore
            for degree, expr in higher_order.items()  # type: ignore
        }
        return new_linear, quadratic.coefficients, higher_order, "==", new_rhs, name

    def interpret(self, x: np.ndarray | list[float]) -> np.ndarray:
        """Convert a result of a converted problem into that of the original problem.

        Args:
            x: The result of the converted problem or the given result in case of FAILURE.


        Returns:
            The result of the original problem.
        """
        # convert back the optimization result into that of the original problem
        assert self._dst is not None
        names = [var.name for var in self._dst.variables]

        # interpret slack variables
        assert self._src is not None
        sol = {name: x[i] for i, name in enumerate(names)}
        new_x = np.zeros(self._src.get_num_vars())
        for i, var in enumerate(self._src.variables):
            new_x[i] = sol[var.name]
        return new_x

    @staticmethod
    def _any_float(values: np.ndarray) -> bool:
        """Check whether the list contains float or not.

        This method is used to check whether a constraint contain float coefficients or not.

        Args:
            values: Coefficients of the constraint


        Returns:
            bool: If the constraint contains float coefficients, this returns True, else False.
        """
        return any(isinstance(v, float) and not v.is_integer() for v in values)

    @property
    def mode(self) -> str:
        """Returns the mode of the converter.

        Returns:
            The mode of the converter used for additional slack variables
        """
        return self._mode

    @mode.setter
    def mode(self, mode: str) -> None:
        """Set a new mode for the converter.

        Args:
            mode: The new mode for the converter
        """
        self._mode = mode
