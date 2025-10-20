# This code is part of a Qiskit project.
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

"""The converter to map integer variables in a optimization problem to binary variables."""

import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..exceptions import OptimizationError
from ..problems.optimization_objective import OptimizationObjective
from ..problems.optimization_problem import OptimizationProblem
from ..problems.variable import Variable
from .optimization_problem_converter import OptimizationProblemConverter


class IntegerToBinary(OptimizationProblemConverter):
    """Convert a :class:`~qiskit_addon_opt_mapper.problems.OptimizationProblem`
    into new one by encoding integers with binary variables.

    This bounded-coefficient encoding used in this converted is proposed in [1], Eq. (5).

    Examples:
        >>> from qiskit_addon_opt_mapper.problems import OptimizationProblem
        >>> from qiskit_addon_opt_mapper.converters import IntegerToBinary
        >>> problem = OptimizationProblem()
        >>> var = problem.integer_var(name='x', lowerbound=0, upperbound=10)
        >>> conv = IntegerToBinary()
        >>> problem2 = conv.convert(problem)

    References:
        [1]: Sahar Karimi, Pooya Ronagh (2017), Practical Integer-to-Binary Mapping for Quantum
            Annealers. arxiv.org:1706.01945.
    """

    _delimiter = "@"  # users are supposed not to use this character in variable names

    def __init__(self) -> None:
        self._src: Optional[OptimizationProblem] = None
        self._dst: Optional[OptimizationProblem] = None
        self._conv: Dict[Variable, List[Tuple[str, int]]] = {}
        # e.g., self._conv = {x: [('x@1', 1), ('x@2', 2)]}

    def convert(self, problem: OptimizationProblem) -> OptimizationProblem:
        """Convert an integer problem into a new problem with binary variables.

        Args:
            problem: The problem to be solved, that may contain integer variables.

        Returns:
            The converted problem, that contains no integer variables.

        Raises:
            OptimizationError: if variable or constraint type is not supported.
        """

        # Copy original QP as reference.
        self._src = copy.deepcopy(problem)

        if self._src.get_num_integer_vars() > 0:

            # Initialize new QP
            self._dst = OptimizationProblem(name=problem.name)

            # Declare variables
            for x in self._src.variables:
                if x.vartype == Variable.Type.INTEGER:
                    new_vars = self._convert_var(x.name, x.lowerbound, x.upperbound)
                    self._conv[x] = new_vars
                    for var_name, _ in new_vars:
                        self._dst._add_variable(
                            name=var_name,
                            vartype=Variable.Type.BINARY,
                            lowerbound=0,
                            upperbound=1,
                            internal=True,
                        )
                else:
                    if x.vartype == Variable.Type.CONTINUOUS:
                        self._dst._add_variable(
                            name=x.name,
                            vartype=Variable.Type.CONTINUOUS,
                            lowerbound=x.lowerbound,
                            upperbound=x.upperbound,
                            internal=True,
                        )
                    elif x.vartype == Variable.Type.BINARY:
                        self._dst._add_variable(
                            name=x.name,
                            vartype=Variable.Type.BINARY,
                            lowerbound=0,
                            upperbound=1,
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

            self._substitute_int_var()

        else:
            # just copy the problem if no integer variables exist
            self._dst = copy.deepcopy(problem)

        return self._dst

    def _convert_var(
        self, name: str, lowerbound: float, upperbound: float
    ) -> List[Tuple[str, int]]:
        var_range = upperbound - lowerbound
        power = int(np.log2(var_range)) if var_range > 0 else 0
        bounded_coef = var_range - (2**power - 1)

        coeffs = [2**i for i in range(power)] + [bounded_coef]
        return [(name + self._delimiter + str(i), coef) for i, coef in enumerate(coeffs)]

    def _convert_linear_coefficients_dict(
        self, coefficients: Dict[str, float]
    ) -> Tuple[Dict[str, float], float]:
        constant = 0.0
        linear: Dict[str, float] = {}
        for name, v in coefficients.items():
            x = self._src.get_variable(name)
            if x in self._conv:
                for y, coeff in self._conv[x]:
                    linear[y] = v * coeff
                constant += v * x.lowerbound
            else:
                linear[x.name] = v

        return linear, constant

    def _convert_quadratic_coefficients_dict(
        self, coefficients: Dict[Tuple[str, str], float]
    ) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float], float]:
        constant = 0.0
        linear: Dict[str, float] = {}
        quadratic = {}
        for (name_i, name_j), v in coefficients.items():
            x = self._src.get_variable(name_i)
            y = self._src.get_variable(name_j)

            if x in self._conv and y not in self._conv:
                for z_x, coeff_x in self._conv[x]:
                    quadratic[z_x, y.name] = v * coeff_x
                linear[y.name] = linear.get(y.name, 0.0) + v * x.lowerbound

            elif x not in self._conv and y in self._conv:
                for z_y, coeff_y in self._conv[y]:
                    quadratic[x.name, z_y] = v * coeff_y
                linear[x.name] = linear.get(x.name, 0.0) + v * y.lowerbound

            elif x in self._conv and y in self._conv:
                for z_x, coeff_x in self._conv[x]:
                    for z_y, coeff_y in self._conv[y]:
                        quadratic[z_x, z_y] = v * coeff_x * coeff_y

                for z_x, coeff_x in self._conv[x]:
                    linear[z_x] = linear.get(z_x, 0.0) + v * coeff_x * y.lowerbound
                for z_y, coeff_y in self._conv[y]:
                    linear[z_y] = linear.get(z_y, 0.0) + v * coeff_y * x.lowerbound

                constant += v * x.lowerbound * y.lowerbound

            else:
                quadratic[x.name, y.name] = v

        return quadratic, linear, constant

    def _convert_higher_order_coefficients_dict(
        self, coefficients: Dict[int, Dict[Tuple[str, ...], float]]
    ) -> Tuple[Dict[int, Dict[Tuple[str, ...], float]], Dict[str, float], float]:
        """
        Expand higher-order terms containing integer variables into binary/constant/linear terms.

        Each integer variable x is represented as:
            x = lowerbound_x + sum_i coeff_i * z_i
        where z_i are newly introduced binary variables.

        We expand the polynomial by convolution across all variables in 'names'.
        """

        constant = 0.0
        linear: Dict[str, float] = {}
        higher_order: Dict[int, Dict[Tuple[str, ...], float]] = {}

        def add_to_degree(d: int, key: Tuple[str, ...], val: float):
            """Helper: add coefficient 'val' to the monomial 'key' at degree 'd'."""
            if d not in higher_order:
                higher_order[d] = {}
            higher_order[d][key] = higher_order[d].get(key, 0.0) + val

        for _degree, terms in coefficients.items():
            for names, v in terms.to_dict(use_name=True).items():
                # Build candidate monomials for each variable in 'names'
                # Each entry is a list of (vars_tuple, coeff)
                factor_options: List[List[Tuple[Tuple[str, ...], float]]] = []
                for name in names:
                    var = self._src.get_variable(name)
                    if var in self._conv:
                        # Integer variable: represented as (lb) + sum(coeff * z)
                        opts: List[Tuple[Tuple[str, ...], float]] = []
                        if var.lowerbound != 0:
                            opts.append(((), float(var.lowerbound)))  # constant shift
                        for z_name, coeff in self._conv[var]:
                            opts.append(((z_name,), float(coeff)))
                        factor_options.append(opts)
                    else:
                        # Non-integer variable: remains as-is
                        factor_options.append([((var.name,), 1.0)])

                # Convolution: combine all options across factors
                monomials: List[Tuple[Tuple[str, ...], float]] = [((), 1.0)]
                for opts in factor_options:
                    next_monomials: List[Tuple[Tuple[str, ...], float]] = []
                    for vars_so_far, coef_so_far in monomials:
                        for vars_new, coef_new in opts:
                            next_monomials.append((vars_so_far + vars_new, coef_so_far * coef_new))
                    monomials = next_monomials

                # Assign expanded terms to constant / linear / higher-order
                for vars_tuple, coef in monomials:
                    total_coef = v * coef
                    if not vars_tuple:
                        constant += total_coef
                    elif len(vars_tuple) == 1:
                        name = vars_tuple[0]
                        linear[name] = linear.get(name, 0.0) + total_coef
                    else:
                        d = len(vars_tuple)
                        add_to_degree(d, vars_tuple, total_coef)

        return higher_order, linear, constant

    def _substitute_int_var(self):

        # set objective
        linear, linear_constant = self._convert_linear_coefficients_dict(
            self._src.objective.linear.to_dict(use_name=True)
        )
        (
            quadratic,
            q_linear,
            q_constant,
        ) = self._convert_quadratic_coefficients_dict(
            self._src.objective.quadratic.to_dict(use_name=True)
        )
        ho, ho_linear, ho_constant = self._convert_higher_order_coefficients_dict(
            self._src.objective.higher_order
        )

        constant = self._src.objective.constant + linear_constant + q_constant + ho_constant
        for i, v in q_linear.items():
            linear[i] = linear.get(i, 0) + v

        for i, v in ho_linear.items():
            linear[i] = linear.get(i, 0) + v

        if 2 in ho:
            for i, v in ho[2].items():
                if (i[0], i[1]) in quadratic:
                    quadratic[(i[0], i[1])] += v
                elif (i[1], i[0]) in quadratic:
                    quadratic[(i[1], i[0])] += v
                else:
                    quadratic[(i[0], i[1])] = v
            ho.pop(2)

        if self._src.objective.sense == OptimizationObjective.Sense.MINIMIZE:
            self._dst.minimize(constant, linear, quadratic, ho)
        else:
            self._dst.maximize(constant, linear, quadratic, ho)

        # set linear constraints
        for constraint in self._src.linear_constraints:
            linear, constant = self._convert_linear_coefficients_dict(
                constraint.linear.to_dict(use_name=True)
            )
            self._dst.linear_constraint(
                linear, constraint.sense, constraint.rhs - constant, constraint.name
            )

        # set quadratic constraints
        for constraint in self._src.quadratic_constraints:
            linear, linear_constant = self._convert_linear_coefficients_dict(
                constraint.linear.to_dict(use_name=True)
            )
            quadratic, q_linear, q_constant = self._convert_quadratic_coefficients_dict(
                constraint.quadratic.to_dict(use_name=True)
            )

            constant = linear_constant + q_constant
            for i, v in q_linear.items():
                linear[i] = linear.get(i, 0) + v

            self._dst.quadratic_constraint(
                linear,
                quadratic,
                constraint.sense,
                constraint.rhs - constant,
                constraint.name,
            )

        # set higher-order constraints
        for constraint in self._src.higher_order_constraints:
            linear, linear_constant = self._convert_linear_coefficients_dict(
                constraint.linear.to_dict(use_name=True)
            )
            quadratic, q_linear, q_constant = self._convert_quadratic_coefficients_dict(
                constraint.quadratic.to_dict(use_name=True)
            )
            ho, ho_linear, ho_constant = self._convert_higher_order_coefficients_dict(
                constraint.higher_order
            )

            constant = linear_constant + q_constant + ho_constant
            for i, v in q_linear.items():
                linear[i] = linear.get(i, 0) + v
            for i, v in ho_linear.items():
                linear[i] = linear.get(i, 0) + v

            if 2 in ho:
                for i, v in ho[2].items():
                    if (i[0], i[1]) in quadratic:
                        quadratic[(i[0], i[1])] += v
                    elif (i[1], i[0]) in quadratic:
                        quadratic[(i[1], i[0])] += v
                    else:
                        quadratic[(i[0], i[1])] = v
                ho.pop(2)

            self._dst.higher_order_constraint(
                linear,
                quadratic,
                ho,
                constraint.sense,
                constraint.rhs - constant,
                constraint.name,
            )

    def interpret(self, x: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Convert back the converted problem (binary variables)
        to the original (integer variables).

        Args:
            x: The result of the converted problem or the given result in case of FAILURE.

        Returns:
            The result of the original problem.
        """
        # interpret integer values
        sol = {var.name: x[i] for i, var in enumerate(self._dst.variables)}
        new_x = np.zeros(self._src.get_num_vars())
        for i, var in enumerate(self._src.variables):
            if var in self._conv:
                new_x[i] = sum(sol[aux] * coef for aux, coef in self._conv[var]) + var.lowerbound
            else:
                new_x[i] = sol[var.name]
        return np.array(new_x)
