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

"""Substitute variables of OptimizationProblem."""

import logging
from dataclasses import dataclass
from math import isclose

from ..converters.util import (
    Monomial,
    Poly,
    _norm,
    _poly_add,
    _poly_from_higher,
    _poly_from_linear,
    _poly_from_quadratic,
    _poly_split,
)
from ..exceptions import OptimizationError
from ..infinity import INFINITY
from .constraint import ConstraintSense
from .linear_expression import LinearExpression
from .optimization_problem import OptimizationProblem
from .quadratic_expression import QuadraticExpression

logger = logging.getLogger(__name__)


@dataclass
class SubstitutionExpression:
    """Represents a substitution of a variable with a linear expression.

    If ``variable`` is ``None``, it substitutes a variable with the constant value.
    Otherwise, it substitutes a variable with (``constant + coefficient * new_variable``).
    """

    const: float = 0.0
    """Constant value"""
    coeff: float = 0.0
    """Coefficient of the new variable"""
    variable: str | None = None
    """Variable name or `None`"""


def substitute_variables(
    optimization_problem: OptimizationProblem,
    constants: dict[str | int, float] | None = None,
    variables: dict[str | int, tuple[str | int, float]] | None = None,
) -> OptimizationProblem:
    """Substitutes variables with constants or other variables.

    Args:
        optimization_problem: a optimization problem whose variables are substituted.

        constants: replace variable by constant
            e.g., ``{'x': 2}`` means ``x`` is substituted with 2

        variables: replace variables by weighted other variable
            need to copy everything using name reference to make sure that indices are matched
            correctly. The lower and upper bounds are updated accordingly.
            e.g., ``{'x': ('y', 2)}`` means ``x`` is substituted with ``y`` * 2

    Returns:
        An optimization problem by substituting variables with constants or other variables.
        If the substitution is valid, ``OptimizationProblem.status`` is still
        ``OptimizationProblem.Status.VALID``.
        Otherwise, it gets ``OptimizationProblem.Status.INFEASIBLE``.

    Raises:
        OptimizationError: if the substitution is invalid as follows.

            - Same variable is substituted multiple times.
            - Coefficient of variable substitution is zero.
    """
    # guarantee that there is no overlap between variables to be replaced and combine input
    subs = {}
    if constants:
        for i, v in constants.items():
            # substitute i <- v
            i_2 = optimization_problem.get_variable(i).name
            if i_2 in subs:
                raise OptimizationError(f"Cannot substitute the same variable twice: {i} <- {v}")
            subs[i_2] = SubstitutionExpression(const=v)

    if variables:
        for i, (j, v) in variables.items():
            if v == 0:
                raise OptimizationError(f"coefficient must be non-zero: {i} {j} {v}")
            # substitute i <- j * v
            i_2 = optimization_problem.get_variable(i).name
            j_2 = optimization_problem.get_variable(j).name
            if i_2 == j_2:
                raise OptimizationError(f"Cannot substitute the same variable: {i} <- {j} {v}")
            if i_2 in subs:
                raise OptimizationError(
                    f"Cannot substitute the same variable twice: {i} <- {j} {v}"
                )
            if j_2 in subs:
                raise OptimizationError(
                    f"Cannot substitute by variable that gets substituted itself: {i} <- {j} {v}"
                )
            subs[i_2] = SubstitutionExpression(variable=j_2, coeff=v)

    return _SubstituteVariables().substitute_variables(optimization_problem, subs)


class _SubstituteVariables:
    """A class to substitute variables of an optimization problem with constants."""

    def __init__(self) -> None:
        self._src: OptimizationProblem | None = None
        self._dst: OptimizationProblem | None = None
        self._subs: dict[str, SubstitutionExpression] = {}

    def substitute_variables(
        self,
        optimization_problem: OptimizationProblem,
        subs: dict[str, SubstitutionExpression],
    ) -> OptimizationProblem:
        """Substitutes variables with constants or other variables.

        Args:
            optimization_problem: a optimization problem whose variables are substituted.

            subs: substitution expressions as a dictionary.
                e.g., {'x': SubstitutionExpression(const=1, coeff=2, variable='y'} means
                `x` is substituted with `1 + 2 * y`.


        Returns:
            An optimization problem by substituting variables with constants or other variables.
            If the substitution is valid, `OptimizationProblem.status` is still
            `OptimizationProblem.Status.VALID`.
            Otherwise, it gets `OptimizationProblem.Status.INFEASIBLE`.
        """
        self._src = optimization_problem
        self._dst = OptimizationProblem(optimization_problem.name)
        self._subs = subs
        results = [
            self._variables(),
            self._objective(),
            self._linear_constraints(),
            self._quadratic_constraints(),
            self._higher_order_constraints(),
        ]
        if not all(results):
            self._dst._status = OptimizationProblem.Status.INFEASIBLE
        return self._dst

    @staticmethod
    def _feasible(sense: ConstraintSense, rhs: float) -> bool:
        """Checks feasibility of the following condition: 0 `sense` rhs."""
        if sense == ConstraintSense.EQ:
            if rhs == 0:
                return True
        elif sense == ConstraintSense.LE:
            if rhs >= 0:
                return True
        elif sense == ConstraintSense.GE and rhs <= 0:
            return True
        return False

    def _variables(self) -> bool:
        # copy variables that are not replaced
        assert self._src is not None and self._dst is not None
        feasible = True
        for var in self._src.variables:
            name = var.name
            vartype = var.vartype
            lowerbound = var.lowerbound
            upperbound = var.upperbound
            if name not in self._subs:
                self._dst._add_variable(lowerbound, upperbound, vartype, name)

        for i, expr in self._subs.items():
            lb_i = self._src.get_variable(i).lowerbound
            ub_i = self._src.get_variable(i).upperbound
            # substitute x_i <- x_j * coeff + const
            # lb_i <= x_i <= ub_i  -->
            #   (lb_i - const) / coeff <=  x_j  <= (ub_i - const) / coeff    if coeff > 0
            #   (ub_i - const) / coeff <=  x_j  <= (lb_i - const) / coeff    if coeff < 0
            #                     lb_i <= const <= ub_i                      if coeff == 0
            if isclose(expr.coeff, 0.0, abs_tol=1e-10):
                if not lb_i <= expr.const <= ub_i:
                    logger.warning("Infeasible substitution for variable: %s", i)
                    feasible = False
            else:
                if abs(lb_i) < INFINITY:
                    new_lb_i = (lb_i - expr.const) / expr.coeff
                else:
                    new_lb_i = lb_i if expr.coeff > 0 else -lb_i
                if abs(ub_i) < INFINITY:
                    new_ub_i = (ub_i - expr.const) / expr.coeff
                else:
                    new_ub_i = ub_i if expr.coeff > 0 else -ub_i
                var_j = self._dst.get_variable(expr.variable)  # type: ignore
                lb_j = var_j.lowerbound
                ub_j = var_j.upperbound
                if expr.coeff > 0:
                    var_j.lowerbound = max(lb_j, new_lb_i)
                    var_j.upperbound = min(ub_j, new_ub_i)
                else:
                    var_j.lowerbound = max(lb_j, new_ub_i)
                    var_j.upperbound = min(ub_j, new_lb_i)

        for var in self._dst.variables:
            if var.lowerbound > var.upperbound:
                logger.warning(
                    "Infeasible lower and upper bounds: %s %f %f",
                    var,
                    var.lowerbound,
                    var.upperbound,
                )
                feasible = False

        return feasible

    # ---- unified polynomial substitution (linear + quadratic + higher) ----
    def _poly_apply_substitution(self, lin_expr, quad_expr, higher_map):
        """Apply x_i -> const_i + coeff_i * x_{j(i)} to (linear + quadratic + higher)."""
        # Build original polynomial f
        f: Poly = {}
        _poly_add(f, _poly_from_linear(lin_expr))
        _poly_add(f, _poly_from_quadratic(quad_expr))
        if higher_map:
            _poly_add(f, _poly_from_higher(higher_map))

        # Substitute each monomial
        out: Poly = {}
        for m, coef in f.items():
            # For each variable in monomial, build options: const and/or variable term
            factors: list[list[tuple[Monomial, float]]] = []
            zero_flag = False
            for name in m:
                expr = self._subs.get(
                    name, SubstitutionExpression(const=0.0, coeff=1.0, variable=name)
                )
                opts: list[tuple[Monomial, float]] = []
                if expr.const != 0.0:
                    opts.append(((), expr.const))
                if expr.variable is not None and expr.coeff != 0.0:
                    opts.append(((expr.variable,), expr.coeff))
                if not opts:
                    zero_flag = True
                    break
                factors.append(opts)
            if zero_flag:
                continue

            monoms: list[tuple[Monomial, float]] = [((), 1.0)]
            for opts in factors:
                nxt: list[tuple[Monomial, float]] = []
                for m0, c0 in monoms:
                    for m1, c1 in opts:
                        nxt.append((_norm(m0 + m1), c0 * c1))
                monoms = nxt

            for mm, cc in monoms:
                out[mm] = out.get(mm, 0.0) + coef * cc

        return _poly_split(out)

    def _objective(self) -> bool:
        assert self._src is not None and self._dst is not None
        obj = self._src.objective
        const, lin, quad, higher = self._poly_apply_substitution(
            obj.linear, obj.quadratic, getattr(obj, "higher_order", {})
        )

        constant = obj.constant + const
        lin_expr = LinearExpression(optimization_problem=self._dst, coefficients=lin if lin else {})
        quad_expr = QuadraticExpression(
            optimization_problem=self._dst, coefficients=quad if quad else {}
        )

        if obj.sense == obj.sense.MINIMIZE:
            self._dst.minimize(
                constant=constant,
                linear=lin_expr.coefficients,
                quadratic=quad_expr.coefficients,
                higher_order=higher if higher else {},
            )
        else:
            self._dst.maximize(
                constant=constant,
                linear=lin_expr.coefficients,
                quadratic=quad_expr.coefficients,
                higher_order=higher if higher else {},
            )
        return True

    def _linear_constraints(self) -> bool:
        assert self._src is not None and self._dst is not None
        feasible = True
        for lin_cst in self._src.linear_constraints:
            const, lin, _quad, _higher = self._poly_apply_substitution(
                lin_cst.linear,
                QuadraticExpression(optimization_problem=self._dst, coefficients={}),
                {},
            )

            rhs = lin_cst.rhs - const
            if lin:
                self._dst.linear_constraint(
                    name=lin_cst.name,
                    linear=lin,
                    sense=lin_cst.sense,
                    rhs=rhs,
                )
            else:
                if not self._feasible(lin_cst.sense, rhs):
                    logger.warning("constraint %s is infeasible due to substitution", lin_cst.name)
                    feasible = False
        return feasible

    def _quadratic_constraints(self) -> bool:
        assert self._src is not None and self._dst is not None
        feasible = True
        for quad_cst in self._src.quadratic_constraints:
            const, lin, quad, _higher = self._poly_apply_substitution(
                quad_cst.linear, quad_cst.quadratic, {}
            )
            rhs = quad_cst.rhs - const

            if quad:
                self._dst.quadratic_constraint(
                    name=quad_cst.name,
                    linear=lin if lin else {},
                    quadratic=quad,
                    sense=quad_cst.sense,
                    rhs=rhs,
                )
            elif lin:
                # Demote to linear; avoid name clash
                name = quad_cst.name
                lin_names = {c.name for c in self._dst.linear_constraints}
                while name in lin_names:
                    name = "_" + name
                self._dst.linear_constraint(name=name, linear=lin, sense=quad_cst.sense, rhs=rhs)
            else:
                if not self._feasible(quad_cst.sense, rhs):
                    logger.warning("constraint %s is infeasible due to substitution", quad_cst.name)
                    feasible = False
        return feasible

    def _higher_order_constraints(self) -> bool:
        assert self._src is not None and self._dst is not None
        feasible = True
        for ho_cst in getattr(self._src, "higher_order_constraints", []):
            const, lin, quad, higher = self._poly_apply_substitution(
                ho_cst.linear, ho_cst.quadratic, ho_cst.higher_order
            )
            rhs = ho_cst.rhs - const

            if higher:
                self._dst.higher_order_constraint(
                    name=ho_cst.name,
                    linear=lin if lin else {},
                    quadratic=quad if quad else {},
                    higher_order=higher,
                    sense=ho_cst.sense,
                    rhs=rhs,
                )
            elif quad:
                self._dst.quadratic_constraint(
                    name=ho_cst.name,
                    linear=lin if lin else {},
                    quadratic=quad,
                    sense=ho_cst.sense,
                    rhs=rhs,
                )
            elif lin:
                name = ho_cst.name
                lin_names = {c.name for c in self._dst.linear_constraints}
                while name in lin_names:
                    name = "_" + name
                self._dst.linear_constraint(name=name, linear=lin, sense=ho_cst.sense, rhs=rhs)
            else:
                if not self._feasible(ho_cst.sense, rhs):
                    logger.warning("constraint %s is infeasible due to substitution", ho_cst.name)
                    feasible = False
        return feasible


# ---- small helper to provide an empty quadratic for linear constraints path ----


def _zero_quad(dst_problem: OptimizationProblem) -> QuadraticExpression:
    return QuadraticExpression(optimization_problem=dst_problem, coefficients={})
