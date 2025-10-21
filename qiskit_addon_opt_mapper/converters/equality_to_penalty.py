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

"""Converter to convert a problem with equality constraints to unconstrained with penalty terms."""

import copy
import logging
from collections import defaultdict
from collections.abc import Mapping
from typing import cast

import numpy as np

from ..exceptions import OptimizationError
from ..problems.constraint import Constraint
from ..problems.optimization_objective import OptimizationObjective
from ..problems.optimization_problem import OptimizationProblem
from ..problems.variable import Variable
from .optimization_problem_converter import OptimizationProblemConverter

logger = logging.getLogger(__name__)

Monomial = tuple[str, ...]  # ()=const, ('x',)=linear, ('x','y')=quadratic, etc.
Poly = dict[Monomial, float]


class EqualityToPenalty(OptimizationProblemConverter):
    """Convert a problem with only equality constraints to unconstrained with penalty terms."""

    def __init__(self, penalty: float | None = None) -> None:
        """Init method.

        Args:
            penalty: Penalty factor to scale equality constraints that are added to objective.
                     If None is passed, a penalty factor will be automatically calculated on
                     every conversion.
                     The penalty factor is calculated as follows:
                        1 + (upperbound - lowerbound) of objective function.

        """
        self._src_num_vars: int | None = None
        self._penalty: float | None = penalty
        self._should_define_penalty: bool = penalty is None
        self._src = OptimizationProblem()
        self._dst = OptimizationProblem()

    def convert(self, problem: OptimizationProblem) -> OptimizationProblem:
        """Convert a problem with equality constraints into an unconstrained problem.

        Args:
            problem: The problem to be solved, that does not contain inequality constraints.


        Returns:
            The converted problem, that is an unconstrained problem.

        Raises:
            OptimizationError: If an inequality constraint exists.
        """
        self._src = copy.deepcopy(problem)
        self._src_num_vars = problem.get_num_vars()
        self._dst = OptimizationProblem()

        for var in problem.variables:
            if var.vartype == Variable.Type.BINARY:
                self._dst._add_variable(
                    name=var.name,
                    vartype=Variable.Type.BINARY,
                    lowerbound=var.lowerbound,
                    upperbound=var.upperbound,
                    internal=True,
                )
            elif var.vartype == Variable.Type.INTEGER:
                self._dst._add_variable(
                    name=var.name,
                    vartype=Variable.Type.INTEGER,
                    lowerbound=var.lowerbound,
                    upperbound=var.upperbound,
                    internal=True,
                )
            elif var.vartype == Variable.Type.CONTINUOUS:
                self._dst._add_variable(
                    name=var.name,
                    vartype=Variable.Type.CONTINUOUS,
                    lowerbound=var.lowerbound,
                    upperbound=var.upperbound,
                    internal=True,
                )
            elif var.vartype == Variable.Type.SPIN:
                self._dst._add_variable(
                    name=var.name,
                    vartype=Variable.Type.SPIN,
                    lowerbound=-1,
                    upperbound=1,
                    internal=True,
                )
            else:
                raise OptimizationError(f"Unknown variable type: {var.vartype}")

        if self._should_define_penalty:
            penalty = self._auto_define_penalty(problem)
            logger.info("Automatically defined penalty factor: %f", penalty)
        else:
            penalty = cast(float, self._penalty)
            if penalty <= 0:
                raise OptimizationError("Penalty factor must be positive.")
            logger.info("Using user-defined penalty factor: %f", penalty)

        offset = problem.objective.constant
        linear = problem.objective.linear.to_dict(use_name=True)
        quadratic = problem.objective.quadratic.to_dict(use_name=True)
        ho = {d: expr.to_dict(use_name=True) for d, expr in problem.objective.higher_order.items()}
        sense = 1 if problem.objective.sense == OptimizationObjective.Sense.MINIMIZE else -1

        def add_poly_to_objective(poly: Poly, scale: float):
            """Add `scale * poly` into (offset, linear, quadratic, ho).

            Poly may include any degree.
            """
            # Poly may have constant term
            const, linc, quadc, hoc = _poly_split_by_degree(poly)
            nonlocal offset
            offset += sense * scale * const
            for x, c in linc.items():
                if c:
                    linear[x] = linear.get(x, 0.0) + sense * scale * c
            for (i, j), c in quadc.items():
                if c:
                    tup = (i, j)
                    quadratic[tup] = quadratic.get(tup, 0.0) + sense * scale * c
            for d, terms in hoc.items():
                tgt = ho.setdefault(d, {})
                for names, c in terms.items():
                    if c:
                        tgt[names] = tgt.get(names, 0.0) + sense * scale * c

        # --- Handle each equality constraint (linear, quadratic, higher order) ---
        def handle_eq_constraint(constraint: Constraint):
            # Check sense of constraint
            if con.sense != Constraint.Sense.EQ:
                raise OptimizationError(
                    "An inequality constraint exists. Only equality constraints are supported."
                )

            c = float(constraint.rhs)
            # Build f(x) = linear + quadratic + higher
            f: Poly = {}
            _poly_add(f, _poly_from_constraint_linear(constraint))
            if getattr(constraint, "quadratic", None):
                _poly_add(f, _poly_from_constraint_quadratic(constraint))
            if getattr(constraint, "higher_order", None):
                _poly_add(f, _poly_from_constraint_higher(constraint))

            # (1) +P * c^2  (Constant)
            offset_nonlocal = c * c
            nonlocal offset
            offset += sense * penalty * offset_nonlocal

            # (2) -2Pc * f(x)  (Linear terms)
            add_poly_to_objective(f, scale=(-2.0 * penalty * c))

            # (3) +P * f(x)^2  (Quadratic terms)
            f2 = _poly_mul(f, f)
            add_poly_to_objective(f2, scale=penalty)

        # Check if any inequality constraint exists
        for con in problem.linear_constraints:
            handle_eq_constraint(con)

        for con in problem.quadratic_constraints:  # type: ignore
            handle_eq_constraint(con)  # type: ignore

        for con in problem.higher_order_constraints:  # type: ignore
            handle_eq_constraint(con)  # type: ignore

        if problem.objective.sense == OptimizationObjective.Sense.MINIMIZE:
            self._dst.minimize(offset, linear, quadratic, ho)  # type: ignore
        else:
            self._dst.maximize(offset, linear, quadratic, ho)  # type: ignore

        # Update the penalty to the one just used
        self._penalty = penalty

        return self._dst

    @staticmethod
    def _auto_define_penalty(problem: OptimizationProblem) -> float:
        """Automatically define the penalty coefficient.

        Returns:
            Return the minimum valid penalty factor calculated
            from the upper bound and the lower bound of the objective function.
            If a constraint has a float coefficient,
            return the default value for the penalty factor.
        """
        default_penalty = 1e5

        # Check coefficients of constraints.
        # If a constraint has a float coefficient, return the default value for the penalty factor.
        terms = []
        for constraint in problem.linear_constraints:
            terms.append(constraint.rhs)
            terms.extend(constraint.linear.to_array().tolist())
        if any(isinstance(term, float) and not term.is_integer() for term in terms):
            logger.warning(
                "Warning: Using %f for the penalty coefficient because "
                "a float coefficient exists in constraints. \n"
                "The value could be too small. "
                "If so, set the penalty coefficient manually.",
                default_penalty,
            )
            return default_penalty

        lin_b = problem.objective.linear.bounds
        quad_b = problem.objective.quadratic.bounds
        ho_b = [ho_exp.bounds for ho_exp in problem.objective.higher_order.values()]
        return (
            1.0
            + (lin_b.upperbound - lin_b.lowerbound)
            + (quad_b.upperbound - quad_b.lowerbound)
            + sum((b.upperbound - b.lowerbound) for b in ho_b)
        )

    def interpret(self, x: np.ndarray | list[float]) -> np.ndarray:
        """Convert the result of the converted problem back to that of the original problem.

        Args:
            x: The result of the converted problem or the given result in case of FAILURE.


        Returns:
            The result of the original problem.

        Raises:
            OptimizationError: if the number of variables in the result differs from
                                     that of the original problem.
        """
        if len(x) != self._src_num_vars:
            raise OptimizationError(
                "The number of variables in the passed result differs from "
                "that of the original problem."
            )
        return np.asarray(x)

    @property
    def penalty(self) -> float | None:
        """Returns the penalty factor used in conversion.

        Returns:
            The penalty factor used in conversion.
        """
        return self._penalty

    @penalty.setter
    def penalty(self, penalty: float | None) -> None:
        """Set a new penalty factor.

        Args:
            penalty: The new penalty factor.
                     If None is passed, a penalty factor will be automatically calculated
                     on every conversion.
        """
        self._penalty = penalty
        self._should_define_penalty = penalty is None


def _normalize_monomial(m: Monomial) -> Monomial:
    """Normalize monomial key for symmetric terms.

    - Sort variable names for a canonical key: ('y','x')->('x','y')
    - Keep duplicates as-is; we do NOT enforce z^2=z (no multilinear reduction here).
    """
    return tuple(sorted(m))


def _poly_add(dst: Poly, src: Mapping[Monomial, float], scale: float = 1.0) -> None:
    for m, c in src.items():
        if c:
            mm = _normalize_monomial(m)
            dst[mm] = dst.get(mm, 0.0) + scale * c


def _poly_mul(a: Poly, b: Poly) -> Poly:
    """Convolution of two sparse polynomials represented as {monomial: coeff}."""
    out: Poly = {}
    for m1, c1 in a.items():
        for m2, c2 in b.items():
            m = _normalize_monomial(m1 + m2)
            out[m] = out.get(m, 0.0) + c1 * c2
    return out


def _poly_from_constraint_linear(constraint) -> Poly:
    """Build f(x) from a *linear* constraint part: sum_j a_j x_j."""
    row = constraint.linear.to_dict(use_name=True)
    return {(name,): coef for name, coef in row.items() if coef != 0.0}


def _poly_from_constraint_quadratic(constraint) -> Poly:
    """Build f(x) from *quadratic* part: sum_{ij} q_ij x_i x_j."""
    q = constraint.quadratic.to_dict(use_name=True)
    poly: Poly = {}
    for (i, j), coef in q.items():
        if coef != 0.0:
            m = _normalize_monomial((i, j))
            poly[m] = poly.get(m, 0.0) + coef
    return poly


def _poly_from_constraint_higher(constraint) -> Poly:
    """Build f(x) from higher-order parts: sum_{deg>=3} sum_{S} h_S prod_{v in S} v."""
    poly: Poly = {}
    # constraint.higher_order: dict[int, Expr]; each Expr has to_dict(use_name=True)
    for _deg, expr in getattr(constraint, "higher_order", {}).items():
        terms = expr.to_dict(use_name=True)  # dict[tuple[str,...], float]
        for names, coef in terms.items():
            if coef != 0.0:
                m = _normalize_monomial(tuple(names))
                poly[m] = poly.get(m, 0.0) + coef
    return poly


def _poly_split_by_degree(poly: Poly):
    """Split poly into constant/linear/quadratic/higher dicts for objective accumulation."""
    const = poly.get((), 0.0)
    linear: dict[str, float] = {}
    quadratic: dict[tuple[str, str], float] = {}
    higher: dict[int, dict[tuple[str, ...], float]] = defaultdict(dict)
    for m, c in poly.items():
        if not m:  # ()
            continue
        d = len(m)
        if d == 1:
            (x,) = m
            linear[x] = linear.get(x, 0.0) + c
        elif d == 2:
            quadratic[(m[0], m[1])] = quadratic.get((m[0], m[1]), 0.0) + c
        else:
            hd = higher[d]
            hd[m] = hd.get(m, 0.0) + c
    return const, linear, quadratic, dict(higher)
