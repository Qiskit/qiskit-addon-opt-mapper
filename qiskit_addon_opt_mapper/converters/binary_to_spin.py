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

"""Converter that converts all binary variables to spin variables."""

import copy

import numpy as np

from ..exceptions import OptimizationError
from ..problems.optimization_objective import OptimizationObjective
from ..problems.optimization_problem import OptimizationProblem
from ..problems.variable import Variable
from .optimization_problem_converter import OptimizationProblemConverter
from .util import (
    Monomial,
    Poly,
    _norm,
    _poly_add,
    _poly_from_higher,
    _poly_from_linear,
    _poly_from_quadratic,
    _poly_split,
    _Subst,
)


class BinaryToSpin(OptimizationProblemConverter):
    """Convert all binary variables in the problem to spin variables.

    The conversion is done by the relation
    b_i = (1 - s_i)/2
    where b_i is a binary variable and s_i is a spin variable.
    """

    _delimiter = "@"

    def __init__(self) -> None:
        """Initialize converter."""
        self._src: OptimizationProblem | None = None
        self._dst: OptimizationProblem | None = None
        self._b2s: dict[str, str] = {}  # original binary name -> new spin name
        self._subst: dict[str, _Subst] = {}  # name -> (const, coeff, spin_name)
        self._src_num_vars: int | None = None

    # ---- public API ----

    def convert(self, problem: OptimizationProblem) -> OptimizationProblem:
        """Convert all binary variables in the problem to spin variables."""
        self._src = copy.deepcopy(problem)
        self._src_num_vars = problem.get_num_vars()
        self._dst = OptimizationProblem(name=problem.name)

        # 1) variables: create spins for binaries, copy others as-is
        for x in self._src.variables:
            if x.vartype == Variable.Type.BINARY:
                s_name = f"{x.name}{self._delimiter}spin"
                self._dst._add_variable(
                    name=s_name,
                    vartype=Variable.Type.SPIN,
                    lowerbound=-1,
                    upperbound=1,
                    internal=True,
                )
                self._b2s[x.name] = s_name
                self._subst[x.name] = _Subst(const=0.5, coeff=-0.5, var=s_name)
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
                raise OptimizationError(f"Unsupported vartype: {x.vartype}")

        # 2) objective: substitute and expand
        self._convert_objective()

        # 3) constraints: substitute and add with proper order
        self._convert_linear_constraints()
        self._convert_quadratic_constraints()
        self._convert_higher_order_constraints()

        return self._dst

    def interpret(self, x: np.ndarray | list[float]) -> np.ndarray:
        """Convert a solution of the converted problem back to a solution of the original problem.

        For binaries that became spins, we use b = (1 - s)/2.
        """
        assert self._dst is not None and self._src_num_vars and self._src is not None
        # Build a name->value map from dst solution order
        if len(x) != self._dst.get_num_vars():
            raise OptimizationError("Result length does not match converted problem.")
        dst_vals: dict[str, float] = {}
        for i, var in enumerate(self._dst.variables):
            dst_vals[var.name] = float(x[i])

        # Compose original vector order
        out = np.zeros(self._src_num_vars)
        for i, var in enumerate(self._src.variables):
            if var.vartype == Variable.Type.BINARY:
                s = dst_vals[self._b2s[var.name]]
                out[i] = (1.0 - s) / 2.0
            else:
                out[i] = dst_vals[var.name]
        return out

    # ---- private methods ----

    def _apply_b2s_subst(self, f: Poly) -> Poly:
        """Apply b = 0.5 - 0.5 s to every variable in polynomial f.

        Where b is binary variable and s is spin variable.
        """
        out: Poly = {}
        for m, coef in f.items():
            # Build options for each factor in monomial
            factors: list[list[tuple[Monomial, float]]] = []
            zero_flag = False
            for name in m:
                subst = self._subst.get(name, _Subst(const=0.0, coeff=1.0, var=name))
                opts: list[tuple[Monomial, float]] = []
                if subst.const != 0.0:
                    opts.append(((), subst.const))
                if subst.var is not None and subst.coeff != 0.0:
                    opts.append(((subst.var,), subst.coeff))
                if not opts:
                    zero_flag = True
                    break
                factors.append(opts)
            if zero_flag:
                continue

            # Convolution
            monoms: list[tuple[Monomial, float]] = [((), 1.0)]
            for opts in factors:
                nxt: list[tuple[Monomial, float]] = []
                for m0, c0 in monoms:
                    for m1, c1 in opts:
                        nxt.append((_norm(m0 + m1), c0 * c1))
                monoms = nxt

            for mm, cc in monoms:
                out[mm] = out.get(mm, 0.0) + coef * cc

        return out

    def _convert_objective(self) -> None:
        """Convert the objective of the source problem and set it to the destination problem."""
        assert self._dst is not None and self._src is not None
        obj = self._src.objective
        # Build polynomial f from original objective
        f: Poly = {}
        _poly_add(f, _poly_from_linear(obj.linear))
        _poly_add(f, _poly_from_quadratic(obj.quadratic))
        # higher order (if any)
        if obj.higher_order:
            # obj.higher_order: dict[int, Expr], each Expr.to_dict(use_name=True)
            for _, expr in obj.higher_order.items():
                for names, coef in expr.to_dict(use_name=True).items():
                    if coef != 0.0:
                        _poly_add(f, {tuple(names): float(coef)})  # type: ignore

        # Apply substitution
        g = self._apply_b2s_subst(f)
        c0, ldict, qdict, hdict = _poly_split(g)
        c0 += obj.constant

        # Dump to dst objective
        if obj.sense == OptimizationObjective.Sense.MINIMIZE:
            self._dst.minimize(c0, ldict, qdict, hdict)
        else:
            self._dst.maximize(c0, ldict, qdict, hdict)

    def _emit_constraint_from_poly(self, name: str, sense, rhs: float, poly: Poly) -> None:
        """Emit a constraint to the destination problem from a polynomial form."""
        assert self._dst is not None
        c0, ldict, qdict, hdict = _poly_split(poly)
        rhs2 = rhs - c0
        if hdict:
            self._dst.higher_order_constraint(ldict, qdict, hdict, sense, rhs2, name)
        elif qdict:
            self._dst.quadratic_constraint(ldict, qdict, sense, rhs2, name)
        elif ldict:
            self._dst.linear_constraint(ldict, sense, rhs2, name)

    def _convert_linear_constraints(self) -> None:
        """Convert linear constraints of the source problem.

        Add them to the destination problem.
        """
        assert self._src is not None
        for c in self._src.linear_constraints:
            f: Poly = _poly_from_linear(c.linear)
            g = self._apply_b2s_subst(f)
            self._emit_constraint_from_poly(c.name, c.sense, c.rhs, g)

    def _convert_quadratic_constraints(self) -> None:
        """Convert quadratic constraints of the source problem.

        Add them to the destination problem.
        """
        assert self._src is not None
        for c in self._src.quadratic_constraints:
            f: Poly = {}
            _poly_add(f, _poly_from_linear(c.linear))
            _poly_add(f, _poly_from_quadratic(c.quadratic))
            g = self._apply_b2s_subst(f)
            self._emit_constraint_from_poly(c.name, c.sense, c.rhs, g)

    def _convert_higher_order_constraints(self) -> None:
        """Convert higher-order constraints of the source problem.

        Add them to the destination problem.
        """
        for c in getattr(self._src, "higher_order_constraints", []):
            f: Poly = {}
            _poly_add(f, _poly_from_linear(c.linear))
            _poly_add(f, _poly_from_quadratic(c.quadratic))
            _poly_add(f, _poly_from_higher(c.higher_order))
            g = self._apply_b2s_subst(f)
            self._emit_constraint_from_poly(c.name, c.sense, c.rhs, g)
