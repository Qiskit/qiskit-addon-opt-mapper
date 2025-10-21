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
"""Converter that converts all spin variables to binary variables."""

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


class SpinToBinary(OptimizationProblemConverter):
    """Convert all spin variables in the problem to binary variables.

    The conversion is done by the relation::

        s_i = 1 - 2 b_i

    where s_i is a spin variable (in {-1, +1}) and b_i is a binary variable (in {0, 1}).
    """

    _delimiter = "@"

    def __init__(self) -> None:
        """Class initializer."""
        self._src: OptimizationProblem | None = None
        self._dst: OptimizationProblem | None = None
        self._s2b: dict[str, str] = {}  # original spin name -> new binary name
        self._subst: dict[str, _Subst] = {}  # name -> (const, coeff, bin_name)
        self._src_num_vars: int | None = None

    # ---- public API ----

    def convert(self, problem: OptimizationProblem) -> OptimizationProblem:
        """Convert all spin variables in the problem to binary variables."""
        self._src = copy.deepcopy(problem)
        self._src_num_vars = problem.get_num_vars()
        self._dst = OptimizationProblem(name=problem.name)

        # 1) Variables: create binaries for spins; copy others as-is
        for x in self._src.variables:
            if x.vartype == Variable.Type.SPIN:
                b_name = f"{x.name}{self._delimiter}bin"
                self._dst._add_variable(
                    name=b_name,
                    vartype=Variable.Type.BINARY,
                    lowerbound=0,
                    upperbound=1,
                    internal=True,
                )
                self._s2b[x.name] = b_name
                # s = 1 - 2 b  →  const=1.0, coeff=-2.0, var=b_name
                self._subst[x.name] = _Subst(const=1.0, coeff=-2.0, var=b_name)
            elif x.vartype == Variable.Type.BINARY:
                self._dst._add_variable(
                    name=x.name,
                    vartype=Variable.Type.BINARY,
                    lowerbound=0,
                    upperbound=1,
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
            else:
                raise OptimizationError(f"Unsupported vartype: {x.vartype}")

        # 2) Objective: substitute & expand
        self._convert_objective()

        # 3) Constraints: substitute & add in the right order bucket
        self._convert_linear_constraints()
        self._convert_quadratic_constraints()
        self._convert_higher_order_constraints()

        return self._dst

    def interpret(self, x: np.ndarray | list[float]) -> np.ndarray:
        """Convert a solution of the converted (binary) problem back to the original (spin) space.

        For spins we use s = 1 - 2 b.
        """
        if len(x) != self._dst.get_num_vars():  # type: ignore[union-attr]
            raise OptimizationError("Result length does not match converted problem.")

        dst_vals: dict[str, float] = {}
        for i, var in enumerate(self._dst.variables):  # type: ignore[union-attr]
            dst_vals[var.name] = float(x[i])

        out = np.zeros(self._src_num_vars)  # type: ignore[arg-type]
        for i, var in enumerate(self._src.variables):  # type: ignore[union-attr]
            if var.vartype == Variable.Type.SPIN:
                b = dst_vals[self._s2b[var.name]]
                out[i] = 1.0 - 2.0 * b
            else:
                out[i] = dst_vals[var.name]
        return out

    # ---- private methods ----

    def _apply_s2b_subst(self, f: Poly) -> Poly:
        """Apply s = 1 - 2 b to every variable in polynomial f.

        Where s is spin variable and b is binary variable.
        """
        out: Poly = {}
        for m, coef in f.items():
            # Build olist[s for each factor in monomial
            factors: list[list[tuple[Monomial, float]]] = []
            zero_flag = False
            for name in m:
                # default: identity for non-spin variables
                subst = self._subst.get(name, _Subst(const=0.0, coeff=1.0, var=name))
                opts: list[tuple[Monomial, float]] = []
                if subst.const != 0.0:
                    opts.append(((), subst.const))  # constant part
                if subst.var is not None and subst.coeff != 0.0:
                    opts.append(((subst.var,), subst.coeff))  # variable part
                if not opts:
                    zero_flag = True
                    break
                factors.append(opts)
            if zero_flag:
                continue

            # Convolution (distributive expansion)
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
        """Convert the objective of the source problem.

        Set it to the destination problem.
        """
        assert self._src is not None and self._dst is not None
        obj = self._src.objective

        # Build polynomial from original objective
        f: Poly = {}
        _poly_add(f, _poly_from_linear(obj.linear))
        _poly_add(f, _poly_from_quadratic(obj.quadratic))
        if obj.higher_order:
            for _, expr in obj.higher_order.items():
                for names, coef in expr.to_dict(use_name=True).items():
                    if coef != 0.0:
                        _poly_add(f, {tuple(names): float(coef)})  # type: ignore[arg-type]

        # Apply substitution
        g = self._apply_s2b_subst(f)
        c0, ldict, qdict, hdict = _poly_split(g)
        c0 += obj.constant

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
        else:
            # Constant-only; keep it explicit as 0 sense rhs2
            self._dst.linear_constraint({}, sense, rhs2, name)

    def _convert_linear_constraints(self) -> None:
        """Convert linear constraints of the source problem.

        Add them to the destination problem.
        """
        for c in self._src.linear_constraints:  # type: ignore[union-attr]
            f: Poly = _poly_from_linear(c.linear)
            g = self._apply_s2b_subst(f)
            self._emit_constraint_from_poly(c.name, c.sense, c.rhs, g)

    def _convert_quadratic_constraints(self) -> None:
        """Convert quadratic constraints of the source problem.

        Add them to the destination problem.
        """
        for c in self._src.quadratic_constraints:  # type: ignore[union-attr]
            f: Poly = {}
            _poly_add(f, _poly_from_linear(c.linear))
            _poly_add(f, _poly_from_quadratic(c.quadratic))
            g = self._apply_s2b_subst(f)
            self._emit_constraint_from_poly(c.name, c.sense, c.rhs, g)

    def _convert_higher_order_constraints(self) -> None:
        """Convert higher-order constraints of the source problem.

        Add them to the destination problem.
        """
        for c in getattr(self._src, "higher_order_constraints", []):  # type: ignore[union-attr]
            f: Poly = {}
            _poly_add(f, _poly_from_linear(c.linear))
            _poly_add(f, _poly_from_quadratic(c.quadratic))
            _poly_add(f, _poly_from_higher(c.higher_order))
            g = self._apply_s2b_subst(f)
            self._emit_constraint_from_poly(c.name, c.sense, c.rhs, g)
