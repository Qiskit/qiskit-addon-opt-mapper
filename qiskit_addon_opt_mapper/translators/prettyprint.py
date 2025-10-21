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
"""Translate ``OptimizationProblem`` into a pretty-printed string."""

from io import StringIO
from math import isclose
from typing import cast

import numpy as np

from qiskit_addon_opt_mapper import INFINITY, OptimizationError
from qiskit_addon_opt_mapper.problems import (
    HigherOrderExpression,
    LinearExpression,
    OptimizationObjective,
    OptimizationProblem,
    QuadraticExpression,
    VarType,
)

DEFAULT_TRUNCATE = 50


def _int_if_close(val: int | float | np.integer | np.floating) -> int | float:
    """Convert a value into an integer if possible.

    Note: if abs(val) is too large, int(val) is not correct
          e.g., int(1e16 - 1) -> 10000000000000000
    """
    if isinstance(val, np.integer):
        val = int(val)
    elif isinstance(val, np.floating):
        val = float(val)

    if isinstance(val, int):
        return val
    if abs(val) <= 1e10 and val.is_integer():
        return int(val)
    return val


def _term2str(coeff: float, term: str, is_head: bool) -> str:
    """Translate a pair of a coefficient and a term to a string.

    Args:
        coeff: a coefficient.
        term: a term. This can be empty and `coeff` is treated as a constant.
        is_head: Whether this coefficient appears in the head of the string or not.

    Returns:
        A strings representing the term.
    """
    if term:
        if is_head:
            if isclose(coeff, 1.0):
                ret = term
            elif isclose(coeff, -1.0):
                ret = f"-{term}"
            else:
                ret = f"{_int_if_close(coeff)}*{term}"
        else:
            sign = "-" if coeff < 0.0 else "+"
            abs_val = abs(coeff)
            if isclose(abs_val, 1.0):
                ret = f"{sign} {term}"
            else:
                ret = f"{sign} {_int_if_close(abs_val)}*{term}"
    else:
        if is_head:
            ret = f"{_int_if_close(coeff)}"
        else:
            sign = "-" if coeff < 0.0 else "+"
            abs_val = abs(coeff)
            ret = f"{sign} {_int_if_close(abs_val)}"
    return ret


def _check_name(name: str, name_type: str) -> None:
    """Check a name is printable or not.

    Args:
        name: a variable name.
        name_type: the type associated with the name.

    Raises:
        OptimizationError: if the name is not printable.
    """
    if not name.isprintable():
        raise OptimizationError(f"{name_type} name is not printable: {name!r}")


def _concatenate_terms(terms: list[str], wrap: int, indent: int) -> str:
    ind = " " * indent
    if wrap == 0:
        return ind + " ".join(terms)
    buf = ind
    cur = indent
    for term in terms:
        if cur + len(term) >= wrap:
            buf += "\n"
            buf += ind
            cur = indent
        if cur != indent:  # if the position is not the start of the line
            buf += " "
            cur += 1
        buf += term
        cur += len(term)
    return buf


def expr2str(
    constant: float = 0.0,
    linear: LinearExpression | None = None,
    quadratic: QuadraticExpression | None = None,
    higher_order: (HigherOrderExpression | dict[int, HigherOrderExpression] | None) = None,
    truncate: int = 0,
    suffix: str = "",
    wrap: int = 0,
    indent: int = 0,
) -> str:
    """Translate constant/linear/quadratic/higher-order into a string."""
    if truncate < 0:
        raise ValueError(f"Invalid truncate value: {truncate}")

    terms: list[str] = []
    is_head = True

    lin_dict = linear.to_dict(use_name=True) if linear else {}
    quad_dict = quadratic.to_dict(use_name=True) if quadratic else {}

    # --- collect higher-order terms into {k: {tuple[str,...]: coeff}} ---
    ho_by_k: dict[int, dict[tuple[str, ...], float]] = {}
    if higher_order is not None:
        for k, expr in higher_order.items():  # type: ignore
            ho_by_k[k] = expr.to_dict(use_name=True)  # type: ignore

    # --- higher-order (descending by order k) ---
    for k in sorted(ho_by_k.keys(), reverse=True):
        for vars_tuple, coeff in sorted(ho_by_k[k].items()):
            mono = _monomial_tuple_to_str(cast(tuple[str, ...], vars_tuple))
            terms.append(_term2str(float(coeff), mono, is_head))
            is_head = False

    # --- quadratic ---
    for (var1, var2), coeff in sorted(quad_dict.items()):
        _check_name(cast(str, var1), "Variable")
        _check_name(cast(str, var2), "Variable")
        if var1 == var2:
            terms.append(_term2str(coeff, f"{var1}^2", is_head))
        else:
            terms.append(_term2str(coeff, f"{var1}*{var2}", is_head))
        is_head = False

    # --- linear ---
    for var, coeff in sorted(lin_dict.items()):
        _check_name(cast(str, var), "Variable")
        terms.append(_term2str(coeff, f"{var}", is_head))
        is_head = False

    # --- constant ---
    if not isclose(constant, 0.0, abs_tol=1e-10):
        terms.append(_term2str(constant, "", is_head))
    elif not lin_dict and not quad_dict and not ho_by_k:
        terms.append(_term2str(0, "", is_head))

    # --- suffix ---
    if suffix:
        terms.append(suffix)

    ret = _concatenate_terms(terms, wrap, indent)
    if 0 < truncate < len(ret):
        ret = ret[:truncate] + "..."
    return ret


def _monomial_tuple_to_str(vars_tuple: tuple[str, ...]) -> str:
    """Turn a tuple like ('x0','x0','x2','x3','x3','x3') into 'x0^2*x2*x3^3'."""
    if not vars_tuple:
        return ""
    # count multiplicities while keeping deterministic order
    counts: dict[str, int] = {}
    for v in vars_tuple:
        _check_name(v, "Variable")
        counts[v] = counts.get(v, 0) + 1
    # stable order by variable name
    parts = []
    for name in sorted(counts.keys()):
        pow_ = counts[name]
        if pow_ == 1:
            parts.append(name)
        else:
            parts.append(f"{name}^{pow_}")
    return "*".join(parts)


def prettyprint(optimization_problem: OptimizationProblem, wrap: int = 80) -> str:
    """Translate an OptimizationProblem into a pretty-printed string (higher-order aware)."""
    with StringIO() as buf:
        _check_name(optimization_problem.name, "Problem")
        buf.write(f"Problem name: {optimization_problem.name}\n\n")

        if optimization_problem.objective.sense == OptimizationObjective.Sense.MINIMIZE:
            buf.write("Minimize\n")
        else:
            buf.write("Maximize\n")

        # --- Objective (support higher_orders / higher_order) ---
        obj = optimization_problem.objective

        # prefer multiple blocks if available
        buf.write(
            expr2str(
                obj.constant,
                obj.linear,
                obj.quadratic,
                obj.higher_order,
                wrap=wrap,
                indent=2,
            )
        )

        # --- Constraints header ---
        buf.write("\n\nSubject to")
        num_lin_csts = optimization_problem.get_num_linear_constraints()
        num_quad_csts = optimization_problem.get_num_quadratic_constraints()

        # higher-order constraints API is optional: support both attribute and getter styles
        num_higher_csts = optimization_problem.get_num_higher_order_constraints()

        if num_lin_csts == 0 and num_quad_csts == 0 and num_higher_csts == 0:
            buf.write("\n  No constraints\n")

        # --- Linear constraints ---
        if num_lin_csts > 0:
            buf.write(f"\n  Linear constraints ({num_lin_csts})\n")
            for cst in optimization_problem.linear_constraints:
                _check_name(cst.name, "Linear constraint")
                suffix = f"{cst.sense.label} {_int_if_close(cst.rhs)}  '{cst.name}'\n"
                buf.write(expr2str(linear=cst.linear, suffix=suffix, wrap=wrap, indent=4))
        if num_quad_csts > 0:
            buf.write(f"\n  Quadratic constraints ({num_quad_csts})\n")
            for cst2 in optimization_problem.quadratic_constraints:
                _check_name(cst2.name, "Quadratic constraint")
                suffix = f"{cst2.sense.label} {_int_if_close(cst2.rhs)}  '{cst2.name}'\n"
                buf.write(
                    expr2str(
                        linear=cst2.linear,
                        quadratic=cst2.quadratic,
                        suffix=suffix,
                        wrap=wrap,
                        indent=4,
                    )
                )

        # --- Higher-order constraints (k>=3), multiple blocks supported ---
        if num_higher_csts > 0:
            buf.write(f"\n  Higher-order constraints ({num_higher_csts})\n")

            for csth in optimization_problem.higher_order_constraints:
                _check_name(csth.name, "Higher-order constraint")
                suffix = f"{csth.sense.label} {_int_if_close(csth.rhs)}  '{csth.name}'\n"

                buf.write(
                    expr2str(
                        linear=csth.linear,
                        quadratic=csth.quadratic,
                        higher_order=csth.higher_order,
                        suffix=suffix,
                        wrap=wrap,
                        indent=4,
                    )
                )

        # --- Variables section (unchanged) ---
        if optimization_problem.get_num_vars() == 0:
            buf.write("\n  No variables\n")
        bin_vars: list[str] = []
        int_vars = []
        con_vars = []
        spin_vars = []
        for var in optimization_problem.variables:
            if var.vartype is VarType.BINARY:
                _check_name(var.name, "Variable")
                bin_vars.append(var.name)
            elif var.vartype is VarType.INTEGER:
                int_vars.append(var)
            elif var.vartype is VarType.CONTINUOUS:
                con_vars.append(var)
            else:
                spin_vars.append(var)
        if int_vars:
            buf.write(f"\n  Integer variables ({len(int_vars)})\n")
            for var in int_vars:
                buf.write("    ")
                if var.lowerbound > -INFINITY:
                    buf.write(f"{_int_if_close(var.lowerbound)} <= ")
                _check_name(var.name, "Variable")
                buf.write(var.name)
                if var.upperbound < INFINITY:
                    buf.write(f" <= {_int_if_close(var.upperbound)}")
                buf.write("\n")

        if con_vars:
            buf.write(f"\n  Continuous variables ({len(con_vars)})\n")
            for var in con_vars:
                buf.write("    ")
                if var.lowerbound > -INFINITY:
                    buf.write(f"{_int_if_close(var.lowerbound)} <= ")
                _check_name(var.name, "Variable")
                buf.write(var.name)
                if var.upperbound < INFINITY:
                    buf.write(f" <= {_int_if_close(var.upperbound)}")
                buf.write("\n")

        if bin_vars:
            buf.write(f"\n  Binary variables ({len(bin_vars)})\n")
            buf.write(_concatenate_terms(bin_vars, wrap=wrap, indent=4))
            buf.write("\n")

        if spin_vars:
            buf.write(f"\n  Spin variables ({len(spin_vars)})\n")
            for var in spin_vars:
                buf.write("    ")
                if var.lowerbound > -INFINITY:
                    buf.write(f"{_int_if_close(var.lowerbound)} <= ")
                _check_name(var.name, "Variable")
                buf.write(var.name)
                if var.upperbound < INFINITY:
                    buf.write(f" <= {_int_if_close(var.upperbound)}")
                buf.write("\n")

        ret = buf.getvalue()
    return ret
