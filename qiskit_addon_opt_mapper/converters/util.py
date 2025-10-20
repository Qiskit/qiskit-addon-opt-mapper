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

"""Utility functions for converters."""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

# We use sparse polynomial representation {monomial: coefficient} throughout the converters.
# Monomials are tuples of variable names, e.g., ('x', 'y') for xy term.
# The empty tuple () represents the constant term.
Monomial = tuple[str, ...]  # ()=const, ('x',)=linear, ('x','y')=quadratic, ('x','y','z')=cubic, ...
# Poly represents a polynomial as a mapping from monomials to coefficients.
Poly = dict[Monomial, float]


def _norm(m: Monomial) -> Monomial:
    """Normalize a monomial by sorting variable names. E.g., ('y','x','y') -> ('x','y','y')."""
    return tuple(sorted(m))


def _poly_add(dst: Poly, src: Mapping[Monomial, float], scale: float = 1.0) -> None:
    """Add a scaled polynomial `scale * src` to `dst` in place."""
    for m, c in src.items():
        if c:
            mm = _norm(m)
            dst[mm] = dst.get(mm, 0.0) + scale * c


def _poly_mul(a: Poly, b: Poly) -> Poly:
    """Multiply two polynomials and return the result."""
    out: Poly = {}
    for m1, c1 in a.items():
        for m2, c2 in b.items():
            m = _norm(m1 + m2)
            out[m] = out.get(m, 0.0) + c1 * c2
    return out


def _poly_split(poly: Poly):
    """Split a polynomial into constant, linear, quadratic, and higher-order parts.

    Poly -> (const, linear{name:coef}, quadratic{(i,j):coef}, higher{deg: {monomial:coef}}).
    """
    const = poly.get((), 0.0)
    lin: dict[str, float] = {}
    quad: dict[tuple[str, str], float] = {}
    higher: dict[int, dict[tuple[str, ...], float]] = defaultdict(dict)
    for m, c in poly.items():
        if not m:
            continue
        d = len(m)
        if d == 1:
            x = m[0]
            lin[x] = lin.get(x, 0.0) + c
        elif d == 2:
            key = (m[0], m[1])
            quad[key] = quad.get(key, 0.0) + c
        else:
            hd = higher[d]
            hd[m] = hd.get(m, 0.0) + c
    return const, lin, quad, dict(higher)


def _poly_from_linear(lin_expr) -> Poly:
    """Convert a linear expression to a polynomial."""
    row = lin_expr.to_dict(use_name=True)
    return {(str(name),): float(coef) for name, coef in row.items() if coef != 0.0}


def _poly_from_quadratic(quad_expr) -> Poly:
    """Convert a quadratic expression to a polynomial."""
    poly: Poly = {}
    for (i, j), coef in quad_expr.to_dict(use_name=True).items():
        if coef != 0.0:
            m = _norm((str(i), str(j)))
            poly[m] = poly.get(m, 0.0) + float(coef)
    return poly


def _poly_from_higher(higher_map) -> Poly:
    """Convert a higher-order expression to a polynomial."""
    poly: Poly = {}
    for _, expr in higher_map.items():
        for names, coef in expr.to_dict(use_name=True).items():
            if coef != 0.0:
                m = _norm(tuple(str(n) for n in names))
                poly[m] = poly.get(m, 0.0) + float(coef)
    return poly


# Substitution of a variable with a polynomial.
# Used in SpinToBinary and BinaryToSpin converters.
# Represents x = coeff * var + const, where var can be None for pure constant substitution.
# For example, in SpinToBinary, x is a spin variable, var is a binary variable,
# coeff=2, const=-1, representing x = 2*var - 1
# In BinaryToSpin, coeff=0.5, const=0.5, representing x = 0.5*var + 0.5 where var is a
# spin variable and var is a spin variable.
@dataclass
class _Subst:
    const: float
    coeff: float
    var: str | None  # None if pure constant
