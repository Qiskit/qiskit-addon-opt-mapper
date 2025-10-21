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

"""Higher-order (k>=3) expression interface.

This module defines :class:`HigherOrderExpression`, a symmetric k-th order polynomial
container that stores coefficients in a canonical, compressed dictionary form.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
from numpy import ndarray

from ..exceptions import OptimizationError
from ..infinity import INFINITY
from .linear_expression import ExpressionBounds
from .optimization_problem_element import OptimizationProblemElement

Index = int | str
Key = tuple[Index, ...]
_IntKey = tuple[int, ...]


class HigherOrderExpression(OptimizationProblemElement):
    r"""Representation of a symmetric k-th order expression by its coefficients.

    We represent a symmetric polynomial term of order k as
    ``f(x) = sum_{t} C[t] * prod_{i in t} x[i]``,
    where t is a multiset of variable indices of length k.

    When dealing with multidimensional array indices, the value is stored only at the
    lexicographically smallest permutation of the indices (obtained by sorting them in ascending
    order).

    For example, for a 4th-order term ``2⋅x1⋅x2⋅x3⋅x4``, the coefficient "2" is stored at
    ``dict((1, 2, 3, 4))``. Other permutations like ``dict((2, 1, 4, 3))`` or
    ``dict((4, 3, 2, 1))`` are left empty.
    """

    def __init__(
        self,
        optimization_problem: Any,
        coefficients: (ndarray | dict[Key, float] | list),  # nested list acting like ndarray
    ) -> None:
        """Creates a new higher-order expression.

        Args:
            optimization_problem (Any): The parent OptimizationProblem.
            coefficients (ndarray | dict[Key, float] | list): Coefficients as either:
                - A dense ndarray or list with shape (n,)*k, or
                - A dict mapping a tuple of variable indices/names (length k) to float.
                Keys are canonicalized to ascending order and summed.
        """
        super().__init__(optimization_problem)
        self._n = int(self.optimization_problem.get_num_vars())
        self._coeffs: dict[_IntKey, float] = {}
        self.coefficients: dict[_IntKey, float] = coefficients  # via setter

    def __getitem__(self, key: Key) -> float:
        """Returns the coefficient for a given (sorted or unsorted) key."""
        idx = self._normalize_key(key)
        return self._coeffs.get(idx, 0.0)

    def __setitem__(self, key: Key, value: float) -> None:
        """Sets the coefficient for a given (sorted or unsorted) key."""
        idx = self._normalize_key(key)
        if value == 0.0:
            self._coeffs.pop(idx, None)
        else:
            self._coeffs[idx] = float(value)

    @property
    def order(self) -> int:
        """Returns the order of the polynomial (k >= 3)."""
        return self._order

    @property
    def num_variables(self) -> int:
        """Returns the number of variables in this expression."""
        return self._n

    @property
    def coefficients(self) -> dict[tuple[int, ...], float]:
        """Returns a copy of internal (canonical) coefficient dictionary."""
        return dict(self._coeffs)

    @coefficients.setter
    def coefficients(self, coefficients: ndarray | dict[Key, float] | list) -> None:
        """Set coefficients; accepts dict, and dense array/list. Also infers order.

        Args:
            coefficients: Coefficients as:
                - dense ndarray/list with shape (n,)*k, or
                - dict mapping tuple of variable indices/names (len=k) to float.
                Keys are canonicalized to ascending order and summed.
        """
        # --- dict path ---
        acc: dict[_IntKey, float] = {}
        if isinstance(coefficients, dict):
            if len(coefficients) == 0:
                raise ValueError("Cannot infer order from empty dict coefficients.")

            order = None
            for key in coefficients:
                if not isinstance(key, tuple):
                    raise ValueError(f"Dict keys must be tuples, got {type(key).__name__}: {key!r}")
                k = len(key)
                if order is None:
                    order = k
                elif k != order:
                    raise ValueError(
                        f"All dict keys must have the same length (uniform order). "
                        f"Found lengths { {order, k} } (offending key: {key!r})"
                    )
            if order is None or order < 3:
                raise ValueError(f"order must be >= 3, got {order}")
            self._order = order

            for key, v in coefficients.items():
                idx = self._normalize_key(key)
                fv = float(v)
                if fv == 0.0:
                    continue
                acc[idx] = acc.get(idx, 0.0) + fv
            self._coeffs = acc

        # --- ndarray / list ---
        elif isinstance(coefficients, ndarray | list):
            arr = np.array(coefficients, dtype=float)
            if arr.ndim < 3:
                raise ValueError(f"order must be >= 3, got ndim={arr.ndim}")

            # all axes must match the number of variables
            expected = (self._n,) * arr.ndim
            if arr.shape != expected:
                raise ValueError(f"coefficients shape must be {expected}, got {arr.shape}")

            self._order = int(arr.ndim)

            nz = np.argwhere(arr != 0.0)
            for idx_tuple in map(tuple, nz):
                val = float(arr[idx_tuple])
                if val == 0.0:
                    continue
                idx = self._normalize_key(idx_tuple)
                acc[idx] = acc.get(idx, 0.0) + val

            self._coeffs = acc
        else:
            raise TypeError(
                f"Unsupported coefficients type {type(coefficients).__name__}; "
                "expected dict, ndarray, list."
            )

    def to_dict(self, use_name: bool = False) -> dict[tuple[int, ...] | tuple[str, ...], float]:
        """Returns the internal coefficients as a dictionary."""
        if not use_name:
            return dict(self._coeffs)  # type: ignore
        # map index -> name
        return {
            tuple(self.optimization_problem.variables[i].name for i in k): v
            for k, v in self._coeffs.items()
        }

    def to_array(self, symmetric: bool = False) -> ndarray:
        """Returns a dense tensor.

        Args:
            symmetric: If False, returns a tensor with coefficients at the
                lexicographically smallest index only (others zero).
                If True, distributes coefficients equally over all permutations
                of the multiset key.
        """
        shape = (self._n,) * self._order
        arr = np.zeros(shape, dtype=float)
        if symmetric:
            for idx, c in self._coeffs.items():
                arr[idx] += c
            return arr

        for idx, c in self._coeffs.items():
            # count multiplicities
            counts: dict[int, int] = {}
            for i in idx:
                counts[i] = counts.get(i, 0) + 1
            # number of distinct permutations: k! / Π m_r!
            k = self._order
            denom = 1
            for r in counts.values():
                denom *= math.factorial(r)

            weight = c / (math.factorial(k) / denom)
            for perm in set(itertools.permutations(idx)):
                arr[perm] += weight
        return arr

    def evaluate(self, x: ndarray | list | dict[Index, float]) -> float:
        """Evaluate the expression: sum_{t} C[t] * prod_{i in t} x[i].

        Args:
            x: The values of the variables to be evaluated.


        Returns:
            The value of the higher order expression given the variable values.
        """
        x = self._cast_as_array(x)
        val = 0.0
        for idx, c in self._coeffs.items():
            prod = 1.0
            for i in idx:
                prod *= x[i]
            val += c * prod
        return float(val)

    def evaluate_gradient(self, x: ndarray | list | dict[Index, float]) -> ndarray:
        """Evaluate gradient wrt x.

        For each m, g[m] = sum_{t} C[t] * (count_m_in_t) * prod_{i in t / {one m}}.

        Args:
            x: The values of the variables to be evaluated.


        Returns:
            The value of the gradient of the higher order expression given the variable values.
        """
        x = self._cast_as_array(x)
        g = np.zeros(self._n, dtype=float)
        for idx, c in self._coeffs.items():
            # multiplicities of each variable in this term
            counts: dict[int, int] = {}
            for i in idx:
                counts[i] = counts.get(i, 0) + 1

            # total product of this term
            term_prod = 1.0
            for i in idx:
                term_prod *= x[i]

            for m, r in counts.items():
                if x[m] != 0.0:
                    # derivative = c * r * term_prod / x[m]
                    g[m] += c * r * (term_prod / x[m])
                else:
                    # avoid division by zero: recompute product excluding one m
                    # (i.e., multiply others; m appears r times -> remove 1 factor)
                    prod_excl = 1.0
                    removed = 0
                    for i in idx:
                        if i == m and removed < 1:
                            removed += 1
                        else:
                            prod_excl *= x[i]
                    g[m] += c * r * prod_excl
        return g

    @property
    def bounds(self) -> ExpressionBounds:
        """Returns the lower bound and the upper bound of the linear expression.

        Returns:
            The lower bound and the upper bound of the linear expression
        """
        l_b = 0.0
        u_b = 0.0

        # validate bounds
        for idx, c in self._coeffs.items():
            # collect variable bounds with multiplicity
            bounds: list[tuple[float, float]] = []
            for i in idx:
                var = self.optimization_problem.get_variable(i)
                lb, ub = var.lowerbound, var.upperbound
                if lb == -INFINITY or ub == INFINITY:
                    raise OptimizationError(
                        f"Higher-order expression contains an unbounded variable: {var.name}"
                    )
                bounds.append((float(lb), float(ub)))

            # enumerate corners
            products = []
            for choices in itertools.product(*((lb, ub) for (lb, ub) in bounds)):
                prod = 1.0
                for val in choices:
                    prod *= val
                products.append(c * prod)

            l_b += min(products, default=0.0)
            u_b += max(products, default=0.0)

        return ExpressionBounds(lowerbound=l_b, upperbound=u_b)

    def _normalize_key(self, key: Key) -> _IntKey:
        """Normalizes the key to a sorted tuple of integers.

        Args:
            key: A tuple of variable indices or names.


        Returns:
            A sorted tuple of integers representing the indices of the variables.
        """
        idxs: list[int] = []
        for k in key:
            kk = k
            if isinstance(kk, str):
                kk = self.optimization_problem.variables_index[kk]
            kk = int(kk)
            if kk < 0 or kk >= self._n:
                raise ValueError(f"index out of range: {kk}")
            idxs.append(kk)
        return tuple(sorted(idxs))

    def _cast_as_array(self, x: ndarray | list | dict[Index, float]) -> np.ndarray:
        """Casts the input x to a 1D numpy array.

        This method handles different input types and ensures the output is a 1D numpy array.

        Args:
            x: The input variable values, which can be a numpy array, list, or dict.

        Returns:
            A 1D numpy array of variable values.
        """
        if isinstance(x, dict):
            arr = np.zeros(self._n, dtype=float)
            for i, v in x.items():
                ii = self.optimization_problem.variables_index[i] if isinstance(i, str) else int(i)
                arr[ii] = float(v)
            return arr
        arr = np.asarray(x, dtype=float)  # type: ignore
        if arr.ndim != 1 or arr.shape[0] != self._n:
            raise ValueError("x must be a 1D array with length equal to the number of variables")
        return arr

    def __repr__(self):
        """Repr. for higher order expression."""
        # pylint: disable=cyclic-import
        from ..translators.prettyprint import DEFAULT_TRUNCATE, expr2str

        rep_str = f"<{self.__class__.__name__}(k={self._order}):"
        rep_str += f"{expr2str(higher_order={self._order: self}, truncate=DEFAULT_TRUNCATE)}>"
        return rep_str

    def __str__(self):
        """Str. for higher order expression."""
        from ..translators.prettyprint import expr2str

        return f"{expr2str(higher_order=self)}"
