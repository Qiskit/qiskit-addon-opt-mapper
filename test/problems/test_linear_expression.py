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

"""Tests for LinearExpression class."""

import numpy as np
import pytest
from qiskit_addon_opt_mapper.exceptions import OptimizationError
from qiskit_addon_opt_mapper.problems import LinearExpression, OptimizationProblem
from scipy.sparse import dok_matrix


def MAKE_INSTANCE(coeffs, nvars=4):
    """Helper to create LinearExpression instance with a fake OptimizationProblem."""
    op = OptimizationProblem()
    op.binary_var_list(range(nvars))
    return LinearExpression(optimization_problem=op, coefficients=coeffs)


# ---------- happy path ----------
@pytest.mark.parametrize("as_array", [list, np.array])
def test_accepts_1d_and_correct_length_for_dense(as_array):
    n = 4
    coeffs = as_array([1.0, 0.0, -2.5, 3.0])
    expr = MAKE_INSTANCE(coeffs, nvars=n)
    assert isinstance(expr.coefficients, dok_matrix)
    assert expr.coefficients.shape == (1, n)


def test_accepts_sparse_row_vector_correct_length():
    n = 4
    m = dok_matrix((1, n))
    m[0, 0] = 1.0
    m[0, 2] = -2.5
    expr = MAKE_INSTANCE(m, nvars=n)
    assert isinstance(expr.coefficients, dok_matrix)
    assert expr.coefficients.shape == (1, n)


# ---------- error cases ----------
@pytest.mark.parametrize(
    "bad_dense",
    [
        [1.0, 2.0, 3.0],  # list, short
        np.array([1.0, 2.0, 3.0]),  # ndarray, short
        [1, 2, 3, 4, 5],  # list, long
        np.array([1, 2, 3, 4, 5]),  # ndarray, long
    ],
)
def test_rejects_wrong_length_for_dense(bad_dense):
    n = 4
    with pytest.raises(OptimizationError) as e:
        MAKE_INSTANCE(bad_dense, nvars=n)
    msg = str(e.value).lower()
    assert "one-dimensional" in msg or "row vector" in msg
    assert "length" in msg and "match" in msg


def test_rejects_ndarray_not_1d():
    n = 4
    bad = np.array([[1, 2], [3, 4]])  # 2D
    with pytest.raises(OptimizationError) as e:
        MAKE_INSTANCE(bad, nvars=n)
    assert "row vector" in str(e.value).lower()


def test_dot_evaluation_consistency():
    n = 4
    coeffs = [1, 2, 3, 4]
    expr = MAKE_INSTANCE(coeffs, nvars=n)

    x = np.array([[1, 1, 1, 1]])  # shape: (1, n)
    expected = sum(coeffs)  # 1+2+3+4=10
    assert expr.evaluate(x[0]) == expected
