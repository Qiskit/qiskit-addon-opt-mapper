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

"""Tests for QuadraticExpression class."""

import numpy as np
import pytest
from qiskit_addon_opt_mapper.exceptions import OptimizationError
from qiskit_addon_opt_mapper.problems import OptimizationProblem, QuadraticExpression
from scipy.sparse import csc_matrix, csr_matrix, dok_matrix


def MAKE_INSTANCE(coeffs, nvars=4):
    """Helper to create LinearExpression instance with a fake OptimizationProblem."""
    op = OptimizationProblem()
    op.binary_var_list(range(nvars))
    return QuadraticExpression(optimization_problem=op, coefficients=coeffs)


# ---------------- happy path ----------------
def test_result_is_dok_matrix_and_square():
    n = 2
    dense = [[1.0, 2.0], [2.0, 5.0]]
    expr = MAKE_INSTANCE(dense, nvars=n)
    assert isinstance(expr.coefficients, dok_matrix)
    assert expr.coefficients.shape == (n, n)


def test_accepts_list_of_lists_n_by_n_and_normalizes_to_dok():
    n = 4
    dense = [
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 4.5, 0.0],
        [2.0, 0.0, 0.0, 5.0],
    ]
    expr = MAKE_INSTANCE(dense, nvars=n)
    assert isinstance(expr.coefficients, dok_matrix)
    assert expr.coefficients.shape == (n, n)
    # Coefficients are stored in a upper-triangular manner
    assert expr.coefficients[0, 0] == 1.0
    assert expr.coefficients[0, 3] == 4.0
    assert expr.coefficients[2, 2] == 4.5
    assert expr.coefficients[1, 2] == 0


def test_accepts_ndarray_n_by_n_and_normalizes_to_dok():
    n = 3
    arr = np.array([[1, 2, 0], [0, 3, 4], [0, 0, 5]], dtype=float)
    expr = MAKE_INSTANCE(arr, nvars=n)
    assert isinstance(expr.coefficients, dok_matrix)
    assert expr.coefficients.shape == (n, n)
    assert expr.coefficients[0, 1] == 2
    assert expr.coefficients[1, 2] == 4
    assert expr.coefficients[2, 2] == 5


@pytest.mark.parametrize("sparse_fmt", [dok_matrix, csr_matrix, csc_matrix])
def test_accepts_sparse_n_by_n_and_normalizes_to_dok(sparse_fmt):
    n = 3
    sp = sparse_fmt((n, n))
    sp[0, 0] = 1.0
    sp[1, 1] = 2.0
    sp[2, 2] = 3.0

    expr = MAKE_INSTANCE(sp, nvars=n)
    assert isinstance(expr.coefficients, dok_matrix)
    assert expr.coefficients.shape == (n, n)
    assert expr.coefficients[1, 1] == 2.0


# ---------------- error/exception cases ----------------


def test_rejects_list_outer_length_mismatch():
    n = 4
    # outer list length wrong
    bad = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    with pytest.raises(OptimizationError):
        MAKE_INSTANCE(bad, nvars=n)


def test_rejects_list_inner_length_mismatch():
    n = 4
    bad = [
        [0, 0, 0, 0],
        [0, 0, 0],  # short
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    with pytest.raises(OptimizationError):
        MAKE_INSTANCE(bad, nvars=n)


def test_rejects_ndarray_wrong_shape_rectangular():
    n = 4
    bad = np.zeros((n, n + 1))
    with pytest.raises(OptimizationError):
        MAKE_INSTANCE(bad, nvars=n)


def test_rejects_ndarray_not_2d():
    n = 3
    bad = np.array([1, 2, 3])  # 1D
    with pytest.raises(OptimizationError):
        MAKE_INSTANCE(bad, nvars=n)
