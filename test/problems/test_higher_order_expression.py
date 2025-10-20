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

import math

import numpy as np
import pytest
from qiskit_addon_opt_mapper import INFINITY, OptimizationError
from qiskit_addon_opt_mapper.problems import HigherOrderExpression, OptimizationProblem


@pytest.fixture
def op3():
    op = OptimizationProblem()
    op.integer_var(-10.0, 10.0, "x0")
    op.integer_var(-10.0, 10.0, "x1")
    op.integer_var(-10.0, 10.0, "x2")
    return op


def test_dict_input_and_aggregation(op3):
    expr = HigherOrderExpression(
        op3,
        coefficients={
            (0, 1, 2): 1.5,
            (2, 1, 0): 2.5,
            ("x0", "x0", "x1"): 3.0,  # -> (0,0,1)
        },
    )
    d = expr.to_dict()
    assert d[(0, 1, 2)] == 4.0
    assert d[(0, 0, 1)] == 3.0
    assert expr[(1, 0, 2)] == 4.0
    assert expr[(0, 2, 1)] == 4.0
    assert expr[(0, 0, 2)] == 0.0


def test_evaluate_and_gradient(op3):
    # f(x) = 2*x0^2*x1 + 3*x0*x1*x2 - 5*x1^3
    expr = HigherOrderExpression(
        op3,
        coefficients={
            (0, 0, 1): 2.0,
            (0, 1, 2): 3.0,
            (1, 1, 1): -5.0,
        },
    )
    x = np.array([2.0, 1.0, -1.0])  # x0=2, x1=1, x2=-1

    # 2*(2^2*1) + 3*(2*1*-1) - 5*(1^3) = 8 - 6 - 5 = -3
    assert math.isclose(expr.evaluate(x), -3.0, rel_tol=1e-12, abs_tol=1e-12)

    # ∂/∂x0: 2*2*(x0*x1) + 3*(x1*x2) = 4*(2*1) + 3*(1*-1) = 8 - 3 = 5
    # ∂/∂x1: 2*1*(x0^2) + 3*(x0*x2) - 5*3*(x1^2) = 8 - 6 - 15 = -13
    # ∂/∂x2: 3*(x0*x1) = 6
    g = expr.evaluate_gradient(x)
    assert np.allclose(g, np.array([5.0, -13.0, 6.0]))


def test_bounds_bruteforce():
    op = OptimizationProblem()
    op.integer_var(-1, 3, "x0")
    op.integer_var(2, 5, "x1")

    expr = HigherOrderExpression(op, coefficients={(0, 0, 1): 2.0})
    b = expr.bounds
    # 2*x0^2*x1 → min=-30, max=90
    assert math.isclose(b.lowerbound, -30.0)
    assert math.isclose(b.upperbound, 90.0)


def test_bounds_raises_on_unbounded():
    op = OptimizationProblem()
    op.integer_var(-INFINITY, 1.0, "x0")
    op.integer_var(0.0, 2.0, "x1")
    expr = HigherOrderExpression(op, coefficients={(0, 0, 1): 1.0})
    with pytest.raises(OptimizationError):
        _ = expr.bounds


def test_to_array_distribution(op3):
    expr = HigherOrderExpression(op3, coefficients={(0, 0, 1): 6.0})

    A = expr.to_array(symmetric=True)
    assert A.shape == (op3.get_num_vars(),) * 3
    assert A[0, 0, 1] == 6.0
    assert np.count_nonzero(A) == 1

    B = expr.to_array(symmetric=False)
    # representative_only=False → distribute value equally to all permutations
    # (0,0,1) has 3 permutations: (0,0,1), (0,1,0), (1,0,0)
    # → value 6.0 divided equally → each gets 2.0
    assert B[0, 0, 1] == 2.0
    assert B[0, 1, 0] == 2.0
    assert B[1, 0, 0] == 2.0
    assert np.count_nonzero(B) == 3
