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

# --- adjust imports to your package layout ---
from qiskit_addon_opt_mapper.problems import (
    ObjSense,
    OptimizationObjective,
    OptimizationProblem,
)

# ---------------------------------------------


@pytest.fixture
def op3():
    op = OptimizationProblem()
    op.integer_var(-10.0, 10.0, "x0")
    op.integer_var(-10.0, 10.0, "x1")
    op.integer_var(-10.0, 10.0, "x2")
    return op


def test_objective_multiple_higher_orders_eval_and_grad(op3):
    """
    Objective with linear + quadratic + {cubic, quartic} blocks via higher_orders.
    Verify value and gradient at a test point.
    """
    linear = {0: 1.5, 1: -2.0}
    quadratic = {(2, 2): 3.0}
    higher_orders = {
        3: {(0, 0, 1): 2.0},  # 2 * x0^2 * x1
        4: {(0, 1, 1, 2): -0.5},  # -0.5 * x0 * x1^2 * x2
    }

    obj = OptimizationObjective(
        optimization_problem=op3,
        constant=0.0,
        linear=linear,
        quadratic=quadratic,
        higher_order=higher_orders,  # multiple blocks style
        sense=ObjSense.MINIMIZE,
    )

    x = [2.0, 1.0, -1.0]

    # Value:
    # linear:    1.5*2 + (-2)*1 = 1
    # quadratic: 3 * (-1)^2 = 3
    # cubic:     2 * (2^2 * 1) = 8
    # quartic:  -0.5 * (2 * 1^2 * -1) = +1
    # total: 1 + 3 + 8 + 1 = 13
    expected_val = 13.0
    assert math.isclose(obj.evaluate(x), expected_val, rel_tol=1e-12, abs_tol=1e-12)

    # Gradient:
    # linear grad = [1.5, -2.0, 0]
    # quadratic grad = [0, 0, 6*x2] = [0, 0, -6]
    # cubic grad (2*x0^2*x1):
    #   [8, 8, 0]
    # quartic grad (-0.5*x0*x1^2*x2):
    #   d/dx0 = -0.5 * x1^2 * x2 = 0.5
    #   d/dx1 = -0.5 * 2*x0*x1*x2 = 2.0
    #   d/dx2 = -0.5 * x0*x1^2 = -1.0
    expected_grad = np.array(
        [
            1.5 + 8.0 + 0.5,  # = 10.0
            -2.0 + 8.0 + 2.0,  # = 8.0
            0.0 - 6.0 - 1.0,  # = -7.0
        ]
    )
    g = obj.evaluate_gradient(x)
    assert np.allclose(g, expected_grad)
