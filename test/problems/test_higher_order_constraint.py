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
from qiskit_addon_opt_mapper.problems import (
    Constraint,
    HigherOrderConstraint,
    OptimizationProblem,
)


@pytest.fixture
def op3():
    op = OptimizationProblem()
    op.integer_var(-10.0, 10.0, "x0")
    op.integer_var(-10.0, 10.0, "x1")
    op.integer_var(-10.0, 10.0, "x2")
    return op


def test_higher_order_constraint_evaluate(op3):
    """Test HigherOrderConstraint with linear, quadratic, and cubic terms."""

    # linear term:  2*x0 + (-1)*x1
    linear = {0: 2.0, 1: -1.0}

    # quadratic term: x0*x1 + 3*x2^2
    quadratic = {
        (0, 1): 1.0,
        (2, 2): 3.0,
    }

    # cubic term:  2*x0^2*x1
    higher_order = {(0, 0, 1): 2.0}

    con = HigherOrderConstraint(
        optimization_problem=op3,
        name="cubic_constr",
        linear=linear,
        quadratic=quadratic,
        higher_order={3: higher_order},
        sense=Constraint.Sense.LE,
        rhs=10.0,
    )

    # Pick a test point
    x = [1.0, 2.0, -1.0]

    # Manually compute:
    # Linear: 2*1.0 + (-1)*2.0 = 2 - 2 = 0
    # Quadratic: 1.0*(1.0*2.0) + 3.0*((-1.0)**2) = 2.0 + 3.0*1.0 = 5.0
    # Cubic: 2.0*(1.0**2 * 2.0) = 4.0
    expected_value = 0 + 5.0 + 4.0  # = 9.0

    assert np.isclose(con.evaluate(x), expected_value)
    assert con.sense == Constraint.Sense.LE
    assert con.rhs == 10.0


def test_higher_order_constraint_with_multiple_orders_evaluate(op3):
    """
    Build a constraint with linear + quadratic + {cubic, quartic} and
    check that evaluate() equals the manually computed value.
    """
    # linear:  1.5*x0 - 2.0*x2
    linear = {0: 1.5, 2: -2.0}

    # quadratic: 3*x0*x1 - 1*x2^2
    quadratic = {(0, 1): 3.0, (2, 2): -1.0}

    # higher orders: k=3,4
    higher_order = {
        3: {(0, 0, 1): 2.0},  # 2 * x0^2 * x1
        4: {(0, 1, 1, 2): -0.5},  # -0.5 * x0 * x1^2 * x2
    }

    con = HigherOrderConstraint(
        optimization_problem=op3,
        name="mix_cubic_quartic",
        linear=linear,
        quadratic=quadratic,
        higher_order=higher_order,
        sense=Constraint.Sense.LE,
        rhs=100.0,
    )

    # x = [2, 1, -1]
    x = [2.0, 1.0, -1.0]

    # Manual value:
    # linear:    1.5*2 + (-2)*(-1) = 3 + 2 = 5
    # quadratic: 3*(2*1) + (-1)*( (-1)**2 ) = 6 - 1 = 5
    # cubic:     2*(2**2 * 1) = 8
    # quartic:  -0.5*(2 * 1**2 * -1) = -0.5*(-2) = 1
    # total: 5 + 5 + 8 + 1 = 19
    expected = 19.0
    assert math.isclose(con.evaluate(x), expected, rel_tol=1e-12, abs_tol=1e-12)

    # Check the orders are registered
    hos = con.higher_order
    assert set(hos.keys()) == {3, 4}
    assert hos[3].order == 3 and hos[4].order == 4
