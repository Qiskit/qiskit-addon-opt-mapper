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
"""Test OptimizationProblem"""

import math
import unittest

import numpy as np
import pytest

# from docplex.mp.model import DOcplexException
from qiskit_addon_opt_mapper import INFINITY, OptimizationError, OptimizationProblem
from qiskit_addon_opt_mapper.problems import (
    Constraint,
    ObjSense,
    OptimizationObjective,
    Variable,
    VarType,
)

from ..optimization_test_case import OptimizationTestCase


@pytest.fixture
def op3():
    """OptimizationProblem with 3 integer vars x0, x1, x2 in [-10, 10]."""
    op = OptimizationProblem()
    op.integer_var(-10.0, 10.0, "x0")
    op.integer_var(-10.0, 10.0, "x1")
    op.integer_var(-10.0, 10.0, "x2")
    return op


def test_minimize_with_multiple_higher_orders(op3):
    """
    Set objective via minimize() with higher_orders={3,4}.
    Check evaluate() and evaluate_gradient().
    """
    # linear:  1.5*x0 - 2*x1
    linear = {0: 1.5, 1: -2.0}
    # quadratic: 3*x2^2
    quadratic = {(2, 2): 3.0}
    # higher orders
    higher_orders = {
        3: {(0, 0, 1): 2.0},  # 2*x0^2*x1
        4: {(0, 1, 1, 2): -0.5},  # -0.5*x0*x1^2*x2
    }

    op3.minimize(constant=0.0, linear=linear, quadratic=quadratic, higher_order=higher_orders)
    assert op3.objective.sense == ObjSense.MINIMIZE

    x = [2.0, 1.0, -1.0]

    # Value:
    # linear:    1.5*2 + (-2)*1 = 1
    # quadratic: 3 * (-1)^2 = 3
    # cubic:     2*(2^2 * 1) = 8
    # quartic:  -0.5*(2*1*1*-1) = +1
    # total = 13
    assert math.isclose(op3.objective.evaluate(x), 13.0, rel_tol=1e-12, abs_tol=1e-12)

    # Gradient:
    # linear grad = [ 1.5, -2.0, 0]
    # quadratic   = [ 0,    0,   6*x2] = [0, 0, -6]
    # cubic (2*x0^2*x1): [8, 8, 0]
    # quartic (-0.5*x0*x1^2*x2):
    #   d/dx0 = -0.5*x1^2*x2 = 0.5
    #   d/dx1 = -0.5*2*x0*x1*x2 = 2.0
    #   d/dx2 = -0.5*x0*x1^2 = -1.0
    expected_grad = np.array([1.5 + 8.0 + 0.5, -2.0 + 8.0 + 2.0, 0.0 - 6.0 - 1.0])
    g = op3.objective.evaluate_gradient(x)
    assert np.allclose(g, expected_grad)


def test_maximize_single_higher_order_then_replace(op3):
    """
    Set objective via maximize() with single (order, higher_order),
    then replace with set_higher_orders() from the objective.
    """
    op3.maximize(
        constant=0.0,
        linear={0: 1.0},
        quadratic={(1, 1): 2.0},  # 2*x1^2
        higher_order={3: {(0, 0, 2): 1.0}},  # x0^2 * x2
    )
    assert op3.objective.sense == ObjSense.MAXIMIZE

    x = [2.0, 1.0, -1.0]
    # Value: lin=2, quad=2, cubic=1*(4*-1)=-4 => total = 0
    assert math.isclose(op3.objective.evaluate(x), 0.0, rel_tol=1e-12)

    # Replace all higher-orders via API on the objective:
    op3.objective.higher_order = {3: {(0, 0, 1): 2.0}}
    # New value: lin=2, quad=2, cubic=2*(4*1)=8 => 12
    assert math.isclose(op3.objective.evaluate(x), 12.0, rel_tol=1e-12)


def test_add_higher_order_constraints_single_and_multiple(op3):
    """
    Add a single higher-order constraint (legacy style), then a multiple-block constraint.
    Evaluate LHS and check feasibility detection.
    """
    # c1: single block (order=3), plus linear+quadratic
    c1 = op3.higher_order_constraint(
        linear=None,
        quadratic=None,
        higher_order={3: {(0, 0, 1): 2.0}},  # 2*x0^2*x1
        sense=Constraint.Sense.LE,
        rhs=8.0,
        name="c_single",
    )

    # c2: multiple blocks (k=3,4)
    c2 = op3.higher_order_constraint(
        linear=None,
        quadratic=None,
        higher_order={
            3: {(0, 0, 1): 2.0},
            4: {(0, 1, 1, 2): -0.5},
        },
        sense=Constraint.Sense.LE,
        rhs=9.0,
        name="c_multi",
    )

    x = [2.0, 1.0, -1.0]

    # Check LHS numerics for c1
    assert math.isclose(c1.evaluate(x), 8.0, rel_tol=1e-12)

    # Check LHS numerics for c2
    assert math.isclose(c2.evaluate(x), 9.0, rel_tol=1e-12)

    # Feasibility: with rhs set as above, both constraints satisfied at x
    feasible, violated_vars, violated_csts = op3.get_feasibility_info(x)
    assert feasible
    assert violated_vars == []
    assert violated_csts == []

    # Now tighten c2 to force a violation
    c2.rhs = 7.0
    feasible, violated_vars, violated_csts = op3.get_feasibility_info(x)
    assert not feasible
    assert c2 in violated_csts


def test_higher_order_constraint_name_autoincrement(op3):
    """
    Names should auto-increment as h0, h1, ... when not provided.
    """
    c0 = op3.higher_order_constraint(linear={}, quadratic={}, higher_order={3: {(0, 0, 1): 1.0}})
    c1 = op3.higher_order_constraint(linear={}, quadratic={}, higher_order={3: {(0, 1, 2): -1.0}})
    assert c0.name.startswith("h0")
    assert c1.name.startswith("h1")
    # index map must be consistent
    assert op3.higher_order_constraints_index[c0.name] == 0
    assert op3.higher_order_constraints_index[c1.name] == 1


def test_clear_resets_problem(op3):
    """
    clear() should remove variables, constraints, and reset objective.
    """
    # add constraints to ensure they are removed
    op3.higher_order_constraint(higher_order={3: {(0, 0, 1): 1.0}})
    op3.linear_constraint(linear={0: 1.0}, rhs=1.0)
    op3.quadratic_constraint(linear={}, quadratic={(0, 0): 1.0}, rhs=0.0)

    assert op3.get_num_vars() == 3
    assert op3.get_num_higher_order_constraints() == 1
    assert op3.get_num_linear_constraints() == 1
    assert op3.get_num_quadratic_constraints() == 1

    op3.clear()

    assert op3.get_num_vars() == 0
    assert op3.get_num_higher_order_constraints() == 0
    assert op3.get_num_linear_constraints() == 0
    assert op3.get_num_quadratic_constraints() == 0


def test_prettyprint_with_higher_orders(op3):
    """
    Ensure prettyprint() returns a string that mentions Higher-order constraints when present.
    """
    # objective with higher_orders
    op3.minimize(
        linear={0: 1.0},
        quadratic={(1, 1): 2.0},
        higher_order={3: {(0, 0, 2): 1.0}},
    )
    # add one higher-order constraint so the section appears
    op3.higher_order_constraint(
        higher_order={3: {(0, 0, 1): 1.0}}, sense=Constraint.Sense.LE, rhs=0.0
    )

    s = op3.prettyprint(wrap=80)
    assert isinstance(s, str)
    # The exact formatting may vary; check key phrases exist
    assert "Minimize" in s
    assert "Higher-order constraints" in s


class TestOptimizationProblem(OptimizationTestCase):
    """Test OptimizationProblem without the members that have separate test classes
    (VariablesInterface, etc)."""

    def test_constructor(self):
        """test constructor"""
        quadratic_program = OptimizationProblem()
        self.assertEqual(quadratic_program.name, "")
        self.assertEqual(quadratic_program.status, OptimizationProblem.Status.VALID)
        self.assertEqual(quadratic_program.get_num_vars(), 0)
        self.assertEqual(quadratic_program.get_num_linear_constraints(), 0)
        self.assertEqual(quadratic_program.get_num_quadratic_constraints(), 0)
        self.assertEqual(quadratic_program.objective.constant, 0)
        self.assertDictEqual(quadratic_program.objective.linear.to_dict(), {})
        self.assertDictEqual(quadratic_program.objective.quadratic.to_dict(), {})

    def test_clear(self):
        """test clear"""
        q_p = OptimizationProblem("test")
        q_p.binary_var("x")
        q_p.binary_var("y")
        q_p.minimize(constant=1, linear={"x": 1, "y": 2}, quadratic={("x", "x"): 1})
        q_p.linear_constraint({"x": 1}, "==", 1)
        q_p.quadratic_constraint({"x": 1}, {("y", "y"): 2}, "<=", 1)
        q_p.clear()
        self.assertEqual(q_p.name, "")
        self.assertEqual(q_p.status, OptimizationProblem.Status.VALID)
        self.assertEqual(q_p.get_num_vars(), 0)
        self.assertEqual(q_p.get_num_linear_constraints(), 0)
        self.assertEqual(q_p.get_num_quadratic_constraints(), 0)
        self.assertEqual(q_p.objective.constant, 0)
        self.assertDictEqual(q_p.objective.linear.to_dict(), {})
        self.assertDictEqual(q_p.objective.quadratic.to_dict(), {})

    def test_name_setter(self):
        """test name setter"""
        q_p = OptimizationProblem()
        self.assertEqual(q_p.name, "")
        name = "test name"
        q_p.name = name
        self.assertEqual(q_p.name, name)

    def assert_equal(self, x: Variable, y: Variable):
        """asserts variable equality"""
        self.assertEqual(x.name, y.name)
        self.assertEqual(x.lowerbound, y.lowerbound)
        self.assertEqual(x.upperbound, y.upperbound)
        self.assertEqual(x.vartype, y.vartype)

    def test_var_dict(self):
        """test {binary,integer,continuous, spin}_var_dict"""
        op = OptimizationProblem()
        spin_d = op.spin_var_dict(name="s", keys=3)
        self.assertSetEqual(set(spin_d.keys()), {"s0", "s1", "s2"})
        self.assertSetEqual({var.name for var in op.variables}, {"s0", "s1", "s2"})

        q_p = OptimizationProblem()

        d_0 = q_p.continuous_var_dict(name="a", key_format="_{}", keys=3)
        self.assertSetEqual(set(d_0.keys()), {"a_0", "a_1", "a_2"})
        self.assertSetEqual({var.name for var in q_p.variables}, {"a_0", "a_1", "a_2"})
        for var in q_p.variables:
            self.assertAlmostEqual(var.lowerbound, 0)
            self.assertAlmostEqual(var.upperbound, INFINITY)
            self.assertEqual(var.vartype, VarType.CONTINUOUS)
            self.assertTupleEqual(var.as_tuple(), d_0[var.name].as_tuple())

        d_1 = q_p.binary_var_dict(name="b", keys=5)
        self.assertSetEqual(set(d_1.keys()), {"b3", "b4", "b5", "b6", "b7"})
        self.assertSetEqual(
            {var.name for var in q_p.variables},
            {"a_0", "a_1", "a_2", "b3", "b4", "b5", "b6", "b7"},
        )
        for var in q_p.variables[-5:]:
            self.assertAlmostEqual(var.lowerbound, 0)
            self.assertAlmostEqual(var.upperbound, 1)
            self.assertEqual(var.vartype, VarType.BINARY)
            self.assertTupleEqual(var.as_tuple(), d_1[var.name].as_tuple())

        d_2 = q_p.integer_var_dict(keys=1, key_format="-{}", lowerbound=-4, upperbound=10)
        self.assertSetEqual(set(d_2.keys()), {"x-8"})
        self.assertSetEqual(
            {var.name for var in q_p.variables},
            {"a_0", "a_1", "a_2", "b3", "b4", "b5", "b6", "b7", "x-8"},
        )
        for var in q_p.variables[-1:]:
            self.assertAlmostEqual(var.lowerbound, -4)
            self.assertAlmostEqual(var.upperbound, 10)
            self.assertEqual(var.vartype, VarType.INTEGER)
            self.assertTupleEqual(var.as_tuple(), d_2[var.name].as_tuple())

        d_3 = q_p.binary_var_dict(name="c", keys=range(3))
        self.assertSetEqual(set(d_3.keys()), {"c0", "c1", "c2"})
        self.assertSetEqual(
            {var.name for var in q_p.variables},
            {
                "a_0",
                "a_1",
                "a_2",
                "b3",
                "b4",
                "b5",
                "b6",
                "b7",
                "x-8",
                "c0",
                "c1",
                "c2",
            },
        )
        for var in q_p.variables[-3:]:
            self.assertAlmostEqual(var.lowerbound, 0)
            self.assertAlmostEqual(var.upperbound, 1)
            self.assertEqual(var.vartype, VarType.BINARY)
            self.assertTupleEqual(var.as_tuple(), d_3[var.name].as_tuple())

        with self.assertRaises(OptimizationError):
            q_p.binary_var_dict(name="c", keys=range(3))

        d_4 = q_p.binary_var_dict(1, "x", "_")
        self.assertSetEqual(set(d_4.keys()), {"x_"})
        self.assertSetEqual(
            {var.name for var in q_p.variables},
            {
                "a_0",
                "a_1",
                "a_2",
                "b3",
                "b4",
                "b5",
                "b6",
                "b7",
                "x-8",
                "c0",
                "c1",
                "c2",
                "x_",
            },
        )
        for var in q_p.variables[-1:]:
            self.assertAlmostEqual(var.lowerbound, 0)
            self.assertAlmostEqual(var.upperbound, 1)
            self.assertEqual(var.vartype, VarType.BINARY)
            self.assertTupleEqual(var.as_tuple(), d_4[var.name].as_tuple())

        with self.assertRaises(OptimizationError):
            q_p.binary_var_dict(1, "x", "_")

        with self.assertRaises(OptimizationError):
            q_p.binary_var("x_")

        d_5 = q_p.continuous_var_dict(1, -1, 2, "", "")
        self.assertSetEqual(set(d_5.keys()), {"x"})
        self.assertSetEqual(
            {var.name for var in q_p.variables},
            {
                "a_0",
                "a_1",
                "a_2",
                "b3",
                "b4",
                "b5",
                "b6",
                "b7",
                "x-8",
                "c0",
                "c1",
                "c2",
                "x_",
                "x",
            },
        )
        for var in q_p.variables[-1:]:
            self.assertAlmostEqual(var.lowerbound, -1)
            self.assertAlmostEqual(var.upperbound, 2)
            self.assertEqual(var.vartype, VarType.CONTINUOUS)
            self.assertTupleEqual(var.as_tuple(), d_5[var.name].as_tuple())

        with self.assertRaises(OptimizationError):
            q_p.binary_var_dict(1, "", "")

        with self.assertRaises(OptimizationError):
            q_p = OptimizationProblem()
            q_p.binary_var_dict(keys=1, key_format="{}{}")

        with self.assertRaises(OptimizationError):
            q_p = OptimizationProblem()
            q_p.binary_var_dict(keys=0)

        with self.assertRaises(OptimizationError):
            q_p = OptimizationProblem()
            q_p.binary_var_dict(keys=1, key_format="_{{}}")

        with self.assertRaises(OptimizationError):
            q_p = OptimizationProblem()
            q_p.binary_var_dict(keys=2, key_format="")

        with self.assertRaises(OptimizationError):
            q_p = OptimizationProblem()
            q_p.binary_var_dict(keys=range(2), key_format="")

    def test_var_list(self):
        """test {binary,integer,continuous, spin}_var_list"""
        op = OptimizationProblem()
        spin_d = op.spin_var_list(name="s", keys=3)
        names = ["s0", "s1", "s2"]
        self.assertSetEqual({var.name for var in op.variables}, {"s0", "s1", "s2"})
        for i, var in enumerate(op.variables):
            self.assertEqual(var.name, names[i])
            self.assertAlmostEqual(var.lowerbound, -1)
            self.assertAlmostEqual(var.upperbound, 1)
            self.assertEqual(var.vartype, VarType.SPIN)
            self.assertTupleEqual(var.as_tuple(), spin_d[i].as_tuple())
        q_p = OptimizationProblem()

        d_0 = q_p.continuous_var_list(name="a", key_format="_{}", keys=3)
        names = ["a_0", "a_1", "a_2"]
        self.assertSetEqual({var.name for var in q_p.variables}, {"a_0", "a_1", "a_2"})
        for i, var in enumerate(q_p.variables):
            self.assertEqual(var.name, names[i])
            self.assertAlmostEqual(var.lowerbound, 0)
            self.assertAlmostEqual(var.upperbound, INFINITY)
            self.assertEqual(var.vartype, VarType.CONTINUOUS)
            self.assertTupleEqual(var.as_tuple(), d_0[i].as_tuple())

        d_1 = q_p.binary_var_list(name="b", keys=5)
        names = ["b3", "b4", "b5", "b6", "b7"]
        self.assertSetEqual(
            {var.name for var in q_p.variables},
            {"a_0", "a_1", "a_2", "b3", "b4", "b5", "b6", "b7"},
        )
        for i, var in enumerate(q_p.variables[-5:]):
            self.assertEqual(var.name, names[i])
            self.assertAlmostEqual(var.lowerbound, 0)
            self.assertAlmostEqual(var.upperbound, 1)
            self.assertEqual(var.vartype, VarType.BINARY)
            self.assertTupleEqual(var.as_tuple(), d_1[i].as_tuple())

        d_2 = q_p.integer_var_list(keys=1, key_format="-{}", lowerbound=-4, upperbound=10)
        names = ["x-8"]
        self.assertSetEqual(
            {var.name for var in q_p.variables},
            {"a_0", "a_1", "a_2", "b3", "b4", "b5", "b6", "b7", "x-8"},
        )
        for i, var in enumerate(q_p.variables[-1:]):
            self.assertEqual(var.name, names[i])
            self.assertAlmostEqual(var.lowerbound, -4)
            self.assertAlmostEqual(var.upperbound, 10)
            self.assertEqual(var.vartype, VarType.INTEGER)
            self.assertTupleEqual(var.as_tuple(), d_2[i].as_tuple())

        d_3 = q_p.binary_var_list(name="c", keys=range(3))
        names = ["c0", "c1", "c2"]
        self.assertSetEqual(
            {var.name for var in q_p.variables},
            {
                "a_0",
                "a_1",
                "a_2",
                "b3",
                "b4",
                "b5",
                "b6",
                "b7",
                "x-8",
                "c0",
                "c1",
                "c2",
            },
        )
        for i, var in enumerate(q_p.variables[-3:]):
            self.assertEqual(var.name, names[i])
            self.assertAlmostEqual(var.lowerbound, 0)
            self.assertAlmostEqual(var.upperbound, 1)
            self.assertEqual(var.vartype, VarType.BINARY)
            self.assertTupleEqual(var.as_tuple(), d_3[i].as_tuple())

        with self.assertRaises(OptimizationError):
            q_p.binary_var_list(name="c", keys=range(3))

        d_4 = q_p.binary_var_dict(1, "x", "_")
        names = ["x_"]
        self.assertSetEqual(
            {var.name for var in q_p.variables},
            {
                "a_0",
                "a_1",
                "a_2",
                "b3",
                "b4",
                "b5",
                "b6",
                "b7",
                "x-8",
                "c0",
                "c1",
                "c2",
                "x_",
            },
        )
        for i, var in enumerate(q_p.variables[-1:]):
            self.assertEqual(var.name, names[i])
            self.assertAlmostEqual(var.lowerbound, 0)
            self.assertAlmostEqual(var.upperbound, 1)
            self.assertEqual(var.vartype, VarType.BINARY)
            self.assertTupleEqual(var.as_tuple(), d_4[var.name].as_tuple())

        with self.assertRaises(OptimizationError):
            q_p.binary_var_list(1, "x", "_")

        with self.assertRaises(OptimizationError):
            q_p.binary_var("x_")

        d_5 = q_p.integer_var_list(1, -1, 2, "", "")
        names = ["x"]
        self.assertSetEqual(
            {var.name for var in q_p.variables},
            {
                "a_0",
                "a_1",
                "a_2",
                "b3",
                "b4",
                "b5",
                "b6",
                "b7",
                "x-8",
                "c0",
                "c1",
                "c2",
                "x_",
                "x",
            },
        )
        for i, var in enumerate(q_p.variables[-1:]):
            self.assertEqual(var.name, names[i])
            self.assertAlmostEqual(var.lowerbound, -1)
            self.assertAlmostEqual(var.upperbound, 2)
            self.assertEqual(var.vartype, VarType.INTEGER)
            self.assertTupleEqual(var.as_tuple(), d_5[i].as_tuple())

        with self.assertRaises(OptimizationError):
            q_p.binary_var_list(1, "", "")

        with self.assertRaises(OptimizationError):
            q_p = OptimizationProblem()
            q_p.binary_var_list(keys=1, key_format="{}{}")

        with self.assertRaises(OptimizationError):
            q_p = OptimizationProblem()
            q_p.binary_var_list(keys=0)

        with self.assertRaises(OptimizationError):
            q_p = OptimizationProblem()
            q_p.binary_var_list(keys=1, key_format="_{{}}")

        with self.assertRaises(OptimizationError):
            q_p = OptimizationProblem()
            q_p.binary_var_list(keys=2, key_format="")

        with self.assertRaises(OptimizationError):
            q_p = OptimizationProblem()
            q_p.binary_var_list(keys=range(2), key_format="")

    def test_variables_handling(self):
        """test add variables"""
        op = OptimizationProblem()
        s_0 = op.spin_var()
        self.assertEqual(s_0.name, "x0")
        self.assertEqual(s_0.lowerbound, -1)
        self.assertEqual(s_0.upperbound, 1)
        self.assertEqual(s_0.vartype, Variable.Type.SPIN)

        quadratic_program = OptimizationProblem()

        self.assertEqual(quadratic_program.get_num_vars(), 0)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 0)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 0)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 0)

        x_0 = quadratic_program.continuous_var()
        self.assertEqual(x_0.name, "x0")
        self.assertEqual(x_0.lowerbound, 0)
        self.assertEqual(x_0.upperbound, INFINITY)
        self.assertEqual(x_0.vartype, Variable.Type.CONTINUOUS)

        self.assertEqual(quadratic_program.get_num_vars(), 1)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 1)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 0)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 0)

        x_1 = quadratic_program.continuous_var(name="x1", lowerbound=5, upperbound=10)
        self.assertEqual(x_1.name, "x1")
        self.assertEqual(x_1.lowerbound, 5)
        self.assertEqual(x_1.upperbound, 10)
        self.assertEqual(x_1.vartype, Variable.Type.CONTINUOUS)

        self.assertEqual(quadratic_program.get_num_vars(), 2)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 2)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 0)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 0)

        x_2 = quadratic_program.binary_var()
        self.assertEqual(x_2.name, "x2")
        self.assertEqual(x_2.lowerbound, 0)
        self.assertEqual(x_2.upperbound, 1)
        self.assertEqual(x_2.vartype, Variable.Type.BINARY)

        self.assertEqual(quadratic_program.get_num_vars(), 3)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 2)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 1)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 0)

        x_3 = quadratic_program.binary_var(name="x3")
        self.assertEqual(x_3.name, "x3")
        self.assertEqual(x_3.lowerbound, 0)
        self.assertEqual(x_3.upperbound, 1)
        self.assertEqual(x_3.vartype, Variable.Type.BINARY)

        self.assertEqual(quadratic_program.get_num_vars(), 4)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 2)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 2)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 0)

        x_4 = quadratic_program.integer_var()
        self.assertEqual(x_4.name, "x4")
        self.assertEqual(x_4.lowerbound, 0)
        self.assertEqual(x_4.upperbound, INFINITY)
        self.assertEqual(x_4.vartype, Variable.Type.INTEGER)

        self.assertEqual(quadratic_program.get_num_vars(), 5)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 2)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 2)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 1)

        x_5 = quadratic_program.integer_var(name="x5", lowerbound=5, upperbound=10)
        self.assertEqual(x_5.name, "x5")
        self.assertEqual(x_5.lowerbound, 5)
        self.assertEqual(x_5.upperbound, 10)
        self.assertEqual(x_5.vartype, Variable.Type.INTEGER)

        self.assertEqual(quadratic_program.get_num_vars(), 6)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 2)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 2)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 2)

        with self.assertRaises(OptimizationError):
            quadratic_program.continuous_var(name="x0")

        with self.assertRaises(OptimizationError):
            quadratic_program.binary_var(name="x0")

        with self.assertRaises(OptimizationError):
            quadratic_program.integer_var(name="x0")

        variables = [x_0, x_1, x_2, x_3, x_4, x_5]
        for i, x in enumerate(variables):
            y = quadratic_program.get_variable(i)
            z = quadratic_program.get_variable(x.name)
            self.assertEqual(x.name, y.name)
            self.assertEqual(x.name, z.name)
        self.assertDictEqual(quadratic_program.variables_index, {"x" + str(i): i for i in range(6)})

    def test_linear_constraints_handling(self):
        """test linear constraints handling"""
        q_p = OptimizationProblem()
        q_p.binary_var("x")
        q_p.binary_var("y")
        q_p.binary_var("z")
        q_p.linear_constraint({"x": 1}, "==", 1)
        q_p.linear_constraint({"y": 1}, "<=", 1)
        q_p.linear_constraint({"z": 1}, ">=", 1)
        self.assertEqual(q_p.get_num_linear_constraints(), 3)
        lin = q_p.linear_constraints
        self.assertEqual(len(lin), 3)

        self.assertDictEqual(lin[0].linear.to_dict(), {0: 1})
        self.assertDictEqual(lin[0].linear.to_dict(use_name=True), {"x": 1})
        self.assertListEqual(lin[0].linear.to_array().tolist(), [1, 0, 0])
        self.assertEqual(lin[0].sense, Constraint.Sense.EQ)
        self.assertEqual(lin[0].rhs, 1)
        self.assertEqual(lin[0].name, "c0")
        self.assertEqual(q_p.get_linear_constraint(0).name, "c0")
        self.assertEqual(q_p.get_linear_constraint("c0").name, "c0")

        self.assertDictEqual(lin[1].linear.to_dict(), {1: 1})
        self.assertDictEqual(lin[1].linear.to_dict(use_name=True), {"y": 1})
        self.assertListEqual(lin[1].linear.to_array().tolist(), [0, 1, 0])
        self.assertEqual(lin[1].sense, Constraint.Sense.LE)
        self.assertEqual(lin[1].rhs, 1)
        self.assertEqual(lin[1].name, "c1")
        self.assertEqual(q_p.get_linear_constraint(1).name, "c1")
        self.assertEqual(q_p.get_linear_constraint("c1").name, "c1")

        self.assertDictEqual(lin[2].linear.to_dict(), {2: 1})
        self.assertDictEqual(lin[2].linear.to_dict(use_name=True), {"z": 1})
        self.assertListEqual(lin[2].linear.to_array().tolist(), [0, 0, 1])
        self.assertEqual(lin[2].sense, Constraint.Sense.GE)
        self.assertEqual(lin[2].rhs, 1)
        self.assertEqual(lin[2].name, "c2")
        self.assertEqual(q_p.get_linear_constraint(2).name, "c2")
        self.assertEqual(q_p.get_linear_constraint("c2").name, "c2")

        with self.assertRaises(OptimizationError):
            q_p.linear_constraint(name="c0")
        with self.assertRaises(OptimizationError):
            q_p.linear_constraint(name="c1")
        with self.assertRaises(OptimizationError):
            q_p.linear_constraint(name="c2")
        with self.assertRaises(IndexError):
            q_p.get_linear_constraint(4)
        with self.assertRaises(KeyError):
            q_p.get_linear_constraint("c3")

        q_p.remove_linear_constraint("c1")
        lin = q_p.linear_constraints
        self.assertEqual(len(lin), 2)
        self.assertDictEqual(lin[1].linear.to_dict(), {2: 1})
        self.assertDictEqual(lin[1].linear.to_dict(use_name=True), {"z": 1})
        self.assertListEqual(lin[1].linear.to_array().tolist(), [0, 0, 1])
        self.assertEqual(lin[1].sense, Constraint.Sense.GE)
        self.assertEqual(lin[1].rhs, 1)
        self.assertEqual(lin[1].name, "c2")
        self.assertEqual(q_p.get_linear_constraint(1).name, "c2")
        self.assertEqual(q_p.get_linear_constraint("c2").name, "c2")

        with self.assertRaises(KeyError):
            q_p.remove_linear_constraint("c1")
        with self.assertRaises(IndexError):
            q_p.remove_linear_constraint(9)

        q_p.linear_constraint(sense="E")
        self.assertEqual(q_p.linear_constraints[-1].sense, Constraint.Sense.EQ)
        q_p.linear_constraint(sense="G")
        self.assertEqual(q_p.linear_constraints[-1].sense, Constraint.Sense.GE)
        q_p.linear_constraint(sense="L")
        self.assertEqual(q_p.linear_constraints[-1].sense, Constraint.Sense.LE)
        q_p.linear_constraint(sense="EQ")
        self.assertEqual(q_p.linear_constraints[-1].sense, Constraint.Sense.EQ)
        q_p.linear_constraint(sense="GE")
        self.assertEqual(q_p.linear_constraints[-1].sense, Constraint.Sense.GE)
        q_p.linear_constraint(sense="LE")
        self.assertEqual(q_p.linear_constraints[-1].sense, Constraint.Sense.LE)
        q_p.linear_constraint(sense="=")
        self.assertEqual(q_p.linear_constraints[-1].sense, Constraint.Sense.EQ)
        q_p.linear_constraint(sense=">")
        self.assertEqual(q_p.linear_constraints[-1].sense, Constraint.Sense.GE)
        q_p.linear_constraint(sense="<")

        with self.assertRaises(OptimizationError):
            q_p.linear_constraint(sense="=>")

    def test_quadratic_constraints_handling(self):
        """test quadratic constraints handling"""
        q_p = OptimizationProblem()
        q_p.binary_var("x")
        q_p.binary_var("y")
        q_p.binary_var("z")
        q_p.quadratic_constraint({"x": 1}, {("x", "y"): 1}, "==", 1)
        q_p.quadratic_constraint({"y": 1}, {("y", "z"): 1}, "<=", 1)
        q_p.quadratic_constraint({"z": 1}, {("z", "x"): 1}, ">=", 1)
        self.assertEqual(q_p.get_num_quadratic_constraints(), 3)
        quad = q_p.quadratic_constraints
        self.assertEqual(len(quad), 3)

        self.assertDictEqual(quad[0].linear.to_dict(), {0: 1})
        self.assertDictEqual(quad[0].linear.to_dict(use_name=True), {"x": 1})
        self.assertListEqual(quad[0].linear.to_array().tolist(), [1, 0, 0])
        self.assertDictEqual(quad[0].quadratic.to_dict(), {(0, 1): 1})
        self.assertDictEqual(quad[0].quadratic.to_dict(symmetric=True), {(0, 1): 0.5, (1, 0): 0.5})
        self.assertDictEqual(quad[0].quadratic.to_dict(use_name=True), {("x", "y"): 1})
        self.assertDictEqual(
            quad[0].quadratic.to_dict(use_name=True, symmetric=True),
            {("x", "y"): 0.5, ("y", "x"): 0.5},
        )
        self.assertListEqual(
            quad[0].quadratic.to_array().tolist(), [[0, 1, 0], [0, 0, 0], [0, 0, 0]]
        )
        self.assertListEqual(
            quad[0].quadratic.to_array(symmetric=True).tolist(),
            [[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]],
        )
        self.assertEqual(quad[0].sense, Constraint.Sense.EQ)
        self.assertEqual(quad[0].rhs, 1)
        self.assertEqual(quad[0].name, "q0")
        self.assertEqual(q_p.get_quadratic_constraint(0).name, "q0")
        self.assertEqual(q_p.get_quadratic_constraint("q0").name, "q0")

        self.assertDictEqual(quad[1].linear.to_dict(), {1: 1})
        self.assertDictEqual(quad[1].linear.to_dict(use_name=True), {"y": 1})
        self.assertListEqual(quad[1].linear.to_array().tolist(), [0, 1, 0])
        self.assertDictEqual(quad[1].quadratic.to_dict(), {(1, 2): 1})
        self.assertDictEqual(quad[1].quadratic.to_dict(symmetric=True), {(1, 2): 0.5, (2, 1): 0.5})
        self.assertDictEqual(quad[1].quadratic.to_dict(use_name=True), {("y", "z"): 1})
        self.assertDictEqual(
            quad[1].quadratic.to_dict(use_name=True, symmetric=True),
            {("y", "z"): 0.5, ("z", "y"): 0.5},
        )
        self.assertListEqual(
            quad[1].quadratic.to_array().tolist(), [[0, 0, 0], [0, 0, 1], [0, 0, 0]]
        )
        self.assertListEqual(
            quad[1].quadratic.to_array(symmetric=True).tolist(),
            [[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0]],
        )
        self.assertEqual(quad[1].sense, Constraint.Sense.LE)
        self.assertEqual(quad[1].rhs, 1)
        self.assertEqual(quad[1].name, "q1")
        self.assertEqual(q_p.get_quadratic_constraint(1).name, "q1")
        self.assertEqual(q_p.get_quadratic_constraint("q1").name, "q1")

        self.assertDictEqual(quad[2].linear.to_dict(), {2: 1})
        self.assertDictEqual(quad[2].linear.to_dict(use_name=True), {"z": 1})
        self.assertListEqual(quad[2].linear.to_array().tolist(), [0, 0, 1])
        self.assertDictEqual(quad[2].quadratic.to_dict(), {(0, 2): 1})
        self.assertDictEqual(quad[2].quadratic.to_dict(symmetric=True), {(0, 2): 0.5, (2, 0): 0.5})
        self.assertDictEqual(quad[2].quadratic.to_dict(use_name=True), {("x", "z"): 1})
        self.assertDictEqual(
            quad[2].quadratic.to_dict(use_name=True, symmetric=True),
            {("x", "z"): 0.5, ("z", "x"): 0.5},
        )
        self.assertListEqual(
            quad[2].quadratic.to_array().tolist(), [[0, 0, 1], [0, 0, 0], [0, 0, 0]]
        )
        self.assertListEqual(
            quad[2].quadratic.to_array(symmetric=True).tolist(),
            [[0, 0, 0.5], [0, 0, 0], [0.5, 0, 0]],
        )
        self.assertEqual(quad[2].sense, Constraint.Sense.GE)
        self.assertEqual(quad[2].rhs, 1)
        self.assertEqual(quad[2].name, "q2")
        self.assertEqual(q_p.get_quadratic_constraint(2).name, "q2")
        self.assertEqual(q_p.get_quadratic_constraint("q2").name, "q2")

        with self.assertRaises(OptimizationError):
            q_p.quadratic_constraint(name="q0")
        with self.assertRaises(OptimizationError):
            q_p.quadratic_constraint(name="q1")
        with self.assertRaises(OptimizationError):
            q_p.quadratic_constraint(name="q2")
        with self.assertRaises(IndexError):
            q_p.get_quadratic_constraint(4)
        with self.assertRaises(KeyError):
            q_p.get_quadratic_constraint("q3")

        q_p.remove_quadratic_constraint("q1")
        quad = q_p.quadratic_constraints
        self.assertEqual(len(quad), 2)
        self.assertDictEqual(quad[1].linear.to_dict(), {2: 1})
        self.assertDictEqual(quad[1].linear.to_dict(use_name=True), {"z": 1})
        self.assertListEqual(quad[1].linear.to_array().tolist(), [0, 0, 1])
        self.assertDictEqual(quad[1].quadratic.to_dict(), {(0, 2): 1})
        self.assertDictEqual(quad[1].quadratic.to_dict(symmetric=True), {(0, 2): 0.5, (2, 0): 0.5})
        self.assertDictEqual(quad[1].quadratic.to_dict(use_name=True), {("x", "z"): 1})
        self.assertDictEqual(
            quad[1].quadratic.to_dict(use_name=True, symmetric=True),
            {("x", "z"): 0.5, ("z", "x"): 0.5},
        )
        self.assertListEqual(
            quad[1].quadratic.to_array().tolist(), [[0, 0, 1], [0, 0, 0], [0, 0, 0]]
        )
        self.assertListEqual(
            quad[1].quadratic.to_array(symmetric=True).tolist(),
            [[0, 0, 0.5], [0, 0, 0], [0.5, 0, 0]],
        )
        self.assertEqual(quad[1].sense, Constraint.Sense.GE)
        self.assertEqual(quad[1].rhs, 1)
        self.assertEqual(quad[1].name, "q2")
        self.assertEqual(q_p.get_quadratic_constraint(1).name, "q2")
        self.assertEqual(q_p.get_quadratic_constraint("q2").name, "q2")

        with self.assertRaises(KeyError):
            q_p.remove_quadratic_constraint("q1")
        with self.assertRaises(IndexError):
            q_p.remove_quadratic_constraint(9)

    def test_objective_handling(self):
        """test objective handling"""
        q_p = OptimizationProblem()
        q_p.binary_var("x")
        q_p.binary_var("y")
        q_p.binary_var("z")
        q_p.minimize()
        obj = q_p.objective
        self.assertEqual(obj.sense, OptimizationObjective.Sense.MINIMIZE)
        self.assertEqual(obj.constant, 0)
        self.assertDictEqual(obj.linear.to_dict(), {})
        self.assertDictEqual(obj.quadratic.to_dict(), {})
        q_p.maximize(1, {"y": 1}, {("z", "x"): 1, ("y", "y"): 1})
        obj = q_p.objective
        self.assertEqual(obj.sense, OptimizationObjective.Sense.MAXIMIZE)
        self.assertEqual(obj.constant, 1)
        self.assertDictEqual(obj.linear.to_dict(), {1: 1})
        self.assertDictEqual(obj.linear.to_dict(use_name=True), {"y": 1})
        self.assertListEqual(obj.linear.to_array().tolist(), [0, 1, 0])
        self.assertDictEqual(obj.quadratic.to_dict(), {(0, 2): 1, (1, 1): 1})
        self.assertDictEqual(
            obj.quadratic.to_dict(symmetric=True), {(0, 2): 0.5, (2, 0): 0.5, (1, 1): 1}
        )
        self.assertDictEqual(obj.quadratic.to_dict(use_name=True), {("x", "z"): 1, ("y", "y"): 1})
        self.assertDictEqual(
            obj.quadratic.to_dict(use_name=True, symmetric=True),
            {("x", "z"): 0.5, ("z", "x"): 0.5, ("y", "y"): 1},
        )
        self.assertListEqual(obj.quadratic.to_array().tolist(), [[0, 0, 1], [0, 1, 0], [0, 0, 0]])
        self.assertListEqual(
            obj.quadratic.to_array(symmetric=True).tolist(),
            [[0, 0, 0.5], [0, 1, 0], [0.5, 0, 0]],
        )

    def test_empty_objective(self):
        """test empty objective"""
        with self.assertRaises(OptimizationError):
            q_p = OptimizationProblem()
            q_p.binary_var_list(3)
            _ = q_p.objective.evaluate([0, 0, 0])

        with self.assertRaises(OptimizationError):
            q_p = OptimizationProblem()
            q_p.binary_var_list(3)
            _ = q_p.objective.evaluate_gradient([0, 0, 0])

        with self.assertRaises(OptimizationError):
            q_p = OptimizationProblem()
            q_p.binary_var_list(3)
            _ = q_p.objective.evaluate({})

        with self.assertRaises(OptimizationError):
            q_p = OptimizationProblem()
            q_p.binary_var_list(3)
            _ = q_p.objective.evaluate_gradient({})

    def test_substitute_variables(self):
        """test substitute variables"""
        q_p = OptimizationProblem("test")
        q_p.binary_var(name="x")
        q_p.integer_var(name="y", lowerbound=-2, upperbound=4)
        q_p.continuous_var(name="z", lowerbound=-1.5, upperbound=3.2)
        q_p.minimize(
            constant=1,
            linear={"x": 1, "y": 2},
            quadratic={("x", "y"): -1, ("z", "z"): 2},
        )
        q_p.linear_constraint({"x": 2, "z": -1}, "==", 1)
        q_p.quadratic_constraint({"x": 2, "z": -1}, {("y", "z"): 3}, "<=", -1)

        with self.subTest("x <- -1"):
            q_p2 = q_p.substitute_variables(constants={"x": -1})
            self.assertEqual(q_p2.status, OptimizationProblem.Status.INFEASIBLE)
            q_p2 = q_p.substitute_variables(constants={"y": -3})
            self.assertEqual(q_p2.status, OptimizationProblem.Status.INFEASIBLE)
            q_p2 = q_p.substitute_variables(constants={"x": 1, "z": 2})
            self.assertEqual(q_p2.status, OptimizationProblem.Status.INFEASIBLE)
            q_p2.clear()
            self.assertEqual(q_p2.status, OptimizationProblem.Status.VALID)

        with self.subTest("x <- 0"):
            q_p2 = q_p.substitute_variables(constants={"x": 0})
            self.assertEqual(q_p2.status, OptimizationProblem.Status.VALID)
            self.assertDictEqual(q_p2.objective.linear.to_dict(use_name=True), {"y": 2})
            self.assertDictEqual(q_p2.objective.quadratic.to_dict(use_name=True), {("z", "z"): 2})
            self.assertEqual(q_p2.objective.constant, 1)
            self.assertEqual(len(q_p2.linear_constraints), 1)
            self.assertEqual(len(q_p2.quadratic_constraints), 1)

            cst = q_p2.linear_constraints[0]
            self.assertDictEqual(cst.linear.to_dict(use_name=True), {"z": -1})
            self.assertEqual(cst.sense.name, "EQ")
            self.assertEqual(cst.rhs, 1)

            cst = q_p2.quadratic_constraints[0]
            self.assertDictEqual(cst.linear.to_dict(use_name=True), {"z": -1})
            self.assertDictEqual(cst.quadratic.to_dict(use_name=True), {("y", "z"): 3})
            self.assertEqual(cst.sense.name, "LE")
            self.assertEqual(cst.rhs, -1)

        with self.subTest("z <- -1"):
            q_p2 = q_p.substitute_variables(constants={"z": -1})
            self.assertEqual(q_p2.status, OptimizationProblem.Status.VALID)
            self.assertDictEqual(q_p2.objective.linear.to_dict(use_name=True), {"x": 1, "y": 2})
            self.assertDictEqual(q_p2.objective.quadratic.to_dict(use_name=True), {("x", "y"): -1})
            self.assertEqual(q_p2.objective.constant, 3)
            self.assertEqual(len(q_p2.linear_constraints), 2)
            self.assertEqual(len(q_p2.quadratic_constraints), 0)

            cst = q_p2.linear_constraints[0]
            self.assertDictEqual(cst.linear.to_dict(use_name=True), {"x": 2})
            self.assertEqual(cst.sense.name, "EQ")
            self.assertEqual(cst.rhs, 0)

            cst = q_p2.linear_constraints[1]
            self.assertDictEqual(cst.linear.to_dict(use_name=True), {"x": 2, "y": -3})
            self.assertEqual(cst.sense.name, "LE")
            self.assertEqual(cst.rhs, -2)

        with self.subTest("y <- -0.5 * x"):
            q_p2 = q_p.substitute_variables(variables={"y": ("x", -0.5)})
            self.assertEqual(q_p2.status, OptimizationProblem.Status.VALID)
            self.assertDictEqual(q_p2.objective.linear.to_dict(use_name=True), {})
            self.assertDictEqual(
                q_p2.objective.quadratic.to_dict(use_name=True),
                {("x", "x"): 0.5, ("z", "z"): 2},
            )
            self.assertEqual(q_p2.objective.constant, 1)
            self.assertEqual(len(q_p2.linear_constraints), 1)
            self.assertEqual(len(q_p2.quadratic_constraints), 1)

            cst = q_p2.linear_constraints[0]
            self.assertDictEqual(cst.linear.to_dict(use_name=True), {"x": 2, "z": -1})
            self.assertEqual(cst.sense.name, "EQ")
            self.assertEqual(cst.rhs, 1)

            cst = q_p2.quadratic_constraints[0]
            self.assertDictEqual(cst.linear.to_dict(use_name=True), {"x": 2, "z": -1})
            self.assertDictEqual(cst.quadratic.to_dict(use_name=True), {("x", "z"): -1.5})
            self.assertEqual(cst.sense.name, "LE")
            self.assertEqual(cst.rhs, -1)

    def test_feasibility(self):
        """Tests feasibility methods."""
        q_p = OptimizationProblem("test")
        _ = q_p.continuous_var(-1, 1, "x")
        _ = q_p.continuous_var(-10, 10, "y")
        q_p.minimize(linear={"x": 1, "y": 1})
        q_p.linear_constraint({"x": 1, "y": 1}, "<=", 10, "c0")
        q_p.linear_constraint({"x": 1, "y": 1}, ">=", -10, "c1")
        q_p.linear_constraint({"x": 1, "y": 1}, "==", 5, "c2")
        q_p.quadratic_constraint({"y": 1}, {("x", "x"): 1}, "<=", 10, "c3")
        q_p.quadratic_constraint({"y": 1}, {("x", "x"): 1}, ">=", 5, "c4")
        q_p.quadratic_constraint(None, {("x", "x"): 1, ("y", "y"): 1}, "==", 25, "c5")

        self.assertTrue(q_p.is_feasible([0, 5]))
        self.assertFalse(q_p.is_feasible([1, 10]))
        self.assertFalse(q_p.is_feasible([1, -12]))
        self.assertFalse(q_p.is_feasible([1, 5]))
        self.assertFalse(q_p.is_feasible([5, 0]))
        self.assertFalse(q_p.is_feasible([1, 1]))
        self.assertFalse(q_p.is_feasible([0, 0]))

        feasible, variables, constraints = q_p.get_feasibility_info([10, 0])
        self.assertFalse(feasible)
        self.assertIsNotNone(variables)
        self.assertEqual(1, len(variables))
        self.assertEqual("x", variables[0].name)

        self.assertIsNotNone(constraints)
        self.assertEqual(3, len(constraints))
        self.assertEqual("c2", constraints[0].name)
        self.assertEqual("c3", constraints[1].name)
        self.assertEqual("c5", constraints[2].name)

    def test_empty_name(self):
        """Test empty names"""

        with self.subTest("problem name"):
            q_p = OptimizationProblem("")
            self.assertEqual(q_p.name, "")

        with self.subTest("variable name"):
            q_p = OptimizationProblem()
            x = q_p.binary_var(name="")
            y = q_p.integer_var(name="")
            z = q_p.continuous_var(name="")
            self.assertEqual(x.name, "x0")
            self.assertEqual(y.name, "x1")
            self.assertEqual(z.name, "x2")

        with self.subTest("variable name 2"):
            q_p = OptimizationProblem()
            w = q_p.binary_var(name="w")
            x = q_p.binary_var(name="")
            y = q_p.integer_var(name="")
            z = q_p.continuous_var(name="")
            self.assertEqual(w.name, "w")
            self.assertEqual(x.name, "x1")
            self.assertEqual(y.name, "x2")
            self.assertEqual(z.name, "x3")

        with self.subTest("variable name list"):
            q_p = OptimizationProblem()
            x = q_p.binary_var_list(2, name="")
            y = q_p.integer_var_list(2, name="")
            z = q_p.continuous_var_list(2, name="")
            self.assertListEqual([v.name for v in x], ["x0", "x1"])
            self.assertListEqual([v.name for v in y], ["x2", "x3"])
            self.assertListEqual([v.name for v in z], ["x4", "x5"])

        with self.subTest("variable name dict"):
            q_p = OptimizationProblem()
            x = q_p.binary_var_dict(2, name="")
            y = q_p.integer_var_dict(2, name="")
            z = q_p.continuous_var_dict(2, name="")
            self.assertDictEqual({k: v.name for k, v in x.items()}, {"x0": "x0", "x1": "x1"})
            self.assertDictEqual({k: v.name for k, v in y.items()}, {"x2": "x2", "x3": "x3"})
            self.assertDictEqual({k: v.name for k, v in z.items()}, {"x4": "x4", "x5": "x5"})

        with self.subTest("linear constraint name"):
            q_p = OptimizationProblem()
            x = q_p.linear_constraint(name="")
            y = q_p.linear_constraint(name="")
            self.assertEqual(x.name, "c0")
            self.assertEqual(y.name, "c1")

        with self.subTest("quadratic constraint name"):
            q_p = OptimizationProblem()
            x = q_p.quadratic_constraint(name="")
            y = q_p.quadratic_constraint(name="")
            self.assertEqual(x.name, "q0")
            self.assertEqual(y.name, "q1")

    def test_printable_name(self):
        """Test non-printable names"""
        name = "\n"

        with self.assertWarns(UserWarning):
            _ = OptimizationProblem(name)

        q_p = OptimizationProblem()

        with self.assertWarns(UserWarning):
            q_p.binary_var(name + "bin")

        with self.assertWarns(UserWarning):
            q_p.binary_var_list(10, name)

        with self.assertWarns(UserWarning):
            q_p.binary_var_dict(10, name)

        with self.assertWarns(UserWarning):
            q_p.integer_var(0, 1, name + "int")

        with self.assertWarns(UserWarning):
            q_p.integer_var_list(10, 0, 1, name)

        with self.assertWarns(UserWarning):
            q_p.integer_var_dict(10, 0, 1, name)

        with self.assertWarns(UserWarning):
            q_p.continuous_var(0, 1, name + "cont")

        with self.assertWarns(UserWarning):
            q_p.continuous_var_list(10, 0, 1, name)

        with self.assertWarns(UserWarning):
            q_p.continuous_var_dict(10, 0, 1, name)

        with self.assertWarns(UserWarning):
            q_p.linear_constraint(name=name)

        with self.assertWarns(UserWarning):
            q_p.quadratic_constraint(name=name)

    def test_str_repr(self):
        """Test str and repr"""
        q_p = OptimizationProblem("my problem")
        q_p.binary_var("x")
        q_p.integer_var(-1, 5, "y")
        q_p.continuous_var(-1, 5, "z")
        q_p.minimize(1, {"x": 1, "y": -1, "z": 10}, {("x", "x"): 0.5, ("y", "z"): -1})
        q_p.linear_constraint({"x": 1, "y": 2}, "==", 1, "lin_eq")
        q_p.linear_constraint({"x": 1, "y": 2}, "<=", 1, "lin_leq")
        q_p.linear_constraint({"x": 1, "y": 2}, ">=", 1, "lin_geq")
        q_p.quadratic_constraint(
            {"x": 1, "y": 1},
            {("x", "x"): 1, ("y", "z"): -1, ("z", "z"): 2},
            "==",
            1,
            "quad_eq",
        )
        q_p.quadratic_constraint(
            {"x": 1, "y": 1},
            {("x", "x"): 1, ("y", "z"): -1, ("z", "z"): 2},
            "<=",
            1,
            "quad_leq",
        )
        q_p.quadratic_constraint(
            {"x": 1, "y": 1},
            {("x", "x"): 1, ("y", "z"): -1, ("z", "z"): 2},
            ">=",
            1,
            "quad_geq",
        )
        self.assertEqual(
            str(q_p),
            "minimize 0.5*x^2 - y*z + x - y + 10*z + 1 (3 variables, 6 constraints, 'my problem')",
        )
        self.assertEqual(
            repr(q_p),
            "<OptimizationProblem: minimize 0.5*x^2 - y*z + x - y + 10*z + 1, "
            "3 variables, 6 constraints, 'my problem'>",
        )


if __name__ == "__main__":
    unittest.main()
