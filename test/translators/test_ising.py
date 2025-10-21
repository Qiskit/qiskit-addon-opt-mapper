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

"""Test to_ising"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_opt_mapper import OptimizationError, OptimizationProblem
from qiskit_addon_opt_mapper.converters import EqualityToPenalty
from qiskit_addon_opt_mapper.problems import Constraint
from qiskit_addon_opt_mapper.translators import to_ising

from ..optimization_test_case import OptimizationTestCase

QUBIT_OP_MAXIMIZE_SAMPLE = SparsePauliOp.from_list(
    [
        ("IIIZ", -199999.5),
        ("IIZI", -399999.5),
        ("IZII", -599999.5),
        ("ZIII", -799999.5),
        ("IIZZ", 100000),
        ("IZIZ", 150000),
        ("IZZI", 300000),
        ("ZIIZ", 200000),
        ("ZIZI", 400000),
        ("ZZII", 600000),
    ]
)
OFFSET_MAXIMIZE_SAMPLE = 1149998


class TestIsingTranslator(OptimizationTestCase):
    """Test to_ising"""

    def test_to_ising(self):
        """test to_ising"""

        with self.subTest("minimize"):
            # minimize: x + x * y
            # subject to: x, y \in {0, 1}
            q_p = OptimizationProblem("test")
            q_p.binary_var(name="x")
            q_p.binary_var(name="y")
            q_p.minimize(linear={"x": 1}, quadratic={("x", "y"): 1})
            op, offset = to_ising(q_p)
            op_ref = SparsePauliOp.from_list([("ZI", -0.25), ("IZ", -0.75), ("ZZ", 0.25)])
            np.testing.assert_allclose(op.to_matrix(), op_ref.to_matrix())
            self.assertAlmostEqual(offset, 0.75)

        with self.subTest("maximize"):
            # maximize: x + x * y
            # subject to: x, y \in {0, 1}
            q_p = OptimizationProblem("test")
            q_p.binary_var(name="x")
            q_p.binary_var(name="y")
            q_p.maximize(linear={"x": 1}, quadratic={("x", "y"): 1})
            op, offset = to_ising(q_p)
            op_ref = SparsePauliOp.from_list([("ZI", 0.25), ("IZ", 0.75), ("ZZ", -0.25)])
            np.testing.assert_allclose(op.to_matrix(), op_ref.to_matrix())
            self.assertAlmostEqual(offset, -0.75)

    def test_to_ising2(self):
        """test to_ising 2"""

        with self.subTest("minimize"):
            # minimize: 1 - 2 * x1 - 2 * x2 + 4 * x1 * x2
            # subject to: x, y \in {0, 1}
            q_p = OptimizationProblem("test")
            q_p.binary_var(name="x")
            q_p.binary_var(name="y")
            q_p.minimize(constant=1, linear={"x": -2, "y": -2}, quadratic={("x", "y"): 4})
            op, offset = to_ising(q_p)
            op_ref = SparsePauliOp.from_list([("ZZ", 1.0)])
            np.testing.assert_allclose(op.to_matrix(), op_ref.to_matrix())
            self.assertAlmostEqual(offset, 0.0)

        with self.subTest("maximize"):
            # maximize: 1 - 2 * x1 - 2 * x2 + 4 * x1 * x2
            # subject to: x, y \in {0, 1}
            q_p = OptimizationProblem("test")
            q_p.binary_var(name="x")
            q_p.binary_var(name="y")
            q_p.maximize(constant=1, linear={"x": -2, "y": -2}, quadratic={("x", "y"): 4})
            op, offset = to_ising(q_p)
            op_ref = SparsePauliOp.from_list([("ZZ", -1.0)])
            np.testing.assert_allclose(op.to_matrix(), op_ref.to_matrix())
            self.assertAlmostEqual(offset, 0.0)

    def test_to_ising_wo_variable(self):
        """test to_ising with problems without variables"""
        with self.subTest("empty problem"):
            q_p = OptimizationProblem()
            op, offset = to_ising(q_p)
            np.testing.assert_allclose(op.to_matrix(), np.zeros((2, 2)))
            self.assertAlmostEqual(offset, 0)

        with self.subTest("min 3"):
            q_p = OptimizationProblem()
            q_p.minimize(constant=3)
            op, offset = to_ising(q_p)
            np.testing.assert_allclose(op.to_matrix(), np.zeros((2, 2)))
            self.assertAlmostEqual(offset, 3)

        with self.subTest("max -1"):
            q_p = OptimizationProblem()
            q_p.maximize(constant=-1)
            op, offset = to_ising(q_p)
            np.testing.assert_allclose(op.to_matrix(), np.zeros((2, 2)))
            self.assertAlmostEqual(offset, 1)

    def test_optimizationproblem_to_ising(self):
        """Test optimization problem to operators"""
        op = OptimizationProblem()
        for i in range(4):
            op.binary_var(name=f"x{i}")
        linear = {}
        for x in op.variables:
            linear[x.name] = 1
        op.maximize(0, linear, {})
        linear = {}
        for i, x in enumerate(op.variables):
            linear[x.name] = i + 1
        op.linear_constraint(linear, Constraint.Sense.EQ, 3, "sum1")
        penalize = EqualityToPenalty(penalty=1e5)
        op2 = penalize.convert(op)
        qubitop, offset = op2.to_ising()
        self.assertTrue(qubitop.equiv(QUBIT_OP_MAXIMIZE_SAMPLE))
        self.assertEqual(offset, OFFSET_MAXIMIZE_SAMPLE)

    def test_valid_variable_type(self):
        """Validate the types of the variables for QuadraticProgram.to_ising."""
        # Integer variable
        with self.assertRaises(OptimizationError):
            op = OptimizationProblem()
            op.integer_var(0, 10, "int_var")
            _ = op.to_ising()
        # Continuous variable
        with self.assertRaises(OptimizationError):
            op = OptimizationProblem()
            op.continuous_var(0, 10, "continuous_var")
            _ = op.to_ising()

    def test_to_ising_higher_order(self):
        """test to_ising with higher order terms"""

        with self.subTest("cubic term"):
            # minimize: x0 * x1 * x2
            o_p = OptimizationProblem("test")
            o_p.binary_var(name="x0")
            o_p.binary_var(name="x1")
            o_p.binary_var(name="x2")
            o_p.minimize(
                higher_order={3: {(0, 1, 2): 2}},
            )
            op, offset = to_ising(o_p)
            op_ref = SparsePauliOp.from_list(
                [
                    ("IIZ", -0.25),
                    ("IZI", -0.25),
                    ("ZII", -0.25),
                    ("IZZ", 0.25),
                    ("ZIZ", 0.25),
                    ("ZZI", 0.25),
                    ("ZZZ", -0.25),
                ]
            )
            np.testing.assert_allclose(op.to_matrix(), op_ref.to_matrix())
            self.assertAlmostEqual(offset, 0.25)

        with self.subTest("cubic term 2"):
            # minimize: 2 * x0^2 * x1 + 3 * x1 * x2^2
            o_p = OptimizationProblem("test")
            o_p.binary_var(name="x0")
            o_p.binary_var(name="x1")
            o_p.binary_var(name="x2")
            o_p.minimize(
                higher_order={
                    3: {
                        (0, 0, 1): 2.0,
                        (1, 2, 2): 3.0,
                    }
                },
            )
            op, offset = to_ising(o_p)
            op_ref = SparsePauliOp.from_list(
                [
                    ("IIZ", -0.5),
                    ("IZI", -1.25),
                    ("ZII", -0.75),
                    ("IZZ", 0.5),
                    ("ZZI", 0.75),
                    ("III", 0.625),
                ]
            )
            np.testing.assert_allclose(op.to_matrix(), op_ref.to_matrix())
            self.assertAlmostEqual(offset, 0.625)

        with self.subTest("cubic and quartic terms"):
            # minimize: x0 * x1 * x2 + x0 * x1 * x2 * x3
            o_p = OptimizationProblem("test")
            o_p.binary_var(name="x0")
            o_p.binary_var(name="x1")
            o_p.binary_var(name="x2")
            o_p.binary_var(name="x3")
            o_p.minimize(
                higher_order={
                    3: {(0, 1, 2): 1},
                    4: {(0, 1, 2, 3): 1},
                }
            )
            op, offset = to_ising(o_p)
            op_ref = SparsePauliOp.from_list(
                [
                    ("IIIZ", -0.1875),
                    ("IIZI", -0.1875),
                    ("IZII", -0.1875),
                    ("ZIII", -0.0625),
                    ("IIZZ", 0.1875),
                    ("IZIZ", 0.1875),
                    ("IZZI", 0.1875),
                    ("ZIIZ", 0.0625),
                    ("ZIZI", 0.0625),
                    ("ZZII", 0.0625),
                    ("IZZZ", -0.1875),
                    ("ZIZZ", -0.0625),
                    ("ZZIZ", -0.0625),
                    ("ZZZI", -0.0625),
                    ("ZZZZ", 0.0625),
                ]
            )
            np.testing.assert_allclose(op.to_matrix(), op_ref.to_matrix())
            self.assertAlmostEqual(offset, 0.1875)
