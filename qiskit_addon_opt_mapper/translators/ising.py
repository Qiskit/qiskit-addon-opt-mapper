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

"""Translator between an Ising Hamiltonian and a optimization problem."""

import itertools

import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp

from qiskit_addon_opt_mapper.exceptions import OptimizationError
from qiskit_addon_opt_mapper.problems.optimization_problem import OptimizationProblem


def to_ising(optimization_problem: OptimizationProblem) -> tuple[SparsePauliOp, float]:
    """Return the Ising Hamiltonian of this problem.

    Variables are mapped to qubits in the same order, i.e.,
    i-th variable is mapped to i-th qubit.
    See https://github.com/Qiskit/qiskit-terra/issues/1148 for details.

    Args:
        optimization_problem: The problem to be translated.

    Returns:
        A tuple (qubit_op, offset) comprising the qubit operator for the problem
        and offset for the constant value in the Ising Hamiltonian.

    Raises:
        OptimizationError: If an integer variable or a continuous variable exists
            in the problem.
        OptimizationError: If constraints exist in the problem.
    """
    # if constraints exist, raise an error
    if optimization_problem.linear_constraints or optimization_problem.quadratic_constraints:
        raise OptimizationError(
            "There must be no constraint in the problem. "
            "You can use `OptimizationProblemToQubo` converter "
            "to convert constraints to penalty terms of the objective function."
        )

    if optimization_problem.get_num_vars() == optimization_problem.get_num_binary_vars():
        # if all variables are binary variables
        # initialize Hamiltonian.
        num_vars = optimization_problem.get_num_vars()
        pauli_list = []
        offset = 0.0
        zero = np.zeros(num_vars, dtype=bool)

        # set a sign corresponding to a maximized or minimized problem.
        # sign == 1 is for minimized problem. sign == -1 is for maximized problem.
        sense = optimization_problem.objective.sense.value

        # convert a constant part of the objective function into Hamiltonian.
        offset += optimization_problem.objective.constant * sense

        # convert linear terms of the objective function into Hamiltonian.
        for idx, coef in optimization_problem.objective.linear.to_dict().items():
            z_p = zero.copy()
            weight = coef * sense / 2
            z_p[idx] = True

            pauli_list.append(SparsePauliOp(Pauli((z_p, zero)), -weight))
            offset += weight

        # convert quadratic terms of the objective function into Hamiltonian.
        for (
            i,
            j,
        ), quadratic_exp in optimization_problem.objective.quadratic.to_dict().items():
            weight = quadratic_exp * sense / 4

            if i == j:
                offset += weight
            else:
                z_p = zero.copy()
                z_p[i] = True
                z_p[j] = True
                pauli_list.append(SparsePauliOp(Pauli((z_p, zero)), weight))

            z_p = zero.copy()
            z_p[i] = True
            pauli_list.append(SparsePauliOp(Pauli((z_p, zero)), -weight))

            z_p = zero.copy()
            z_p[j] = True
            pauli_list.append(SparsePauliOp(Pauli((z_p, zero)), -weight))

            offset += weight

        # convert higher order terms of the objective function into Hamiltonian.
        for (  # type: ignore
            degree,
            high_order_exp,
        ) in optimization_problem.objective.higher_order.items():
            # For each binary term of order k we get Pauli spins with orders ranging from 0 to k due
            # to the expansion (1-z0)(1-z1)(1-z2) ... / 2**k for a term x0x1x2..., for example.
            for variables, coef in high_order_exp.to_dict().items():  # type: ignore
                for i in range(1, degree + 1):
                    for comb in itertools.combinations(variables, i):
                        weight = coef * sense * (-1) ** (i) / (2 ** (degree))
                        # enumerate all combinations of i elements in variables
                        # and add corresponding Pauli terms.
                        # For example, if variables=(0,1,2) and i=1,
                        # comb takes (0,), (1,), and (2,)
                        # and we add IIZ, IZI, and ZII to pauli_list.
                        z_p = zero.copy()
                        for idx in comb:  # type: ignore
                            z_p[idx] = not z_p[idx]
                        pauli_list.append(SparsePauliOp(Pauli((z_p, zero)), weight))
                offset += coef * sense / (2 ** (degree))

    elif optimization_problem.get_num_vars() == optimization_problem.get_num_spin_vars():
        # if all variables are spin variables
        # initialize Hamiltonian.
        num_vars = optimization_problem.get_num_vars()
        pauli_list = []
        offset = 0.0
        zero = np.zeros(num_vars, dtype=bool)

        # set a sign corresponding to a maximized or minimized problem.
        # sign == 1 is for minimized problem. sign == -1 is for maximized problem.
        sense = optimization_problem.objective.sense.value

        # convert a constant part of the objective function into Hamiltonian.
        offset += optimization_problem.objective.constant * sense

        # convert linear terms of the objective function into Hamiltonian.
        for idx, coef in optimization_problem.objective.linear.to_dict().items():
            z_p = zero.copy()
            z_p[idx] = True

            pauli_list.append(SparsePauliOp(Pauli((z_p, zero)), coef * sense))

        # convert quadratic terms of the objective function into Hamiltonian.
        for (
            i,
            j,
        ), coef in optimization_problem.objective.quadratic.to_dict().items():
            if i == j:
                offset += coef * sense
            else:
                z_p = zero.copy()
                z_p[i] = True
                z_p[j] = True
                pauli_list.append(SparsePauliOp(Pauli((z_p, zero)), coef * sense))

        # convert higher order terms of the objective function into Hamiltonian.
        for (  # type: ignore
            _degree,
            high_order_exp,
        ) in optimization_problem.objective.higher_order.items():
            for variables, coef in high_order_exp.to_dict().items():  # type: ignore
                z_p = zero.copy()
                for idx in variables:
                    z_p[idx] = not z_p[idx]
                pauli_list.append(SparsePauliOp(Pauli((z_p, zero)), coef * sense))

    else:
        raise OptimizationError(
            "The problem must contain either only binary variables or only spin variables. "
            "You can use `OptimizationProblemToHubo` converter "
            "to convert integer variables to binary variables. "
        )

    if pauli_list:
        # Remove paulis whose coefficients are zeros.
        qubit_op = sum(pauli_list).simplify(atol=0)
    else:
        # If there is no variable, we set num_nodes=1 so that qubit_op should be an operator.
        # If num_nodes=0, I^0 = 1 (int).
        num_vars = max(1, num_vars)
        qubit_op = SparsePauliOp("I" * num_vars, 0)

    return qubit_op, offset
