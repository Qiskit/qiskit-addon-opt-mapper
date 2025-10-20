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

"""Translator between a gurobipy model and an optimization problem."""

from typing import cast

import qiskit_addon_opt_mapper.optionals as _optionals
from qiskit_addon_opt_mapper.exceptions import OptimizationError
from qiskit_addon_opt_mapper.problems.constraint import Constraint
from qiskit_addon_opt_mapper.problems.optimization_objective import (
    OptimizationObjective,
)
from qiskit_addon_opt_mapper.problems.optimization_problem import OptimizationProblem
from qiskit_addon_opt_mapper.problems.variable import Variable

if _optionals.HAS_GUROBIPY:
    # pylint: disable=import-error,no-name-in-module
    from gurobipy import LinExpr, Model, QuadExpr
else:

    class Model:  # type: ignore
        """Empty Model class.

        Replacement if gurobipy.Model is not present.
        """

        pass

    class LinExpr:  # type: ignore
        """Empty LinExpr class.

        Replacement if gurobipy.LinExpr is not present.
        """

        pass

    class QuadExpr:  # type: ignore
        """Empty QuadExpr class.

        Replacement if gurobipy.QuadExpr is not present.
        """

        pass


@_optionals.HAS_GUROBIPY.require_in_call
def to_gurobipy(optimization_problem: OptimizationProblem) -> Model:
    """Returns a gurobipy model corresponding to an optimization problem.

    Args:
        optimization_problem: The optimization problem to be translated.

    Returns:
        The gurobipy model corresponding to an optimization problem.

    Raises:
        OptimizationError: if non-supported elements (should never happen).
    """
    # initialize model
    # pylint: disable=import-error
    import gurobipy as gp

    mdl = gp.Model(optimization_problem.name)

    # add variables
    var = {}
    for idx, x in enumerate(optimization_problem.variables):
        if x.vartype == Variable.Type.CONTINUOUS:
            var[idx] = mdl.addVar(
                vtype=gp.GRB.CONTINUOUS, lb=x.lowerbound, ub=x.upperbound, name=x.name
            )
        elif x.vartype == Variable.Type.BINARY:
            var[idx] = mdl.addVar(vtype=gp.GRB.BINARY, name=x.name)
        elif x.vartype == Variable.Type.INTEGER:
            var[idx] = mdl.addVar(
                vtype=gp.GRB.INTEGER, lb=x.lowerbound, ub=x.upperbound, name=x.name
            )
        else:
            # should never happen
            raise OptimizationError(f"Unsupported variable type: {x.vartype}")

    # add objective
    objective = QuadExpr(optimization_problem.objective.constant)
    for i, v in optimization_problem.objective.linear.to_dict().items():
        objective += v * var[cast(int, i)]
    for (i, j), v in optimization_problem.objective.quadratic.to_dict().items():
        objective += v * var[cast(int, i)] * var[cast(int, j)]
    if optimization_problem.objective.sense == OptimizationObjective.Sense.MINIMIZE:
        mdl.setObjective(objective, sense=gp.GRB.MINIMIZE)
    else:
        mdl.setObjective(objective, sense=gp.GRB.MAXIMIZE)

    # add linear constraints
    for l_constraint in optimization_problem.linear_constraints:
        name = l_constraint.name
        rhs = l_constraint.rhs
        if rhs == 0 and l_constraint.linear.coefficients.nnz == 0:
            continue
        linear_expr = LinExpr(0)
        for j, v in l_constraint.linear.to_dict().items():
            linear_expr += v * var[cast(int, j)]
        sense = l_constraint.sense
        if sense == Constraint.Sense.EQ:
            mdl.addConstr(linear_expr == rhs, name=name)
        elif sense == Constraint.Sense.GE:
            mdl.addConstr(linear_expr >= rhs, name=name)
        elif sense == Constraint.Sense.LE:
            mdl.addConstr(linear_expr <= rhs, name=name)
        else:
            # should never happen
            raise OptimizationError(f"Unsupported constraint sense: {sense}")

    # add quadratic constraints
    for q_constraint in optimization_problem.quadratic_constraints:
        name = q_constraint.name
        rhs = q_constraint.rhs
        if (
            rhs == 0
            and q_constraint.linear.coefficients.nnz == 0
            and q_constraint.quadratic.coefficients.nnz == 0
        ):
            continue
        quadratic_expr = QuadExpr(0)
        for j, v in q_constraint.linear.to_dict().items():
            quadratic_expr += v * var[cast(int, j)]
        for (j, k), v in q_constraint.quadratic.to_dict().items():
            quadratic_expr += v * var[cast(int, j)] * var[cast(int, k)]
        sense = q_constraint.sense
        if sense == Constraint.Sense.EQ:
            mdl.addConstr(quadratic_expr == rhs, name=name)
        elif sense == Constraint.Sense.GE:
            mdl.addConstr(quadratic_expr >= rhs, name=name)
        elif sense == Constraint.Sense.LE:
            mdl.addConstr(quadratic_expr <= rhs, name=name)
        else:
            # should never happen
            raise OptimizationError(f"Unsupported constraint sense: {sense}")

    mdl.update()
    return mdl


@_optionals.HAS_GUROBIPY.require_in_call
def from_gurobipy(model: Model) -> OptimizationProblem:
    """Translate a gurobipy model into an optimization problem.

    Note that this supports only basic functions of gurobipy as follows:

    - quadratic objective function
    - linear / quadratic constraints
    - binary / integer / continuous variables

    Args:
        model: The gurobipy model to be loaded.

    Returns:
        The optimization problem corresponding to the model.

    Raises:
        OptimizationError: if the model contains unsupported elements.
    """
    # pylint: disable=import-error
    import gurobipy as gp

    if not isinstance(model, Model):
        raise OptimizationError(f"The model is not compatible: {model}")

    optimization_problem = OptimizationProblem()

    # Update the model to make sure everything works as expected
    model.update()

    # get name
    optimization_problem.name = model.ModelName

    # get variables
    # keep track of names separately, since gurobipy allows to have None names.
    var_names = {}
    for x in model.getVars():
        if x.VType == gp.GRB.CONTINUOUS:
            x_new = optimization_problem.continuous_var(x.LB, x.UB, x.VarName)
        elif x.VType == gp.GRB.BINARY:
            x_new = optimization_problem.binary_var(x.VarName)
        elif x.VType == gp.GRB.INTEGER:
            x_new = optimization_problem.integer_var(x.LB, x.UB, x.VarName)
        else:
            raise OptimizationError(f"Unsupported variable type: {x.VarName} {x.VType}")
        var_names[x] = x_new.name

    # objective sense
    minimize = model.ModelSense == gp.GRB.MINIMIZE

    # Retrieve the objective
    objective = model.getObjective()
    has_quadratic_objective = False

    # Retrieve the linear part in case it is a quadratic objective
    if isinstance(objective, gp.QuadExpr):
        linear_part = objective.getLinExpr()
        has_quadratic_objective = True
    else:
        linear_part = objective

    # Get the constant
    constant = linear_part.getConstant()

    # get linear part of objective
    linear = {}
    for i in range(linear_part.size()):
        linear[var_names[linear_part.getVar(i)]] = linear_part.getCoeff(i)

    # get quadratic part of objective
    quadratic = {}
    if has_quadratic_objective:
        for i in range(objective.size()):
            var1 = var_names[cast(QuadExpr, objective).getVar1(i)]
            var2 = var_names[cast(QuadExpr, objective).getVar2(i)]
            coeff = objective.getCoeff(i)
            quadratic[var1, var2] = coeff

    # set objective
    if minimize:
        optimization_problem.minimize(constant, linear, quadratic)
    else:
        optimization_problem.maximize(constant, linear, quadratic)

    # check whether there are any general constraints
    if model.NumSOS > 0 or model.NumGenConstrs > 0:
        raise OptimizationError("Unsupported constraint: SOS or General Constraint")

    # get linear constraints
    for l_constraint in model.getConstrs():
        name = l_constraint.ConstrName
        sense = l_constraint.Sense

        l_left_expr = model.getRow(l_constraint)
        rhs = l_constraint.RHS

        lhs = {}
        for i in range(l_left_expr.size()):
            lhs[var_names[l_left_expr.getVar(i)]] = l_left_expr.getCoeff(i)

        if sense == gp.GRB.EQUAL:
            optimization_problem.linear_constraint(lhs, "==", rhs, name)
        elif sense == gp.GRB.GREATER_EQUAL:
            optimization_problem.linear_constraint(lhs, ">=", rhs, name)
        elif sense == gp.GRB.LESS_EQUAL:
            optimization_problem.linear_constraint(lhs, "<=", rhs, name)
        else:
            raise OptimizationError(f"Unsupported constraint sense: {l_constraint}")

    # get quadratic constraints
    for q_constraint in model.getQConstrs():
        name = q_constraint.QCName
        sense = q_constraint.QCSense

        q_left_expr = model.getQCRow(q_constraint)
        rhs = q_constraint.QCRHS

        linear = {}
        quadratic = {}

        linear_part = q_left_expr.getLinExpr()
        for i in range(linear_part.size()):
            linear[var_names[linear_part.getVar(i)]] = linear_part.getCoeff(i)

        for i in range(q_left_expr.size()):
            var1 = var_names[q_left_expr.getVar1(i)]
            var2 = var_names[q_left_expr.getVar2(i)]
            coeff = q_left_expr.getCoeff(i)
            quadratic[var1, var2] = coeff

        if sense == gp.GRB.EQUAL:
            optimization_problem.quadratic_constraint(linear, quadratic, "==", rhs, name)
        elif sense == gp.GRB.GREATER_EQUAL:
            optimization_problem.quadratic_constraint(linear, quadratic, ">=", rhs, name)
        elif sense == gp.GRB.LESS_EQUAL:
            optimization_problem.quadratic_constraint(linear, quadratic, "<=", rhs, name)
        else:
            raise OptimizationError(f"Unsupported constraint sense: {q_constraint}")

    return optimization_problem
