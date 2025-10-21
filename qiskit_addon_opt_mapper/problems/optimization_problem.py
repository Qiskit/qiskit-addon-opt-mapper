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

"""Optimization Problem."""

import logging
import warnings
from collections.abc import Sequence
from enum import Enum
from math import isclose
from typing import cast
from warnings import warn

import numpy as np
from numpy import ndarray
from qiskit.quantum_info import SparsePauliOp
from scipy.sparse import spmatrix

from ..exceptions import OptimizationError
from ..infinity import INFINITY
from .constraint import Constraint, ConstraintSense
from .higher_order_constraint import HigherOrderConstraint
from .linear_constraint import LinearConstraint
from .optimization_objective import OptimizationObjective
from .optimization_problem_element import OptimizationProblemElement
from .quadratic_constraint import QuadraticConstraint
from .variable import Variable, VarType

logger = logging.getLogger(__name__)


CoeffLike = ndarray | dict[tuple[int | str, ...], float] | list


class OptimizationProblemStatus(Enum):
    """Status of OptimizationProblem."""

    VALID = 0
    INFEASIBLE = 1


class OptimizationProblem:
    """Optimization problem representation.

    This representation supports inequality and equality constraints,
    as well as continuous, binary, integer, and spin variables.
    """

    Status = OptimizationProblemStatus

    def __init__(self, name: str = "") -> None:
        """Init method.

        Args:
            name: The name of the optimization problem.
        """
        if not name.isprintable():
            warn("Problem name is not printable", stacklevel=2)
        self._name = ""
        self.name = name
        self._status = OptimizationProblem.Status.VALID

        self._variables: list[Variable] = []
        self._variables_index: dict[str, int] = {}

        self._linear_constraints: list[LinearConstraint] = []
        self._linear_constraints_index: dict[str, int] = {}

        self._quadratic_constraints: list[QuadraticConstraint] = []
        self._quadratic_constraints_index: dict[str, int] = {}

        self._higher_order_constraints: list[HigherOrderConstraint] = []
        self._higher_order_constraints_index: dict[str, int] = {}

        self._objective = OptimizationObjective(self)

    def __repr__(self) -> str:
        """Repr. for OptimizationProblem."""
        from ..translators.prettyprint import DEFAULT_TRUNCATE, expr2str

        objective = expr2str(
            constant=self._objective.constant,
            linear=self.objective.linear,
            quadratic=self._objective.quadratic,
            higher_order=self._objective.higher_order,
            truncate=DEFAULT_TRUNCATE,
        )
        num_constraints = (
            self.get_num_linear_constraints()
            + self.get_num_quadratic_constraints()
            + self.get_num_higher_order_constraints()
        )
        return (
            f"<{self.__class__.__name__}: "
            f"{self.objective.sense.name.lower()} "
            f"{objective}, "
            f"{self.get_num_vars()} variables, "
            f"{num_constraints} constraints, "
            f"'{self._name}'>"
        )

    def __str__(self) -> str:
        """Str. for OptimizationProblem."""
        num_constraints = (
            self.get_num_linear_constraints()
            + self.get_num_quadratic_constraints()
            + self.get_num_higher_order_constraints()
        )
        return (
            f"{self.objective!s} "
            f"({self.get_num_vars()} variables, "
            f"{num_constraints} constraints, "
            f"'{self._name}')"
        )

    def clear(self) -> None:
        """Clears the optimization problem.

        i.e., deletes all variables, constraints, the
        objective function as well as the name.
        """
        self._name = ""
        self._status = OptimizationProblem.Status.VALID

        self._variables.clear()
        self._variables_index.clear()

        self._linear_constraints.clear()
        self._linear_constraints_index.clear()

        self._quadratic_constraints.clear()
        self._quadratic_constraints_index.clear()

        self._higher_order_constraints.clear()
        self._higher_order_constraints_index.clear()

        self._objective = OptimizationObjective(self)

    @property
    def name(self) -> str:
        """Returns the name of the optimization problem.

        Returns:
            The name of the optimization problem.
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """Sets the name of the optimization problem.

        Args:
            name: The name of the optimization problem.
        """
        self._check_name(name, "Problem")
        self._name = name

    @property
    def status(self) -> OptimizationProblemStatus:
        """Status of the optimization problem.

        It can be infeasible due to variable substitution.

        Returns:
            The status of the optimization problem
        """
        return self._status

    @property
    def variables(self) -> list[Variable]:
        """Returns the list of variables of the optimization problem.

        Returns:
            list of variables.
        """
        return self._variables

    @property
    def variables_index(self) -> dict[str, int]:
        """Returns the dictionary that maps the name of a variable to its index.

        Returns:
            The variable index dictionary.
        """
        return self._variables_index

    def _add_variable(
        self,
        lowerbound: float | int,
        upperbound: float | int,
        vartype: VarType,
        name: str | None,
        internal: bool = False,
    ) -> Variable:
        if not name:
            name = "x"
            key_format = "{}"
        else:
            key_format = ""
        return self._add_variables(1, lowerbound, upperbound, vartype, name, key_format, internal)[
            1
        ][0]

    def _add_variables(  # pylint: disable=too-many-positional-arguments
        self,
        keys: int | Sequence,
        lowerbound: float | int,
        upperbound: float | int,
        vartype: VarType,
        name: str | None,
        key_format: str,
        internal: bool = False,
    ) -> tuple[list[str], list[Variable]]:
        """Add variables."""
        if isinstance(keys, int) and keys < 1:
            raise OptimizationError(f"Cannot create non-positive number of variables: {keys}")
        if not name:
            name = "x"
        if "@" in name and not internal:
            warnings.warn(
                f"Variable name '{name}' contains '@', which is reserved for internal variables.",
                stacklevel=2,
            )

        if "{{}}" in key_format:
            raise OptimizationError(f"Formatter cannot contain nested substitutions: {key_format}")
        if key_format.count("{}") > 1:
            raise OptimizationError(
                f"Formatter cannot contain more than one substitution: {key_format}"
            )

        def _find_name(name, key_format, k):
            prev = None
            while True:
                new_name = name + key_format.format(k)
                if new_name == prev:
                    raise OptimizationError(f"Variable name already exists: {new_name}")
                if new_name in self._variables_index:
                    k += 1
                    prev = new_name
                else:
                    break
            return new_name, k + 1

        names = []
        variables = []
        k = self.get_num_vars()
        lst = keys if isinstance(keys, Sequence) else range(keys)
        for key in lst:
            if isinstance(keys, Sequence):
                indexed_name = name + key_format.format(key)
            else:
                indexed_name, k = _find_name(name, key_format, k)
            if indexed_name in self._variables_index:
                raise OptimizationError(f"Variable name already exists: {indexed_name}")
            self._check_name(indexed_name, "Variable")
            names.append(indexed_name)
            self._variables_index[indexed_name] = self.get_num_vars()
            variable = Variable(self, indexed_name, lowerbound, upperbound, vartype)
            self._variables.append(variable)
            variables.append(variable)
        return names, variables

    def _var_dict(  # pylint: disable=too-many-positional-arguments
        self,
        keys: int | Sequence,
        lowerbound: float | int,
        upperbound: float | int,
        vartype: VarType,
        name: str | None,
        key_format: str,
    ) -> dict[str, Variable]:
        """Adds a positive number of variables to the variable list and index.

        Returns a dictionary mapping the variable names to their instances.
        If 'key_format' is present,
        the next 'var_count' available indices are substituted into 'key_format' and appended
        to 'name'.

        Args:
            lowerbound: The lower bound of the variable(s).
            upperbound: The upper bound of the variable(s).
            vartype: The type of the variable(s).
            name: The name(s) of the variable(s).
            key_format: The format used to name/index the variable(s).
            keys: If keys: int, it is interpreted as the number of variables to construct.
                  Otherwise, the elements of the sequence are converted to strings via 'str' and
                  substituted into `key_format`.


        Returns:
            A dictionary mapping the variable names to variable instances.

        Raises:
            OptimizationError: if the variable name is already taken.
            OptimizationError: if less than one variable instantiation is attempted.
            OptimizationError: if `key_format` has more than one substitution or a
                                     nested substitution.
        """
        return dict(
            zip(
                *self._add_variables(keys, lowerbound, upperbound, vartype, name, key_format),
                strict=False,
            )
        )

    def _var_list(  # pylint: disable=too-many-positional-arguments
        self,
        keys: int | Sequence,
        lowerbound: float | int,
        upperbound: float | int,
        vartype: VarType,
        name: str | None,
        key_format: str,
    ) -> list[Variable]:
        """Adds a positive number of variables to the variable list and index.

        Returns a list of variable instances.

        Args:
            lowerbound: The lower bound of the variable(s).
            upperbound: The upper bound of the variable(s).
            vartype: The type of the variable(s).
            name: The name(s) of the variable(s).
            key_format: The format used to name/index the variable(s).
            keys: If keys: int, it is interpreted as the number of variables to construct.
                  Otherwise, the elements of the sequence are converted to strings via 'str' and
                  substituted into `key_format`.


        Returns:
            A dictionary mapping the variable names to variable instances.

        Raises:
            OptimizationError: if the variable name is already taken.
            OptimizationError: if less than one variable instantiation is attempted.
            OptimizationError: if `key_format` has more than one substitution or a
                                     nested substitution.
        """
        return self._add_variables(keys, lowerbound, upperbound, vartype, name, key_format)[1]

    def continuous_var(
        self,
        lowerbound: float | int = 0,
        upperbound: float | int = INFINITY,
        name: str | None = None,
    ) -> Variable:
        """Adds a continuous variable to the optimization problem.

        Args:
            lowerbound: The lowerbound of the variable.
            upperbound: The upperbound of the variable.
            name: The name of the variable.
                If it's ``None`` or empty ``""``, the default name, e.g., ``x0``, is used.


        Returns:
            The added variable.

        Raises:
            OptimizationError: if the variable name is already occupied.
        """
        return self._add_variable(lowerbound, upperbound, Variable.Type.CONTINUOUS, name)

    def continuous_var_dict(  # pylint: disable=too-many-positional-arguments
        self,
        keys: int | Sequence,
        lowerbound: float | int = 0,
        upperbound: float | int = INFINITY,
        name: str | None = None,
        key_format: str = "{}",
    ) -> dict[str, Variable]:
        """Uses 'var_dict' to construct a dictionary of continuous variables.

        Args:
            lowerbound: The lower bound of the variable(s).
            upperbound: The upper bound of the variable(s).
            name: The name(s) of the variable(s).
                If it's ``None`` or empty ``""``, the default name, e.g., ``x0``, is used.
            key_format: The format used to name/index the variable(s).
            keys: If keys: int, it is interpreted as the number of variables to construct.
                  Otherwise, the elements of the sequence are converted to strings via 'str' and
                  substituted into `key_format`.


        Returns:
            A dictionary mapping the variable names to variable instances.

        Raises:
            OptimizationError: if the variable name is already taken.
            OptimizationError: if less than one variable instantiation is attempted.
            OptimizationError: if `key_format` has more than one substitution or a
                                     nested substitution.
        """
        return self._var_dict(
            keys=keys,
            lowerbound=lowerbound,
            upperbound=upperbound,
            vartype=Variable.Type.CONTINUOUS,
            name=name,
            key_format=key_format,
        )

    def continuous_var_list(  # pylint: disable=too-many-positional-arguments
        self,
        keys: int | Sequence,
        lowerbound: float | int = 0,
        upperbound: float | int = INFINITY,
        name: str | None = None,
        key_format: str = "{}",
    ) -> list[Variable]:
        """Uses 'var_list' to construct a list of continuous variables.

        Args:
            lowerbound: The lower bound of the variable(s).
            upperbound: The upper bound of the variable(s).
            name: The name(s) of the variable(s).
                If it's ``None`` or empty ``""``, the default name, e.g., ``x0``, is used.
            key_format: The format used to name/index the variable(s).
            keys: If keys: int, it is interpreted as the number of variables to construct.
                  Otherwise, the elements of the sequence are converted to strings via 'str' and
                  substituted into `key_format`.


        Returns:
            A list of variable instances.

        Raises:
            OptimizationError: if the variable name is already taken.
            OptimizationError: if less than one variable instantiation is attempted.
            OptimizationError: if `key_format` has more than one substitution or a
                                     nested substitution.
        """
        return self._var_list(
            keys, lowerbound, upperbound, Variable.Type.CONTINUOUS, name, key_format
        )

    def binary_var(self, name: str | None = None) -> Variable:
        """Adds a binary variable to the optimization problem.

        Args:
            name: The name of the variable.
                If it's ``None`` or empty ``""``, the default name, e.g., ``x0``, is used.


        Returns:
            The added variable.

        Raises:
            OptimizationError: if the variable name is already occupied.
        """
        return self._add_variable(0, 1, Variable.Type.BINARY, name)

    def binary_var_dict(
        self,
        keys: int | Sequence,
        name: str | None = None,
        key_format: str = "{}",
    ) -> dict[str, Variable]:
        """Uses 'var_dict' to construct a dictionary of binary variables.

        Args:
            name: The name(s) of the variable(s).
                If it's ``None`` or empty ``""``, the default name, e.g., ``x0``, is used.
            key_format: The format used to name/index the variable(s).
            keys: If keys: int, it is interpreted as the number of variables to construct.
                  Otherwise, the elements of the sequence are converted to strings via 'str' and
                  substituted into `key_format`.


        Returns:
            A dictionary mapping the variable names to variable instances.

        Raises:
            OptimizationError: if the variable name is already taken.
            OptimizationError: if less than one variable instantiation is attempted.
            OptimizationError: if `key_format` has more than one substitution or a
                                     nested substitution.
        """
        return self._var_dict(
            keys=keys,
            lowerbound=0,
            upperbound=1,
            vartype=Variable.Type.BINARY,
            name=name,
            key_format=key_format,
        )

    def binary_var_list(
        self,
        keys: int | Sequence,
        name: str | None = None,
        key_format: str = "{}",
    ) -> list[Variable]:
        """Uses 'var_list' to construct a list of binary variables.

        Args:
            name: The name(s) of the variable(s).
                If it's ``None`` or empty ``""``, the default name, e.g., ``x0``, is used.
            key_format: The format used to name/index the variable(s).
            keys: If keys: int, it is interpreted as the number of variables to construct.
                  Otherwise, the elements of the sequence are converted to strings via 'str' and
                  substituted into `key_format`.


        Returns:
            A list of variable instances.

        Raises:
            OptimizationError: if the variable name is already taken.
            OptimizationError: if less than one variable instantiation is attempted.
            OptimizationError: if `key_format` has more than one substitution or a
                                     nested substitution.
        """
        return self._var_list(keys, 0, 1, Variable.Type.BINARY, name, key_format)

    def integer_var(
        self,
        lowerbound: float | int = 0,
        upperbound: float | int = INFINITY,
        name: str | None = None,
    ) -> Variable:
        """Adds an integer variable to the optimization problem.

        Args:
            lowerbound: The lowerbound of the variable.
            upperbound: The upperbound of the variable.
            name: The name of the variable.
                If it's ``None`` or empty ``""``, the default name, e.g., ``x0``, is used.


        Returns:
            The added variable.

        Raises:
            OptimizationError: if the variable name is already occupied.
        """
        return self._add_variable(lowerbound, upperbound, Variable.Type.INTEGER, name)

    def integer_var_dict(  # pylint: disable=too-many-positional-arguments
        self,
        keys: int | Sequence,
        lowerbound: float | int = 0,
        upperbound: float | int = INFINITY,
        name: str | None = None,
        key_format: str = "{}",
    ) -> dict[str, Variable]:
        """Uses 'var_dict' to construct a dictionary of integer variables.

        Args:
            lowerbound: The lower bound of the variable(s).
            upperbound: The upper bound of the variable(s).
            name: The name(s) of the variable(s).
                If it's ``None`` or empty ``""``, the default name, e.g., ``x0``, is used.
            key_format: The format used to name/index the variable(s).
            keys: If keys: int, it is interpreted as the number of variables to construct.
                  Otherwise, the elements of the sequence are converted to strings via 'str' and
                  substituted into `key_format`.


        Returns:
            A dictionary mapping the variable names to variable instances.

        Raises:
            OptimizationError: if the variable name is already taken.
            OptimizationError: if less than one variable instantiation is attempted.
            OptimizationError: if `key_format` has more than one substitution or a
                                     nested substitution.
        """
        return self._var_dict(
            keys=keys,
            lowerbound=lowerbound,
            upperbound=upperbound,
            vartype=Variable.Type.INTEGER,
            name=name,
            key_format=key_format,
        )

    def integer_var_list(  # pylint: disable=too-many-positional-arguments
        self,
        keys: int | Sequence,
        lowerbound: float | int = 0,
        upperbound: float | int = INFINITY,
        name: str | None = None,
        key_format: str = "{}",
    ) -> list[Variable]:
        """Uses 'var_list' to construct a list of integer variables.

        Args:
            lowerbound: The lower bound of the variable(s).
            upperbound: The upper bound of the variable(s).
            name: The name(s) of the variable(s).
                If it's ``None`` or empty ``""``, the default name, e.g., ``x0``, is used.
            key_format: The format used to name/index the variable(s).
            keys: If keys: int, it is interpreted as the number of variables to construct.
                  Otherwise, the elements of the sequence are converted to strings via 'str' and
                  substituted into `key_format`.


        Returns:
            A list of variable instances.

        Raises:
            OptimizationError: if the variable name is already taken.
            OptimizationError: if less than one variable instantiation is attempted.
            OptimizationError: if `key_format` has more than one substitution or a
                                     nested substitution.
        """
        return self._var_list(keys, lowerbound, upperbound, Variable.Type.INTEGER, name, key_format)

    def spin_var(self, name: str | None = None) -> Variable:
        """Adds a spin variable to the optimization program.

        Args:
            name: The name of the variable.
                If it's ``None`` or empty ``""``, the default name, e.g., ``x0``, is used.


        Returns:
            The added variable.

        Raises:
            OptimizationError: if the variable name is already occupied.
        """
        return self._add_variable(-1, 1, Variable.Type.SPIN, name)

    def spin_var_dict(
        self,
        keys: int | Sequence,
        name: str | None = None,
        key_format: str = "{}",
    ) -> dict[str, Variable]:
        """Uses 'var_dict' to construct a dictionary of spin variables.

        Args:
            keys: If keys: int, it is interpreted as the number of variables to construct.
                Otherwise, the elements of the sequence are converted to strings via 'str' and
                substituted into `key_format`.
            name: The name(s) of the variable(s).
                If it's ``None`` or empty ``""``, the default name, e.g., ``x0``, is used.
            key_format: The format used to name/index the variable(s).



        Returns:
            A dictionary mapping the variable names to variable instances.

        Raises:
            OptimizationError: if the variable name is already taken.
            OptimizationError: if less than one variable instantiation is attempted.
            OptimizationError: if `key_format` has more than one substitution or a
                                     nested substitution.
        """
        return self._var_dict(
            keys=keys,
            lowerbound=-1,
            upperbound=1,
            vartype=Variable.Type.SPIN,
            name=name,
            key_format=key_format,
        )

    def spin_var_list(
        self,
        keys: int | Sequence,
        name: str | None = None,
        key_format: str = "{}",
    ) -> list[Variable]:
        """Uses 'var_list' to construct a list of spin variables.

        Args:
            keys: If keys: int, it is interpreted as the number of variables to construct.
                Otherwise, the elements of the sequence are converted to strings via 'str' and
                substituted into `key_format`.
            name: The name(s) of the variable(s).
                If it's ``None`` or empty ``""``, the default name, e.g., ``x0``, is used.
            key_format: The format used to name/index the variable(s).


        Returns:
            A list of variable instances.

        Raises:
            OptimizationError: if the variable name is already taken.
            OptimizationError: if less than one variable instantiation is attempted.
            OptimizationError: if `key_format` has more than one substitution or a
                                     nested substitution.
        """
        return self._var_list(keys, -1, 1, Variable.Type.SPIN, name, key_format)

    def get_variable(self, i: int | str) -> Variable:
        """Returns a variable for a given name or index.

        Args:
            i: the index or name of the variable.


        Returns:
            The corresponding variable.
        """
        if isinstance(i, int | np.integer):
            return self.variables[i]
        return self.variables[self._variables_index[i]]

    def get_num_vars(self, vartype: VarType | None = None) -> int:
        """Returns the total number of variables or the number of variables of the specified type.

        Args:
            vartype: The type to be filtered on. All variables are counted if None.


        Returns:
            The total number of variables.
        """
        if vartype:
            return sum(variable.vartype == vartype for variable in self._variables)
        return len(self._variables)

    def get_num_continuous_vars(self) -> int:
        """Returns the total number of continuous variables.

        Returns:
            The total number of continuous variables.
        """
        return self.get_num_vars(Variable.Type.CONTINUOUS)

    def get_num_binary_vars(self) -> int:
        """Returns the total number of binary variables.

        Returns:
            The total number of binary variables.
        """
        return self.get_num_vars(Variable.Type.BINARY)

    def get_num_integer_vars(self) -> int:
        """Returns the total number of integer variables.

        Returns:
            The total number of integer variables.
        """
        return self.get_num_vars(Variable.Type.INTEGER)

    def get_num_spin_vars(self) -> int:
        """Returns the total number of spin variables.

        Returns:
            The total number of spin variables.
        """
        return self.get_num_vars(Variable.Type.SPIN)

    @property
    def linear_constraints(self) -> list[LinearConstraint]:
        """Returns the list of linear constraints of the optimization problem.

        Returns:
            List of linear constraints.
        """
        return self._linear_constraints

    @property
    def linear_constraints_index(self) -> dict[str, int]:
        """Returns the dictionary that maps the name of a linear constraint to its index.

        Returns:
            The linear constraint index dictionary.
        """
        return self._linear_constraints_index

    def linear_constraint(
        self,
        linear: ndarray | spmatrix | list[float] | dict[int | str, float] = None,
        sense: str | ConstraintSense = "<=",
        rhs: float = 0.0,
        name: str | None = None,
    ) -> LinearConstraint:
        """Adds a linear equality constraint to the optimization problem.

        The constraint is of the form:
            ``(linear * x) sense rhs``.

        Args:
            linear: The linear coefficients of the left-hand side of the constraint.
            sense: The sense of the constraint,

              - ``==``, ``=``, ``E``, and ``EQ`` denote 'equal to'.
              - ``>=``, ``>``, ``G``, and ``GE`` denote 'greater-than-or-equal-to'.
              - ``<=``, ``<``, ``L``, and ``LE`` denote 'less-than-or-equal-to'.

            rhs: The right-hand side of the constraint.
            name: The name of the constraint.
                If it's ``None`` or empty ``""``, the default name, e.g., ``c0``, is used.


        Returns:
            The added constraint.

        Raises:
            OptimizationError: if the constraint name already exists or the sense is not
                valid.
        """
        if name:
            if name in self.linear_constraints_index:
                raise OptimizationError(f"Linear constraint's name already exists: {name}")
            self._check_name(name, "Linear constraint")
        else:
            k = self.get_num_linear_constraints()
            while f"c{k}" in self.linear_constraints_index:
                k += 1
            name = f"c{k}"
        self.linear_constraints_index[name] = len(self.linear_constraints)
        if linear is None:
            linear = {}
        constraint = LinearConstraint(self, name, linear, Constraint.Sense.convert(sense), rhs)
        self.linear_constraints.append(constraint)
        return constraint

    def get_linear_constraint(self, i: int | str) -> LinearConstraint:
        """Returns a linear constraint for a given name or index.

        Args:
            i: the index or name of the constraint.


        Returns:
            The corresponding constraint.

        Raises:
            IndexError: if the index is out of the list size
            KeyError: if the name does not exist
        """
        if isinstance(i, int):
            return self._linear_constraints[i]
        return self._linear_constraints[self._linear_constraints_index[i]]

    def get_num_linear_constraints(self) -> int:
        """Returns the number of linear constraints.

        Returns:
            The number of linear constraints.
        """
        return len(self._linear_constraints)

    @property
    def quadratic_constraints(self) -> list[QuadraticConstraint]:
        """Returns the list of quadratic constraints of the optimization problem.

        Returns:
            List of quadratic constraints.
        """
        return self._quadratic_constraints

    @property
    def quadratic_constraints_index(self) -> dict[str, int]:
        """Returns the dictionary that maps the name of a quadratic constraint to its index.

        Returns:
            The quadratic constraint index dictionary.
        """
        return self._quadratic_constraints_index

    def quadratic_constraint(  # pylint: disable=too-many-positional-arguments
        self,
        linear: ndarray | spmatrix | list[float] | dict[int | str, float] = None,
        quadratic: (
            ndarray | spmatrix | list[list[float]] | dict[tuple[int | str, int | str], float]
        ) = None,
        sense: str | ConstraintSense = "<=",
        rhs: float = 0.0,
        name: str | None = None,
    ) -> QuadraticConstraint:
        """Adds a quadratic equality constraint to the optimization problem.

        The constraint is of the form:
            ``(x * quadratic * x + linear * x) sense rhs``.

        Args:
            linear: The linear coefficients of the constraint.
            quadratic: The quadratic coefficients of the constraint.
            sense: The sense of the constraint,

              - ``==``, ``=``, ``E``, and ``EQ`` denote 'equal to'.
              - ``>=``, ``>``, ``G``, and ``GE`` denote 'greater-than-or-equal-to'.
              - ``<=``, ``<``, ``L``, and ``LE`` denote 'less-than-or-equal-to'.

            rhs: The right-hand side of the constraint.
            name: The name of the constraint.
                If it's ``None`` or empty ``""``, the default name, e.g., ``q0``, is used.


        Returns:
            The added constraint.

        Raises:
            OptimizationError: if the constraint name already exists.
        """
        if name:
            if name in self.quadratic_constraints_index:
                raise OptimizationError(f"Quadratic constraint name already exists: {name}")
            self._check_name(name, "Quadratic constraint")
        else:
            k = self.get_num_quadratic_constraints()
            while f"q{k}" in self.quadratic_constraints_index:
                k += 1
            name = f"q{k}"
        self.quadratic_constraints_index[name] = len(self.quadratic_constraints)
        if linear is None:
            linear = {}
        if quadratic is None:
            quadratic = {}
        constraint = QuadraticConstraint(
            self, name, linear, quadratic, Constraint.Sense.convert(sense), rhs
        )
        self.quadratic_constraints.append(constraint)
        return constraint

    def get_quadratic_constraint(self, i: int | str) -> QuadraticConstraint:
        """Returns a quadratic constraint for a given name or index.

        Args:
            i: the index or name of the constraint.


        Returns:
            The corresponding constraint.

        Raises:
            IndexError: if the index is out of the list size
            KeyError: if the name does not exist
        """
        if isinstance(i, int):
            return self._quadratic_constraints[i]
        return self._quadratic_constraints[self._quadratic_constraints_index[i]]

    def get_num_quadratic_constraints(self) -> int:
        """Returns the number of quadratic constraints.

        Returns:
            The number of quadratic constraints.
        """
        return len(self._quadratic_constraints)

    @property
    def higher_order_constraints(self) -> list[HigherOrderConstraint]:
        """Returns the list of higher-order constraints."""
        return self._higher_order_constraints

    @property
    def higher_order_constraints_index(self) -> dict[str, int]:
        """Returns the name -> index map for higher-order constraints."""
        return self._higher_order_constraints_index

    def higher_order_constraint(  # pylint: disable=too-many-positional-arguments
        self,
        linear: ndarray | spmatrix | list[float] | dict[int | str, float] = None,
        quadratic: (
            ndarray | spmatrix | list[list[float]] | dict[tuple[int | str, int | str], float]
        ) = None,
        higher_order: dict[int, CoeffLike] | None = None,
        sense: str | ConstraintSense = "<=",
        rhs: float = 0.0,
        name: str | None = None,
    ) -> HigherOrderConstraint:
        """Adds a higher-order constraint.

        e.g. linear(x) + x^T Q x + sum_{k>=3}  sum_{|t|=k} C_k[t] * prod_{i in t} x[i] `sense` `rhs`
        where `sense` is one of the ConstraintSense values (e.g., LE, <=) and `rhs` is a float.
        Supports both a single higher-order term (order+coeffs) and multiple via
        higher_orders={k: coeffs}.

        Args:
            linear: The coefficients for the linear part of the constraint.
            quadratic: The coefficients for the quadratic part of the constraint.
            higher_order: A single higher-order expression or a dictionary of {order: coeffs}
                for multiple orders (k>=3).
            sense: The sense of the constraint (e.g., LE, <=).
            rhs: The right-hand-side value of the constraint.
            name: The name of the constraint.
                If it's ``None`` or empty ``""``, the default name, e.g., ``h0``, is used.
        """
        # Handle constraint name
        if name:
            if name in self.higher_order_constraints_index:
                raise OptimizationError(f"Higher-order constraint name already exists: {name}")
            self._check_name(name, "Higher-order constraint")
        else:
            k = self.get_num_higher_order_constraints()
            while f"h{k}" in self.higher_order_constraints_index:
                k += 1
            name = f"h{k}"

        # Default empty expressions if None
        if linear is None:
            linear = {}
        if quadratic is None:
            quadratic = {}

        # Build the constraint
        if higher_order is not None:
            con = HigherOrderConstraint(
                self,
                name=name,
                linear=linear,
                quadratic=quadratic,
                higher_order=higher_order,
                sense=Constraint.Sense.convert(sense),
                rhs=rhs,
            )

        # Register the constraint
        self.higher_order_constraints_index[name] = len(self.higher_order_constraints)
        self.higher_order_constraints.append(con)
        return con

    def get_higher_order_constraint(self, i: int | str) -> HigherOrderConstraint:
        """Returns a higher-order constraint for a given name or index.

        Args:
            i: the index or name of the constraint.


        Returns:
            The corresponding constraint.
        """
        if isinstance(i, int):
            return self._higher_order_constraints[i]
        return self._higher_order_constraints[self._higher_order_constraints_index[i]]

    def get_num_higher_order_constraints(self) -> int:
        """Returns the number of higher-order constraints.

        Returns:
        The number of higher-order constraints.
        """
        return len(self._higher_order_constraints)

    # (optional) iterator helper used by pretty-printer fallback
    def iter_higher_order_constraints(self):
        """Yields all higher-order constraints (for pretty printing)."""
        return iter(self._higher_order_constraints)

    def remove_linear_constraint(self, i: str | int) -> None:
        """Remove a linear constraint.

        Args:
            i: an index or a name of a linear constraint

        Raises:
            KeyError: if name does not exist
            IndexError: if index is out of range
        """
        if isinstance(i, str):
            i = self._linear_constraints_index[i]
        del self._linear_constraints[i]
        self._linear_constraints_index = {
            cst.name: j for j, cst in enumerate(self._linear_constraints)
        }

    def remove_quadratic_constraint(self, i: str | int) -> None:
        """Remove a quadratic constraint.

        Args:
            i: an index or a name of a quadratic constraint

        Raises:
            KeyError: if name does not exist
            IndexError: if index is out of range
        """
        if isinstance(i, str):
            i = self._quadratic_constraints_index[i]
        del self._quadratic_constraints[i]
        self._quadratic_constraints_index = {
            cst.name: j for j, cst in enumerate(self._quadratic_constraints)
        }

    def remove_higher_order_constraint(self, i: str | int) -> None:
        """Remove a higher-order constraint.

        Args:
            i: an index or a name of a higher-order constraint
        Raises:
            KeyError: if name does not exist
            IndexError: if index is out of range.
        """
        if isinstance(i, str):
            i = self._higher_order_constraints_index[i]
        del self._higher_order_constraints[i]
        self._higher_order_constraints_index = {
            cst.name: j for j, cst in enumerate(self._higher_order_constraints)
        }

    @property
    def objective(self) -> OptimizationObjective:
        """Returns the quadratic objective.

        Returns:
            The quadratic objective.
        """
        return self._objective

    def minimize(
        self,
        constant: float = 0.0,
        linear: ndarray | spmatrix | list[float] | dict[int | str, float] = None,
        quadratic: (
            ndarray | spmatrix | list[list[float]] | dict[tuple[int | str, int | str], float]
        ) = None,
        higher_order: CoeffLike | dict[int, CoeffLike] | None = None,
    ) -> None:
        """Sets an objective to be minimized."""
        self._objective = OptimizationObjective(
            self,
            constant=constant,
            linear=linear,
            quadratic=quadratic,
            sense=OptimizationObjective.Sense.MINIMIZE,
            higher_order=higher_order,  # type: ignore
        )

    def maximize(
        self,
        constant: float = 0.0,
        linear: ndarray | spmatrix | list[float] | dict[int | str, float] = None,
        quadratic: (
            ndarray | spmatrix | list[list[float]] | dict[tuple[int | str, int | str], float]
        ) = None,
        higher_order: CoeffLike | dict[int, CoeffLike] | None = None,
    ) -> None:
        """Sets an objective to be maximized."""
        self._objective = OptimizationObjective(
            self,
            constant=constant,
            linear=linear,
            quadratic=quadratic,
            sense=OptimizationObjective.Sense.MAXIMIZE,
            higher_order=higher_order,  # type: ignore
        )

    def _copy_from(self, other: "OptimizationProblem", include_name: bool) -> None:
        """Copy another OptimizationProblem to this updating OptimizationProblemElement.

        Note: this breaks the consistency of `other`. You cannot use `other` after the copy.

        Args:
            other: The optimization problem to be copied from.
            include_name: Whether this method copies the problem name or not.
        """
        for attr, val in vars(other).items():
            if attr == "_name" and not include_name:
                continue
            if isinstance(val, OptimizationProblemElement):
                val.optimization_problem = self
            if isinstance(val, list):
                for elem in val:
                    if isinstance(elem, OptimizationProblemElement):
                        elem.optimization_problem = self
            setattr(self, attr, val)

    def substitute_variables(
        self,
        constants: dict[str | int, float] | None = None,
        variables: dict[str | int, tuple[str | int, float]] | None = None,
    ) -> "OptimizationProblem":
        """Substitutes variables with constants or other variables.

        Args:
            constants: replace variable by constant
                e.g., ``{'x': 2}`` means ``x`` is substituted with 2

            variables: replace variables by weighted other variable
                need to copy everything using name reference to make sure that indices are matched
                correctly. The lower and upper bounds are updated accordingly.
                e.g., ``{'x': ('y', 2)}`` means ``x`` is substituted with ``y * 2``


        Returns:
            An optimization problem by substituting variables with constants or other variables.
            If the substitution is valid, ``OptimizationProblem.status`` is still
            ``OptimizationProblem.Status.VALID``.
            Otherwise, it gets ``OptimizationProblem.Status.INFEASIBLE``.

        Raises:
            OptimizationError: if the substitution is invalid as follows.

                - Same variable is substituted multiple times.
                - Coefficient of variable substitution is zero.
        """
        # pylint: disable=cyclic-import
        from .substitute_variables import substitute_variables

        return substitute_variables(self, constants, variables)

    def to_ising(self) -> tuple[SparsePauliOp, float]:
        """Return the Ising Hamiltonian of this problem.

        Variables are mapped to qubits in the same order, i.e.,
        i-th variable is mapped to i-th qubit.
        See https://github.com/Qiskit/qiskit-terra/issues/1148 for details.


        Returns:
            qubit_op: The qubit operator for the problem
            offset: The constant value in the Ising Hamiltonian.

        Raises:
            OptimizationError: If a variable type is not binary.
            OptimizationError: If constraints exist in the problem.
        """
        # pylint: disable=cyclic-import
        from ..translators.ising import to_ising

        return to_ising(self)

    def get_feasibility_info(
        self, x: list[float] | np.ndarray
    ) -> tuple[bool, list[Variable], list[Constraint]]:
        """Returns whether a solution is feasible or not along with the violations.

        Args:
            x: a solution value, such as returned in an optimizer result.

        Returns:
            feasible: Whether the solution provided is feasible or not.
            list[Variable]: List of variables which are violated.
            list[Constraint]: List of constraints which are violated.

        Raises:
            OptimizationError: If the input `x` is not same len as total vars
        """
        # if input `x` is not the same len as the total vars, raise an error
        if len(x) != self.get_num_vars():
            raise OptimizationError(
                f"The size of solution `x`: {len(x)}, does not match the number of "
                f"problem variables: {self.get_num_vars()}"
            )

        # check whether the input satisfy the bounds of the problem
        violated_variables = []
        for i, val in enumerate(x):
            variable = self.get_variable(i)
            if val < variable.lowerbound or variable.upperbound < val:
                violated_variables.append(variable)

        # check whether the input satisfy the constraints of the problem
        violated_constraints = []
        for constraint in cast(list[Constraint], self._linear_constraints) + cast(
            list[Constraint], self._quadratic_constraints
        ):
            lhs = constraint.evaluate(x)
            if (
                (constraint.sense == ConstraintSense.LE and lhs > constraint.rhs)
                or (constraint.sense == ConstraintSense.GE and lhs < constraint.rhs)
                or (constraint.sense == ConstraintSense.EQ and not isclose(lhs, constraint.rhs))
            ):
                violated_constraints.append(constraint)

        for constraint in cast(list[Constraint], self._higher_order_constraints):
            lhs = constraint.evaluate(x)
            if (
                (constraint.sense == ConstraintSense.LE and lhs > constraint.rhs)
                or (constraint.sense == ConstraintSense.GE and lhs < constraint.rhs)
                or (constraint.sense == ConstraintSense.EQ and not isclose(lhs, constraint.rhs))
            ):
                violated_constraints.append(constraint)

        feasible = not violated_variables and not violated_constraints

        return feasible, violated_variables, violated_constraints

    def is_feasible(self, x: list[float] | np.ndarray) -> bool:
        """Returns whether a solution is feasible or not.

        Args:
            x: a solution value, such as returned in an optimizer result.


        Returns:
            ``True`` if the solution provided is feasible otherwise ``False``.

        """
        feasible, _, _ = self.get_feasibility_info(x)

        return feasible

    def prettyprint(self, wrap: int = 80) -> str:
        """Returns a pretty printed string of this problem.

        Args:
            wrap: The text width to wrap the output strings. It is disabled by setting 0.
                Note that some strings might exceed this value, for example, a long variable
                name won't be wrapped. The default value is 80.


        Returns:
            A pretty printed string representing the problem.

        Raises:
            OptimizationError: if there is a non-printable name.
        """
        # pylint: disable=cyclic-import
        from qiskit_addon_opt_mapper.translators.prettyprint import prettyprint

        return prettyprint(self, wrap)

    @staticmethod
    def _check_name(name: str, name_type: str) -> None:
        """Displays a warning message if a name string is not printable."""
        if not name.isprintable():
            warn(f"{name_type} name is not printable: {name!r}", stacklevel=2)
