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

"""Abstract Constraint."""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import Any

from numpy import ndarray

from ..exceptions import OptimizationError
from .optimization_problem_element import OptimizationProblemElement


class ConstraintSense(Enum):
    """Constraint Sense Type."""

    # pylint: disable=invalid-name
    LE = 0
    GE = 1
    EQ = 2

    @staticmethod
    def convert(sense: str | ConstraintSense) -> ConstraintSense:
        """Convert a string into a corresponding sense of constraints.

        Args:
            sense: A string or sense of constraints


        Returns:
            The sense of constraints

        Raises:
            OptimizationError: if the input string is invalid.
        """
        if isinstance(sense, ConstraintSense):
            return sense
        sense = sense.upper()
        if sense not in [
            "E",
            "L",
            "G",
            "EQ",
            "LE",
            "GE",
            "=",
            "==",
            "<=",
            "<",
            ">=",
            ">",
        ]:
            raise OptimizationError(f"Invalid sense: {sense}")
        if sense in ["E", "EQ", "=", "=="]:
            return ConstraintSense.EQ
        if sense in ["L", "LE", "<=", "<"]:
            return ConstraintSense.LE
        return ConstraintSense.GE

    @property
    def label(self) -> str:
        """Label of the constraint sense.

        Returns:
            The label of the constraint sense ('<=', '>=', or '==')
        """
        if self is ConstraintSense.LE:
            return "<="
        if self is ConstraintSense.GE:
            return ">="
        return "=="


class Constraint(OptimizationProblemElement):
    """Abstract Constraint Class."""

    Sense = ConstraintSense

    def __init__(
        self, optimization_problem: Any, name: str, sense: ConstraintSense, rhs: float
    ) -> None:
        """Initializes the constraint.

        Args:
            optimization_problem: The parent OptimizationProblem.
            name: The name of the constraint.
            sense: The sense of the constraint.
            rhs: The right-hand-side of the constraint.
        """
        super().__init__(optimization_problem)
        self._name = name
        self._sense = sense
        self._rhs = rhs

    @property
    def name(self) -> str:
        """Returns the name of the constraint.

        Returns:
            The name of the constraint.
        """
        return self._name

    @property
    def sense(self) -> ConstraintSense:
        """Returns the sense of the constraint.

        Returns:
            The sense of the constraint.
        """
        return self._sense

    @sense.setter
    def sense(self, sense: ConstraintSense) -> None:
        """Sets the sense of the constraint.

        Args:
            sense: The sense of the constraint.
        """
        self._sense = sense

    @property
    def rhs(self) -> float:
        """Returns the right-hand-side of the constraint.

        Returns:
            The right-hand-side of the constraint.
        """
        return self._rhs

    @rhs.setter
    def rhs(self, rhs: float) -> None:
        """Sets the right-hand-side of the constraint.

        Args:
            rhs: The right-hand-side of the constraint.
        """
        self._rhs = rhs

    @abstractmethod
    def evaluate(self, x: ndarray | list | dict[str | int, float]) -> float:
        """Evaluate left-hand-side of constraint for given values of variables.

        Args:
            x: The values to be used for the variables.


        Returns:
            The left-hand-side of the constraint.
        """
        raise NotImplementedError()
