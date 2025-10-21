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

"""An abstract class for optimization application classes."""

from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.result import QuasiDistribution

from qiskit_addon_opt_mapper.problems.optimization_problem import OptimizationProblem


class OptimizationApplication(ABC):
    """An abstract class for optimization applications."""

    @abstractmethod
    def to_optimization_problem(self) -> OptimizationProblem:
        """Represent as an optimization problem.

        Convert a problem instance into a
        :class:`~qiskit_addon_opt_mapper.problems.OptimizationProblem`
        """
        pass

    @abstractmethod
    def interpret(self, result: np.ndarray):
        """Interpret the problem.

        Convert the calculation result of the problem
        (:class:`~qiskit_addon_opt_mapper.solvers.SolverResult` or a binary array using
        np.ndarray) to the answer of the problem in an easy-to-understand format.

        Args:
            result: The calculated result of the problem
        """
        pass

    def _result_to_x(self, result: np.ndarray) -> np.ndarray:
        """Hook to support different result formats in the future."""
        if isinstance(result, np.ndarray):
            x = result
        else:
            raise TypeError(
                "Unsupported format of result. Provide a",
                f" binary array using np.ndarray instead of {type(result)}",
            )
        return x

    @staticmethod
    def sample_most_likely(
        state_vector: QuasiDistribution | Statevector | np.ndarray | dict,
    ) -> np.ndarray:
        """Compute the most likely binary string from state vector.

        Args:
            state_vector: state vector or counts or quasi-probabilities.


        Returns:
            binary string as numpy.ndarray of ints.

        Raises:
            ValueError: if state_vector is not QuasiDistribution, Statevector,
                np.ndarray, or dict.
        """
        if isinstance(state_vector, QuasiDistribution):
            probabilities = state_vector.binary_probabilities()
            binary_string = max(probabilities.items(), key=lambda kv: kv[1])[0]
            x = np.asarray([int(y) for y in reversed(list(binary_string))])
            return x
        if isinstance(state_vector, Statevector):
            probabilities = state_vector.probabilities()
            n = state_vector.num_qubits
            k = np.argmax(np.abs(probabilities))
            x = np.zeros(n)
            for i in range(n):
                x[i] = k % 2
                k >>= 1
            return x
        if isinstance(state_vector, OrderedDict | dict):
            # get the binary string with the largest count
            binary_string = max(state_vector.items(), key=lambda kv: kv[1])[0]
            x = np.asarray([int(y) for y in reversed(list(binary_string))])
            return x
        if isinstance(state_vector, np.ndarray):
            n = int(np.log2(state_vector.shape[0]))
            k = np.argmax(np.abs(state_vector))
            x = np.zeros(n)
            for i in range(n):
                x[i] = k % 2
                k >>= 1
            return x
        raise ValueError(
            "state vector should be QuasiDistribution, Statevector, ndarray, or dict. "
            f"But it is {type(state_vector)}."
        )
