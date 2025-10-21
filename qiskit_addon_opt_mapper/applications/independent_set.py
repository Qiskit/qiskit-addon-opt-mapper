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

"""An application class for the independent set."""

import numpy as np
from docplex.mp.model import Model
from rustworkx import visualization as rx_visualization

from qiskit_addon_opt_mapper.problems.optimization_problem import OptimizationProblem
from qiskit_addon_opt_mapper.translators import from_docplex_mp

from .graph_optimization_application import GraphOptimizationApplication


class IndependentSet(GraphOptimizationApplication):
    """Optimization application for the "independent set" [1] problem based on a NetworkX graph.

    References:
        [1]: "Independent set (graph theory)",
        `https://en.wikipedia.org/wiki/Independent_set_(graph_theory)
        <https://en.wikipedia.org/wiki/Independent_set_(graph_theory)>`_
    """

    def to_optimization_problem(self) -> OptimizationProblem:
        """Represent as an optimization problem.

        Convert a independent set instance into a
        :class:`~qiskit_addon_opt_mapper.problems.OptimizationProblem`


        Returns:
            The :class:`~qiskit_addon_opt_mapper.problems.OptimizationProblem` created
            from the independent set instance.
        """
        mdl = Model(name="Independent set")
        n = self._graph.num_nodes()
        x = {i: mdl.binary_var(name=f"x_{i}") for i in range(n)}
        objective = mdl.sum(x[i] for i in x)
        for w, v in self._graph.edge_list():
            mdl.add_constraint(x[w] + x[v] <= 1)
        mdl.maximize(objective)
        op = from_docplex_mp(mdl)
        return op

    def interpret(self, result: np.ndarray) -> list[int]:
        """Interpret a result as a list of node indices.

        Args:
            result : The calculated result of the problem


        Returns:
            A list of node indices whose corresponding variable is 1
        """
        x = self._result_to_x(result)
        independent_set = []
        for i, value in enumerate(x):
            if value:
                independent_set.append(i)
        return independent_set

    def _draw_result(
        self,
        result: np.ndarray,
        pos: dict[int, np.ndarray] | None = None,
    ) -> None:
        """Draw the result with colors.

        Args:
            result : The calculated result for the problem
            pos: The positions of nodes
        """
        x = self._result_to_x(result)
        rx_visualization.mpl_draw(
            self._graph,
            node_color=self._node_colors(x),
            pos=pos,  # type: ignore
            with_labels=True,
        )

    def _node_colors(self, x: np.ndarray):
        # Return a list of strings for draw.
        # Color a node with red when the corresponding variable is 1.
        # Otherwise color it with dark gray.
        return ["r" if x[node] == 1 else "darkgrey" for node in self._graph.node_indices()]
