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

"""An application class for the graph partitioning."""

import numpy as np
from docplex.mp.model import Model
from rustworkx import visualization as rx_visualization

from qiskit_addon_opt_mapper.problems.optimization_problem import OptimizationProblem
from qiskit_addon_opt_mapper.translators import from_docplex_mp

from .graph_optimization_application import GraphOptimizationApplication


class GraphPartition(GraphOptimizationApplication):
    """Optimization application for the "graph partition" [1] problem based on a NetworkX graph.

    References:
        [1]: "Graph partition", https://en.wikipedia.org/wiki/Graph_partition
    """

    def to_optimization_problem(self) -> OptimizationProblem:
        """Represent as an optimization problem.

        Convert a graph partition instance into a
        :class:`~qiskit_addon_opt_mapper.problems.OptimizationProblem`


        Returns:
            The :class:`~qiskit_addon_opt_mapper.problems.OptimizationProblem` created
            from the graph partition instance.
        """
        mdl = Model(name="Graph partition")
        n = self._graph.num_nodes()
        x = {i: mdl.binary_var(name=f"x_{i}") for i in range(n)}
        objective = mdl.sum(
            (self._graph.get_edge_data(i, j) if self._graph.get_edge_data(i, j) is not None else 1)
            * (x[i] + x[j] - 2 * x[i] * x[j])
            for i, j in self._graph.edge_list()
        )
        mdl.minimize(objective)
        mdl.add_constraint(mdl.sum([x[i] for i in x]) == n // 2)
        op = from_docplex_mp(mdl)
        return op

    def interpret(self, result: np.ndarray) -> list[list[int]]:
        """Interpret a result as a list of node indices.

        Args:
            result : The calculated result of the problem


        Returns:
            A list of node indices divided into two groups.
        """
        x = self._result_to_x(result)
        partition = [[], []]  # type: list[list[int]]
        for i, value in enumerate(x):
            if value == 0:
                partition[0].append(i)
            else:
                partition[1].append(i)
        return partition

    def _draw_result(
        self,
        result: np.ndarray,
        pos: dict[int, np.ndarray] | None = None,
    ) -> None:
        """Draw the result with colors.

        Args:
            result : The calculated result for the pr∂oblem
            pos: The positions of nodes
        """
        x = self._result_to_x(result)
        rx_visualization.mpl_draw(
            self._graph,
            node_color=self._node_colors(x),
            pos=pos,  # type: ignore
            with_labels=True,
        )

    def _node_colors(self, x: np.ndarray) -> list[str]:
        # Return a list of strings for draw.
        # Color a node with red when the corresponding variable is 1.
        # Otherwise color it with blue.
        return ["r" if x[node] else "b" for node in self._graph.node_indices()]
