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

"""An application class for the clique."""

import networkx as nx
import numpy as np
import rustworkx as rx
from docplex.mp.model import Model
from rustworkx import visualization as rx_visualization

from qiskit_addon_opt_mapper.problems.optimization_problem import OptimizationProblem
from qiskit_addon_opt_mapper.translators import from_docplex_mp

from .graph_optimization_application import GraphOptimizationApplication


class Clique(GraphOptimizationApplication):
    """Optimization application for the "clique" [1] problem based on a NetworkX graph.

    References:
        [1]: "Clique (graph theory)",
        `https://en.wikipedia.org/wiki/Clique_(graph_theory)
        <https://en.wikipedia.org/wiki/Clique_(graph_theory)>`_
    """

    def __init__(self, graph: nx.Graph | np.ndarray | list, size: int | None = None) -> None:
        """Init method.

        Args:
            graph: A graph representing a problem. It can be specified directly as a
                `NetworkX <https://networkx.org/>`_ graph,
                or as an array or list format suitable to build out a NetworkX graph.
            size: The size of the clique. When it's `None`, the default, this class makes an
                optimization model for a maximal clique instead of the specified size of a clique.
        """
        super().__init__(graph)
        self._size = size

    def to_optimization_problem(self) -> OptimizationProblem:
        """Represent as an optimization problem.

        Convert a clique problem instance into a
        :class:`~qiskit_addon_opt_mapper.problems.OptimizationProblem`.
        When "size" is None, this makes an optimization model for a maximal clique
        instead of the specified size of a clique.


        Returns:
            The :class:`~qiskit_addon_opt_mapper.problems.OptimizationProblem` created
            from the clique problem instance.
        """
        complement_g = rx.graph_complement(self._graph)

        mdl = Model(name="Clique")
        n = self._graph.num_nodes()
        x = {i: mdl.binary_var(name=f"x_{i}") for i in range(n)}
        for w, v in complement_g.edge_list():
            mdl.add_constraint(x[w] + x[v] <= 1)
        if self.size is None:
            mdl.maximize(mdl.sum(x[i] for i in x))
        else:
            mdl.add_constraint(mdl.sum(x[i] for i in x) == self.size)
        op = from_docplex_mp(mdl)
        return op

    def interpret(self, result: np.ndarray) -> list[int]:
        """Interpret a result as a list of node indices.

        Args:
            result : The calculated result of the problem


        Returns:
            The list of node indices whose corresponding variable is 1
        """
        x = self._result_to_x(result)
        clique = []
        for i, value in enumerate(x):
            if value:
                clique.append(i)
        return clique

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

    def _node_colors(self, x: np.ndarray) -> list[str]:
        # Return a list of strings for draw.
        # Color a node with red when the corresponding variable is 1.
        # Otherwise color it with dark gray.
        return ["r" if x[node] else "darkgrey" for node in self._graph.node_indices()]

    @property
    def size(self) -> int | None:
        """Getter of size.

        Returns:
            The size of the clique, `None` when maximal clique
        """
        return self._size

    @size.setter
    def size(self, size: int | None) -> None:
        """Setter of size.

        Args:
            size: The size of the clique, `None` for maximal clique
        """
        self._size = size
