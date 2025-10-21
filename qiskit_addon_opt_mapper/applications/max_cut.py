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


"""An application class for the Max-cut."""

import numpy as np
from docplex.mp.model import Model
from rustworkx import visualization as rx_visualization

from qiskit_addon_opt_mapper.problems.optimization_problem import OptimizationProblem
from qiskit_addon_opt_mapper.translators import from_docplex_mp

from .graph_optimization_application import GraphOptimizationApplication


class Maxcut(GraphOptimizationApplication):
    """Optimization application for the "max-cut" [1] problem based on a NetworkX graph.

    References:
        [1]: "Maximum cut",
        https://en.wikipedia.org/wiki/Maximum_cut
    """

    def to_optimization_problem(self) -> OptimizationProblem:
        """Represent as an optimization problem.

        Convert a Max-cut problem instance into a
        :class:`~qiskit_addon_opt_mapper.problems.OptimizationProblem`


        Returns:
            The :class:`~qiskit_addon_opt_mapper.problems.OptimizationProblem` created
            from the Max-cut problem instance.
        """
        mdl = Model(name="Max-cut")
        x = {i: mdl.binary_var(name=f"x_{i}") for i in range(self._graph.num_nodes())}
        for edge_index, (i, j) in enumerate(self._graph.edge_list()):
            weight = self._graph.get_edge_data(i, j)
            if weight is None:
                weight = 1
                self._graph.update_edge_by_index(edge_index, weight)
        objective = mdl.sum(
            self._graph.get_edge_data(i, j) * (x[i] * (1 - x[j]) + x[j] * (1 - x[i]))
            for _, (i, j) in enumerate(self._graph.edge_list())
        )
        mdl.maximize(objective)
        op = from_docplex_mp(mdl)
        return op

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
            node_color=self._node_color(x),
            pos=pos,  # type: ignore
            with_labels=True,
        )

    def interpret(self, result: np.ndarray) -> list[list[int]]:
        """Interpret a result as two lists of node indices.

        Args:
            result : The calculated result of the problem


        Returns:
            Two lists of node indices correspond to two node sets for the Max-cut
        """
        x = self._result_to_x(result)
        cut = [[], []]  # type: list[list[int]]
        for i, value in enumerate(x):
            if value == 0:
                cut[0].append(i)
            else:
                cut[1].append(i)
        return cut

    def _node_color(self, x: np.ndarray) -> list[str]:
        # Return a list of strings for draw.
        # Color a node with red when the corresponding variable is 1.
        # Otherwise color it with blue.
        return ["b" if x[node] == 1 else "r" for node in self._graph.node_indices()]

    @staticmethod
    def parse_gset_format(filename: str) -> np.ndarray:
        """Read graph in Gset format from file.

        Args:
            filename: the name of the file.


        Returns:
            An adjacency matrix as a 2D numpy array.
        """
        n = -1
        with open(filename, encoding="utf8") as infile:
            header = True
            m = -1
            count = 0
            for line in infile:
                v = (int(e) for e in line.split())
                if header:
                    n, m = v
                    w = np.zeros((n, n))
                    header = False
                else:
                    s__, t__, w__ = v
                    s__ -= 1  # adjust 1-index
                    t__ -= 1  # ditto
                    w[s__, t__] = w__
                    count += 1
            assert m == count
        w += w.T
        return w

    @staticmethod
    def get_gset_result(x: np.ndarray) -> dict[int, int]:
        """Get graph solution in Gset format from binary string.

        Args:
            x: binary string as numpy array.


        Returns:
            A graph solution in Gset format.
        """
        return {i + 1: 1 - x[i] for i in range(len(x))}
