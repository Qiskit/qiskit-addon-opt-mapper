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

"""An application class for Traveling salesman problem (TSP)."""

import numpy as np
import rustworkx as rx
from docplex.mp.model import Model
from rustworkx import visualization as rx_visualization

from qiskit_addon_opt_mapper.exceptions import OptimizationError
from qiskit_addon_opt_mapper.problems import OptimizationProblem
from qiskit_addon_opt_mapper.translators import from_docplex_mp

from .graph_optimization_application import GraphOptimizationApplication


# pylint: disable=wrong-spelling-in-docstring
class Tsp(GraphOptimizationApplication):
    """Optimization application for the "traveling salesman problem" [1] based on a NetworkX graph.

    References:
        [1]: "Travelling salesman problem",
        https://en.wikipedia.org/wiki/Travelling_salesman_problem
    """

    def to_optimization_problem(self) -> OptimizationProblem:
        """Represent as an optimization problem.

        Convert a traveling salesman problem instance into a
        :class:`~qiskit_addon_opt_mapper.problems.OptimizationProblem`


        Returns:
            The :class:`~qiskit_addon_opt_mapper.problems.OptimizationProblem` created
            from the traveling salesman problem instance.
        """
        mdl = Model(name="TSP")
        n = self._graph.num_nodes()
        x = {(i, k): mdl.binary_var(name=f"x_{i}_{k}") for i in range(n) for k in range(n)}

        # Only sum over existing edges in the graph
        tsp_func = mdl.sum(
            self._graph.get_edge_data(i, j) * x[(i, k)] * x[(j, (k + 1) % n)]
            for i, j in self._graph.edge_list()
            for k in range(n)
        )

        # Add reverse edges since we have an undirected graph
        tsp_func += mdl.sum(
            self._graph.get_edge_data(i, j) * x[(j, k)] * x[(i, (k + 1) % n)]
            for i, j in self._graph.edge_list()
            for k in range(n)
        )

        mdl.minimize(tsp_func)
        for i in range(n):
            mdl.add_constraint(mdl.sum(x[(i, k)] for k in range(n)) == 1)
        for k in range(n):
            mdl.add_constraint(mdl.sum(x[(i, k)] for i in range(n)) == 1)

        def non_edges(graph):
            node_indices = list(graph.node_indices())
            remaining = set(node_indices)
            while remaining:
                u = remaining.pop()
                neighbors = set(graph.neighbors(u))
                for v in remaining - neighbors:
                    yield (u, v)

        # Add constraints for non-edges
        for i, j in non_edges(self._graph):
            for k in range(n):
                mdl.add_constraint(x[i, k] + x[j, (k + 1) % n] <= 1)
                mdl.add_constraint(x[j, k] + x[i, (k + 1) % n] <= 1)
        op = from_docplex_mp(mdl)
        return op

    def interpret(self, result: np.ndarray) -> list[int | list[int]]:
        """Interpret a result as a list of node indices.

        Args:
            result : The calculated result of the problem


        Returns:
            A list of nodes whose indices correspond to its order in a prospective cycle.
        """
        x = self._result_to_x(result)
        n = self._graph.num_nodes()
        route = []  # type: list[int | list[int]]
        for p__ in range(n):
            p_step = []
            for i in range(n):
                if x[i * n + p__]:
                    p_step.append(i)
            if len(p_step) == 1:
                route.extend(p_step)
            else:
                route.append(p_step)
        return route

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
            edge_list=self._edgelist(x),
            with_labels=True,
            pos=pos,  # type: ignore
            width=8,
            alpha=0.5,
            edge_color="tab:red",
        )

    def _edgelist(self, x: np.ndarray):
        # Arrange route and return the list of the edges for the edge list of nx.draw_networkx_edges
        route = self.interpret(x)
        return [(route[i], route[(i + 1) % len(route)]) for i in range(len(route))]

    @staticmethod
    # pylint: disable=undefined-variable
    def create_random_instance(
        n: int, low: int = 0, high: int = 100, seed: int | None = None
    ) -> "Tsp":
        """Create a random instance of the traveling salesman problem.

        Args:
            n: the number of nodes.
            low: The minimum value for the coordinate of a node.
            high: The maximum value for the coordinate of a node.
            seed: the seed for the random coordinates.


        Returns:
             A Tsp instance created from the input information
        """
        rng = np.random.default_rng(seed)
        coord = rng.uniform(low, high, (n, 2))
        # pos = {i: (coord_[0], coord_[1]) for i, coord_ in enumerate(coord)}
        # graph = nx.random_geometric_graph(n, np.hypot(high - low, high - low) + 1, pos=pos)
        # for w, v in graph.edges:
        #     delta = [graph.nodes[w]["pos"][i] - graph.nodes[v]["pos"][i] for i in range(2)]
        #     graph.edges[w, v]["weight"] = np.rint(np.hypot(delta[0], delta[1]))

        graph = rx.PyGraph()
        graph.add_nodes_from([None] * n)
        threshold = np.hypot(high - low, high - low) + 1
        for i in range(n):
            for j in range(i + 1, n):
                delta = coord[i] - coord[j]
                dist = np.hypot(delta[0], delta[1])
                if dist <= threshold:
                    graph.add_edge(i, j, round(dist))

        return Tsp(graph)

    @staticmethod
    def parse_tsplib_format(filename: str) -> "Tsp":
        """Read a graph in TSPLIB format from file and return a Tsp instance.

        Only the EUC_2D edge weight format is supported.

        Args:
            filename: the name of the file.

        Raises:
            OptimizationError: If the type is not "TSP"
            OptimizationError: If the edge weight type is not "EUC_2D"


        Returns:
            A Tsp instance data.
        """
        name = ""
        coord = []  # type: ignore
        with open(filename, encoding="utf8") as infile:
            coord_section = False
            for line in infile:
                if line.startswith("NAME"):
                    name = line.split(":")[1]
                    name = name.strip()
                elif line.startswith("TYPE"):
                    typ = line.split(":")[1]
                    typ = typ.strip()
                    if typ != "TSP":
                        raise OptimizationError(f'This supports only "TSP" type. Actual: {typ}')
                elif line.startswith("EOF"):
                    # End Of File tag
                    break
                elif line.startswith("DIMENSION"):
                    dim = int(line.split(":")[1])
                    coord = np.zeros((dim, 2))  # type: ignore
                elif line.startswith("EDGE_WEIGHT_TYPE"):
                    typ = line.split(":")[1]
                    typ = typ.strip()
                    if typ != "EUC_2D":
                        raise OptimizationError(
                            f'This supports only "EUC_2D" edge weight. Actual: {typ}'
                        )
                elif line.startswith("NODE_COORD_SECTION"):
                    coord_section = True
                elif coord_section:
                    v = line.split()
                    index = int(v[0]) - 1
                    coord[index][0] = float(v[1])
                    coord[index][1] = float(v[2])
        x_max = max(coord_[0] for coord_ in coord)
        x_min = min(coord_[0] for coord_ in coord)
        y_max = max(coord_[1] for coord_ in coord)
        y_min = min(coord_[1] for coord_ in coord)
        # pos = dict(enumerate(coord))
        # graph = nx.random_geometric_graph(
        #     len(coord), np.hypot(x_max - x_min, y_max - y_min) + 1, pos=pos
        # )
        # for w, v in graph.edges:
        #     delta = [graph.nodes[w]["pos"][i] - graph.nodes[v]["pos"][i] for i in range(2)]
        #     graph.edges[w, v]["weight"] = np.rint(np.hypot(delta[0], delta[1]))

        # Create empty graph
        graph = rx.PyGraph()
        graph.add_nodes_from([None] * len(coord))

        # Distance threshold
        threshold = np.hypot(x_max - x_min, y_max - y_min) + 1

        # Add edges with weights
        for i in range(len(coord)):
            for j in range(i + 1, len(coord)):
                delta = coord[i] - coord[j]
                dist = np.hypot(delta[0], delta[1])
                if dist <= threshold:
                    graph.add_edge(i, j, round(dist))

        return Tsp(graph)

    @staticmethod
    def tsp_value(z: list[int], adj_matrix: np.ndarray) -> float:
        """Compute the TSP value of a solution.

        Args:
            z: list of cities.
            adj_matrix: adjacency matrix.


        Returns:
            value of the total length
        """
        ret = 0.0
        for i in range(len(z) - 1):
            ret += adj_matrix[z[i], z[i + 1]]
        ret += adj_matrix[z[-1], z[0]]
        return float(ret)
