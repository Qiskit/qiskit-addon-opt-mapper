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

"""An abstract class for graph optimization application classes."""

import copy
from abc import abstractmethod

import networkx as nx
import numpy as np
import rustworkx as rx
from rustworkx import visualization as rx_visualization

import qiskit_addon_opt_mapper.optionals as _optionals

from .optimization_application import OptimizationApplication


class GraphOptimizationApplication(OptimizationApplication):
    """An abstract class for graph optimization applications."""

    def __init__(self, graph: nx.Graph | np.ndarray | list | rx.PyGraph) -> None:
        """Init method.

        Args:
            graph: A graph representing a problem. It can be specified in the following formats:

                - A Rustworkx undirected graph (``rx.PyGraph``)
                - A NetworkX undirected graph (``nx.Graph``)
                - A NumPy adjacency matrix (``np.ndarray``)
                - A list of edges or adjacency list (``list``)

            The input graph will be internally normalized to a ``rx.PyGraph``.
        """
        if isinstance(graph, rx.PyGraph):
            self._graph = copy.deepcopy(graph)
        elif isinstance(graph, nx.Graph):
            self._graph = self._from_networkx(graph)
        elif isinstance(graph, np.ndarray | list):
            nx_graph = nx.Graph(graph)
            self._graph = self._from_networkx(nx_graph)
        else:
            raise TypeError(f"Unsupported graph type: {type(graph)}")

    def _from_networkx(self, nx_graph: nx.Graph) -> rx.PyGraph:
        """Convert a NetworkX graph to a Rustworkx PyGraph."""
        rx_graph = rx.PyGraph()
        rx_graph.add_nodes_from([None] * nx_graph.number_of_nodes())
        # Use 1 as default edge weight
        edges = [(u, v, d.get("weight", 1)) for u, v, d in nx_graph.edges(data=True)]
        rx_graph.add_edges_from(edges)
        return rx_graph

    @_optionals.HAS_MATPLOTLIB.require_in_call
    def draw(
        self,
        result: np.ndarray | None = None,
        pos: dict[int, np.ndarray] | None = None,
    ) -> None:
        """Draw a graph with the result.

        When the result is None, draw an original graph without
        colors.

        Args:
            result: The calculated result for the problem
            pos: The positions of nodes
        """
        if result is None:
            rx_visualization.mpl_draw(self._graph, pos=pos, with_labels=True)  # type: ignore
        else:
            self._draw_result(result, pos)

    @abstractmethod
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
        pass

    @property
    def graph(self) -> rx.PyGraph:
        """Getter of the graph.

        Returns:
            A graph for a problem
        """
        return self._graph

    @property
    def nx_graph(self) -> nx.Graph:
        """Getter of the graph in Networkx format.

        Returns:
            A graph for a problem
        """
        nx_graph = nx.Graph()

        # Add nodes
        for node_index in self._graph.node_indices():
            node_data = self._graph[node_index]
            nx_graph.add_node(node_index, data=node_data)

        # Add edges
        for edge in self._graph.edge_list():
            source, target = edge
            edge_data = self._graph.get_edge_data(source, target)
            nx_graph.add_edge(source, target, data=edge_data)

        return nx_graph
