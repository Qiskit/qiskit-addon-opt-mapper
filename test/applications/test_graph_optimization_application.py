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

"""Test GraphOptimizationApplication class"""

import unittest
from unittest.mock import patch

import networkx as nx
import pytest
from qiskit_addon_opt_mapper.applications import GraphOptimizationApplication


@pytest.mark.parametrize(
    "nodes, edges",
    [
        ([0, 1], [(0, 1)]),
        ([1, 2], [(1, 2)]),
        ([1, 2, 3], [(1, 2)]),
    ],
    ids=["from0", "from1", "from1_extra"],
)
@patch.multiple(GraphOptimizationApplication, __abstractmethods__=set())
def test_from_nx_graph(nodes, edges):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    test_graph = GraphOptimizationApplication(graph).graph
    assert test_graph.num_nodes() == len(nodes)
    assert test_graph.num_edges() == len(edges)


if __name__ == "__main__":
    unittest.main()
