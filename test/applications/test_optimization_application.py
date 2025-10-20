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

"""Test OptimizationApplication class"""

import unittest

import numpy as np
import pytest
from qiskit.result import QuasiDistribution
from qiskit_addon_opt_mapper.applications import OptimizationApplication


@pytest.mark.parametrize(
    "state_vector",
    [
        np.array([0, 0, 1, 0]),
        {"10": 0.8, "01": 0.2},
        QuasiDistribution({"10": 0.8, "01": 0.2}),
    ],
    ids=["array", "dict", "quasi"],
)
def test_sample_most_likely(state_vector):
    result = OptimizationApplication.sample_most_likely(state_vector)
    np.testing.assert_allclose(result, [0, 1])


if __name__ == "__main__":
    unittest.main()
