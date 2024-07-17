"""Benchmark tests.
"""

import numpy as np

from pyfea.benchmarks import rastrigin__


def test_rastrigin():
    """Rastrigin at minima."""
    a = rastrigin__(np.array([0, 0, 0, 0, 0, 0]))
    assert a == 0
