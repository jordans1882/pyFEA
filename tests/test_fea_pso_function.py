
from examples.fea import FEA
from examples.base_pso import PSO
from examples.function import Function
import numpy as np
import pytest
import time
import numba
from numpy import cos, sqrt, pi, e, exp, sum

@numba.jit
def rastrigin__(solution = None):
    return sum(solution**2 - 10 * cos(2 * pi * solution) + 10)

@pytest.mark.benchmark(
    group="random",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_fea_builder():
    array = np.zeros((10))
    array[0] = 1
    array[1] = 2
    domain = np.zeros((10, 2))
    domain[:,0] = -5
    domain[:,1] = 5
    function = Function(array, rastrigin__, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    fea = FEA([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]], rastrigin__, 10, 10, "PSO", domain)
    assert fea.run() == pytest.approx(0.0)