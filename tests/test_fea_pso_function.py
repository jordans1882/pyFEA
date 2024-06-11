
from examples.fea import FEA
from examples.base_pso import PSO, rastrigin
from examples.function import Function
import numpy as np
import pytest
import time
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
    domain = np.zeros((2, 2))
    domain[:,0] = -5
    domain[:,1] = 5
    function = Function(array, rastrigin, [0, 1])
    fea = FEA([[0], [1], [0, 1]], function, 100, 2, "PSO", domain)
    assert fea.run() == pytest.approx(0.0)