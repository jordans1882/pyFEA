"""Provides some benchmarks of random number generation.

Tests numpy versus built-in implementation.
"""

import time
import random
import numpy as np
import pytest
import splipy
import scipy.sparse.linalg as splinalg
import math


# Comment this next line to remove benchmark from test suite
# @pytest.mark.skip(reason="Done benchmarking")
@pytest.mark.benchmark(
    group="random",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_random_python_theta(benchmark):
    """Benchmark python list random unif(10000)."""

    def mdoppler(x: float) -> float:
        return math.sin(20 / (x + 0.15))

    # Parameters
    n = 100000
    k = 1000

    # Create testing data
    x = np.random.rand(n)
    ytrue = [mdoppler(xval) for xval in x]
    y = ytrue + np.random.normal(0.0, 0.2, n)

    # plt.scatter(x, y)
    # plt.show()

    knots = np.concatenate((np.array([0, 0]), np.linspace(0.0, 1.0, k + 1), np.array([1, 1])))

    bsp = splipy.BSplineBasis(3, knots, -1)
    xmat = bsp.evaluate(x, 0, True, True)
    xt = xmat.transpose()

    def random1():
        theta = (splinalg.inv(xt @ xmat) @ xt) @ y

    benchmark(random1)


# Comment this next line to remove benchmark from test suite
# @pytest.mark.skip(reason="Done benchmarking")
@pytest.mark.benchmark(
    group="random",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_random_python_evaluate(benchmark):
    """Benchmark python list random unif(10000)."""

    def mdoppler(x: float) -> float:
        return math.sin(20 / (x + 0.15))

    # Parameters
    n = 100000
    k = 1000

    # Create testing data
    x = np.random.rand(n)
    ytrue = [mdoppler(xval) for xval in x]
    y = ytrue + np.random.normal(0.0, 0.2, n)

    # plt.scatter(x, y)
    # plt.show()

    knots = np.concatenate((np.array([0, 0]), np.linspace(0.0, 1.0, k + 1), np.array([1, 1])))

    bsp = splipy.BSplineBasis(3, knots, -1)

    def random2():
        xmat = bsp.evaluate(x, 0, True, True)

    benchmark(random2)
    assert int(0) == 0


# Comment this next line to remove benchmark from test suite
