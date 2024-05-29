"""Provides some benchmarks of random number generation.

Tests numpy versus built-in implementation.
"""

import time
import random
import numpy as np
import pytest


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
def test_random_python_builtin(benchmark):
    """Benchmark python list random unif(10000)."""

    def random1():
        randomlist = []
        for i in range(0, 10000):
            print(i)
            rand_val = random.random()
            randomlist.append(rand_val)
        return randomlist

    benchmark(random1)
    assert int(0) == 0


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
def test_random_numpy(benchmark):
    """Benchmark np random unif(10000)."""

    def random2():
        return np.random.random(10000)

    benchmark(random2)
    assert int(0) == 0
