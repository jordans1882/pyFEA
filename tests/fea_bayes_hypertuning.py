from bayes_opt import BayesianOptimization
from examples.fea import FEA
from examples.base_pso import PSO
import numba
import numpy as np
import pytest
import time

@numba.jit
def rastrigin__(solution = None):
    return sum(solution**2 - 10 * np.cos(2 * np.pi * solution) + 10)

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
    fea = FEA([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]], rastrigin__, 10, 10, "PSO", domain)
    assert fea.run() == pytest.approx(0.0)

def linear_factorizer(fact_size, overlap, dim):
    smallest = 0
    if fact_size <= overlap:
        temp = fact_size
        fact_size = overlap
        overlap = temp
    largest = fact_size
    factors = []
    while largest <= dim:
        factors.append([x for x in range(smallest, largest)])
        smallest = largest - overlap
        largest += fact_size - overlap
    factors.append([x for x in range(smallest, dim)])
    return factors

def bayes_input(fact_size, overlap, iters, generations, phi_p, phi_g, omega):
    fact_size = int(fact_size)
    overlap = int(overlap)
    iters = int(iters)
    generations = int(generations)
    domain = np.zeros((10, 2))
    dim = 10
    domain[:,0] = -5
    domain[:,1] = 5
    factors = linear_factorizer(fact_size, overlap, dim)
    fea = FEA(factors, rastrigin__, dim, iters, "PSO", domain, generations=generations, phi_p=phi_p, phi_g=phi_g, omega=omega)
    return -fea.run()

pbounds = {"fact_size": (1,5), "overlap": (0,3), "iters": (20,200), "generations":(20,200), "phi_p":(0,4), "phi_g":(0,4), "omega":(0,1)}
optimizer = BayesianOptimization(bayes_input, pbounds)
optimizer.maximize()
print(optimizer.max)