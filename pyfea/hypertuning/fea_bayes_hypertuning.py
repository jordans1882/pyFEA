import time

import numpy as np
import pytest
from bayes_opt import BayesianOptimization

import pyfea.benchmarks as benchmarks
from pyfea import FEA, PSO
from pyfea.base_algos import FeaPso

"""@pytest.mark.benchmark
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
    assert fea.run() == pytest.approx(0.0)"""


def linear_factorizer(fact_size, overlap, dim):
    smallest = 0
    # if fact_size < overlap:
    #    temp = fact_size
    #    fact_size = overlap
    #    overlap = temp
    # if fact_size == overlap:
    #    fact_size += 1
    largest = fact_size
    factors = []
    while largest <= dim:
        factors.append([x for x in range(smallest, largest)])
        smallest = largest - overlap
        largest += fact_size - overlap
    if smallest < dim:
        factors.append([x for x in range(smallest, dim)])
    return factors


def bayes_input(generations, iterations, pop_size):  # , fact_size, overlap, phi_p, phi_g, omega):
    # fact_size = int(fact_size)
    # overlap = int(overlap)
    # if fact_size <= overlap:
    #    return -999999999
    # factors = linear_factorizer(fact_size, overlap, dim)
    generations = int(generations)
    pop_size = int(pop_size)
    iterations = int(iterations)
    domain = np.zeros((10, 2))
    dim = 10
    domain[:, 0] = -5
    domain[:, 1] = 5
    # print("pre-constructor")
    factors = [
        [0],
        [0, 1],
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [6, 7, 8],
        [7, 8, 9],
        [8, 9],
        [9],
    ]
    fea = FEA(
        factors,
        benchmarks.rastrigin__,
        iterations,
        dim,
        FeaPso,
        domain,
        pop_size=pop_size,
        generations=generations,
    )  # , phi_p=phi_p, phi_g=phi_g, omega=omega)
    return -fea.run()


pbounds = {
    "generations": (10, 30),
    "iterations": (250, 450),
    "pop_size": (10, 30),
}  # , "fact_size": (1,5), "overlap": (0,3), "phi_p":(1.25,1.75), "phi_g":(1.25,1.75), "omega":(0.5,.9)}
optimizer = BayesianOptimization(bayes_input, pbounds)
optimizer.maximize(init_points=10, n_iter=40)
print(optimizer.max)
