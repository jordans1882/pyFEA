from bayes_opt import BayesianOptimization
from feareu import FEA
from feareu import PSO
import feareu.benchmarks as benchmarks
import numpy as np
import pytest
import time


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
    if fact_size < overlap:
        temp = fact_size
        fact_size = overlap
        overlap = temp
    if fact_size == overlap:
        fact_size += 1
    largest = fact_size
    factors = []
    while largest <= dim:
        factors.append([x for x in range(smallest, largest)])
        smallest = largest - overlap
        largest += fact_size - overlap
    if smallest < dim:
        factors.append([x for x in range(smallest, dim)])
    return factors

def bayes_input(fact_size, overlap, phi_p, phi_g, omega, generations, iterations, pop_size):
    #print("reached bayes_input")
    fact_size = int(fact_size)
    overlap = int(overlap)
    generations = int(generations)
    pop_size = int(pop_size)
    iterations = int(iterations)
    domain = np.zeros((10, 2))
    dim = 10
    domain[:,0] = -5
    domain[:,1] = 5
    factors = linear_factorizer(fact_size, overlap, dim)
    #print("pre-constructor")
    fea = FEA(factors, benchmarks.rastrigin__, iterations, dim, "PSO", domain, pop_size=pop_size, generations=generations, phi_p=phi_p, phi_g=phi_g, omega=omega)
    return -fea.run()

pbounds = {"generations":(10,50), "iterations":(100,300), "pop_size":(10,50), "fact_size": (1,5), "overlap": (0,3), "phi_p":(0,4), "phi_g":(0,4), "omega":(0,1)}
optimizer = BayesianOptimization(bayes_input, pbounds)
optimizer.maximize()
print(optimizer.max)
