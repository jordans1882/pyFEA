import matplotlib.pyplot as plt
import numpy as np
from feareu.base_algos import FeaDE, DE
from feareu import Function
from feareu.fea import FEA
from bayes_opt import BayesianOptimization
import feareu.benchmarks as benchmarks
import pytest
import time

ndims = 5 


array = np.zeros((ndims))

domain = np.zeros((ndims, 2))
domain[:, 0] = -5
domain[:, 1] = 5
domain
# function = Function(array, rastrigin__, [0, 1])

#function = Function(array, benchmarks.rastrigin__, [0, 1, 2, 3, 4])
#de = FeaDE(function = function, domain = domain, generations = 400, pop_size=40)
#de_og = DE(function = function, domain = domain, generations = 400, pop_size=40)
#print(de.run())
#print(de_og.run())
#diag_plots = de.diagnostic_plots()
#diag_plots = de_og.diagnostic_plots()
#plt.show()

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
    #if fact_size < overlap:
    #    temp = fact_size
    #    fact_size = overlap
    #    overlap = temp
    #if fact_size == overlap:
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

def bayes_input(fact_size, overlap, generations, iterations, pop_size, mutation_factor, crossover_rate):
    #print("reached bayes_input")
    fact_size = int(fact_size)
    overlap = int(overlap)
    generations = int(generations)
    pop_size = int(pop_size)
    iterations = int(iterations)
    if overlap>=fact_size:
        return -9999999999
    domain = np.zeros((10, 2))
    dim = 10
    domain[:,0] = -5
    domain[:,1] = 5
    factors = linear_factorizer(fact_size, overlap, dim)
    #print("pre-constructor")
    fea = FEA(factors, benchmarks.rastrigin__, iterations, dim, FeaDE, domain, pop_size=pop_size, generations=generations, mutation_factor=mutation_factor, crossover_rate=crossover_rate)
    return -fea.run()

pbounds = {"generations":(10,30), "iterations":(30,80), "pop_size":(10,40), "fact_size": (1,5), "overlap": (0,3), "mutation_factor": (0.1,1), "crossover_rate": (0.1,1)}
optimizer = BayesianOptimization(bayes_input, pbounds)
optimizer.maximize()
print(optimizer.max)
