from bayes_opt import BayesianOptimization
from feareu import FEA
from feareu import PSO
import feareu.benchmarks as benchmarks
import numpy as np
import pytest
import time



def random_factorizer(fact_size, overlap, dim, num_factors):
    if fact_size < overlap:
        temp = fact_size
        fact_size = overlap
        overlap = temp
    if fact_size == overlap:
        fact_size += 1
    factors = []
    factors.append(np.random.choice(range(dim), fact_size, replace=False))
    for i in range(num_factors):
        if overlap > 0:
            temp = np.random.choice(factors[i], overlap, replace=False)
            ranger = np.random.choice(np.delete(range(dim), temp), fact_size-overlap, replace=False)
            ranger = np.concatenate((temp,ranger))
        else:
            ranger = np.random.choice(range(dim), fact_size, replace=False)
        factors.append(ranger)
    return factors
    
def bayes_input(fact_size, overlap, phi_p, phi_g, omega, num_factors, iterations, generations, pop_size):
    #print("reached bayes_input")
    fact_size = int(fact_size)
    overlap = int(overlap)
    num_factors = int(num_factors)
    iterations = int(iterations)
    generations = int(generations)
    pop_size = int(pop_size)
    domain = np.zeros((10, 2))
    dim = 10
    domain[:,0] = -5
    domain[:,1] = 5
    factors = random_factorizer(fact_size, overlap, dim, num_factors)
    #print("pre-constructor")
    fea = FEA(factors, benchmarks.rastrigin__, iterations, dim, "PSO", domain, pop_size=pop_size, generations=generations, phi_p=phi_p, phi_g=phi_g, omega=omega)
    return -fea.run()

pbounds = {"generations": (10, 50), "iterations": (100,300), "fact_size": (1,5), "pop_size":(10,50), "num_factors": (20, 35), "overlap": (0,3), "phi_p":(0,4), "phi_g":(0,4), "omega":(0,1)}
optimizer = BayesianOptimization(bayes_input, pbounds)
optimizer.maximize()
print(optimizer.max)

