from bayes_opt import BayesianOptimization
from feareu import FEA
from feareu import PSO
import pickle
import feareu.benchmarks as benchmarks
from feareu.benchmarks import *
import numpy as np
import pytest
import time
from feareu.base_algos.fea_pso import FeaPso

function = benchmarks.rastrigin__

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

def bayes_input(fact_size, overlap, phi_p, phi_g, omega, generations, iterations, pop_size):
    #print("reached bayes_input")
    fact_size = int(fact_size)
    overlap = int(overlap)
    generations = int(generations)
    pop_size = int(pop_size)
    iterations = int(iterations)
    if fact_size <= overlap:
        return -999999999
    domain = np.zeros((10, 2))
    dim = 10
    domain[:,0] = -5
    domain[:,1] = 5
    factors = linear_factorizer(fact_size, overlap, dim)
    #print("pre-constructor")
    #print(function)
    fea = FEA(factors, function, iterations, dim, FeaPso, domain, pop_size=pop_size, generations=generations, phi_p=phi_p, phi_g=phi_g, omega=omega)
    ret = -fea.run()
    return ret

def bayes_run(init_points=5, n_iter=25):
    for benchmark in benchmarks.__all__:
        global function
        function = globals()[benchmark]
        print(function)
        pbounds = {"generations":(10,50), "iterations":(100,300), "pop_size":(10,50), "fact_size": (1,5), "overlap": (0,3), "phi_p":(0,4), "phi_g":(0,4), "omega":(0,1)}
        optimizer = BayesianOptimization(bayes_input, pbounds)
        optimizer.maximize(init_points, n_iter)
        print(optimizer.max)
        storage = open(f'results/{benchmark}','wb')
        pickle.dump(optimizer.max, storage)
        storage.close()

bayes_run(2, 8)

#pbounds = {"generations":(10,50), "iterations":(100,300), "pop_size":(10,50), "fact_size": (1,5), "overlap": (0,3), "phi_p":(0,4), "phi_g":(0,4), "omega":(0,1)}
#optimizer = BayesianOptimization(bayes_input, pbounds)
#optimizer.maximize()
#print(optimizer.max)

