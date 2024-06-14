from feareu import FEA
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
    largest = fact_size
    factors = []
    while largest <= dim:
        factors.append([x for x in range(smallest, largest)])
        smallest = largest - overlap
        largest += fact_size - overlap
    if smallest < dim:
        factors.append([x for x in range(smallest, dim)])
    return factors

def parameter_input(fact_size, overlap, phi_p, phi_g, omega, generations, iterations, pop_size):
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
    fea = FEA(factors, function, iterations, dim, FeaPso, domain, pop_size=pop_size, generations=generations, phi_p=phi_p, phi_g=phi_g, omega=omega)
    ret = fea.run()
    return ret

def parameter_run():
    for benchmark in benchmarks.__all__:
        global function
        function = globals()[benchmark]
        print(benchmark)
        storage = open(f'results/{benchmark}','rb')
        data = pickle.load(storage)
        print(data['params'])
        result = parameter_input(data['params']['fact_size'], 
                                 data['params']['overlap'], 
                                  data['params']['phi_p'], 
                                   data['params']['phi_g'], 
                                    data['params']['omega'], 
                                     data['params']['generations'], 
                                      data['params']['iterations'], 
                                       data['params']['pop_size'])
        print(result)
        storage.close()

parameter_run()
