from multiprocessing import freeze_support
from feareu import BsplineFEA, FEA, ParallelBsplineFEA
from feareu.base_algos import FeaPso, FeaGA, FeaDE
from feareu import Function
import numpy as np
import pytest
import time
import numba

import matplotlib.pyplot as plt
from numpy import cos, sqrt, pi, e, exp, sum

@numba.jit
def rastrigin__(solution = None):
    #return sum(solution**2)
    return sum(solution**2 - 10 * cos(2 * pi * solution) + 10)
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

array = np.zeros((10))
array[0] = 1
array[1] = 2
domain = np.zeros((10, 2))
domain[:,0] = -5
domain[:,1] = 5

if __name__ == '__main__':
    freeze_support()
    function = Function(array, rastrigin__, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    #rand_factors = [np.random.choice(range(10), replace=False, size=3) for x in range(10)]
    fct1 = [[0],[0,1],[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9],[9]]
    fct2 = [[0],[0,1],[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[11,12,13],[12,13,14],[13,14],[14]]
    factor_number = 5000
    factors = linear_factorizer(2, 1, factor_number)
    start = time.time()
    fea1 = ParallelBsplineFEA(factors=factors, function = rastrigin__, iterations = 5, dim = factor_number, base_algo_name=FeaPso, domain=(-5, 5), process_count=3, generations= 5, pop_size=20)
    fea1.run()
    end = time.time()
    print("parallel time: ", end-start)
    
    start = time.time()
    fea2 = BsplineFEA(factors=factors, function = rastrigin__, iterations = 5, dim = factor_number, base_algo_name=FeaPso, domain=(-5, 5), generations= 5, pop_size=20)
    fea2.run()
    end = time.time()
    print("non-parallel time: ", end-start)
    
    daig_plt1 = fea1.diagnostic_plots()
    plt.show()