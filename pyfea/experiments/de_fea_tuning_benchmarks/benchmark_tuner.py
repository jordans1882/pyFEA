import pickle
import time

import numpy as np
import pytest
from bayes_opt import BayesianOptimization

import pyfea.benchmarks as benchmarks
from pyfea import FEA, PSO
from pyfea.base_algos import FeaGA
from pyfea.benchmarks import *


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


def bayes_input(
    fact_size, overlap, generations, iterations, pop_size, mutation_rate, mutation_range, b
):
    # print("reached bayes_input")
    fact_size = int(fact_size)
    overlap = int(overlap)
    generations = int(generations)
    pop_size = int(pop_size)
    iterations = int(iterations)
    if fact_size <= overlap:
        return -999999999
    domain = np.zeros((10, 2))
    dim = 10
    domain[:, 0] = -5
    domain[:, 1] = 5
    factors = linear_factorizer(fact_size, overlap, dim)
    # print("pre-constructor")
    # print(function)
    fea = FEA(
        factors,
        function,
        iterations,
        dim,
        FeaGA,
        domain,
        pop_size=pop_size,
        generations=generations,
        mutation_rate=mutation_rate,
        mutation_range=mutation_range,
        b=b,
    )
    ret = -fea.run()
    return ret


def bayes_run(init_points=5, n_iter=25):
    for benchmark in benchmarks.__all__:
        global function
        function = globals()[benchmark]
        print(benchmark)
        pbounds = {
            "generations": (10, 35),
            "iterations": (30, 100),
            "pop_size": (10, 35),
            "fact_size": (1, 5),
            "overlap": (0, 3),
            "mutation_rate": (0.1, 1),
            "b": (0.1, 0.9),
            "mutation_range": (0.1, 2),
        }
        optimizer = BayesianOptimization(bayes_input, pbounds)
        optimizer.maximize(init_points, n_iter)
        print(optimizer.max)
        storage = open(f"results/{benchmark}", "wb")
        pickle.dump(optimizer.max, storage)
        storage.close()


bayes_run(2, 8)
