import math
import pickle
import time

import numpy as np
from bayes_opt import BayesianOptimization

import pyfea

# set the number of processors and the chunksizes used by parallel base algorithms.
processes = 4
chunksize = 1

# set the number of processes and threads used by a parallel FEA.
process_count = 4
thread_count = 5

# assign the fitness function you want to evaluate below.
fitness = pyfea.rastrigin__

# set the bounds for your FEA's Bayesian run.
# IMPORTANT: Only set the variables you want to use as hyperparameters. Comment out the others.

pbounds = {
    "generations": (10, 35),
    "iterations": (10, 50),
    "pop_size": (10, 35),
    "fact_size": (1, 5),
    "overlap": (0, 3),
    "num_covers": (1, 5),
    "num_clamps": (0, 5),
}

pso_bounds = {"phi_p": (0.8, 3), "phi_g": (0.8, 3), "omega": (0.01, 0.99)}

ga_bounds = {
    "mutation_rate": (0.01, 0.99),
    "mutation_range": (0.1, 2),
}

de_bounds = {
    "mutation_factor": (0.1, 1),
    "crossover_rate": (0.1, 1),
}

# assign the base algorithm to test FEA on.
base_alg = pyfea.FeaPso
# add that algorithm's parameters to pbounds.
pbounds.update(pso_bounds)

# choose your factorizer.
factorizer = pyfea.linear_factorizer

# set the kind of FEA you want to use.
fea = pyfea.BsplineFEA


def bayes_input(
    fact_size=None,
    overlap=1,
    num_covers=2,
    num_clamps=0,
    generations=20,
    iterations=100,
    pop_size=20,
    mutation_factor=0.5,
    crossover_rate=0.9,
    mutation_rate=0.05,
    mutation_range=0.5,
    phi_p=math.sqrt(2),
    phi_g=math.sqrt(2),
    omega=1 / math.sqrt(2),
    dim=10,
):
    fact_size = int(fact_size)
    dim = int(dim)
    overlap = int(overlap)
    generations = int(generations)
    pop_size = int(pop_size)
    num_covers = int(num_covers)
    num_clamps = int(num_clamps)
    iterations = int(iterations)

    # code for generating domain on normal FEA.
    # domain = np.zeros((dim,2))
    # domain[:,0] = -5
    # domain[:,1] = 5

    # code for generating domain on Bspline FEA. Comment out if you're working on normal FEA.
    domain = (-5, 5)

    factors = factorizer(fact_size=fact_size, overlap=overlap, dim=dim, num_covers=num_covers)
    if factors is None:
        return -999999999
    if num_clamps > 0:
        pyfea.clamp_factor_ends(dim, factors, num_clamps)
    objective = fea(
        factors,
        fitness,
        iterations,
        dim,
        base_alg,
        domain,
        process_count=process_count,
        thread_count=thread_count,
        pop_size=pop_size,
        generations=generations,
        mutation_factor=mutation_factor,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        mutation_range=mutation_range,
        phi_p=phi_p,
        phi_g=phi_g,
        omega=omega,
        processes=processes,
        chunksize=chunksize,
    )
    ret = -objective.run()
    return ret


def bayes_run(init_points=5, n_iter=25):
    optimizer = BayesianOptimization(bayes_input, pbounds)
    optimizer.maximize(init_points, n_iter)
    storage = open(f"results/{fitness}_{base_alg}", "wb")
    pickle.dump(optimizer.max, storage)
    storage.close()


if __name__ == "__main__":
    bayes_run(2, 8)
