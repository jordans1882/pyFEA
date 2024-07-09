import math
import pickle
import time
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from bayes_opt import BayesianOptimization

import feareu

# set the number of processors and the chunksizes used by parallel base algorithms.
processes = 4
chunksize = 5

# set the number of processes and threads used by a parallel FEA.
process_count = 4
thread_count = 16

# set the number of iterations at which we record data to be printed.
diagnostics_amount = 5

# set the bounds for your FEA's Bayesian run.
# IMPORTANT: Only set the variables you want to use as hyperparameters. Comment out the others.

pbounds = {
    "generations": (2, 10),
    "iterations": (2, 70),
    "pop_size": (10, 35),
    "fact_size": (1, 15),
    "overlap": (0, 8),
    # "num_covers":(1,5),
    "num_clamps": (0, 5),
    "dim": (10, 30),
}

base_bounds = {"generations": (10, 20), "pop_size": (10, 35), "dim": (8, 12)}

pso_bounds = {"phi_p": (0.8, 3), "phi_g": (0.8, 3), "omega": (0.000001, 0.99)}

ga_bounds = {
    "mutation_rate": (0.01, 0.99),
    "mutation_range": (0.1, 2),
}

de_bounds = {
    "mutation_factor": (0.1, 1),
    "crossover_rate": (0.1, 1),
}


# choose your factorizer.
factorizer = feareu.linear_factorizer

# set the kind of FEA you want to use.
fea = feareu.ParallelBsplineFEA


def bayes_input_fea(
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
    domain = (0, 1)

    factors = factorizer(fact_size=fact_size, overlap=overlap, dim=dim, num_covers=num_covers)
    if factors is None or fact_size > dim:
        return -999999999
    if num_clamps > 0:
        feareu.clamp_factor_ends(dim, factors, num_clamps)
    objective = fea(
        factors,
        fitness,
        iterations,
        dim,
        base_alg,
        domain,
        diagnostics_amount=diagnostics_amount,
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


def bayes_run_fea(bounds, init_points=5, n_iter=25):
    optimizer = BayesianOptimization(bayes_input_fea, bounds)
    optimizer.maximize(init_points, n_iter)
    #p = Path(f"results/sanity_check")
    #storage = open(p, "wb")
    #pickle.dump(optimizer.max, storage)
    #storage.close()


# Stuff for B-spline experimentation in particular
benchmark = feareu.doppler
sample_size = 500
base_algo_type = feareu.BsplineFeaPSO
bounding = pso_bounds

# TODO: change this when we get a better bspline evaluation method
bspline_eval_class = feareu.SlowBsplineEval

if __name__ == "__main__":
    global base_alg
    x = np.random.random(sample_size)
    y = benchmark(x)
    func_width = np.max(y) - np.min(y)
    noise = func_width / 20
    y = feareu.make_noisy(y, noise)
    # plt.scatter(x, y)
    # plt.show()
    global fitness
    fitness = bspline_eval_class(x, y)
    base_alg = base_algo_type
    bounds = deepcopy(pbounds)
    bounds.update(bounding)
    print(
        "function: ",
        benchmark,
        "\nsample size: ",
        sample_size,
        "\nnoise: ",
        noise,
        "\nalgorithm: FEA",
        base_alg,
    )
    bayes_run_fea(bounds, init_points=1, n_iter=4)
