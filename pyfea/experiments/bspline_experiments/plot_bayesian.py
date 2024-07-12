import logging
import math
import pickle
import time
from copy import deepcopy
from pathlib import Path
from types import NoneType

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import splipy
from bayes_opt import BayesianOptimization
from scipy.stats import gaussian_kde

import pyfea

best = float("-inf")
number_recorded = 0

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
filename = results_dir / f"logging_best.log"
logger = logging.getLogger(__name__)
logging.basicConfig(filename=filename, encoding="utf-8", level=logging.DEBUG)


# set the number of processors and the chunksizes used by parallel base algorithms.
processes = 4
chunksize = 1

# set the number of processes and threads used by a parallel FEA.
process_count = 8
thread_count = 16

# set the number of iterations at which we record data to be printed.
diagnostics_amount = 1

# set the bounds for your FEA's Bayesian run.
# IMPORTANT: Only set the variables you want to use as hyperparameters. Comment out the others.

iterations = 100
generations = 100000
pop_size = 300

pbounds = {
    "generations": (2, 40),
    # "iterations": (2, 70),
    "pop_size": (30, 200),
    "fact_size": (1, 50),
    "overlap": (0, 30),
    # "num_covers":(1,5),
    # "num_clamps": (0, 5),
    "dim": (5, 500),
}

base_bounds = {
    # "generations":(5,200),
    # "pop_size":(30,100),
    "dim": (5, 500)
}

pso_bounds = {"phi_p": (0.8, 3), "phi_g": (0.8, 3), "omega": (0.00000001, 0.99)}

ga_bounds = {
    "mutation_rate": (0.01, 0.99),
    "mutation_range": (0.1, 2),
}

de_bounds = {
    "mutation_factor": (0.1, 1),
    "crossover_rate": (0.1, 1),
}


# choose your factorizer.
factorizer = pyfea.linear_factorizer

# set the kind of FEA you want to use.
fea = pyfea.ParallelBsplineFEA


def bayes_input_fea(
    fact_size=None,
    overlap=1,
    num_covers=2,
    num_clamps=0,
    generations=20,
    iterations=iterations,
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

    global best
    global number_recorded

    # code for generating domain on Bspline FEA. Comment out if you're working on normal FEA.
    domain = (0, 1)

    factors = factorizer(fact_size=fact_size, overlap=overlap, dim=dim, num_covers=num_covers)
    if factors is None or fact_size > dim:
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
    if ret > best:
        best = ret
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        filename = results_dir / f"{number_recorded}_Function_FEA_{base_alg.__name__}.png"
        plt.figure()
        objective.diagnostic_plots()
        plt.savefig(filename)
        logger.info(
            f"NEW BEST - NUMBER {number_recorded} ------------------------------------------------"
        )
        logger.info(f"fitness evaluation: {ret}")
        logger.info(f"fact_size: {fact_size}")
        logger.info(f"overlap: {overlap}")
        logger.info(f"num_clamps: {num_clamps}")
        logger.info(f"dim: {dim}")
        logger.info(f"iterations: {iterations}")
        logger.info(f"base_alg: {base_alg.__name__}")
        logger.info(f"pop_size: {pop_size}")
        logger.info(f"generations: {generations}")
        logger.info(f"mutation_factor: {mutation_factor}")
        logger.info(f"crossover_rate: {crossover_rate}")
        logger.info(f"mutation_rate: {mutation_rate}")
        logger.info(f"mutation_range: {mutation_range}")
        logger.info(f"phi_p: {phi_p}")
        logger.info(f"phi_g: {phi_g}")
        logger.info(f"omega: {omega}")
        knots = objective.context_variable
        plot_results(knots, fea)
        number_recorded += 1
    return ret


def bayes_run_fea(bounds, init_points=5, n_iter=25, sample_size=-1, noise_level=-1, func=-1):
    global best
    best = float("-inf")
    optimizer = BayesianOptimization(bayes_input_fea, bounds)
    optimizer.maximize(init_points, n_iter)


def bayes_input_base(
    generations=generations,
    pop_size=pop_size,
    mutation_factor=0.5,
    crossover_rate=0.9,
    mutation_rate=0.05,
    mutation_range=0.5,
    phi_p=math.sqrt(2),
    phi_g=math.sqrt(2),
    omega=1 / math.sqrt(2),
    dim=10,
):
    dim = int(dim)
    generations = int(generations)
    pop_size = int(pop_size)

    domain = np.zeros((dim, 2))
    domain[:, 0] = 0
    domain[:, 1] = 1

    global best
    global number_recorded

    if base_alg is pyfea.ParallelBsplineFeaPSO:
        objective = base_alg(
            function=fitness,
            generations=generations,
            domain=domain,
            pop_size=pop_size,
            phi_p=phi_p,
            phi_g=phi_g,
            omega=omega,
            processes=processes,
            chunksize=chunksize,
            fitness_terminate=True,
        )
        ret = -objective.run()
        if ret > best:
            best = ret
            results_dir = Path("results")
            results_dir.mkdir(parents=True, exist_ok=True)
            filename = results_dir / f"{number_recorded}_Function_{base_alg.__name__}.png"
            plt.figure()
            objective.diagnostic_plots()
            plt.savefig(filename)
            logger.info(
                f"NEW BEST - NUMBER {number_recorded} ------------------------------------------------"
            )
            logger.info(f"fitness evaluation: {ret}")
            logger.info(f"dim: {dim}")
            logger.info(f"base_alg: {base_alg.__name__}")
            logger.info(f"pop_size: {pop_size}")
            logger.info(f"generations: {generations}")
            logger.info(f"phi_p: {phi_p}")
            logger.info(f"phi_g: {phi_g}")
            logger.info(f"omega: {omega}")
            knots = base_alg.best_position
            plot_results(knots, base_alg)
            number_recorded += 1

    elif base_alg is pyfea.ParallelBsplineFeaDE:
        objective = base_alg(
            function=fitness,
            generations=generations,
            domain=domain,
            pop_size=pop_size,
            mutation_factor=mutation_factor,
            crossover_rate=crossover_rate,
            processes=processes,
            chunksize=chunksize,
            fitness_terminate=True,
        )
        ret = -objective.run()
        if ret > best:
            best = ret
            results_dir = Path("results")
            results_dir.mkdir(parents=True, exist_ok=True)
            filename = results_dir / f"{number_recorded}_Function_{base_alg.__name__}.png"
            plt.figure()
            objective.diagnostic_plots()
            plt.savefig(filename)
            logger.info(
                f"NEW BEST - NUMBER {number_recorded} ------------------------------------------------"
            )
            logger.info(f"fitness evaluation: {ret}")
            logger.info(f"dim: {dim}")
            logger.info(f"base_alg: {base_alg.__name__}")
            logger.info(f"pop_size: {pop_size}")
            logger.info(f"generations: {generations}")
            logger.info(f"mutation_factor: {mutation_factor}")
            logger.info(f"crossover_rate: {crossover_rate}")
            knots = base_alg.best_position
            plot_results(knots, base_alg)
            number_recorded += 1

    elif base_alg is pyfea.ParallelBsplineFeaGA:
        objective = base_alg(
            function=fitness,
            generations=generations,
            domain=domain,
            pop_size=pop_size,
            mutation_rate=mutation_rate,
            mutation_range=mutation_range,
            processes=processes,
            chunksize=chunksize,
            fitness_terminate=True,
        )
        ret = -objective.run()
        if ret > best:
            best = ret
            results_dir = Path("results")
            results_dir.mkdir(parents=True, exist_ok=True)
            filename = results_dir / f"{number_recorded}_Function_{base_alg.__name__}.png"
            plt.figure()
            objective.diagnostic_plots()
            plt.savefig(filename)
            logger.info(
                f"NEW BEST - NUMBER {number_recorded} ------------------------------------------------"
            )
            logger.info(f"fitness evaluation: {ret}")
            logger.info(f"dim: {dim}")
            logger.info(f"base_alg: {base_alg.__name__}")
            logger.info(f"pop_size: {pop_size}")
            logger.info(f"generations: {generations}")
            logger.info(f"mutation_rate: {mutation_rate}")
            logger.info(f"mutation_range: {mutation_range}")
            knots = base_alg.best_position
            plot_results(knots, base_alg)
            number_recorded += 1

    return ret


def bayes_run_base(bounds, init_points=5, n_iter=25, sample_size=-1, noise_level=-1, func=-1):
    global best
    best = float("-inf")
    optimizer = BayesianOptimization(bayes_input_base, bounds)
    optimizer.maximize(init_points, n_iter)


def plot_results(knots, alg):
    knots = pyfea.bspline_clamp(knots, 3)
    bsp = splipy.BSplineBasis(3, knots, -1)
    xmat = bsp.evaluate(x, 0, True, True)
    xseq = np.linspace(0, 1, len(y))
    xmat_seq = bsp.evaluate(xseq, 0, True, True)
    xt = xmat.transpose()
    LHS = xt @ xmat
    RHS = xt @ y
    theta, info = sparse.linalg.bicgstab(LHS, RHS)
    yest_seq = xmat_seq @ theta
    yest = xmat @ theta
    knot_y = np.zeros(knots.shape)
    knot_y[:] = np.min(y) - 0.2

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    filename = results_dir / f"{number_recorded}_Function_est_{alg.__name__}"

    plt.figure()
    plt.plot(xseq, yest_seq, "y")
    plt.scatter(x, y, s=5)
    plt.scatter(knots, knot_y, color="orange", s=5)
    plt.savefig(filename)

    filename = results_dir / f"{number_recorded}_Knot_density_{alg.__name__}"

    density = gaussian_kde(knots)
    xs = np.linspace(0, 1, 200)
    density.covariance_factor = lambda: 0.1
    density._compute_covariance()
    plt.figure()
    plt.plot(xs, density(xs))
    upper = np.max(density(xs))
    plt.ylim((0, upper))
    plt.savefig(filename)


# Stuff for B-spline experimentation in particular
benchmarks = [
    pyfea.big_spike,
    pyfea.discontinuity,
    pyfea.cliff,
    pyfea.smooth_peak,
    pyfea.second_smooth_peak,
    pyfea.doppler,
]
sample_sizes = np.around(np.geomspace(2000, 200000, num=3)).astype(int)
base_algo_types = [
    pyfea.ParallelBsplineFeaPSO,
    pyfea.ParallelBsplineFeaDE,
    pyfea.ParallelBsplineFeaGA,
]
search_types = [
    pyfea.ParallelBsplineFeaPSO,
    pyfea.ParallelBsplineFeaDE,
    pyfea.ParallelBsplineFeaGA,
]
bounding = [pso_bounds, de_bounds, ga_bounds]

# TODO: change this when we get a better bspline evaluation method
bspline_eval_class = pyfea.SlowBsplineEval

if __name__ == "__main__":
    global x
    global y
    global base_alg
    for f, function in enumerate(benchmarks):
        for sample_size in sample_sizes:
            x = np.random.random(sample_size)
            y = function(x)
            func_width = np.max(y) - np.min(y)
            noises = np.linspace(func_width / 100, func_width / 20, num=3)
            for n, noise in enumerate(noises):
                y = pyfea.make_noisy(y, noise)
                global fitness
                results_dir = Path("results")
                results_dir.mkdir(parents=True, exist_ok=True)
                filename = (
                    results_dir
                    / f"Baseline_{function.__name__}_noise_{n}_sample_size_{sample_size}"
                )
                xseq = np.linspace(0, 1, 100000)
                yseq = function(xseq)
                plt.figure()
                plt.scatter(x, y)
                plt.plot(xseq, yseq, "k")
                plt.savefig(filename)
                fitness = bspline_eval_class(x, y)
                before = time.time()
                for i, algo in enumerate(base_algo_types):
                    base_alg = algo
                    bounds = deepcopy(pbounds)
                    bounds.update(bounding[i])
                    print(
                        "function: ",
                        function,
                        "\nsample size: ",
                        sample_size,
                        "\nnoise: ",
                        noise,
                        "\nalgorithm: FEA",
                        algo,
                    )
                    bayes_run_fea(
                        bounds,
                        init_points=20,
                        n_iter=100,
                        sample_size=sample_size,
                        noise_level=n,
                        func=f,
                    )
                for i, algo in enumerate(search_types):
                    base_alg = algo
                    bounds = deepcopy(base_bounds)
                    bounds.update(bounding[i])
                    print(
                        "function: ",
                        function,
                        "\nsample size: ",
                        sample_size,
                        "\nnoise: ",
                        noise,
                        "\nalgorithm: ",
                        algo,
                    )
                    bayes_run_base(
                        bounds,
                        init_points=20,
                        n_iter=100,
                        sample_size=sample_size,
                        noise_level=n,
                        func=f,
                    )

                after = time.time()
                print("Time to run one Bayesian optimizer per algorithm: ", after - before)
