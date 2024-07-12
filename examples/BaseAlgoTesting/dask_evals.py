import itertools
import multiprocessing
import time
from multiprocessing import shared_memory

import numpy as np

from pyfea.base_algos import FeaDE, ParallelFeaDE
from pyfea.benchmarks import *


def parallel_run(run, subpops, processes=4, chunksize=4):
    with multiprocessing.get_context("spawn").Pool(processes=processes) as pool:
        evals = pool.map(run, subpops, chunksize=chunksize)
    pool.close()
    return evals


def combination_factorizer(fact_size, dim):
    factors = list(itertools.combinations(range(dim), fact_size))
    return factors


def run(subpop):
    subpop.run()
    return subpop.best_eval


def slow_eval(solution):
    time.sleep(1e-3)
    return rastrigin__(solution)


if __name__ == "__main__":
    dim = 10
    domain = np.zeros((dim, 2))
    domain[:, 0] = -5
    domain[:, 1] = 5
    fact_size = 2
    processes = 5
    chunksize = 9
    print("about to make factors")
    factors = combination_factorizer(fact_size, dim)
    print("factor len: ", len(factors))
    ret = np.array(
        [
            FeaDE(
                slow_eval,
                domain[subpop, :],
                generations=10,
                pop_size=50,
                crossover_rate=0.9,
                mutation_factor=0.5,
            )
            for subpop in factors
        ]
    )
    shm = shared_memory.SharedMemory(create=True, size=ret.nbytes)
    god_why = np.ndarray(ret.shape, dtype=ret.dtype, buffer=shm.buf)
    god_why[:] = ret[:]
    print("made the stuff")
    before = time.time()
    parallel_run(run, ret, processes=processes, chunksize=chunksize)
    print("original parallel done")

    in_between = time.time()

    with multiprocessing.Pool(processes=processes) as pool:
        pool.map(run, god_why, chunksize=chunksize)
    pool.close()
    shm.close()
    shm.unlink()

    print("shared parallel done")
    please = time.time()
    for subpop in ret:
        subpop.run()
    print("serial done")
    after = time.time()
    # client = Client(n_workers=4, threads_per_worker=2)
    # print("post-client")
    # delays = [dask.delayed(run)(subpop) for subpop in ret]
    # final = dask.compute(*delays)
    # ans = ran.compute()
    # client.close()
    # print("dask done")
    # done = time.time()
    # print(len(final))

    print("original parallel took: ", in_between - before)
    print("shared parallel took: ", please - in_between)
    print("serial took: ", (after - please))
