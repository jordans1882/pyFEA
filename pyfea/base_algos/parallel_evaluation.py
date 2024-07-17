import multiprocessing


def parallel_eval(eval_function, population, processes=4, chunksize=4):
    with multiprocessing.Pool(processes=processes) as pool:
        evals = pool.map(eval_function, population, chunksize=chunksize)
    pool.close()
    return evals
