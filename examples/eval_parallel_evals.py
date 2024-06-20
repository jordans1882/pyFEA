from feareu.benchmarks import rastrigin__, sphere__
import numpy as np
import time
from feareu.base_algos import ParallelFeaDE, ParallelFeaGA, parallel_eval, FeaDE, FeaGA
import math
import multiprocessing
from multiprocessing import shared_memory

def slow_eval(solution):
    time.sleep(3e-4)
    return rastrigin__(solution)
pop_to_eval = np.random.random(size=(300,1000000))
pop_evals = np.zeros((20))
rastrigin__(pop_evals)
#t0 = time.time()
#rastrigin__(np.zeros((50)))
#t1 = time.time()
#
#print("serial eval: ", t1 - t0)
#print(pop_evals)
#
#evals = np.zeros((20))
#
#t2 = time.time()
#evals = parallel_eval(rastrigin__, pop_to_eval)
#t3 = time.time()
#
#print("parallel eval: ", t3-t2)
#print(evals)
#shm = shared_memory.SharedMemory(create=True, size=pop_to_eval.nbytes)
#shared = np.ndarray(pop_to_eval.shape, buffer=shm.buf)
#shared[:] = pop_to_eval[:]

def p_eval(eval_function, population, processes = 4, chunksize = 4):
    with multiprocessing.Pool(processes = processes) as pool:
        evals = pool.map(eval_function, population, chunksize = chunksize)
        time.sleep(5)
    pool.close()
    return evals

def tester(population):
    np.sum(population)
before = time.time()
p_eval(tester, pop_to_eval, processes = 1, chunksize = 300)
after = time.time()
print("single process done")

start = time.time()
p_eval(tester, pop_to_eval, processes = 12, chunksize = 250)
end = time.time()
print("multiprocess done")

print("with one process: ", after - before)
print("with multiple processes: ", end - start)
domain = np.zeros((500,2))
domain[:,0] = -5
domain[:,1] = 5
t4 = time.time()
alg = ParallelFeaGA(slow_eval, domain, generations=200, pop_size = 500, b = 0.7, mutation_range=0.5, mutation_rate = 0.05, processes=5, chunksize=100)
alg.run()
print(alg.best_eval)
t5 = time.time()

alg = FeaGA(slow_eval, domain, generations=200, pop_size = 500, b = 0.7, mutation_range=0.5, mutation_rate = 0.05)
alg.run()
print(alg.best_eval)
t6=time.time()

print("parallel PSO: ", t5-t4)
print("serial PSO: ", t6-t5)
