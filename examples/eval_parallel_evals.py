from feareu.base_algos import parallel_eval
from feareu.benchmarks import rastrigin__
import numpy as np
import time
from feareu.base_algos import ParallelFeaPSO
import math
from feareu.base_algos import FeaPso

def slow_eval(solution):
    time.sleep(1e-5)
    return 0

pop_to_eval = np.random.random(size=(20,25))

pop_evals = np.zeros((20))
rastrigin__(pop_evals)
t0 = time.time()
rastrigin__(np.zeros((50)))
t1 = time.time()

print("serial eval: ", t1 - t0)
print(pop_evals)

evals = np.zeros((20))

t2 = time.time()
evals = parallel_eval(rastrigin__, pop_to_eval)
t3 = time.time()

print("parallel eval: ", t3-t2)
print(evals)

domain = np.zeros((50,2))
domain[:,0] = -5
domain[:,1] = 5
t4 = time.time()
pso = ParallelFeaPSO(slow_eval, domain, generations=10, pop_size = 50, phi_p=math.sqrt(2), phi_g=math.sqrt(2), omega=1 / math.sqrt(2), processes = 5, chunksize = 10)
pso.run()
print(pso.gbest_eval)
t5 = time.time()

pso = FeaPso(slow_eval, domain, generations=10, pop_size = 500, phi_p=math.sqrt(2), phi_g=math.sqrt(2), omega=1 / math.sqrt(2))
pso.run()
print(pso.gbest_eval)
t6=time.time()

print("parallel PSO: ", t5-t4)
print("serial PSO: ", t6-t5)
