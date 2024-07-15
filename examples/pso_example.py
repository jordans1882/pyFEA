import math
import time

import matplotlib.pyplot as plt
import numpy as np

from pyfea import PSO
from pyfea.benchmarks.benchmarks import rastrigin__

dims = 50
dom = np.zeros((dims, 2))
dom[:, 0] = -5.0
dom[:, 1] = 5.0

# construct pso w/ default parms
pso = PSO(
    rastrigin__,
    domain=dom,
    generations=50,
    pop_size=100000,
    phi_p=1.4 * math.sqrt(2),
    phi_g=math.sqrt(2),
    omega=1 / math.sqrt(2),
)

start_time = time.time()
pso.run(parallel=False)
end_time = time.time()
print(end_time - start_time)

# reconstruct pso w/ default parms
pso = PSO(
    rastrigin__,
    domain=dom,
    generations=50,
    pop_size=100000,
    phi_p=1.4 * math.sqrt(2),
    phi_g=math.sqrt(2),
    omega=1 / math.sqrt(2),
)
start_time = time.time()
pso.run(parallel=True, processes=12, chunksize=10)
end_time = time.time()
print(end_time - start_time)

# Get the solution
soln = pso.get_soln()
print(soln)

# Get the fitness of the solution
soln_fitness = pso.get_soln_fitness()
print(soln_fitness)

# Plot diagnostics
pso.diagnostic_plots()
plt.show()
