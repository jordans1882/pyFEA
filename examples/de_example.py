import math
import time

import matplotlib.pyplot as plt
import numpy as np

from pyfea import DE
from pyfea.benchmarks.benchmarks import rastrigin__

dims = 50
dom = np.zeros((dims, 2))
dom[:, 0] = -5.0
dom[:, 1] = 5.0

# construct pso w/ default parms
de = DE(
    rastrigin__,
    domain=dom,
    generations=50,
    pop_size=1000,
    mutation_factor=0.5,
    crossover_rate=0.9,
)

start_time = time.time()
de.run(parallel=False)
end_time = time.time()
print(end_time - start_time)

# reconstruct pso w/ default parms
de = DE(
    rastrigin__,
    domain=dom,
    generations=50,
    pop_size=1000,
    mutation_factor=0.5,
    crossover_rate=0.9,
)
start_time = time.time()
de.run(parallel=True, processes=12, chunksize=10)
end_time = time.time()
print(end_time - start_time)

# Get the solution
soln = de.get_soln()
print(soln)

# Get the fitness of the solution
soln_fitness = de.get_soln_fitness()
print(soln_fitness)

# Plot diagnostics
de.diagnostic_plots()
plt.show()
