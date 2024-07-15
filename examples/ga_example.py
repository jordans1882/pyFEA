import math
import time

import matplotlib.pyplot as plt
import numpy as np

from pyfea import GA
from pyfea.benchmarks.benchmarks import rastrigin__

dims = 50
dom = np.zeros((dims, 2))
dom[:, 0] = -5.0
dom[:, 1] = 5.0

# construct pso w/ default parms
ga = GA(
    rastrigin__,
    domain=dom,
    generations=50,
    pop_size=1000,
    mutation_rate=0.05,
    mutation_range=0.5,
    tournament_options=2,
    number_of_children=2,
)

# Run the algorithm (in serial)
start_time = time.time()
ga.run(parallel=False)
end_time = time.time()
print(end_time - start_time)

# Get the solution
soln = ga.get_soln()
print(soln)

# Get the fitness of the solution
soln_fitness = ga.get_soln_fitness()
print(soln_fitness)

# Plot diagnostics
ga.diagnostic_plots()
plt.show()


# reconstruct pso w/ default parms
ga = GA(
    rastrigin__,
    domain=dom,
    generations=50,
    pop_size=1000,
    mutation_rate=0.05,
    mutation_range=0.5,
    tournament_options=2,
    number_of_children=2,
)

# Run the algorithm (in parallel)
start_time = time.time()
ga.run(parallel=True, processes=12, chunksize=10)
end_time = time.time()
print(end_time - start_time)

# Get the solution
soln = ga.get_soln()
print(soln)

# Get the fitness of the solution
soln_fitness = ga.get_soln_fitness()
print(soln_fitness)

# Plot diagnostics
ga.diagnostic_plots()
plt.show()
