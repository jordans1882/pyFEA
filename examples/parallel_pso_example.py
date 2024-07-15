import math

import matplotlib.pyplot as plt
import numpy as np

from pyfea import PSO
from pyfea.benchmarks.benchmarks import rastrigin__

# construct pso w/ default parms
pso = PSO(
    rastrigin__,
    domain=np.array([[-5.0, 5.0], [-5.0, 5.0]]),
    generations=100,
    pop_size=20,
    phi_p=math.sqrt(2),
    phi_g=math.sqrt(2),
    omega=1 / math.sqrt(2),
)

# Run the algorithm
pso.run(parallel=True)

# Get the solution
soln = pso.get_soln()
print(soln)

# Get the fitness of the solution
soln_fitness = pso.get_soln_fitness()
print(soln_fitness)

# Plot diagnostics
pso.diagnostic_plots()
plt.show()
