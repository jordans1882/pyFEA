import matplotlib.pyplot as plt
import numpy as np

import examples.base_pso
import examples.fea_debug
from examples.base_pso import PSO
from examples.function import Function

ndims = 10


def rastrigin__(solution=None):
    return np.sum(solution**2 - 10 * np.cos(2 * np.pi * solution) + 10)


array = np.zeros((ndims))

domain = np.zeros((ndims, 2))
domain[:, 0] = -5
domain[:, 1] = 5
domain
# function = Function(array, rastrigin__, [0, 1])

pso = PSO(rastrigin__, domain, pop_size=1000)
pso.run()
diag_plots = pso.diagnostic_plots()
plt.show()
print(pso.run())
