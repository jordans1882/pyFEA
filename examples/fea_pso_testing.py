import matplotlib.pyplot as plt
import numpy as np

from feareu.base_algos import FeaPso, PSO
from feareu import Function

ndims = 10


def rastrigin__(solution=None):
    return np.sum(solution**2 - 10 * np.cos(2 * np.pi * solution) + 10)


array = np.zeros((ndims))

domain = np.zeros((ndims, 2))
domain[:, 0] = -5
domain[:, 1] = 5
domain
# function = Function(array, rastrigin__, [0, 1])

function = Function(array, rastrigin__, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
pso = FeaPso(function = function, domain = domain, pop_size=10)
pso_og = PSO(function = function, domain = domain, pop_size=10)
pso.run()
pso_og.run()
diag_plots = pso.diagnostic_plots()
diag_plots = pso_og.diagnostic_plots()
plt.show()