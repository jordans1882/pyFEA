import matplotlib.pyplot as plt
import numpy as np

from feareu.base_algos import GA, FeaGA
from feareu import Function

ndims = 5


def rastrigin__(solution=None):
    return np.sum(solution**2 - 10 * np.cos(2 * np.pi * solution) + 10)


array = np.zeros((ndims))

domain = np.zeros((ndims, 2))
domain[:, 0] = -5
domain[:, 1] = 5
domain
# function = Function(array, rastrigin__, [0, 1])

function = Function(array, rastrigin__, [0, 1, 2, 3, 4])
ga = GA(function = function, domain = domain, pop_size=20)
feaga = FeaGA(function = function, domain = domain, pop_size=20)
ga.run()
feaga.run()
diag_plots = feaga.diagnostic_plots()
diag_plots = ga.diagnostic_plots()
plt.show()