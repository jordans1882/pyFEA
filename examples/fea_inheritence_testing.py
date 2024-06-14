from feareu import FEA
from feareu.base_algos import FeaPso
from feareu import Function
import numpy as np
import pytest
import time
import numba

import matplotlib.pyplot as plt
from numpy import cos, sqrt, pi, e, exp, sum

@numba.jit
def rastrigin__(solution = None):
    #return sum(solution**2)
    return sum(solution**2 - 10 * cos(2 * pi * solution) + 10)

array = np.zeros((10))
array[0] = 1
array[1] = 2
domain = np.zeros((10, 2))
domain[:,0] = -5
domain[:,1] = 5
function = Function(array, rastrigin__, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
#rand_factors = [np.random.choice(range(10), replace=False, size=3) for x in range(10)]
fct = [[0],[0,1],[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9],[9]]
fea1 = FEA(factors=fct, function = rastrigin__, iterations = 50, dim = 10, base_algo_name=FeaPso, domain=domain, generations= 20, pop_size=20)
fea1.run()
daig_plt1 = fea1.diagnostic_plots()
plt.show()