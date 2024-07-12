import time

import matplotlib.pyplot as plt
import numba
import numpy as np
import pytest
from numpy import cos, e, exp, pi, sqrt, sum

from pyfea import FEA, BsplineFEA, BsplineFeaPSO, Function, linear_factorizer
from pyfea.base_algos import FeaDE, FeaGA, FeaPso


@numba.jit
def rastrigin__(solution=None):
    # return sum(solution**2)
    return sum(solution**2 - 10 * cos(2 * pi * solution) + 10)


array = np.zeros((10))
array[0] = 1
array[1] = 2
domain = np.zeros((10, 2))
domain[:, 0] = -5
domain[:, 1] = 5
function = Function(array, rastrigin__, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# rand_factors = [np.random.choice(range(10), replace=False, size=3) for x in range(10)]
# fct = [[0],[0,1],[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[11,12,13],[12,13,14],[13,14],[14]]
fct = linear_factorizer(3, 1, 12)
fea1 = BsplineFEA(
    factors=fct,
    function=rastrigin__,
    iterations=200,
    dim=12,
    base_algo_name=BsplineFeaPSO,
    domain=(-5, 5),
    diagnostics_amount=1,
    generations=5,
    pop_size=15,
)
fea1.run()
daig_plt1 = fea1.diagnostic_plots()
plt.show()

