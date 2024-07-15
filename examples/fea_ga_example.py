import matplotlib.pyplot as plt
import numpy as np

from pyfea import FEA, Function
from pyfea.benchmarks import rastrigin__
from pyfea.fea.base_algos import FeaGA

array = np.zeros((10))
array[0] = 1
array[1] = 2
domain = np.zeros((10, 2))
domain[:, 0] = -5
domain[:, 1] = 5
function = Function(array, rastrigin__, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


fea = FEA(
    factors=[
        [0],
        [0, 1],
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [6, 7, 8],
        [7, 8, 9],
        [8, 9],
        [9],
    ],
    function=rastrigin__,
    iterations=100,
    dim=10,
    base_algo=FeaGA,
    domain=domain,
    generations=20,
    pop_size=40,
)

fea.run()

# fea.get_soln()
# fea.get_soln_fitness()

fea.diagnostic_plots()
plt.show()
