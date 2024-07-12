import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, e, exp, pi, sqrt, sum

from pyfea import BsplineFEA, Function
from pyfea.base_algos import FeaDE, FeaGA, FeaPso
from pyfea.base_algos.bspline_specific.bspline_de import BsplineDE
from pyfea.benchmarks import rastrigin__

# domain = np.zeros((10, 2))
# domain[:,0] = -5
# domain[:,1] = 5
domain = (-5, 5)
# rand_factors = [np.random.choice(range(10), replace=False, size=3) for x in range(10)]
fct = [
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
]
og_knots = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

fea1 = BsplineFEA(
    factors=fct,
    function=rastrigin__,
    iterations=100,
    dim=10,
    base_algo_name=FeaPso,
    domain=domain,
    diagnostics_amount=1,
    generations=5,
    pop_size=20,
)
fea3 = BsplineFEA(
    factors=fct,
    function=rastrigin__,
    iterations=100,
    dim=10,
    base_algo_name=FeaDE,
    domain=domain,
    diagnostics_amount=5,
    generations=5,
    pop_size=20,
)
fea5 = BsplineFEA(
    factors=fct,
    function=rastrigin__,
    iterations=100,
    dim=10,
    base_algo_name=FeaGA,
    domain=domain,
    diagnostics_amount=5,
    generations=5,
    pop_size=20,
)


start = time.time()
print(fea1.run())
end = time.time()
print("PSO: ", end - start)

start = time.time()
print(fea3.run())
end = time.time()
print("DE: ", end - start)

start = time.time()
print(fea5.run())
end = time.time()
print("GA: ", end - start)

daig_plt1 = fea1.diagnostic_plots()
daig_plt1 = fea3.diagnostic_plots()
daig_plt1 = fea5.diagnostic_plots()

plt.show()
