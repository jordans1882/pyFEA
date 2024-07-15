import matplotlib.pyplot as plt
import numpy as np

from pyfea import FEA
from pyfea.benchmarks import rastrigin__
from pyfea.fea import linear_factorizer
from pyfea.fea.base_algos import FeaGA

domain = np.zeros((10, 2))
domain[:, 0] = -5
domain[:, 1] = 5

fea = FEA(
    factors=linear_factorizer(3, 2, 10),
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
