import matplotlib.pyplot as plt
import numpy as np
import feareu.benchmarks as benchmarks
from feareu import PSO
from feareu.function import Function
from feareu import FEA

ndims = 10


array = np.zeros((ndims))

domain = np.zeros((ndims, 2))
domain[:, 0] = -5
domain[:, 1] = 5
# function = Function(array, rastrigin__, [0, 1])

def linear_factorizer(fact_size, overlap, dim):
    smallest = 0
    if fact_size <= overlap:
        temp = fact_size
        fact_size = overlap
        overlap = temp
    largest = fact_size
    factors = []
    while largest <= dim:
        factors.append([x for x in range(smallest, largest)])
        smallest = largest - overlap
        largest += fact_size - overlap
    if smallest < dim:
        factors.append([x for x in range(smallest, dim)])
    return factors
fact_size = 3
overlap = 1
dim = ndims
factors = linear_factorizer(fact_size, overlap, dim)
fea = FEA(factors, benchmarks.rastrigin__, 20, dim, "PSO", domain, pop_size=50, generations=100) 
out = fea.run()
diag_plots = fea.diagnostic_plots()
plt.show()
print(out)
