import numpy as np
from feareu import PSO
from feareu import benchmarks
import pytest
import time
from bayes_opt import BayesianOptimization


def bayes_input(phi_p, phi_g, omega):
    domain = np.zeros((10, 2))
    dim = 10
    domain[:,0] = -5
    domain[:,1] = 5
    pso = PSO(benchmarks.sphere__, domain = domain, generations = 100, pop_size=5000, phi_p=phi_p, phi_g=phi_g, omega=omega)
    gbest = pso.run()
    #print(pso.pop)
    return -benchmarks.sphere__(gbest)

pbounds = {"phi_p":(0,4), "phi_g":(0,4), "omega":(0,1)}
optimizer = BayesianOptimization(bayes_input, pbounds)
optimizer.maximize()
print(optimizer.max)

