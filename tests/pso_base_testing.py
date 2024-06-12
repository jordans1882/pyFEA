import numpy as np
from examples.base_pso import PSO
import numba
import pytest
import time
from bayes_opt import BayesianOptimization

@numba.jit
def rastrigin__(solution = None):
    return np.sum(solution**2 - 10 * np.cos(2 * np.pi * solution) + 10)

@numba.jit
def sphere__(solution=None):
    return np.sum(solution**2)

def bayes_input(generations, phi_p, phi_g, omega):
    generations = int(generations)
    domain = np.zeros((10, 2))
    dim = 10
    domain[:,0] = -5
    domain[:,1] = 5
    pso = PSO(sphere__, domain = domain, generations = generations, pop_size=5000, phi_p=phi_p, phi_g=phi_g, omega=omega)
    gbest = pso.run()
    #print(pso.pop)
    return -sphere__(gbest)

pbounds = {"generations":(200,1000), "phi_p":(0,4), "phi_g":(0,4), "omega":(0,1)}
optimizer = BayesianOptimization(bayes_input, pbounds)
optimizer.maximize()
print(optimizer.max)

#domain = np.zeros((10, 2))
#domain[:,0] = -5
#domain[:,1] = 5
#pso = PSO(rastrigin__, domain = domain, generations = 200, pop_size = 10000)
#gbest = pso.run()
#print(gbest)
#print(rastrigin__(gbest))
