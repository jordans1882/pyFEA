
from fea import FEA
from base_pso import PSO
from function import Function
import numpy as np
import pytest
import time
import numba
from numpy import cos, sqrt, pi, e, exp, sum

def rastrigin__(solution = None):
    return sum(solution**2)

"""context_variable =np.array([-1.20183249,  2.0641521 , -0.95602851,  0.91484061,  0.03741595,
       -0.93080492, -2.02097708, -0.88648763, -1.9462521 ,  2.0437545])

cont_var=np.array([-1.20183249,  2.0641521 , -0.95602851,  0.91484061,  0.03741595,
       -0.93080492,  0.35505035, -0.88648763, -1.9462521 ,  2.0437545])

print(rastrigin__(context_variable))
print(rastrigin__(cont_var))"""

array = np.zeros((10))
array[0] = 1
array[1] = 2
domain = np.zeros((10, 2))
domain[:,0] = -5
domain[:,1] = 5
function = Function(array, rastrigin__, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
fea = FEA([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]], rastrigin__, 10, 10, "PSO", domain)
fea.run()