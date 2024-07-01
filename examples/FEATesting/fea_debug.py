from feareu import FEA
from feareu import PSO
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
#rand_factors = [np.random.choice(range(10), replace=False, size=3) for x in range(10)]

"""fea1 = FEA(factors=[[0],[0,1],[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9],[9]], function = rastrigin__, iterations = 300, dim = 10, base_algo_name="PSO", domain=domain, generations= 20, pop_size =20)
fea2 = FEA(factors=[[0],[0,1],[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9],[9]], function = rastrigin__, iterations = 300, dim = 10, base_algo_name="PSO", domain=domain, generations= 10, pop_size =20)
fea3 = FEA(factors=[[0],[0,1],[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9],[9]], function = rastrigin__, iterations = 300, dim = 10, base_algo_name="PSO", domain=domain, generations= 5, pop_size = 20)"""
fea1 = FEA(factors=[[0],[0,1],[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9],[9]], function = rastrigin__, iterations = 100, dim = 10, base_algo_name="PSO", domain=domain, generations= 20, pop_size =40)
#fea2 = FEA(factors=[[0],[0,1],[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9],[9]], function = rastrigin__, iterations = 300, dim = 10, base_algo_name="PSO", domain=domain, generations= 20, pop_size =20)
#fea3 = FEA(factors=[[0],[0,1],[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9],[9]], function = rastrigin__, iterations = 300, dim = 10, base_algo_name="PSO", domain=domain, generations= 20, pop_size =10)
fea1.run()
#fea2.run()
#fea3.run()
#print(fea1.context_variable)
daig_plt1 = fea1.diagnostic_plots()
#diag_plt2 = fea2.diagnostic_plots()
#diag_plt3 = fea3.diagnostic_plots()
plt.show()