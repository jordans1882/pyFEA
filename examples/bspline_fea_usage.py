from feareu import FEA
from feareu.base_algos import FeaDE, FeaGA
from feareu import Function
import numpy as np
from feareu.benchmarks import rastrigin__
from feareu import BsplineFEA

import matplotlib.pyplot as plt
from numpy import cos, sqrt, pi, e, exp, sum

#domain = np.zeros((10, 2))
#domain[:,0] = -5
#domain[:,1] = 5
domain = (-5,5)
#rand_factors = [np.random.choice(range(10), replace=False, size=3) for x in range(10)]
fct = [[0],[0,1],[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9],[9]]
fea = BsplineFEA(factors=fct, function = rastrigin__, iterations = 75, dim = 10, base_algo_name=FeaDE, domain=domain, generations= 20, pop_size=20)
#fea.context_variable = fea.init_full_global()
#fea.context_variable.sort()
#print(fea.context_variable)
#fea.domain_restriction()
#for dom in fea.domain:
#    print(dom)

print(fea.run())

#daig_plt1 = fea.diagnostic_plots()
#plt.show()

