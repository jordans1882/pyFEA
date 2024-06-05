
from pymoo.problems.single import Rastrigin
#from pymoo.algorithms.soo.nonconvex.ga import GA
#from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
#from pymoo.core.population import Population
#from pymoo.core.result import Result
from skopt import gp_minimize
from bayes_opt import BayesianOptimization

numOfKnots = 1000

rastrigin = Rastrigin()

"""ga = GA() #specify bayseian
de = DE()



gaRes = minimize(rastrigin, ga)
minGA = gaRes.F
print("GA MSE: " + str(minGA))
knotPlacementGA = gaRes.X
print("GA Knot Placement: " + str(knotPlacementGA))
gaTime = gaRes.exec_time
print("GA Timing: " + str(gaTime))
print()

deRes = minimize(rastrigin, de)
minDE = deRes.F
print("DE MSE: " + str(minDE))
knotPlacementDE = deRes.X
print("DE Knot Placement: " + str(knotPlacementDE))
deTime = deRes.exec_time
print("DE Timing: " + str(deTime))
print()



pso = PSO()
psoRes = minimize(rastrigin, pso)
finPSOPop = psoRes.opt
minPSO = psoRes.F
print("PSO MSE: " + str(minDE))
knotPlacementPSO = psoRes.X
print("PSO Knot Placement: " + str(knotPlacementPSO))
psoTime = psoRes.exec_time
print("PSO Timing: " + str(psoTime))
print()
"""

# Bayesian
def bayInput(w, c1, c2):
    pso = PSO(w = w, c1 = c1, c2 = c2)
    psoRes = minimize(rastrigin, pso)
    minVal = psoRes.F
    return -(minVal[0])

#pbounds = {"w": (0, 1), "c1": (0, 1), "c2": (0, 1)}
pbounds = {"w": (0.0, 1.0), "c1": (0.0, 1.0), "c2":(0.0, 1.0)}
obj = BayesianOptimization(bayInput, pbounds)
obj.maximize()
print(obj.max)