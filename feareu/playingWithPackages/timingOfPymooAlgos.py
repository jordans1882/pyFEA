from pymoo.problems.single import Rastrigin
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.core.result import Result

numOfKnots = 1000

rastrigin = Rastrigin()

ga = GA(numOfKnots)
de = DE(numOfKnots)
pso = PSO(numOfKnots)


gaRes = minimize(rastrigin, ga)
finGAPop = gaRes.opt
#FIND MIN print(finPop.get())
gaTime = gaRes.exec_time
print("GA Timing: " + str(gaTime))

deRes = minimize(rastrigin, de)
finDEPop = deRes.opt
#FIND MIN print(finPop.get())
deTime = deRes.exec_time
print("DE Timing: " + str(deTime))

psoRes = minimize(rastrigin, pso)
finPSOPop = psoRes.opt
#FIND MIN print(finPop.get())
psoTime = psoRes.exec_time
print("PSO Timing: " + str(psoTime))