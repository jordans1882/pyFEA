import numpy as np

from FEA.basealgorithms.pso import PSO
from FEA.FEA.factorarchitecture import FactorArchitecture
from FEA.FEA.factorevolution import FEA
from FEA.optimizationproblems.continuous_functions import Function

fa = FactorArchitecture(
    10, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]
)

func = Function(8, -5, 5, shift_data=np.zeros((1, 10)))

fea = FEA(
    func,
    fea_runs=300,
    generations=10,
    pop_size=500,
    factor_architecture=fa,
    base_algorithm=PSO,
)

fea.run()

func.run(fea.global_solution)
fea.global_solution
