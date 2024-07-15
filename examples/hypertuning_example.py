import numpy as np
from bayes_opt import BayesianOptimization

from pyfea import PSO, benchmarks


def bayes_input(phi_p, phi_g, omega):
    domain = np.zeros((10, 2))
    domain[:, 0] = -5
    domain[:, 1] = 5
    pso = PSO(
        benchmarks.sphere__,
        domain=domain,
        generations=100,
        pop_size=5000,
        phi_p=phi_p,
        phi_g=phi_g,
        omega=omega,
    )
    gbest = pso.run()
    return -benchmarks.sphere__(gbest)


pbounds = {"phi_p": (0, 4), "phi_g": (0, 4), "omega": (0, 1)}
optimizer = BayesianOptimization(bayes_input, pbounds)
optimizer.set_gp_params
optimizer.maximize(5, 25)
print(optimizer.max)
