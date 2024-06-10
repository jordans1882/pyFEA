import math
from copy import deepcopy
import numpy as np


def rastrigin(x):
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))



f = rastrigin

niter = 100
phi_p = math.sqrt(2)
phi_g = math.sqrt(2)
dims = 2
npop = 50
bounds = [-5, 5]


rastrigin(np.array([0.0, 0.0]))

# step 1 initialize particles
pop = (np.random.uniform(bounds[0], bounds[1], npop * 2)).reshape((npop, dims))
pbest = pop
print(pop)

# step 2 initialize velocities
velocities = (
    np.random.uniform(
        -math.fabs(bounds[1] - bounds[0]) * 0.5, math.fabs(bounds[1] - bounds[0]) * 0.5, npop * 2
    )
).reshape((npop, dims))

velocities

pbest_evals = [f(pop[i, :]) for i in range(npop)]
# first update

gbest_eval = np.min(pbest_evals)
gbest_idx = np.argmin(pbest_evals)
gbest = pop[gbest_idx, :]

omega = math.sqrt(2) / 2

velocities

for iter in range(niter):
    #print("iter: ", iter, "/", niter)
    for pidx in range(npop):
        # print("popid: ", pidx)
        for dim in range(dims):
            r_p = np.random.random()
            r_g = np.random.random()
            # update velocity
            velocities[pidx][dim] = (
                omega * velocities[pidx][dim]
                + phi_p * r_p * (pbest[pidx][dim] - pop[pidx][dim])
                + phi_g * r_g * (gbest[dim] - pop[pidx][dim])
            )
            if(pidx==1):
                print((gbest[dim]), (pop[pidx][dim]))
                print("velocity of ",  pidx, ", ", dim, velocities[pidx][dim])
            # update position
            if pidx==1:
                print(gbest[dim], pop[pidx][dim])
        pop[pidx, :] = pop[pidx, :] + velocities[pidx, :]
        if(pidx==1):
            print("pop", pop[pidx, :])
        curr_pop = pop[pidx, :]
        curr_eval = f(curr_pop)
        """if(pidx==1):
            print(curr_eval)"""
        # print("f(", curr_pop, ") = ", curr_eval)
        pbe = pbest_evals[pidx]
        """if(pidx%5==0):
            print("pbe", pidx, ": ", pbest_evals[pidx])"""
        if curr_eval < pbe:
            # print("updating pbest from ", pbest_evals[pidx], " to ", curr_eval)
            pbest[pidx, :] = curr_pop
            pbest_evals[pidx] = curr_eval
            if curr_eval < gbest_eval:
                print("updating gbest from ", gbest_eval, " to ", curr_eval)
                gbest = deepcopy(curr_pop)
                # gbest_eval = curr_eval
                gbest_eval = np.min(pbest_evals)
                gbest_idx = np.argmin(pbest_evals)
    print("gbest = ", gbest)
    #print("gbest_eval = ", rastrigin(gbest))
    print("gbe = ", gbest_eval)
    #print("gbest_idx = ", gbest_idx)
    #print("")
    #print("")


# rastrigin(np.array(pop[gbest_idx, :]))
# pop[gbest_idx, :]
# rastrigin(np.array([-1.87283425, -0.13822152]))
# rastrigin(np.array(pop[gbest_idx, :]))
# gbest
# gbest_eval
# pop[gbest_idx, :]
# rastrigin(np.array([-1.87283425 - 0.13822152]))
# rastrigin(np.array([9.71623289 - 3.22594915]))
