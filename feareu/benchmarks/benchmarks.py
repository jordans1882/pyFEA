import numba
import numpy as np

"""
Continuous benchmark functions wrappers to speed up calculations
"""


@numba.jit
def sphere__(solution=None):
    return np.sum(solution**2)


@numba.jit
def elliptic__(solution=None):
    result = 0.0
    for i in range(0, len(solution)):
        result += (10**6) ** (i / (len(solution) - 1)) * solution[i] ** 2
    return result


@numba.jit
def rastrigin__(solution=None):
    return np.sum(solution**2 - 10 * np.cos(2 * np.pi * solution) + 10) 


@numba.jit
def ackley__(solution=None):
    return (
        -20 * np.exp(-0.2 * np.sqrt(np.sum(solution**2) / len(solution)))
        - np.exp(np.sum(np.cos(2 * np.pi * solution)) / len(solution))
        + 20
        + np.e
    )

#@numba.jit
#def schwefel__(solution=None):
#    return 418.9829*len(solution) - np.sum(solution * np.sin(np.sqrt(np.abs(solution)))) 


@numba.jit
def rosenbrock__(solution=None):
    result = 0.0
    for i in range(len(solution) - 1):
        result += 100 * (solution[i] ** 2 - solution[i + 1]) ** 2 + (solution[i] - 1) ** 2
    return result
