# NOTE: use this later
# from cmakeswig.datamunge import pyDatamunge as dm

import math
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import seaborn as sns
import splipy

from feareu.base_algos.bspline_pso import BSplinePSO


# Doppler function
def mdoppler(x: float) -> float:
    return math.sin(20 / (x + 0.15))


# Parameters
n = 50000
k = 400

# Create testing data
x = np.random.rand(n)
ytrue = [mdoppler(xval) for xval in x]
y = ytrue + np.random.normal(0.0, 0.2, n)

plt.scatter(x, y)
plt.show()


knots = np.concatenate((np.array([0, 0]), np.linspace(0.0, 1.0, k + 1), np.array([1, 1])))


def calc_mse(knot_seq, x, y):
    n = len(y)
    bsp = splipy.BSplineBasis(3, knot_seq, -1)
    xmat = bsp.evaluate(x, 0, True, True)
    xt = xmat.transpose()
    try:
        theta = (splinalg.inv(xt @ xmat) @ xt) @ y
    except:
        return 99999999999
    yhat = xmat @ theta
    mse = np.sum(np.square(y - yhat)) / n
    return mse


fitness = partial(calc_mse, x=x, y=y)
fitness(knots)


dom = np.zeros((k, 2))
dom[:, 1] = 1.0


pso = BSplinePSO(fitness, dom, 1000, 100)

soln = pso.run()
soln

bsp = splipy.BSplineBasis(3, soln, -1)
xmat = bsp.evaluate(x, 0, True, True)
xt = xmat.transpose()
theta = (splinalg.inv(xt @ xmat) @ xt) @ y
nx = 1000
xseq = np.linspace(0.0, 1.0, nx)
xseqmat = bsp.evaluate(xseq, 0, True, True)
yest = xseqmat @ theta

fig, ax = plt.subplots(figsize=(8, 4))
plt.plot(xseq, yest)
ax.set_ylim(-2, 2)
ax.plot(soln, [-1.95] * len(soln), "|", color="k")
plt.show()

# nx = 1000
# xseq = np.linspace(0.0, 1.0, nx)
# xseqmat = bsp.evaluate(xseq, 0, True, True)
# yest = xseqmat @ theta
#
# plt.plot(xseq, yest)
# plt.show()
