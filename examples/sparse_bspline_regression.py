import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.sparse as sparse
import splipy
# import xalglib # NOTE: unused... consider using
import math
import scipy.sparse.linalg as splinalg

# Doppler function
def mdoppler(x: float) -> float:
    return math.sin(20/(x+0.15))

# Parameters
n = 100000
k = 20

# Create testing data
x = np.random.rand(n)
ytrue = [mdoppler(xval) for xval in x]
y = ytrue + np.random.normal(0.0, 0.2, n)

# plt.scatter(x, y)
# plt.show()

knots = np.concatenate(
    (np.array([0, 0]),
    np.linspace(0.0, 1.0, k+1),
    np.array([1, 1]))
    )


bsp = splipy.BSplineBasis(3, knots, -1)
xmat = bsp.evaluate(x, 0, True, True)

xt = xmat.transpose()
theta = (splinalg.inv(xt @ xmat) @ xt) @ y

nx = 1000
xseq = np.linspace(0.0, 1.0, nx)
xseqmat = bsp.evaluate(xseq, 0, True, True)

yest = xseqmat @ theta

plt.plot(xseq, yest)
plt.show()
