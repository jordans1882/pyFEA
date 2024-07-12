# import xalglib # NOTE: unused... consider using
import math

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import splipy


# Doppler function
def mdoppler(x: float) -> float:
    return math.sin(20 / (x + 0.15))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Parameters
n = 10000
k = 100
# Create testing data
x = np.random.rand(n)
ytrue = [mdoppler(xval) for xval in x]
y = ytrue + np.random.normal(0.0, 0.2, n)
knots = np.concatenate((np.array([0, 0]), np.linspace(0.0, 1.0, k + 1), np.array([1, 1])))
bsp = splipy.BSplineBasis(3, knots, -1)
xmat = bsp.evaluate(x, 0, True, True)
xt = xmat.transpose()
theta = (splinalg.inv(xt @ xmat) @ xt) @ y
nx = 1000
xseq = np.linspace(0.0, 1.0, nx)
xseqmat = bsp.evaluate(xseq, 0, True, True)
yest = xseqmat @ theta
ax1.scatter(x, y, alpha=0.5)
ax1.set_ylim(-2, 2)
ax1.set_title("B-Spline Regression - k=100, n = 10K")
ax1.plot(xseq, yest, color="pink")
ax1.vlines(knots, ymin=-2, ymax=-1.9, color="orange")
# plt.show()


# Parameters
n = 10000
k = 500
# Create testing data
x = np.random.rand(n)
ytrue = [mdoppler(xval) for xval in x]
y = ytrue + np.random.normal(0.0, 0.2, n)
knots = np.concatenate((np.array([0, 0]), np.linspace(0.0, 1.0, k + 1), np.array([1, 1])))
bsp = splipy.BSplineBasis(3, knots, -1)
xmat = bsp.evaluate(x, 0, True, True)
xt = xmat.transpose()
theta = (splinalg.inv(xt @ xmat) @ xt) @ y
nx = 1000
xseq = np.linspace(0.0, 1.0, nx)
xseqmat = bsp.evaluate(xseq, 0, True, True)
yest = xseqmat @ theta
ax2.scatter(x, y, alpha=0.5)
ax2.set_ylim(-2, 2)
ax2.set_title("B-Spline Regression - k=100, n = 10K")
ax2.plot(xseq, yest, color="pink")
ax2.vlines(knots, ymin=-2, ymax=-1.9, color="orange")
fig.savefig("fixed-width-bspline-doppler-fits.png")
# fig.savefig("fixed-width-bspline-doppler-fits.png", figsize=(6, 12))
