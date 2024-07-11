import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.sparse as sparse
import splipy
import math
import scipy.sparse.linalg as splinalg

class SlowBsplineEval:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, knots):
        all_knots = feareu.bspline_clamp(knots, order)
        bsp = splipy.BSplineBasis(3, knots, -1)
        xmat = bsp.evaluate(self.x, 0, True, True)
        xt = xmat.transpose()
        LHS = xt @ xmat
        RHS = xt @ self.y
        theta, info  = sparse.linalg.bicgstab(LHS, RHS)
        #print("theta: ", theta)
        yest = xmat @ theta
        mse = np.sum((self.y - yest)**2)/len(self.y)
        return mse
