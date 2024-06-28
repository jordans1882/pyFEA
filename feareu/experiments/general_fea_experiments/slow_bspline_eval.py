import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.sparse as sparse
import splipy
import math
import scipy.sparse.linalg as splinalg

class SlowBsplineEval:
    def __init__(x, y):
        self.x = x
        self.y = y

    def __call__(knots):
        bsp = splipy.BsplineBasis(3, knots, -1)
        xmat = bsp.evaluate(self.x, 0, True, True)
        xt = xmat.transpose()
        theta = (splinalg.inv(xt @ xmat) @ xt) @ self.y

        yest = xmat @ theta
        mse = np.sum((self.y - yest)**2)/len(self.y)
        return mse
