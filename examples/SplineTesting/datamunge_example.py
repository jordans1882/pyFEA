import math

import matplotlib.pyplot as plt
from cmakeswig.datamunge import pyDatamunge as dm


def doppler(x):
    return math.sin(20 / (x + 0.15))


n = 100000
k = 700
x = dm.gsl_runif(0.0, 1.0, n)
knots = dm.gsl_seq(0.0, 1.0, k)
xseq = dm.gsl_seq(0.0, 1.0, 700)
xmat = dm.gsl_bspline_eval(x, knots, 2)
xmat_seq = dm.gsl_bspline_eval(xseq, knots, 2)
y = dm.Vector([doppler(xi) for xi in x.to_std_vec_cpy()])
slm = dm.GSLSLM(xmat, y)
yseq = xmat_seq.mult(slm.get_theta())

plt.plot(xseq.to_std_vec_cpy(), yseq.to_std_vec_cpy())
plt.show()
