import feareu
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import splipy
import scipy.sparse.linalg as splinalg
import scipy.sparse as sparse
from scipy.stats import gaussian_kde

k = 500
n = 20000

bsp = splipy.BSplineBasis(3, knots, -1)
xmat = bsp.evaluate(x, 0, True, True)
xseq = np.linspace(0,1,len(y))
xmat_seq = bsp.evaluate(xseq,0,True,True)
xt = xmat.transpose()
LHS = xt @ xmat
RHS = xt @ y
theta, info  = sparse.linalg.bicgstab(LHS, RHS)
yest_seq = xmat_seq @ theta
yest = xmat @ theta
knot_y = np.zeros(knots.shape)
knot_y[:] = np.min(y) - 0.2

results_dir = Path('results')
results_dir.mkdir(parents=True, exist_ok=True)
filename = results_dir / "spline_clamp"

plt.figure()
plt.plot(xseq,yest_seq,'y')
plt.scatter(x,y,s=5)
plt.scatter(knots,knot_y,color='orange', s=5)
plt.savefig(filename)
