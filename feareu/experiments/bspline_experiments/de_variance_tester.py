import numpy as np
import scipy.sparse.linalg as splinalg
from scipy.stats import gaussian_kde
import scipy.sparse as sparse
import splipy
import matplotlib.pyplot as plt
import feareu


processes = 10
chunksize=1

sample_size = 20000

fea = feareu.ParallelBsplineFEA
base_alg = feareu.ParallelBsplineFeaDE

bspline_eval_class = feareu.SlowBsplineEval

x = np.random.random(sample_size)
x.sort()
ytrue = feareu.doppler(x)
y = feareu.make_noisy(ytrue, 0.1)

dim = 200
pop_size = 30
dom = np.zeros((dim,2))
dom[:,1]=1

xseq = np.linspace(0,1,num=y.shape[0])
yseq_true = feareu.doppler(xseq)
plt.figure()
plt.plot(xseq,yseq_true, 'k')
plt.scatter(x,y,s=5)
plt.savefig('results/doppler.png')

func = bspline_eval_class(x,y)
de_alg = base_alg(
        generations=1200,
        pop_size=100,
        function=func,
        domain=dom,
        processes=processes,
        chunksize=chunksize
        )
print(de_alg.pop)
de_alg.run()
print(de_alg.pop)
print("done running")
knots = de_alg.best_solution
bsp = splipy.BSplineBasis(3, knots, -1)
xmat = bsp.evaluate(x, 0, True, True)
xmat_seq = bsp.evaluate(xseq,0,True,True)
xt = xmat.transpose()
LHS = xt @ xmat
RHS = xt @ y
theta, info  = sparse.linalg.bicgstab(LHS, RHS)
yest_seq = xmat_seq @ theta
yest = xmat @ theta
knot_y = np.zeros(knots.shape)
knot_y[:] = np.min(y) - 0.2
print("made vectors to plot")

plt.figure()
plt.plot(xseq,yest_seq,'y')
plt.scatter(x,y,s=5)
plt.scatter(knots,knot_y,color='orange', s=5)
plt.savefig('results/doppler_de_est.png')
print("plotted doppler estimate")

density = gaussian_kde(knots)
xs = np.linspace(0,1,200)
density.covariance_factor = lambda : .1
density._compute_covariance()
plt.figure()
plt.plot(xs,density(xs))
upper = np.max(density(xs))
plt.ylim((0,upper))
plt.savefig('results/doppler_de_density.png')
print("made density plot")

plt.figure()
de_alg.diagnostic_plots()
plt.savefig('results/doppler_diagnostic_de.png')
print("did diagnostic plots")

