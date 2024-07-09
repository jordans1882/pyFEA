from types import NoneType
import splipy
import scipy.sparse.linalg as splinalg
import scipy.sparse as sparse
import feareu
from pathlib import Path
import pickle
import numpy as np
import time
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


process_count = 5
thread_count = 12
diagnostics_amount = 1
processes = 10
chunksize=30

sample_size = 20000

fea = feareu.ParallelBsplineFEA
base_alg = feareu.ParallelBsplineFeaPSO

bspline_eval_class = feareu.SlowBsplineEval

x = np.random.random(sample_size)
x.sort()
ytrue = feareu.doppler(x)
y = feareu.make_noisy(ytrue, 0.1)

dim = 200
fact_size = 50
overlap = 25
num_clamps = 0
factors = feareu.linear_factorizer(fact_size=fact_size, overlap=overlap, dim=dim)
feareu.clamp_factor_ends(dim, factors, num_clamps)
print(factors)
domain = (0,1)
pop_size = 300
generations=10
iterations=30
dom = np.zeros((dim,2))
dom[:,1]=1

xseq = np.linspace(0,1,num=y.shape[0])
yseq_true = feareu.doppler(xseq)
plt.figure()
plt.plot(xseq,yseq_true, 'k')
plt.scatter(x,y,s=5)
plt.savefig('results/doppler.png')

func = bspline_eval_class(x,y)
pso_alg = base_alg(
        generations=60,
        pop_size=300,
        function=func,
        domain=dom,
        processes=processes,
        chunksize=chunksize
        )
pso_alg.run()
knots = pso_alg.gbest
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

plt.figure()
plt.plot(xseq,yest_seq,'y')
plt.scatter(x,y,s=5)
plt.scatter(knots,knot_y,color='orange', s=5)
plt.savefig('results/doppler_pso_est.png')

density = gaussian_kde(knots)
xs = np.linspace(0,1,200)
density.covariance_factor = lambda : .1
density._compute_covariance()
plt.figure()
plt.plot(xs,density(xs))
upper = np.max(density(xs))
plt.ylim((0,upper))
plt.savefig('results/doppler_pso_density.png')

plt.figure()
pso_alg.diagnostic_plots()
plt.savefig('results/doppler_diagnostic_pso.png')

print("ran pso")

alg = fea(
        factors,
        function=func,
        iterations=iterations,
        dim=dim,
        base_algo_name=base_alg,
        domain=domain,
        diagnostics_amount=diagnostics_amount,
        process_count=process_count,
        thread_count=thread_count,
        pop_size=pop_size,
        generations=generations,
        processes=processes,
        chunksize=chunksize
        )
alg.run()
knots = alg.context_variable
bsp = splipy.BSplineBasis(3, knots, -1)
xmat = bsp.evaluate(x, 0, True, True)
xmat_seq = bsp.evaluate(xseq,0,True,True)
xt = xmat.transpose()
LHS = xt @ xmat
RHS = xt @ y
theta, info  = sparse.linalg.bicgstab(LHS, RHS)
yest_seq = xmat_seq @ theta
yest = xmat @ theta


plt.figure()
plt.plot(xseq,yest_seq, 'r')
plt.scatter(x,y, s=5)
plt.scatter(knots,knot_y,color='orange', s=5)
plt.savefig('results/doppler_fea_est.png')

density = gaussian_kde(knots)
xs = np.linspace(0,1,200)
density.covariance_factor = lambda : .1
density._compute_covariance()
plt.figure()
plt.plot(xs,density(xs))
upper = np.max(density(xs))
plt.ylim((0,upper))
plt.savefig('results/doppler_fea_density.png')

plt.figure()
alg.diagnostic_plots()
plt.savefig('results/doppler_diagnostic_fea.png')

#diag_plt = alg.diagnostic_plots()
#diag_dir = Path('ending_diagnostics')
#diag_dir.mkdir(parents=True, exist_ok=True)
#filename = diag_dir / f"function{func}_{base_alg.__name__}_sample{sample_size}_noise{noise_level}.png"
#plt.savefig(filename)
#plt.clf()
