import math
import pickle
import time
from copy import deepcopy
from pathlib import Path
from types import NoneType

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import splipy
from scipy.stats import gaussian_kde

import pyfea

process_count = 8
thread_count = 16
diagnostics_amount = 1
processes = 10
chunksize = 30

sample_size = 20000

fea = pyfea.ParallelBsplineFEA
base_alg = pyfea.ParallelBsplineFeaGA

bspline_eval_class = pyfea.SlowBsplineEval

x = np.random.random(sample_size)
x.sort()
ytrue = pyfea.doppler(x)
y = pyfea.make_noisy(ytrue, 0.1)

dim = 200
fact_size = 80
overlap = 40
num_clamps = 0
factors = pyfea.linear_factorizer(fact_size=fact_size, overlap=overlap, dim=dim)
pyfea.clamp_factor_ends(dim, factors, num_clamps)
print(factors)
domain = (0, 1)
pop_size = 30
generations = 7
iterations = 10
dom = np.zeros((dim, 2))
dom[:, 1] = 1

xseq = np.linspace(0, 1, num=y.shape[0])
yseq_true = pyfea.doppler(xseq)
plt.figure()
plt.plot(xseq, yseq_true, "k")
plt.scatter(x, y, s=5)
plt.savefig("results/doppler.png")

func = bspline_eval_class(x, y)
pso_alg = base_alg(
    generations=1000,
    pop_size=300,
    function=func,
    domain=dom,
    processes=processes,
    chunksize=chunksize,
)
pso_alg.run()
knots = pso_alg.best_position
knots = pyfea.bspline_clamp(knots, 3)
bsp = splipy.BSplineBasis(3, knots, -1)
xmat = bsp.evaluate(x, 0, True, True)
xmat_seq = bsp.evaluate(xseq, 0, True, True)
xt = xmat.transpose()
LHS = xt @ xmat
RHS = xt @ y
theta, info = sparse.linalg.bicgstab(LHS, RHS)
yest_seq = xmat_seq @ theta
yest = xmat @ theta
knot_y = np.zeros(knots.shape)
knot_y[:] = np.min(y) - 0.2

plt.figure()
plt.plot(xseq, yest_seq, "y")
plt.scatter(x, y, s=5)
plt.scatter(knots, knot_y, color="orange", s=5)
plt.savefig("results/doppler_ga_est.png")

density = gaussian_kde(knots)
xs = np.linspace(0, 1, 200)
density.covariance_factor = lambda: 0.1
density._compute_covariance()
plt.figure()
plt.plot(xs, density(xs))
upper = np.max(density(xs))
plt.ylim((0, upper))
plt.savefig("results/doppler_ga_density.png")

plt.figure()
pso_alg.diagnostic_plots()
plt.savefig("results/doppler_diagnostic_ga.png")

print("ran de")

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
    chunksize=chunksize,
)
alg.run()
knots = alg.context_variable
knots = pyfea.bspline_clamp(knots, 3)
bsp = splipy.BSplineBasis(3, knots, -1)
xmat = bsp.evaluate(x, 0, True, True)
xmat_seq = bsp.evaluate(xseq, 0, True, True)
xt = xmat.transpose()
LHS = xt @ xmat
RHS = xt @ y
theta, info = sparse.linalg.bicgstab(LHS, RHS)
yest_seq = xmat_seq @ theta
yest = xmat @ theta


plt.figure()
plt.plot(xseq, yest_seq, "r")
plt.scatter(x, y, s=5)
plt.scatter(knots, knot_y, color="orange", s=5)
plt.savefig("results/doppler_fea_est.png")

density = gaussian_kde(knots)
xs = np.linspace(0, 1, 200)
density.covariance_factor = lambda: 0.1
density._compute_covariance()
plt.figure()
plt.plot(xs, density(xs))
upper = np.max(density(xs))
plt.ylim((0, upper))
plt.savefig("results/doppler_fea_density.png")

plt.figure()
alg.diagnostic_plots()
plt.savefig("results/doppler_diagnostic_fea.png")

# diag_plt = alg.diagnostic_plots()
# diag_dir = Path('ending_diagnostics')
# diag_dir.mkdir(parents=True, exist_ok=True)
# filename = diag_dir / f"function{func}_{base_alg.__name__}_sample{sample_size}_noise{noise_level}.png"
# plt.savefig(filename)
# plt.clf()
