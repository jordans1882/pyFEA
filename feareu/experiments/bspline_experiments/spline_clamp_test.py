import numpy as np
import splipy
import matplotlib.pyplot as plt
from feareu import SlowBsplineEval
from feareu import VectorComparisonBsplineFEA, rastrigin__
from feareu import BsplineFeaPSO
import feareu
import random
from feareu import linear_factorizer
from feareu import BsplineFEA
from pathlib import Path

random.seed(42)
np.random.seed(42)
number_of_knots = 12
number_of_points = 1000
order = 3
max_error = 0.01
thetas = np.random.normal(0.0, 1.0, number_of_knots+order) # coefficients for the curve
#print(thetas)
interior_knots = np.sort(np.random.uniform(0.0, 1.0, number_of_knots)) # knot locations
knots = feareu.bspline_clamp(interior_knots, order)
#print(knots)
bspline_basis = splipy.BSplineBasis(order, knots, -1) # bspline basis

x = np.random.uniform(0.0, 1.0, number_of_points) # x locations for data
xseq = np.linspace(0.01, 0.99, 1000)
xmat = bspline_basis(x)
#print(np.sum(xmat, axis=0))
#print(xmat)
xseq_mat = bspline_basis(xseq)
#print(np.sum(xseq_mat, axis=0))
epsilon = np.random.normal(0.0, 0.1, number_of_points)
true_y = xmat @ thetas
y = true_y + epsilon
true_yseq = xseq_mat @ thetas

results_dir = Path('results')
results_dir.mkdir(parents=True, exist_ok=True)

plt.scatter(x, y)
plt.ylim(-2.0, 2.0)
plt.plot(xseq, true_yseq, color = 'red')
plt.plot(knots, [-2] * (number_of_knots+6), '|', markersize=20)
filename = results_dir / "bspline_test.png"
plt.savefig(filename)

# MSE at this solution.
scatter_plot = SlowBsplineEval(x, y)
print(scatter_plot(knots))
