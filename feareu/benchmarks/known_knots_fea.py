import numpy as np
import splipy
import matplotlib.pyplot as plt
from feareu.experiments.general_fea_experiments.slow_bspline_eval import SlowBsplineEval
from feareu.fea.vector_comparison_bspline_fea import VectorComparisonBsplineFEA
from feareu.experiments.general_fea_experiments.automated_factors import linear_factorizer

# random.seed(42)
# np.random.seed(42)

class KnownKnotsFea():
    def __init__(self, number_of_knots, number_of_points, max_error, delta, base_algo, diagnostics_amount, generations, pop_size):
        self.number_of_knots = number_of_knots
        self.number_of_points = number_of_points
        self.max_error = max_error
        self.delta = delta
        self.base_algo = base_algo
        self.diagnostics_amount = diagnostics_amount
        self.generations = generations
        self.pop_size = pop_size
    def run(self):
        thetas = np.random.normal(0.0, 1.0, self.number_of_knots+3) # coefficients for the curve
        interior_knots = np.sort(np.random.uniform(0.0, 1.0, self.number_of_knots)) # knot locations
        knots = np.concatenate(([0.0, 0.0, 0.0],  interior_knots,  [1.0, 1.0, 1.0]))
        bspline_basis = splipy.BSplineBasis(3, knots, -1) # bspline basis
        x = np.random.uniform(0.0, 1.0, self.number_of_points) # x locations for data
        xseq = np.linspace(0.01, 0.99, 1000)
        xmat = bspline_basis(x)
        xseq_mat = bspline_basis(xseq)
        epsilon = np.random.normal(0.0, 0.1, self.number_of_points)
        true_y = xmat @ thetas
        y = true_y + epsilon
        true_yseq = xseq_mat @ thetas
        plt.scatter(x, y)
        plt.ylim(-2.0, 2.0)
        plt.plot(xseq, true_yseq, color = 'red')
        plt.plot(knots, [-2] * (self.number_of_knots+6), '|', markersize=20)
        plt.show()
        # MSE at this solution.
        scatter_plot = SlowBsplineEval(x, y)
        print(scatter_plot(knots))
        fct = linear_factorizer(3, 1, self.number_of_knots)
        testing = VectorComparisonBsplineFEA(factors=fct, function = scatter_plot, true_error=scatter_plot(knots), delta = self.delta, og_knot_points = interior_knots, dim = self.number_of_knots, base_algo_name=self.base_algo, domain=(0, 1), diagnostics_amount = self.diagnostics_amount, generations= self.generations, pop_size=self.pop_size)
        testing.run()
        testing.diagnostic_plots()
        plt.show()
    
    
"""
number_of_knots = 12
number_of_points = 1000
max_error = 0.01
thetas = np.random.normal(0.0, 1.0, number_of_knots+3) # coefficients for the curve
#print(thetas)
interior_knots = np.sort(np.random.uniform(0.0, 1.0, number_of_knots)) # knot locations
knots = np.concatenate(([0.0, 0.0, 0.0],  interior_knots,  [1.0, 1.0, 1.0]))
#print(knots)
bspline_basis = splipy.BSplineBasis(3, knots, -1) # bspline basis

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

plt.scatter(x, y)
plt.ylim(-2.0, 2.0)
plt.plot(xseq, true_yseq, color = 'red')
plt.plot(knots, [-2] * (number_of_knots+6), '|', markersize=20)
plt.show()

# MSE at this solution.
scatter_plot = SlowBsplineEval(x, y)
print(scatter_plot(knots))
fct = linear_factorizer(3, 1, number_of_knots)
testing = VectorComparisonBsplineFEA(factors=fct, function = scatter_plot, true_error=scatter_plot(knots), delta = 0.01, og_knot_points = interior_knots, dim = number_of_knots, base_algo_name=BsplineFeaPSO, domain=(0, 1), diagnostics_amount = 1, generations= 5, pop_size=15)
testing.run()
diag_plt = testing.diagnostic_plots()
plt.show()
"""