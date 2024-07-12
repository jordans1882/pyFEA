import numpy as np
import feareu
import math
import splipy

from bayes_opt import BayesianOptimization
from feareu.base_algos.bspline_specific.bspline_fea_ga import BsplineFeaGA
from feareu.base_algos.bspline_specific.bspline_fea_pso import BsplineFeaPSO
from feareu.base_algos.bspline_specific.bspline_fea_de import BsplineFeaDE
from feareu.base_algos.bspline_specific.known_knot_bspline_fea_de import KnownKnotBsplineFeaDE
from feareu.base_algos.bspline_specific.known_knot_bspline_fea_ga import KnownKnotBsplineFeaGA
from feareu.base_algos.bspline_specific.known_knot_bspline_fea_pso import KnownKnotBsplineFeaPSO
from feareu.experiments.general_fea_experiments.automated_factors import linear_factorizer
from feareu.experiments.general_fea_experiments.slow_bspline_eval import SlowBsplineEval
from feareu.fea.vector_comparison_bspline_fea import VectorComparisonBsplineFEA


pbounds = {
           "pop_size":(10,35), 
           "fact_size": (1,5), 
           "overlap": (0,3),
           #"num_covers":(1,5),
           "num_clamps":(0,5),
           "dim":(8,12)
           }
base_bounds = {
        "generations":(10,20),
        "pop_size":(10,35),
        "dim":(8,12)
        }

pso_bounds = {
       "phi_p":(0.8,3),
       "phi_g":(0.8,3),
       "omega":(0.01,0.99)
        }

ga_bounds = {
       "mutation_rate":(0.01,0.99),
       "mutation_range":(0.1,2),
        }

de_bounds = {
       "mutation_factor":(0.1,1), 
       "crossover_rate":(0.1,1),
        }

def create_scatter_data(number_of_knots, number_of_points):
    thetas = np.random.normal(0.0, 1.0, number_of_knots+3) # coefficients for the curve
    interior_knots = np.sort(np.random.uniform(0.0, 1.0, number_of_knots)) # knot locations
    knots = np.concatenate(([0.0, 0.0, 0.0],  interior_knots,  [1.0, 1.0, 1.0]))
    bspline_basis = splipy.BSplineBasis(3, knots, -1) # bspline basis
    x = np.random.uniform(0.0, 1.0, number_of_points) # x locations for data
    xmat = bspline_basis(x)
    epsilon = np.random.normal(0.0, 0.1, number_of_points)
    true_y = xmat @ thetas
    y = true_y + epsilon

    scatter_plot = SlowBsplineEval(x, y)
    return [scatter_plot, knots]

def bayes_algo_input_base(base_alg,
            scatter_data,
            number_of_knots=12, 
            delta=0.01,
            generations=20,
            pop_size=20,
            base_algo_early_stop = 1000,
            mutation_factor=0.5,
            crossover_rate=0.9,
            mutation_rate=0.05,
            mutation_range=0.5,
            phi_p=math.sqrt(2),
            phi_g=math.sqrt(2),
            omega=1/math.sqrt(2),
            dim = 10
            ):
    scatter_plot = scatter_data[0]
    knots = scatter_data[1]
    interior_knots = knots[3:-3]
    dim = int(number_of_knots)
    generations = int(generations)
    pop_size = int(pop_size)

    domain = np.zeros((dim,2))
    domain[:,0] = 0
    domain[:,1] = 1
    if base_alg == "pso":
        objective = KnownKnotBsplineFeaPSO(function=scatter_plot, early_stop = base_algo_early_stop, true_error=scatter_plot(knots), delta=delta, domain=domain, og_knot_points=interior_knots, pop_size = pop_size)

    elif base_alg == "de":
        objective = KnownKnotBsplineFeaDE(function=scatter_plot, early_stop = base_algo_early_stop, true_error=scatter_plot(knots), delta=delta, domain=domain, og_knot_points=interior_knots, pop_size = pop_size)
    
    elif base_alg == "ga":
        objective = KnownKnotBsplineFeaGA(function=scatter_plot, early_stop = base_algo_early_stop, true_error=scatter_plot(knots), delta=delta, domain=domain, og_knot_points=interior_knots, pop_size = pop_size)
    
    ret = -objective.run()
    print("ret: ", ret)
    return ret
    
def bayes_algo_input_fea(
            base_alg,
            scatter_data,
            number_of_knots=12, 
            delta=0.01,
            factor_size=3,
            overlap=1,
            num_covers=2,
            num_clamps=0,
            generations=20,
            iterations=100,
            pop_size=20,
            mutation_factor=0.5,
            crossover_rate=0.9,
            mutation_rate=0.05,
            mutation_range=0.5,
            phi_p=math.sqrt(2),
            phi_g=math.sqrt(2),
            omega=1/math.sqrt(2),
            dim = 10
            ):
    scatter_plot = scatter_data[0]
    knots = scatter_data[1]
    interior_knots = knots[3:-3]
    dim = int(number_of_knots)
    fact_size = int(factor_size)
    overlap = int(overlap)
    generations = int(generations)
    num_covers = int(num_covers)
    num_clamps=int(num_clamps)
    domain = (0,1)
    factors = linear_factorizer(fact_size=fact_size, overlap=overlap, dim=dim, num_covers=num_covers)
    if factors is None or fact_size > dim:
        return -999999999
    if num_clamps > 0:
        feareu.clamp_factor_ends(dim, factors, num_clamps)
    if base_alg == "pso":
        objective = VectorComparisonBsplineFEA(factors=factors, function=scatter_plot, early_stop = 50, true_error=scatter_plot(knots), delta=delta, dim=number_of_knots, base_algo_name=BsplineFeaPSO, domain=domain, diagnostics_amount=5, og_knot_points=interior_knots, generations = 5, pop_size = pop_size)

    elif base_alg == "de":
        objective = VectorComparisonBsplineFEA(factors=factors, function=scatter_plot, early_stop = 50, true_error=scatter_plot(knots), delta=delta, dim=number_of_knots, base_algo_name=BsplineFeaDE, domain=domain, diagnostics_amount=5, og_knot_points=interior_knots, generations = 5, pop_size = pop_size)

    elif base_alg == "ga":
        objective = VectorComparisonBsplineFEA(factors=factors, function=scatter_plot, early_stop = 50, true_error=scatter_plot(knots), delta=delta, dim=number_of_knots, base_algo_name=BsplineFeaGA, domain=domain, diagnostics_amount=5, og_knot_points=interior_knots, generations = 5, pop_size = pop_size)
    ret = -objective.run()
    return ret
    
def base_optimizer(bounds, base_alg, scatter_data, init_points, n_iter):
    optimizer = BayesianOptimization(bayes_algo_input_base(base_alg=base_alg, scatter_data=scatter_data), bounds)
    optimizer.maximize(init_points, n_iter)
    return optimizer.max
def fea_optimizer(bounds, base_alg, scatter_data, init_points, n_iter):
    optimizer = BayesianOptimization(bayes_algo_input_fea(base_alg=base_alg, scatter_data=scatter_data), bounds)
    optimizer.maximize(init_points, n_iter)
    return optimizer.max

scd = create_scatter_data(12, 1000)
print(base_optimizer(bounds=pso_bounds, base_alg="pso", scatter_data=scd, init_points=1, n_iter=5))

"""create_single_full_set()
create_single_full_set()"""

