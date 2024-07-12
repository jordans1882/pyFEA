import math
from pathlib import Path
from bayes_opt import BayesianOptimization
import numpy as np
import splipy
import feareu
import matplotlib.pyplot as plt
from feareu.base_algos.bspline_specific.bspline_fea_pso import BsplineFeaPSO
from feareu.base_algos.bspline_specific.bspline_fea_de import BsplineFeaDE
from feareu.base_algos.bspline_specific.bspline_fea_ga import BsplineFeaGA
from feareu.base_algos.bspline_specific.known_knot_bspline_fea_ga import KnownKnotBsplineFeaGA
from feareu.base_algos.bspline_specific.known_knot_bspline_fea_pso import KnownKnotBsplineFeaPSO
from feareu.base_algos.bspline_specific.known_knot_bspline_fea_de import KnownKnotBsplineFeaDE
from feareu.experiments.general_fea_experiments.slow_bspline_eval import SlowBsplineEval
from feareu.fea.vector_comparison_bspline_fea import VectorComparisonBsplineFEA
from feareu.experiments.general_fea_experiments.automated_factors import linear_factorizer


def create_single_full_set(number_of_knots = 12,
        number_of_points = 1000,
        max_error = 0.01,
        delta = 0.05,
        diagnostics_amount = 1,
        fea_pop_size = 30,
        base_algo_pop_size = 200,
        base_algo_gens = 20,
        overlap = 2,
        factor_size = 4,
        fea_early_stop = 5,
        base_algo_early_stop = 20
        ):
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
    
    diag_dir = Path('known_knots_ending_diagnostics')
    diag_dir.mkdir(parents=True, exist_ok=True)
    knot_string = ""
    for k in interior_knots:
        back_k = int(k*100000)
        shortened_k = float(back_k)/100000.0
        knot_string += str(shortened_k)
        knot_string += "-"
    knot_string = knot_string[:-1]
    print("knot_string: ", knot_string)

    #FEAs
    factors = linear_factorizer(factor_size, overlap, number_of_knots)
    psoFEA = VectorComparisonBsplineFEA(factors=factors, function=scatter_plot, early_stop = fea_early_stop, true_error=scatter_plot(knots), delta=delta, dim=number_of_knots, base_algo_name=BsplineFeaPSO, domain=(0,1), diagnostics_amount=diagnostics_amount, og_knot_points=interior_knots, generations = 15, pop_size = fea_pop_size)
    psoFEA.run()
    diag_plt = psoFEA.diagnostic_plots()
    filename = diag_dir / f"_fea_pso_knot_vector{knot_string}.png"
    print("knot_string: ", knot_string)
    plt.savefig(filename)
    plt.clf()
    
    gaFEA = VectorComparisonBsplineFEA(factors=factors, function=scatter_plot, early_stop = fea_early_stop, true_error=scatter_plot(knots), delta=delta, dim=number_of_knots, base_algo_name=BsplineFeaGA, domain=(0,1), diagnostics_amount=diagnostics_amount, og_knot_points=interior_knots, generations = 15, pop_size = fea_pop_size)
    gaFEA.run()
    diag_plt = gaFEA.diagnostic_plots()
    filename = diag_dir / f"_fea_ga_knot_vector{knot_string}.png"
    plt.savefig(filename)
    plt.clf()
    
    deFEA = VectorComparisonBsplineFEA(factors=factors, function=scatter_plot, early_stop = fea_early_stop, true_error=scatter_plot(knots), delta=delta, dim=number_of_knots, base_algo_name=BsplineFeaDE, domain=(0,1), diagnostics_amount=diagnostics_amount, og_knot_points=interior_knots, generations = 15, pop_size = fea_pop_size)
    deFEA.run()
    diag_plt = deFEA.diagnostic_plots()
    filename = diag_dir / f"_fea_de_knot_vector{knot_string}.png"
    plt.savefig(filename)
    plt.clf()


    #Base Algos
    fact_dom = np.zeros((number_of_knots,2))
    fact_dom[:,0] = 0
    fact_dom[:,1] = 1
    pso = KnownKnotBsplineFeaPSO(function=scatter_plot, early_stop = base_algo_early_stop, true_error=scatter_plot(knots), delta=delta, domain=fact_dom, og_knot_points=interior_knots, pop_size = base_algo_pop_size)
    pso.run()
    pso.diagnostic_plots()
    filename = diag_dir / f"_pso_knot_vector{knot_string}.png"
    plt.savefig(filename)
    plt.clf()
    
    ga = KnownKnotBsplineFeaGA(function=scatter_plot, early_stop = base_algo_early_stop, true_error=scatter_plot(knots), delta=delta, domain=fact_dom, og_knot_points=interior_knots, pop_size = base_algo_pop_size)
    ga.run()
    ga.diagnostic_plots()
    filename = diag_dir / f"_ga_knot_vector{knot_string}.png"
    plt.savefig(filename)
    plt.clf()
    
    de = KnownKnotBsplineFeaDE(function=scatter_plot, early_stop = base_algo_early_stop, true_error=scatter_plot(knots), delta=delta, domain=fact_dom, og_knot_points=interior_knots, pop_size = base_algo_pop_size)
    de.run()
    de.diagnostic_plots()
    filename = diag_dir / f"_de_knot_vector{knot_string}.png"
    plt.savefig(filename)
    plt.clf()

