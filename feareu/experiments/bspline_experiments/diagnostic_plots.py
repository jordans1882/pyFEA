from types import NoneType
import feareu
from bayes_opt import BayesianOptimization
from pathlib import Path
import pickle
import numpy as np
import time
import math
from copy import deepcopy
import matplotlib.pyplot as plt


sample_sizes = np.around(np.geomspace(20, 200000, num=5)).astype(int)

alg_list = [
        feareu.BsplineFeaPSO,
        feareu.BsplineFeaDE,
        feareu.BsplineFeaGA
        ]
fea = feareu.BsplineFEA

once = True
bspline_eval_class = feareu.SlowBsplineEval

"""
fea_overarching_winner_fit = float('inf')
fea_overarching_winner = None
fea_base_alg = alg_list[0]
func_overarching_winner_fit = float('inf')
func_overarching_winner = None
func_base_alg = alg_list[0]
"""

def construct_alg(base_alg, parameters):
    dim = parameters['dim']
    domain = np.zeros((int(dim),2))
    domain[:,1] = 1
    parameters['pop_size'] = int(parameters['pop_size'])
    parameters['generations'] = int(parameters['generations'])
    return base_alg.from_kwargs(fitness, domain, parameters)


if __name__ == "__main__":
    benchmarks = [feareu.big_spike, feareu.discontinuity, feareu.cliff, feareu.smooth_peak, feareu.second_smooth_peak, feareu.doppler]
    for func, benchmark in enumerate(benchmarks):
        for sample_size in sample_sizes:
            for noise_level in range(5):
                for base_alg in alg_list:
                    x = np.random.random(sample_size)
                    y = benchmark(x)
                    func_width = np.max(y) - np.min(y)
                    noises = np.linspace(func_width/100,func_width/20,num=5)
                    y = feareu.make_noisy(y, noises[noise_level])
                    global fitness
                    fitness = bspline_eval_class(x, y)
                    results_dir = Path('results')
                    filename = results_dir / f"function{func}_{base_alg.__name__}_sample{sample_size}_noise{noise_level}"
                    params = {}
                    try:
                        with open(filename, mode='rb') as storage:
                            data = pickle.load(storage)
                            params = (data['params'])
                            storage.close()
                            try:
                                alg = construct_alg(base_alg, params)
                                print("running ", base_alg, " with params: ", params)
                                result_vector = alg.run()
                                x = np.random.random(sample_size)
                                y = benchmark(x)
                                func_width = np.max(y) - np.min(y)
                                noises = np.linspace(0,func_width/10,num=6)
                                y = feareu.make_noisy(y, noises[noise_level])
                                fitness = bspline_eval_class(x, y)
                                diag_plt = alg.diagnostic_plots()
                                diag_dir = Path('ending_diagnostics')
                                diag_dir.mkdir(parents=True, exist_ok=True)
                                filename = diag_dir / f"function{func}_{base_alg.__name__}_sample{sample_size}_noise{noise_level}.png"
                                plt.savefig(filename)
                                plt.clf()
                                """
                                if fitness(result_vector, np.arange(params['dim'])) < func_overarching_winner_fit:
                                    print("hello dude")
                                    func_overarching_winner_fit = fitness
                                    func_overarching_winner = params
                                    func_base_alg = base_alg
                                """
                            except:
                                #do nothing
                                hi = 0
                    except FileNotFoundError as e:
                        hi = 0
                        #print(f"Error opening file: {e}")
                    except EOFError as e:
                        hi = 0
                        #print(f"EOFError: {e}. Possible issues with the content or format of the file.")
                    results_dir = Path('results')
                    filename = results_dir / f"FEA_function{func}_{base_alg.__name__}_sample{sample_size}_noise{noise_level}"
                    params = {}
                    try:
                        with open(filename, mode='rb') as storage:
                            data = pickle.load(storage)
                            params = (data['params'])
                            storage.close()
                            factors = feareu.linear_factorizer(fact_size=int(params['fact_size']), overlap=int(params['overlap']), dim=int(params['dim']))
                            try:
                                alg = fea(
                                        factors,
                                        fitness,
                                        iterations=int(params['iterations']),
                                        dim=int(params['dim']),
                                        base_algo_name=base_alg,
                                        domain=(0,1),
                                        diagnostics_amount=1,
                                        pop_size=int(params['pop_size']),
                                        generations=int(params['generations']),
                                        phi_p=params['phi_p'],
                                        phi_g=params['phi_g'],
                                        omega=params['omega'],
                                        )
                            except:
                                try:
                                    alg = fea(
                                        factors,
                                        fitness,
                                        iterations=int(params['iterations']),
                                        dim=int(params['dim']),
                                        base_algo_name=base_alg,
                                        domain=(0,1),
                                        diagnostics_amount=1,
                                        pop_size=int(params['pop_size']),
                                        generations=int(params['generations']),
                                        mutation_factor=params['mutation_factor'],
                                        crossover_rate=params['crossover_rate'],
                                        )
                                except:
                                    alg = fea(
                                        factors,
                                        fitness,
                                        iterations=int(params['iterations']),
                                        dim=int(params['dim']),
                                        base_algo_name=base_alg,
                                        domain=(0,1),
                                        diagnostics_amount=1,
                                        pop_size=int(params['pop_size']),
                                        generations=int(params['generations']),
                                        mutation_rate=params['mutation_rate'],
                                        mutation_range=params['mutation_range'],
                                        )

                            print("running FEA ", base_alg, " with params: ", params)
                            try:
                                result_vector = alg.run()
                                x = np.random.random(sample_size)
                                y = benchmark(x)
                                func_width = np.max(y) - np.min(y)
                                noises = np.linspace(0,func_width/10,num=6)
                                y = feareu.make_noisy(y, noises[noise_level])
                                fitness = bspline_eval_class(x, y)
                                diag_plt = alg.diagnostic_plots()
                                diag_dir = Path('ending_diagnostics')
                                diag_dir.mkdir(parents=True, exist_ok=True)
                                filename = diag_dir / f"FEA_function{func}_{base_alg.__name__}_sample{sample_size}_noise{noise_level}.png"
                                plt.savefig(filename)
                                plt.clf()
                                """
                                if fitness(result_vector, np.arange(params['dim'])) < fea_overarching_winner_fit:
                                    print("hello dude")
                                    fea_overarching_winner_fit = fitness
                                    fea_overarching_winner = params
                                    fea_base_alg = base_alg
                                """
                            except:
                                #do nothing
                                hi = 0

                    except FileNotFoundError as e:
                        hi = 0
                        #print(f"Error opening file: {e}")
                    except EOFError as e:
                        hi = 0
                        #print(f"EOFError: {e}. Possible issues with the content or format of the file.")
    
    """
    print("func overarching winner: ", func_overarching_winner)
    fin_func = construct_alg(func_base_alg, func_overarching_winner)
    factors = feareu.linear_factorizer(fact_size=int(fea_overarching_winner['fact_size']), overlap=int(fea_overarching_winner['overlap']), dim=int(fea_overarching_winner['dim']))
    fin_fea = fea(factors,
                fitness,
                iterations=int(fea_overarching_winner['iterations']),
                dim=int(fea_overarching_winner['dim']),
                base_algo_name=fea_base_alg,
                domain=(0,1),
                diagnostics_amount=1,
                pop_size=int(fea_overarching_winner['pop_size']),
                generations=int(fea_overarching_winner['generations']),
                mutation_factor=fea_overarching_winner['mutation_factor'],
                crossover_rate=fea_overarching_winner['crossover_rate'],)
    fin_func.run()
    fin_fea.run()
    diagplot1 = fin_func.diagnostic_plots()
    diagplot1 = fin_fea.diagnostic_plots()
    plt.show()
    """
