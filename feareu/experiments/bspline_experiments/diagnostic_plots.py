import feareu
from bayes_opt import BayesianOptimization
import pickle
import numpy as np
import time
import math
from copy import deepcopy



sample_sizes = np.around(np.geomspace(20, 200000, num=5)).astype(int)

alg_list = [
        feareu.BsplineFeaPSO,
        feareu.BsplineFeaDE,
        feareu.BsplineFeaGA
        ]
fea = feareu.BsplineFEA

bspline_eval_class = feareu.SlowBsplineEval

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
            for noise_level in range(6):
                for base_alg in alg_list:
                    x = np.random.random(sample_size)
                    y = benchmark(x)
                    func_width = np.max(y) - np.min(y)
                    noises = np.linspace(0,func_width/10,num=6)
                    y = feareu.make_noisy(y, noises[noise_level])
                    global fitness
                    fitness = bspline_eval_class(x, y)
                    storage = open(f'results/function{func}_{base_alg}_sample{sample_size}_noise{noise_level}', 'rb')
                    data = pickle.load(storage)
                    params = data['params']
                    alg = construct_alg(base_alg, params)
                    storage.close()
                    print("running ", base_alg, " with params: ", params)
                    alg.run()
                    x = np.random.random(sample_size)
                    y = benchmark(x)
                    func_width = np.max(y) - np.min(y)
                    noises = np.linspace(0,func_width/10,num=6)
                    y = feareu.make_noisy(y, noises[noise_level])
                    fitness = bspline_eval_class(x, y)
                    storage = open(f'results/FEA_function{func}_{base_alg}_sample{sample_size}_noise{noise_level}', 'rb')
                    data = pickle.load(storage)
                    params = data['params']
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

                    storage.close()
                    print("running FEA ", base_alg, " with params: ", params)
                    alg.run()
