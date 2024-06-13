from operator import sub
import matplotlib.pyplot as plt
import numpy as np
from feareu.function import Function

"""
Factored Evolutionary Architecture

Parameters:
    factors: list of lists, contains the dimensions that each factor of the architecture optimizes over.
    function: the objective function that the FEA minimizes.
    iterations: the number of times that the FEA runs.
    dim: the number of dimensions our function optimizes over.
    base_algo_name: the base algorithm class that we optomize over. Should be a subclass of FeaBaseAlgo.
    domain: the domain of our function over every dimension as a numpy array of shape (dim, 2).
    The first column contains lower bounds, the second contains upper bounds.
    update_worst: determines whether we perform the second half of the share algorithm.
    **kwargs: parameters for the base algorithm.
"""
class FEA:
    def __init__(self, factors, function, iterations, dim, base_algo_name, domain, update_worst = True, **kwargs):
        self.factors = factors
        self.variable_map = self._construct_factor_variable_mapping()
        self.function = function
        self.iterations = iterations
        self.base_algo = base_algo_name
        self.dim = dim
        self.domain = domain
        self.context_variable = None
        self.base_algo_args = kwargs
        self.niterations = 0
        self.update_worst = update_worst
        self.convergences = []
        self.solution_variance_per_dim = []
        self.solution_variance_in_total = []

    def _construct_factor_variable_mapping(self):
        variable_map = [[] for k in range(self.dim)]
        for i in range(len(self.factors)):
            for j in self.factors[i]:
                variable_map[j].append(i)
        return variable_map

    def run(self):
        self.context_variable = self.init_full_global()
        subpopulations = self.initialize_subpops()
        for i in range(self.iterations):
            self.niterations += 1
            for subpop in subpopulations:
                subpop.base_reset()
                subpop.run()
            self.compete(subpopulations)
            self.share(subpopulations)
            self.convergences.append(self.function(self.context_variable))
        return self.function(self.context_variable)

    def compete(self, subpopulations):
        cont_var = self.context_variable
        best_fit = self.function(self.context_variable)
        rand_var_permutation = np.random.permutation(self.dim)
        for i in rand_var_permutation:
            overlapping_factors = self.variable_map[i]
            best_val = cont_var[i]
            best_fit = self.function(cont_var)
            rand_pop_permutation = np.random.permutation(len(overlapping_factors))
            solution_to_measure_variance = []
            for j in rand_pop_permutation:
                s_j = overlapping_factors[j]
                index = np.where(self.factors[s_j] == i)[0][0]
                cont_var[i] = subpopulations[s_j].get_solution_at_index(index)
                solution_to_measure_variance.append(subpopulations[s_j].get_solution_at_index(index))
                current_fit = self.function(cont_var)
                if current_fit < best_fit:
                    best_val = subpopulations[s_j].get_solution_at_index(index)
                    best_fit = current_fit
            cont_var[i] = best_val
            self.solution_variance_per_dim.append(np.var(solution_to_measure_variance))
        self.context_variable = cont_var
        self.solution_variance_in_total.append(np.average(self.solution_variance_per_dim))
        self.solution_variance_per_dim = []

    def share(self, subpopulations):
        for i in range(len(subpopulations)):
            subpopulations[i].func.context = np.copy(self.context_variable)
            subpopulations[i].update_worst(self.context_variable[self.factors[i]])
            subpopulations[i].update_bests()

    def init_full_global(self):
        lbound = self.domain[:, 0]
        area = self.domain[:, 1] - self.domain[:, 0]
        return lbound + area * np.random.random(size=(area.shape[0]))

    def initialize_subpops(self):
        ret = []
        for subpop in self.factors:
            fun = Function(context=self.context_variable, function=self.function, factor=subpop)
            ret.append(self.base_algo.from_kwargs(fun, self.domain[subpop, :], self.base_algo_args))
        return ret

    def diagnostic_plots(self):
        plt.subplot(1, 2, 1)
        ret = plt.plot(range(0, self.niterations), self.convergences)
        plt.title("Convergence")

        plt.subplot(1, 2, 2)
        plt.plot(range(0, self.niterations), self.solution_variance_in_total)
        plt.title("Solution Variance")
        return ret
