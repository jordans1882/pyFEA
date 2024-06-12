from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

# from feareu. import PSO
from feareu.function import Function


class FEA:
    def __init__(self, factors, function, iterations, dim, base_algo_name, domain, **kwargs):
        self.factors = factors
        self.function = function
        self.iterations = iterations
        self.base_algo_name = base_algo_name
        self.dim = dim
        self.domain = domain
        self.context_variable = None
        self.base_algo_args = kwargs
        self.niterations = 0
        self.variable_map = self._construct_factor_variable_mapping()

    # UNIT TEST THIS
    def _construct_factor_variable_mapping(self):
        variable_map = [[] for k in range(self.dim)]
        for i in range(len(self.factors)):
            for j in self.factors[i]:
                variable_map[j].append(i)
        return variable_map

    def run(self):
        self.context_variable = self.init_full_global()
        print("initial context variable: ", self.context_variable)
        subpopulations = self.initialize_subpops()
        # print("initial subpopulations: ", subpopulations)
        convergence = []
        counter = 0
        for i in range(self.iterations):
            self.niterations += 1
            for subpop in subpopulations:
                # FIX DOMAIN
                subpop.velocities = subpop.init_velocities()
                subpop.reset_fitness()
                subpop.run()
            self.compete(subpopulations)
            self.share(subpopulations)
            convergence.append(self.function(self.context_variable))
        print("convergence array: ", convergence)
        return self.function(self.context_variable)

    def compete(self, subpopulations):
        cont_var = deepcopy(self.context_variable)
        best_fit = deepcopy(self.function(self.context_variable))
        rand_var_permutation = np.random.permutation(self.dim)
        for i in rand_var_permutation:
            overlapping_factors = self.variable_map[i]
            best_val = deepcopy(cont_var[i])
            best_fit = self.function(cont_var)
            rand_pop_permutation = np.random.permutation(len(overlapping_factors))
            for j in rand_pop_permutation:
                s_j = overlapping_factors[j]
                index = np.where(self.factors[s_j] == i)[0][0]
                cont_var[i] = deepcopy(subpopulations[s_j].gbest[index])
                current_fit = deepcopy(self.function(cont_var))
                if current_fit < best_fit:
                    best_val = deepcopy(subpopulations[s_j].gbest[index])
                    best_fit = deepcopy(current_fit)
            cont_var[i] = deepcopy(best_val)
        self.context_variable = deepcopy(cont_var)
        for subpop in subpopulations:
            subpop.func.context = deepcopy(self.context_variable)

    def share(self, subpopulations):
        for i in range(len(subpopulations)):
            worst = deepcopy(subpopulations[i].worst)
            subpopulations[i].pop[worst, :] = deepcopy(self.context_variable[self.factors[i]])
            subpopulations[i].update_bests()

    def init_full_global(self):
        lbound = self.domain[:, 0]
        area = self.domain[:, 1] - self.domain[:, 0]
        return lbound + area * np.random.random(size=(area.shape[0]))

    def initialize_subpops(self):
        ret = []
        for subpop in self.factors:
            fun = Function(context=self.context_variable, function=self.function, factor=subpop)
            alg = globals()[self.base_algo_name].from_kwargs(
                fun, self.domain[subpop, :], self.base_algo_args
            )
            ret.append(alg)
        return ret

    def diagnostic_plots(self):
        plt.subplot(1, 3, 1)
        ret = plt.plot(range(0, self.niterations), self.function(self.context_variable))
        plt.title("Convergence")

        plt.subplot(1, 3, 2)
        plt.plot(range(0, self.niterations), self.gbest_evals)
        plt.title("Global Bests")

        plt.subplot(1, 3, 3)
        plt.plot(range(0, self.niterations), self.average_velocities)
        plt.title("Average Velocities")
        plt.tight_layout()
        return ret
