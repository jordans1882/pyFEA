#from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

# from feareu. import PSO
from feareu.function import Function
from feareu import PSO


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
        self.convergences = []
        self.gbest_variance_per_dim = []
        self.gbest_variance_in_total = []
        #self.pop_variances
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
            gbests_to_measure_variance = []
            for j in rand_pop_permutation:
                s_j = overlapping_factors[j]
                index = np.where(self.factors[s_j] == i)[0][0]
                cont_var[i] = subpopulations[s_j].gbest[index]
                gbests_to_measure_variance.append(subpopulations[s_j].gbest[index])
                current_fit = self.function(cont_var)
                if current_fit < best_fit:
                    best_val = subpopulations[s_j].gbest[index]
                    best_fit = current_fit
            cont_var[i] = best_val
            self.gbest_variance_per_dim.append(np.var(gbests_to_measure_variance))
        self.context_variable = cont_var
        for subpop in subpopulations:
            subpop.func.context = np.copy(self.context_variable)
        self.gbest_variance_in_total.append(np.average(self.gbest_variance_per_dim))
        self.gbest_variance_per_dim = []

    def share(self, subpopulations):
        for i in range(len(subpopulations)):
            worst = (subpopulations[i].worst)
            subpopulations[i].pop[worst, :] = (self.context_variable[self.factors[i]])
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
        plt.subplot(1, 2, 1)
        ret = plt.plot(range(0, self.niterations), self.convergences)
        plt.title("Convergence")

        plt.subplot(1, 2, 2)
        plt.plot(range(0, self.niterations), self.gbest_variance_in_total)
        plt.title("Gbest Variance")
        
        """plt.subplot(1, 3, 4)
        plt.plot(range(0, self.niterations), XXXXX)
        plt.title("Pop Variances")
        plt.tight_layout()"""
        
        
        return ret
        


        """plt.subplot(1, 3, 2)
        plt.plot(range(0, self.niterations), self.gbest_evals)
        plt.title("Global Bests")

        plt.subplot(1, 3, 3)
        plt.plot(range(0, self.niterations), self.average_velocities)
        plt.title("Average Velocities")
        plt.tight_layout()
        return ret"""
