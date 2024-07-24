from copy import deepcopy
from operator import sub

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from pyfea.fea.function import Function


class FEA:
    """Factored Evolutionary Architecture, implemented based on the 2017 paper by Strasser et al."""

    def __init__(self, factors, function, iterations, dim, base_algo, domain, fitness_terminate=False, **kwargs=None):
        """
        @param factors: list of lists, contains the dimensions that each factor of the architecture optimizes over.
        @param function: the objective function that the FEA minimizes.
        @param iterations: the number of times that the FEA runs.
        @param dim: the number of dimensions our function optimizes over.
        @param base_algo: the class for the base algo. Must adhere to FeaBaseAlgo interface.
        @param domain: the domain of our function over every dimension as a numpy array of shape (dim, 2).
        @param **kwargs: parameters for the base algorithm.
        """
        self.factors = factors
        self.dim = dim
        self.variable_map = self._construct_factor_variable_mapping()
        self.function = function
        self.iterations = iterations
        self.base_algo = base_algo
        self.fitness_terminate = fitness_terminate
        self.dim = dim
        self.domain = domain
        self.full_fit_func = 0
        self.context_variable = None
        self.base_algo_args = kwargs
        self.niterations = 0
        self.convergences = []
        self.solution_variance_per_dim = []
        self.solution_variance_in_total = []

    def _construct_factor_variable_mapping(self):
        """
        Constructs a list of lists where each list contains the factors that optimize over a given variable.
        Essentially, indeces of variable_map are variables, and its elements are lists of factors.
        For example, variable map = [[0],[0,1],[1,2]] tells us that
        variable 0 is optimized by factor 0 alone, variable 1 is optimized both by factor 0 and factor 1.
        """
        variable_map = [[] for k in range(self.dim)]
        for i in range(len(self.factors)):
            for j in self.factors[i]:
                variable_map[j].append(i)
        return variable_map

    def run(self, progress=True):
        """
        Algorithm 3 from the Strasser et al. paper.
        """
        self.context_variable = self.init_full_global()
        subpopulations = self.initialize_subpops()
        if self.fitness_terminate:
            part_fit_func = 0
            while self.full_fit_func + part_fit_func < self.iterations:
                part_fit_func = 0
                self.niterations += 1
                for subpop in subpopulations:
                    subpop.base_reset()
                    # subpop.run(progress=False)
                    subpop.run()
                    part_fit_func += subpop.nfitness_evals
                self.compete(subpopulations)
                self.share(subpopulations)
                self.convergences.append(self.function(self.context_variable))
        for i in tqdm(range(self.iterations), disable=(not progress)):
            self.niterations += 1
            for subpop in subpopulations:
                subpop.base_reset()
                # subpop.run(progress=False)
                subpop.run()
            self.compete(subpopulations)
            self.share(subpopulations)
            self.convergences.append(self.function(self.context_variable))

    def compete(self, subpopulations):
        """
        Algorithm 1 from the Strasser et al. paper.
        @param subpopulations: the list of base algorithms, each with their own factor.
        """
        cont_var = self.context_variable
        best_fit = self.function(self.context_variable)
        self.full_fit_func += 1
        rand_var_permutation = np.random.permutation(self.dim)
        for i in rand_var_permutation:
            overlapping_factors = self.variable_map[i]
            best_val = np.copy(cont_var[i])
            best_fit = self.function(cont_var)
            self.full_fit_func += 1
            rand_pop_permutation = np.random.permutation(len(overlapping_factors))
            solution_to_measure_variance = []
            for j in rand_pop_permutation:
                s_j = overlapping_factors[j]
                index = np.where(self.factors[s_j] == i)[0][0]
                cont_var[i] = np.copy(subpopulations[s_j].get_solution_at_index(index))
                solution_to_measure_variance.append(
                    subpopulations[s_j].get_solution_at_index(index)
                )
                current_fit = self.function(cont_var)
                self.full_fit_func += 1
                if current_fit < best_fit:
                    best_val = np.copy(subpopulations[s_j].get_solution_at_index(index))
                    best_fit = current_fit
            cont_var[i] = np.copy(best_val)
            self.solution_variance_per_dim.append(np.var(solution_to_measure_variance))
        self.context_variable = cont_var
        self.solution_variance_in_total.append(np.average(self.solution_variance_per_dim))
        self.solution_variance_per_dim = []

    def share(self, subpopulations):
        """
        Algorithm 2 from the Strasser et al. paper.
        @param subpopulations: the list of subpopulations initialized in initialize_subpops.
        """
        for i in range(len(subpopulations)):
            subpopulations[i].func.context = np.copy(self.context_variable)
            subpopulations[i].update_worst(self.context_variable[self.factors[i]])
            subpopulations[i].update_bests()

    def init_full_global(self):
        """
        Randomly initializes the global context vector within the boundaries given by domain.
        """
        lbound = self.domain[:, 0]
        area = self.domain[:, 1] - self.domain[:, 0]
        return lbound + area * np.random.random(size=(area.shape[0]))

    def initialize_subpops(self):
        """
        Initializes some inheritor of FeaBaseAlgo to optimize over each factor.
        """
        ret = []
        for subpop in self.factors:
            fun = Function(context=self.context_variable, function=self.function, factor=subpop)
            ret.append(self.base_algo.from_kwargs(fun, self.domain[subpop, :], self.base_algo_args))
        return ret

    def diagnostic_plots(self):
        """
        Set up plots tracking solution convergence and variance over time.
        """
        plt.subplot(1, 2, 1)
        plt.plot(range(0, self.niterations), self.convergences)
        plt.title("Convergence")
        plt.subplot(1, 2, 2)
        plt.plot(range(0, self.niterations), self.solution_variance_in_total)
        plt.title("Solution Variance")

    def get_solution(self):
        return self.context_variable

    def get_solution_fitness(self):
        return self.function(self.context_variable)
