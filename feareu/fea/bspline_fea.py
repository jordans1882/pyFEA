from feareu.fea import FEA
from copy import deepcopy
from operator import sub
import matplotlib.pyplot as plt
import numpy as np
from feareu.function import Function

class BsplineFEA(FEA):
    """Factored Evolutionary Architecture, implemented based on the 2017 paper by Strasser et al."""

    def __init__(self, factors, function, iterations, dim, base_algo_name, min_max, **kwargs):
        """
        @param factors: list of lists, contains the dimensions that each factor of the architecture optimizes over.
        @param function: the objective function that the FEA minimizes.
        @param iterations: the number of times that the FEA runs.
        @param dim: the number of dimensions our function optimizes over.
        @param base_algo_name: the base algorithm class that we optomize over. Should be a subclass of FeaBaseAlgo.
        @param min_max: the minimum and maximum possible values of our domain for any variable of the context vector.
        @param **kwargs: parameters for the base algorithm.
        """
        self.min_max = min_max
        self.domain = None
        super().__init__(factors, function, iterations, dim, base_algo_name, self.domain, **kwargs)
#        self.factors = factors
#        self.dim = dim
#        self.variable_map = self._construct_factor_variable_mapping()
#        self.function = function
#        self.iterations = iterations
#        self.base_algo = base_algo_name
#        self.dim = dim
#        self.domain = domain
#        self.context_variable = None
#        self.base_algo_args = kwargs
#        self.niterations = 0
#        self.convergences = []
#        self.solution_variance_per_dim = []
#        self.solution_variance_in_total = []

#    def _construct_factor_variable_mapping(self):
#        """
#        Constructs a list of lists where each list contains the factors that optimize over a given variable.
#        Essentially, indeces of variable_map are variables, and its elements are lists of factors.
#        For example, variable map = [[0],[0,1],[1,2]] tells us that
#        variable 0 is optimized by factor 0 alone, variable 1 is optimized both by factor 0 and factor 1.
#        """
#        variable_map = [[] for k in range(self.dim)]
#        for i in range(len(self.factors)):
#            for j in self.factors[i]:
#                variable_map[j].append(i)
#        return variable_map

    def run(self):
        """
        Algorithm 3 from the Strasser et al. paper.
        """
        self.context_variable = self.init_full_global()
        self.context_variable.sort()
        self.domain_restriction()
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
        """
        Algorithm 1 from the Strasser et al. paper.
        @param subpopulations: the list of base algorithms, each with their own factor.
        """
        cont_var = self.context_variable
        best_fit = self.function(self.context_variable)
        rand_var_permutation = np.random.permutation(self.dim)
        for i in rand_var_permutation:
            overlapping_factors = self.variable_map[i]
            best_val = np.copy(cont_var[i])
            best_fit = self.function(cont_var)
            rand_pop_permutation = np.random.permutation(len(overlapping_factors))
            solution_to_measure_variance = []
            for j in rand_pop_permutation:
                s_j = overlapping_factors[j]
                index = np.where(self.factors[s_j] == i)[0][0]
                cont_var[i] = np.copy(subpopulations[s_j].get_solution_at_index(index))
                solution_to_measure_variance.append(subpopulations[s_j].get_solution_at_index(index))
                current_fit = self.function(cont_var)
                if current_fit < best_fit:
                    best_val = np.copy(subpopulations[s_j].get_solution_at_index(index))
                    best_fit = current_fit
            cont_var[i] = np.copy(best_val)
            self.solution_variance_per_dim.append(np.var(solution_to_measure_variance))
        self.context_variable = (cont_var)
        self.context_variable.sort()
        self.solution_variance_in_total.append(np.average(self.solution_variance_per_dim))
        self.solution_variance_per_dim = []

#    def share(self, subpopulations):
#        """
#        Algorithm 2 from the Strasser et al. paper.
#        @param subpopulations: the list of subpopulations initialized in initialize_subpops. 
#        """
#        for i in range(len(subpopulations)):
#            subpopulations[i].func.context = np.copy(self.context_variable)
#            subpopulations[i].update_worst(self.context_variable[self.factors[i]])
#            subpopulations[i].update_bests()
#
    def init_full_global(self):
        """
        Randomly initializes the global context vector within the boundaries given by domain.
        """
        lbound = self.min_max[0]
        area = self.min_max[1] - self.min_max[0]
        return lbound + area * np.random.random(size=(self.dim))

    def initialize_subpops(self):
        """
        Initializes some inheritor of FeaBaseAlgo to optimize over each factor.
        """
        ret = []
        for i, subpop in enumerate(self.factors):
            fun = Function(context=self.context_variable, function=self.function, factor=subpop)
            ret.append(self.base_algo.from_kwargs(fun, self.domain[i], self.base_algo_args))
        return ret

    def domain_restriction(self):
        self.domain = []
        for i, factor in enumerate(self.factors):
            factor.sort()
            fact_dom = np.zeros((len(factor),2))
            if factor[0] == 0:
                fact_dom[:,0] = self.min_max[0]
            else:
                fact_dom[:,0] = self.context_variable[factor[0]-1]
            if factor[-1] == len(self.context_variable)-1:
                fact_dom[:,1] = self.min_max[1]
            else:
                fact_dom[:,1] = self.context_variable[factor[-1]+1]
            self.domain.append(fact_dom)

#    def diagnostic_plots(self):
#        """
#        Set up plots tracking solution convergence and variance over time.
#        """
#        plt.subplot(1, 2, 1)
#        ret = plt.plot(range(0, self.niterations), self.convergences)
#        plt.title("Convergence")
#
#        plt.subplot(1, 2, 2)
#        plt.plot(range(0, self.niterations), self.solution_variance_in_total)
#        plt.title("Solution Variance")
#        return ret

