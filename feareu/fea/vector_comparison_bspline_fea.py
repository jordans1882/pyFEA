from feareu.fea import BsplineFEA
import matplotlib.pyplot as plt
import numpy as np
from feareu.function import Function

class VectorComparisonBsplineFEA(BsplineFEA):
    """Factored Evolutionary Architecture, implemented based on the 2017 paper by Strasser et al.
    Altered so that each factor has its own domain based on the context vector.
    Intended for use in BSpline knot selection problem."""

    def __init__(self, factors, function, true_error, delta, dim, base_algo_name, domain, diagnostics_amount, og_knot_points, **kwargs):
        """
        @param factors: list of lists, contains the dimensions that each factor of the architecture optimizes over.
        @param function: the objective function that the FEA minimizes.
        @param iterations: the number of times that the FEA runs.
        @param dim: the number of dimensions our function optimizes over.
        @param base_algo_name: the base algorithm class that we optomize over. Should be a subclass of FeaBaseAlgo.
        @param domain: the minimum and maximum possible values of our domain for any variable of the context vector. It should be a tuple of size 2.
        @param **kwargs: parameters for the base algorithm.
        """
        self.og_knot_points = og_knot_points
        self.dif_to_og = []
        self.domain = domain
        self.stopping_point = delta + true_error
        self.diagnostic_amount = diagnostics_amount
        self.full_fit_func_array = []
        self.part_fit_func_array = []
        self.factors = factors
        self.dim = dim
        self.variable_map = self._construct_factor_variable_mapping()
        self.function = function
        self.base_algo = base_algo_name
        self.dim = dim
        self.domain = domain
        self.full_fit_func = 0
        self.context_variable = None
        self.base_algo_args = kwargs
        self.niterations = 0
        self.iterations = 0
        self.convergences = []
        self.solution_variance_per_dim = []
        self.solution_variance_in_total = []
        for factor in self.factors:
            factor.sort()
        
    def run(self):
        """
        Algorithm 3 from the Strasser et al. paper, altered to sort the context
        vector on initialization.
        """
        self.context_variable = self.init_full_global()
        self.context_variable.sort()
        subpop_domains = self.domain_evaluation()
        subpopulations = self.initialize_subpops(subpop_domains)
        print("stopping_point: ", self.stopping_point)
        print("current func eval: ", self.function(self.context_variable))
        while self.function(self.context_variable) > self.stopping_point:
            self.niterations += 1
            for subpop in subpopulations:
                self.subpop_compute(subpop)
            self.compete(subpopulations)
            self.share(subpopulations)
            if self.niterations % self.diagnostic_amount == 0:
                self.update_plots(subpopulations)
            self.iterations +=1
            print("delta: ", self.stopping_point)
            print("current func eval: ", self.function(self.context_variable))
            print("full fit func: ", self.full_fit_func)
            print("part fit func: ", self.part_fit_func_array[len(self.part_fit_func_array)-1])
        return self.function(self.context_variable)
    
    def update_plots(self, subpopulations):
        self.convergences.append(self.function(self.context_variable))
        self.full_fit_func_array.append(self.full_fit_func)
        print("og_knot_points: ", self.og_knot_points)
        print("context: ", self.context_variable)
        self.dif_to_og.append(np.linalg.norm(self.og_knot_points - self.context_variable))
        tot_part_fit = 0
        for subpop in subpopulations:
            tot_part_fit += subpop.fitness_functions
        self.part_fit_func_array.append(tot_part_fit)

    def diagnostic_plots(self):
        """
        Set up plots tracking solution convergence and variance over time.
        """
        plt.subplot(1, 4, 1)
        ret = plt.plot(range(0, int(self.iterations/self.diagnostic_amount)), self.convergences)
        plt.title("Convergence")
        
        plt.subplot(1, 4, 2)
        plt.plot(range(0, int(self.iterations/self.diagnostic_amount)), self.full_fit_func_array)
        plt.title("Full Fit Func")
        
        plt.subplot(1, 4, 3)
        plt.plot(range(0, int(self.iterations/self.diagnostic_amount)), self.part_fit_func_array)
        plt.title("Part Fit Func")
        
        plt.subplot(1, 4, 4)
        plt.plot(range(0, int(self.iterations/self.diagnostic_amount)), self.dif_to_og)
        plt.title("Dif to OG Knots")
        return ret