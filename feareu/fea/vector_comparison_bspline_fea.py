from feareu.fea import BsplineFEA
import matplotlib.pyplot as plt
import numpy as np
from feareu.function import Function

class VectorComparisonBsplineFEA(BsplineFEA):
    """Factored Evolutionary Architecture, implemented based on the 2017 paper by Strasser et al.
    Altered so that each factor has its own domain based on the context vector.
    Intended for use in BSpline knot selection problem."""

    def __init__(self, factors, function, iterations, dim, base_algo_name, domain, diagnostics_amount, og_knot_points, **kwargs):
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
        super().__init__(factors, function, iterations, dim, base_algo_name, domain, diagnostics_amount, **kwargs)
        

    def update_plots(self, subpopulations):
        self.convergences.append(self.function(self.context_variable))
        self.full_fit_func_array.append(self.full_fit_func)
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