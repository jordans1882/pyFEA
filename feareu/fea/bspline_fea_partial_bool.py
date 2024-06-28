from feareu.fea import FEA, BsplineFEA
import matplotlib.pyplot as plt
import numpy as np
from feareu.function import Function

class BsplineFEAPartialBool(BsplineFEA):
    """Factored Evolutionary Architecture, implemented based on the 2017 paper by Strasser et al.
    Altered so that each factor has its own domain based on the context vector.
    Intended for use in BSpline knot selection problem."""

    def __init__(self, factors, function, iterations, dim, base_algo_name, domain, partial_reeval = True, **kwargs):
        """
        @param factors: list of lists, contains the dimensions that each factor of the architecture optimizes over.
        @param function: the objective function that the FEA minimizes.
        @param iterations: the number of times that the FEA runs.
        @param dim: the number of dimensions our function optimizes over.
        @param base_algo_name: the base algorithm class that we optomize over. Should be a subclass of FeaBaseAlgo.
        @param domain: the minimum and maximum possible values of our domain for any variable of the context vector. It should be a tuple of size 2.
        @param **kwargs: parameters for the base algorithm.
        """
        self.domain = domain
        super().__init__(factors, function, iterations, dim, base_algo_name, self.domain, **kwargs)
        for factor in self.factors:
            factor.sort()
        self.partial_reeval = partial_reeval

    def run(self):
        """
        Algorithm 3 from the Strasser et al. paper, altered to sort the context
        vector on initialization.
        """
        self.context_variable = self.init_full_global()
        self.context_variable.sort()
        subpop_domains = self.domain_evaluation()
        subpopulations = self.initialize_subpops(subpop_domains)
        for i in range(self.iterations):
            self.niterations += 1
            for subpop in subpopulations:
                self.subpop_compute(subpop)
            self.compete(subpopulations)
            self.share(subpopulations)
            self.convergences.append(self.function(self.context_variable))
        return self.function(self.context_variable)

    def subpop_compute(self, subpop):
        if self.partial_reeval:
            subpop.partial_base_reset()
        else:
            subpop.base_reset()
        subpop.run()