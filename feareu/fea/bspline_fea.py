from feareu.fea import FEA
import matplotlib.pyplot as plt
import numpy as np
from feareu.function import Function

class BsplineFEA(FEA):
    """Factored Evolutionary Architecture, implemented based on the 2017 paper by Strasser et al.
    Altered so that each factor has its own domain based on the context vector.
    Intended for use in BSpline knot selection problem."""

    def __init__(self, factors, function, iterations, dim, base_algo_name, domain, **kwargs):
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

    def init_full_global(self):
        """
        Randomly initializes the global context vector within the boundaries given by domain.
        """
        lbound = np.zeros(self.dim)
        lbound[:] = self.domain[0]
        area = self.domain[1] - lbound
        return lbound + area * np.random.random(size=(self.dim))

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
        subpop.base_reset()
        subpop.run()
    
    def compete(self, subpopulations):
        """
        Algorithm 1 from the Strasser et al. paper, altered to sort the context vector
        when updated.
        @param subpopulations: the list of base algorithms, each with their own factor.
        """
        super().compete(subpopulations)
        self.context_variable.sort()

    def share(self, subpopulations):
        """
        Algorithm 2 from the Strasser et al. paper.
        @param subpopulations: the list of subpopulations initialized in initialize_subpops. 
        """
        subpop_domains = self.domain_evaluation()
        for i in range(len(subpopulations)):
            subpopulations[i].domain = subpop_domains[i]
            subpopulations[i].func.context = np.copy(self.context_variable)
            subpopulations[i].update_bests()

    def initialize_subpops(self, subpop_domains):
        """
        Initializes some inheritor of FeaBaseAlgo to optimize over each factor.
        Slightly altered to call domain differently.
        @param subpop_domains: the domains from domain_restriction.
        """
        ret = []
        for i, subpop in enumerate(self.factors):
            fun = Function(context=self.context_variable, function=self.function, factor=subpop)
            ret.append(self.base_algo.from_kwargs(fun, subpop_domains[i], self.base_algo_args))
        return ret
        
    def domain_evaluation(self):
        """
        Ensures that each factor has its own domain in which its variables can move.
        """
        subpop_domains = []
        for i, factor in enumerate(self.factors):
            fact_dom = np.zeros((len(factor),2))
            if factor[0] == 0:
                fact_dom[:,0] = self.domain[0]
            else:
                fact_dom[:,0] = self.context_variable[factor[0]-1]
            if factor[-1] == len(self.context_variable)-1:
                fact_dom[:,1] = self.domain[1]
            else:
                fact_dom[:,1] = self.context_variable[factor[-1]+1]
            subpop_domains.append(fact_dom)
        return subpop_domains

    def updating_subpop_domains(self, subpop_domains, subpopulations):
        """
        Updates each subpopulation to use the new domains from domain_restriction.
        @param subpop_domains: the domains from domain_restriction.
        @param subpopulations: the base algorithms to update the domains of.
        """
        for i, subpop in enumerate(subpopulations):
            subpop.domain = subpop_domains[i]
