from feareu.fea import BsplineFEA
import matplotlib.pyplot as plt
import numpy as np
from feareu.function import Function
from multiprocessing import Pool

class ParallelBsplineFEA(BsplineFEA):
    """Factored Evolutionary Architecture, implemented based on the 2017 paper by Strasser et al.
    Altered so that each factor has its own domain based on the context vector.
    Intended for use in BSpline knot selection problem."""
    def __init__(self, factors, function, iterations, dim, base_algo_name, domain, process_count, **kwargs):
        """
        @param factors: list of lists, contains the dimensions that each factor of the architecture optimizes over.
        @param function: the objective function that the FEA minimizes.
        @param iterations: the number of times that the FEA runs.
        @param dim: the number of dimensions our function optimizes over.
        @param base_algo_name: the base algorithm class that we optomize over. Should be a subclass of FeaBaseAlgo.
        @param domain: the minimum and maximum possible values of our domain for any variable of the context vector. It should be a tuple of size 2.
        @param **kwargs: parameters for the base algorithm.
        """
        self.process_count = process_count
        super().__init__(factors, function, iterations, dim, base_algo_name, domain, **kwargs)
        self.subpop_domains = []
        
    def run(self):
        """
        Algorithm 3 from the Strasser et al. paper, altered to sort the context
        vector on initialization.
        """
        self.context_variable = self.init_full_global()
        self.context_variable.sort()
        self.domain_evaluation()
        with Pool(self.process_count) as pool:
            subpopulations = pool.map(self.initialize_subpop, np.arange(0, len(self.factors)), chunksize=int(len(self.factors)/self.process_count))
        #subpopulations = self.initialize_subpops(self.subpop_domains)
        for i in range(self.iterations):
            self.niterations += 1
            with Pool(self.process_count) as pool:
                pool.map(self.subpop_compute, subpopulations, chunksize=int(len(self.factors)/self.process_count))
            self.compete(subpopulations)
            self.share(subpopulations)
            self.convergences.append(self.function(self.context_variable))
        return self.function(self.context_variable)
        
    def domain_evaluation(self):
        """
        Ensures that each factor has its own domain in which its variables can move.
        """
        self.subpop_domains = []
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
            self.subpop_domains.append(fact_dom)

    def compete(self, subpopulations):
        """
        Algorithm 1 from the Strasser et al. paper, altered to sort the context vector
        when updated.
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
        self.solution_variance_in_total.append(np.average(self.solution_variance_per_dim))
        self.solution_variance_per_dim = []
        self.context_variable.sort()
        self.domain_evaluation()
        self.updating_subpop_domains(subpopulations)

    def updating_subpop_domains(self, subpopulations):
        """
        Updates each subpopulation to use the new domains from domain_restriction.
        @param subpop_domains: the domains from domain_restriction.
        @param subpopulations: the base algorithms to update the domains of.
        """
        for i, subpop in enumerate(subpopulations):
            subpop.domain = self.subpop_domains[i]

    #def initialize_subpops(self):
        """
        Initializes some inheritor of FeaBaseAlgo to optimize over each factor.
        Slightly altered to call domain differently.
        @param subpop_domains: the domains from domain_restriction.
        """
        #ret = []
        #for i, subpop in enumerate(self.factors):
        #    fun = Function(context=self.context_variable, function=self.function, factor=subpop)
        #    ret.append(self.base_algo.from_kwargs(fun, self.subpop_domains[i], self.base_algo_args))
        #return ret
        
    def initialize_subpop(self, i):
        """
        Initializes some inheritor of FeaBaseAlgo to optimize over each factor.
        Slightly altered to call domain differently.
        @param subpop_domains: the domains from domain_restriction.
        """
        fun = Function(context=self.context_variable, function=self.function, factor=self.factors[i])
        return self.base_algo.from_kwargs(fun, self.subpop_domains[i], self.base_algo_args)