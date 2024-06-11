import numpy as np
from examples.function import Function
from examples.base_pso import PSO

class FEA():
    def __init__(self, factors, function, iterations, dim, base_algo_name, domain, **kwargs):
        self.factors = factors
        self.function = function
        self.iterations = iterations
        self.base_algo_name = base_algo_name
        self.dim = dim
        self.domain = domain
        self.context_variable = None
        self.base_algo_args = kwargs
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
        print("initial subpopulations: ", subpopulations)
        convergence = []
        for i in range(self.iterations):
            for subpop in subpopulations:
               # FIX DOMAIN
               subpop.run()
            self.compete(subpopulations)
            self.share(subpopulations)
            convergence.append(self.function(self.context_variable))
        print("best points: ", self.context_variable)
        print("convergence array: ", convergence)
        return self.function(self.context_variable)
    
    def compete(self, subpopulations):
        rand_var_permutation = np.random.permutation(self.dim)
        best_fit = self.function(self.context_variable)
        for i in rand_var_permutation:
            overlapping_factors = self.variable_map[i]
            best_val = subpopulations[overlapping_factors[0]].gbest_eval
            rand_pop_permutation = np.random.permutation(len(overlapping_factors))
            for j in rand_pop_permutation:
                s_j = overlapping_factors[j]
                index = np.where(self.factors[s_j]==i)[0][0]
                self.context_variable[i] = subpopulations[s_j].gbest[index]
                current_fit = self.function(self.context_variable)
                if(current_fit<best_fit):
                    best_val = subpopulations[s_j].gbest[index]
                    best_fit = current_fit
            self.context_variable[i] = best_val
        for subpop in subpopulations:
            subpop.func.context = self.context_variable
       
    """def compete(self, subpopulations):
        rand_var_permutation = np.random.permutation(self.dim)
        for i in rand_var_permutation:
            best_fit = self.function(self.context_variable)
            overlapping_factors = self.variable_map[i]
            best_val = subpopulations[overlapping_factors[0]].gbest_eval
            rand_pop_permutation = np.random.permutation(len(overlapping_factors))
            for j in rand_pop_permutation:
                s_j = overlapping_factors[j]
                index = np.where(self.factors[s_j]==i)[0][0]
                self.context_variable[i] = subpopulations[s_j].gbest[index]
                current_fit = self.function(self.context_variable)
                if(current_fit<best_fit):
                    best_val = subpopulations[s_j].gbest[index]
                    best_fit = current_fit
            self.context_variable[i] = best_val
        for subpop in subpopulations:
            subpop.func.context = self.context_variable"""
    def share(self, subpopulations):
        for i in range(len(subpopulations)):
            worst = subpopulations[i].worst
            subpopulations[i].pop[worst, :] = self.context_variable[self.factors[i]]
            subpopulations[i].pop_eval[worst] = self.function(self.context_variable)
    
    def init_full_global(self):
        lbound = self.domain[:,0]
        area = self.domain[:,1] - self.domain[:,0]
        return lbound + area * np.random.random(size=(area.shape[0]))
    
    def initialize_subpops(self):
        ret = []
        for subpop in self.factors:
            fun = Function(context = self.context_variable, function = self.function, factor = subpop)
            alg = globals()[self.base_algo_name].from_kwargs(fun, self.domain[subpop, :], self.base_algo_args)
            ret.append(alg)
        return ret
    
    