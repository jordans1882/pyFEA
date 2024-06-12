import numpy as np
from function import Function
from copy import deepcopy
from base_pso import PSO

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

        print("best points: ", self.context_variable)
        print("convergence array: ", convergence)
        return self.function(self.context_variable)
       
    def compete(self, subpopulations):
        print("new compete")
        cont_var = deepcopy(self.context_variable)
        best_fit = deepcopy(self.function(self.context_variable))
        rand_var_permutation = np.random.permutation(self.dim)
        for i in rand_var_permutation:
            #print("Best fit: ", best_fit)
            overlapping_factors = self.variable_map[i]
            best_val = deepcopy(cont_var[i])
            rand_pop_permutation = np.random.permutation(len(overlapping_factors))
            for j in rand_pop_permutation:
                s_j = overlapping_factors[j]
                """curr_pop = subpopulations[s_j]
                pop_var_idx = np.where(self.factors[s_j] == i)
                position = [x for x in curr_pop.gbest]
                var_candidate_value = position[pop_var_idx[0][0]]
                cont_var[i] = var_candidate_value"""
                index = np.where(self.factors[s_j]==i)[0][0]
                cont_var[i] = deepcopy(subpopulations[s_j].gbest[index])
                current_fit = deepcopy(self.function(cont_var))
                #print("Current fit: ", current_fit, "Best fit: ", best_fit)
                if(current_fit<best_fit):
                    #print("accepted")
                    #best_val = var_candidate_value
                    best_val = deepcopy(subpopulations[s_j].gbest[index])
                    best_fit = deepcopy(current_fit)
            cont_var[i] = deepcopy(best_val)
        print("Context Vector before: ", self.context_variable)
        self.context_variable = deepcopy(cont_var)
        print("Context Vector after: ", self.context_variable)
        for subpop in subpopulations:
            subpop.func.context = deepcopy(self.context_variable)
    def share(self, subpopulations):
        for i in range(len(subpopulations)):
            worst = deepcopy(subpopulations[i].worst)
            subpopulations[i].pop[worst, :] = deepcopy(self.context_variable[self.factors[i]])
            #subpopulations[i].pop_eval[worst] = self.function(self.context_variable)
            subpopulations[i].update_bests()
    
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
    
    