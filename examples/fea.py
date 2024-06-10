import numpy as np
from base_pso import PSO
from function import Function

class FEA():
    def __init__(self, factors, function, iterations, dim, base_algo_name, domain, **kwargs):
        self.factors = factors
        self.function = function
        self.iterations = iterations
        self.base_algo_name = base_algo_name
        self.dim = dim
        self.domain = domain
        self.base_algo_args = kwargs
        self.generations = kwargs["generations"]
        self.variable_map = self._construct_factor_variable_mapping()
        
    # UNIT TEST THIS
    def _construct_factor_variable_mapping(self):
        variable_map = [[] for k in range(self.dim)]
        for i in range(len(self.factors)):
            for j in self.factors[i]:
                variable_map[j].append(i)
        return variable_map
    
    def run(self):
       subpopulations = self.initialize_subpops()
       self.context_variable = self.init_full_global()
       for i in range(self.iterations):
           for subpop in subpopulations:
               # FIX DOMAIN
               subpop.run()
            self.compete()
            self.share()
    
    def compete(self):
        rand_var_permutation = np.random.permutation(self.dim)
        for i in rand_var_permutation:
            best_fit = self.function(self.context_variable)
            
            
       
    def share():
        pass
    
    def init_full_global(self):
        lbound = self.domain[:,0]
        area = self.domain[:,1] - self.domain[:,0]
        return lbound + area * np.random.random(size=(1, area.shape[0]))
    
    def initialize_subpops(self):
        ret = []
        for subpop in self.factors:
            fun = Function(self.context_variable, self.function, subpop)
            alg = globals()[self.base_algo_name](fun, self.domain, self.base_algo_args)
            ret.append(alg)
        return ret
    
    """def update():
        
    
    def compete():
        
    
    def share():
        """