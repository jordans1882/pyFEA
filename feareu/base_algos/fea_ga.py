from feareu.base_algos.fea_base_algo import FeaBaseAlgo
from feareu.base_algos.ga import GA
import math
import numpy as np

class FeaGA(GA, FeaBaseAlgo):
    
    def get_solution_at_index(self, idx):
        return self.pop[0, idx]
    
    def update_worst(self, context):
        self.pop[-1, :] = (context)
        
    @classmethod
    def from_kwargs(cls, function, domain, params):
        kwargs = {
            "generations": 100,
            "pop_size": 20,
            "b": 0.7,
            "mutation_rate": 0.05,
        }
        kwargs.update(params)
        return cls(
            function=function,
            domain=domain,
            generations=kwargs["generations"],
            pop_size=kwargs["pop_size"],
            b=kwargs["b"],
            mutation_rate=kwargs["mutation_rate"],
        )
    def base_reset(self):
        self.pop_eval = [self.func(self.pop[i, :]) for i in range(self.pop_size)]