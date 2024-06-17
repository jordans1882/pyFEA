from feareu.base_algos.fea_base_algo import FeaBaseAlgo
from feareu.base_algos.ga import GA
import math
import numpy as np

class FeaGA(GA, FeaBaseAlgo):
    """
    The Genetic Algorithm set up for use in an FEA.
    See the base Genetic Algorithm in ga.py for the full functionality.
    """
    def get_solution_at_index(self, idx):
        """
        The method to retrieve a variable's value from the base algorithm.
        @param idx: the index of the variable to be retrieved.
        """
        return self.best_position
    
    
    def update_worst(self, context):
        """
        The second half of the share step in the FEA algorithm, where we update 
        the worst individual in our population to become the context vector.
        @param context: the context vector of our FEA.
        """
        self.pop[-1, :] = (context)
        
    
    @classmethod
    def from_kwargs(cls, function, domain, params):
        """
        The method for inputting parameters to the DE from FEA.
        @param function: the objective function to minimize.
        @param domain: the domain over which to minimize as a numpy array of size (dim, 2)
        @param params: the remaining parameters, namely generations, pop_size, mutation_rate,
        and b. These are taken in as a dictionary here from the keyword arguments
        passed to FEA's constructor.
        """
        kwargs = {
            "generations": 100,
            "pop_size": 20,
            "b": 0.7,
            "mutation_rate": 0.05,
            "mutation_range": 0.5,
        }
        kwargs.update(params)
        return cls(
            function=function,
            domain=domain,
            generations=kwargs["generations"],
            pop_size=kwargs["pop_size"],
            b=kwargs["b"],
            mutation_rate=kwargs["mutation_rate"],
            mutation_range=kwargs["mutation_range"],
        )
    def base_reset(self):
        """
        Reset the algorithm in preparation for another run.
        """
        self.pop_eval = [self.func(self.pop[i, :]) for i in range(self.pop_size)]