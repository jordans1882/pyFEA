from feareu.base_algos.fea_base_algo import FeaBaseAlgo
from feareu.base_algos.pso import PSO
import math
import numpy as np

class FeaPso(PSO, FeaBaseAlgo):
    """def __init__(self,
        function,
        domain,
        generations=100,
        pop_size=20,
        phi_p=math.sqrt(2),
        phi_g=math.sqrt(2),
        omega=1 / math.sqrt(2),):
        PSO.__init__(self, function = function, domain = domain, generations=generations, pop_size=pop_size, phi_p=phi_p, phi_g=phi_g, omega=omega)
    """    
    """def update_bests(self):
        super().update_bests()
    
    def run(self):
        super().run()"""
    
    def get_solution_at_index(self, idx):
        return self.gbest[idx]
    
    def update_worst(self, context):
        self.pop[self.worst, :] = (context)
        
    @classmethod
    def from_kwargs(cls, function, domain, params):
        kwargs = {
            "generations": 100,
            "pop_size": 20,
            "phi_p": math.sqrt(2),
            "phi_g": math.sqrt(2),
            "omega": 1 / math.sqrt(2),
        }
        kwargs.update(params)
        return cls(
            function=function,
            domain=domain,
            generations=kwargs["generations"],
            pop_size=kwargs["pop_size"],
            phi_p=kwargs["phi_p"],
            phi_g=kwargs["phi_g"],
            omega=kwargs["omega"],
        )
    def base_reset(self):
        self.velocities = super().init_velocities()
        self.reset_fitness()
    
    def reset_fitness(self):
        self.pbest = self.pop
        self.pop_eval = [self.func(self.pop[i, :]) for i in range(self.pop_size)]
        self.pbest_eval = self.pop_eval
        self.worst = np.argmax(self.pop_eval)
        self.gbest_eval = np.min(self.pbest_eval)
        self.gbest = np.copy(self.pbest[np.argmin(self.pbest_eval), :])