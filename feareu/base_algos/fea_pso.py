from feareu.base_algos.fea_base_algo import FeaBaseAlgo
from feareu.base_algos.pso import PSO
import math

class FeaPso(FeaBaseAlgo, PSO):
    def update_bests(self):
        PSO.update_bests(self)
    
    def run(self):
        PSO.run(self)
    
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
        self.init_velocities()
        self.reset_fitness()