from feareu.base_algos import DE
from feareu.base_algos.fea_base_algo import FeaBaseAlgo
import numpy as np

class FeaDE(DE, FeaBaseAlgo):
    def get_solution_at_index(self, idx):
        return self.best_solution[idx]
    
    def update_worst(self, context):
        self.pop[np.argmax(self.pop_eval), :] = (context)

    @classmethod
    def from_kwargs(cls, function, domain, params):
        kwargs = {
            "generations": 100,
            "pop_size": 20,
            "mutation_factor": 0.5,
            "crossover_rate": 0.5
        }
        kwargs.update(params)
        return cls(
            function=function,
            domain=domain,
            generations=kwargs["generations"],
            pop_size=kwargs["pop_size"],
            mutation_factor=kwargs["mutation_factor"],
            crossover_rate=kwargs["crossover_rate"],
        )
    def base_reset(self):
        self.pop_eval = [self.func(self.pop[i, :]) for i in range(self.pop_size)]
        self.best_solution = np.copy(self.pop[np.argmin(self.pop_eval), :])
        self.best_eval = np.min(self.pop_eval)

    def run(self):
        return super().run()

    def update_bests(self):
        for pidx in range(self.pop_size):
            curr_eval = self.func(self.pop[pidx, :])
            self.pop_eval[pidx] = curr_eval
        self.best_eval = np.min(self.pop_eval)
        self.best_solution = np.copy(self.pop[np.argmin(self.pop_eval), :])
