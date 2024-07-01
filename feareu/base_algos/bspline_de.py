import numpy as np
from feareu.base_algos.de import DE

class BsplineDE(DE):

    def init_pop(self):
        """
        Initialize random particles.
        """
        lbound = self.domain[:, 0]
        area = self.domain[:, 1] - self.domain[:, 0]
        pop = lbound + area * np.random.random(size=(self.pop_size, area.shape[0]))
        pop.sort()
        return pop

    def stay_in_domain(self):
        """
        Ensure that the particles don't move outside the domain of our function by projecting
        them to the domain's boundary at that point.
        """
        area = self.domain[:, 1] - self.domain[:, 0]
        self.mutant_pop = np.where(
            self.domain[:, 0] > self.mutant_pop,
            self.domain[:, 0] + 0.1 * area * np.random.random(),
            self.mutant_pop,
        )
        self.mutant_pop = np.where(
            self.domain[:, 1] < self.mutant_pop,
            self.domain[:, 1] - 0.1 * area * np.random.random(),
            self.mutant_pop,
        )

    def selection(self):
        """
        The fitness evaluation and selection. Greedily selects whether to keep or throw out a value.
        Consider implementing and testing more sophisticated selection algorithms.
        """
        self.pop.sort()
        self.mutant_pop.sort()
        for i in range(self.pop_size):
            fella_eval = self.func(self.mutant_pop[i,:])
            self.fitness_functions+=1
            if fella_eval < self.pop_eval[i]:
                self.pop_eval[i] = fella_eval
                self.pop[i,:] = np.copy(self.mutant_pop[i,:])
        self.best_solution = np.copy(self.pop[np.argmin(self.pop_eval),:])
        self.best_eval = np.min(self.pop_eval)
