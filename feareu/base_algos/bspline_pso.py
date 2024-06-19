import numpy as np

from feareu.base_algos.pso import PSO


class BSplinePSO(PSO):

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
        self.pop = np.where(
            self.domain[:, 0] > self.pop,
            self.domain[:, 0] + 0.1 * area * np.random.random(),
            self.pop,
        )
        self.pop = np.where(
            self.domain[:, 1] < self.pop,
            self.domain[:, 1] - 0.1 * area * np.random.random(),
            self.pop,
        )

    def update_bests(self):
        """
        Update the current personal and global best values based on the new positions of the particles.
        """
        self.pop.sort()
        for pidx in range(self.pop_size):
            curr_eval = self.func(self.pop[pidx, :])
            self.pop_eval[pidx] = curr_eval
            if curr_eval < self.pbest_eval[pidx]:
                self.pbest[pidx, :] = np.copy(self.pop[pidx, :])
                self.pbest_eval[pidx] = curr_eval
                if curr_eval < self.gbest_eval:
                    self.gbest = np.copy(self.pop[pidx, :])
                    self.gbest_eval = curr_eval
