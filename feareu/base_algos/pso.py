import math
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np


class PSO:
    def __init__(
        self,
        function,
        domain,
        generations=100,
        pop_size=20,
        phi_p=math.sqrt(2),
        phi_g=math.sqrt(2),
        omega=1 / math.sqrt(2),
    ):
        self.generations = generations
        self.pop_size = pop_size
        self.func = function
        self.domain = domain
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.omega = omega
        self.pop = self.init_pop()
        self.pbest = (self.pop)
        self.pop_eval = [self.func(self.pop[i, :]) for i in range(self.pop_size)]
        self.pbest_eval = deepcopy(self.pop_eval)
        self.worst = np.argmax(self.pop_eval)
        self.gbest_eval = np.min(self.pbest_eval)
        self.gbest = np.copy(self.pbest[np.argmin(self.pbest_eval), :])
        self.velocities = self.init_velocities()
        self.ngenerations = 0
        self.average_velocities = []
        self.average_pop_eval = []
        self.gbest_evals = []

    
    def init_pop(self):
        lbound = self.domain[:, 0]
        area = self.domain[:, 1] - self.domain[:, 0]
        return lbound + area * np.random.random(size=(self.pop_size, area.shape[0]))

    def init_velocities(self):
        area = self.domain[:, 1] - self.domain[:, 0]
        return 0.5 * area * np.random.random(size=(self.pop_size, area.shape[0]))

    def run(self):
        self._append_avg_velocities()
        self._append_avg_evals()
        self._append_gbest_evals()
        self.ngenerations += 1
        for gen in range(self.generations):
            self.update_velocities()
            self.pop = self.pop + self.velocities
            #print("new pop: ", self.pop)
            self.stay_in_domain()
            self.update_bests()
            self._append_avg_velocities()
            self._append_avg_evals()
            self._append_gbest_evals()
            self.ngenerations += 1
        return self.gbest

    def stay_in_domain(self):
        self.pop = np.where(self.domain[:, 0] > self.pop, self.domain[:, 0], self.pop)
        self.pop = np.where(self.domain[:, 1] < self.pop, self.domain[:, 1], self.pop)

    def update_velocities(self):
        r_p = np.random.random(size=self.pop.shape)
        r_g = np.random.random(size=self.pop.shape)
        self.velocities = (
            self.omega * self.velocities
            + self.phi_p * r_p * (self.pbest - self.pop)
            + self.phi_g * r_g * (self.gbest - self.pop)
        )

    def update_bests(self):
        print("pop evals: ", self.pop_eval)
            #Not updating
        for pidx in range(self.pop_size):
            curr_eval = self.func(self.pop[pidx, :])
            self.pop_eval[pidx] = curr_eval
            if curr_eval < self.pbest_eval[pidx]:
                self.pbest[pidx, :] = np.copy(self.pop[pidx, :])
                self.pbest_eval[pidx] = curr_eval
                if curr_eval < self.gbest_eval:
                    self.gbest = np.copy(self.pop[pidx, :])
                    self.gbest_eval = curr_eval
        self.worst = np.argmax(self.pop_eval)

    def _append_gbest_evals(self):
        self.gbest_evals.append(self.gbest_eval)

    def _append_avg_velocities(self):
        self.average_velocities.append(np.average(np.abs(self.velocities)))

    def _append_avg_evals(self):
        self.average_pop_eval.append(np.average(self.pop_eval))
        
    def diagnostic_plots(self):
        plt.subplot(1, 3, 1)
        ret = plt.plot(range(0, self.ngenerations), self.average_pop_eval)
        plt.title("Average pop evals")

        plt.subplot(1, 3, 2)
        plt.plot(range(0, self.ngenerations), self.gbest_evals)
        plt.title("Global Bests")

        plt.subplot(1, 3, 3)
        plt.plot(range(0, self.ngenerations), self.average_velocities)
        plt.title("Average Velocities")
        plt.tight_layout()
        
        return ret


# def main():
#     array = np.zeros((10))
#     array[0] = 1
#     array[1] = 2
#     domain = np.zeros((2, 2))
#     domain[:, 0] = -5
#     domain[:, 1] = 5
#     function = Function(array, rastrigin__, [0, 1])
#     pso = PSO(function, domain)
#     print(pso.run())
#
#
# if __name__ == "__main__":
#     main()
