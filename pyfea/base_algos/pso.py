import math
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .parallel_evaluation import parallel_eval


class PSO:
    """
    Particle Swarm Optimization
    """

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
        """
        @param function: the objective function to be minimized.
        @param domain: the domain on which we explore the function stored as a (dim,2) matrix,
        where dim is the number of dimensions we evaluate the function over.
        @param generations: the number of generations run before the algorithm terminates.
        @param pop_size: the number of particles in the population.
        @param phi_p: the factor by which we multiply our distance from the particle's personal best
        when updating velocities.
        @param phi_g: the factor by which we multiply our distance from the global best solution
        when updating velocities.
        @param omega: the inertia, or the amount of the old velocity that we keep during our update.
        """
        self.generations = generations
        self.pop_size = pop_size
        self.func = function
        self.domain = domain
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.omega = omega
        self.nfitness_evals = self.pop_size
        self.generations_passed = 0
        self.average_velocities = []
        self.average_pop_variance = []
        self.average_pop_eval = []
        self.gbest_evals = []
        self.fitness_list = []
        self.pop = self._init_pop()
        self.pbest = self.pop
        self.pop_eval = [sys.float_info.max] * self.pop_size
        self.velocities = np.zeros((self.pop_size, domain.shape[0]))

    def _init_pop(self):
        """
        Initialize random particles.
        """
        # print(self.generations)
        lbound = self.domain[:, 0]
        area = self.domain[:, 1] - self.domain[:, 0]
        return lbound + area * np.random.random(size=(self.pop_size, area.shape[0]))

    def _init_velocities(self):
        """
        Initialize random velocities.
        """
        # print("hi")
        area = self.domain[:, 1] - self.domain[:, 0]
        return 0.5 * area * np.random.random(size=(self.pop_size, area.shape[0]))

    def _initialize(self, parallel=False, processes=4, chunksize=4):
        if parallel:
            self.pop_eval = parallel_eval(self.func, self.pop, processes, chunksize)
        else:
            self.pop_eval = [self.func(self.pop[i, :]) for i in range(self.pop_size)]
        self.pbest_eval = deepcopy(self.pop_eval)
        self.gbest_eval = np.min(self.pbest_eval)
        self.gbest = np.copy(self.pbest[np.argmin(self.pbest_eval), :])
        self.velocities = self._init_velocities()

    def _update_positions(self):
        self.pop = self.pop + self.velocities

    def run(self, progress=False, parallel=False, processes=4, chunksize=4):
        """
        Run the algorithm.
        """
        self._initialize(parallel, processes, chunksize)
        for gen in tqdm(range(self.generations), disable=(not progress)):
            self._update_velocities()
            self._update_positions()
            self._stay_in_domain()
            self._eval_pop(parallel, processes, chunksize)
            self.update_bests()
            self._track_values()
            self.generations_passed += 1

    def get_soln(self):
        return self.gbest

    def get_soln_fitness(self):
        return self.gbest_eval

    def _stay_in_domain(self):
        """
        Ensure that the particles don't move outside the domain of our function by projecting
        them to the domain's boundary at that point.
        """
        self.pop = np.where(self.domain[:, 0] > self.pop, self.domain[:, 0], self.pop)
        self.pop = np.where(self.domain[:, 1] < self.pop, self.domain[:, 1], self.pop)

    def _update_velocities(self):
        """
        Update the velocities of the particles according to the PSO algorithm.
        """
        r_p = np.random.random(size=self.pop.shape)
        r_g = np.random.random(size=self.pop.shape)
        self.velocities = (
            self.omega * self.velocities
            + self.phi_p * r_p * (self.pbest - self.pop)
            + self.phi_g * r_g * (self.gbest - self.pop)
        )

    def _eval_pop(self, parallel=False, processes=4, chunksize=4):
        if not parallel:
            for pidx in range(self.pop_size):
                curr_eval = self.func(self.pop[pidx, :])
                self.nfitness_evals += 1
                self.pop_eval[pidx] = curr_eval
        else:
            self.pop_eval = parallel_eval(
                self.func, self.pop, processes=processes, chunksize=chunksize
            )
            self.nfitness_evals += self.pop_size

    def update_bests(self):
        """
        Update the current personal and global best values based on the new positions of the particles.
        """
        for pidx in range(self.pop_size):
            curr_eval = self.pop_eval[pidx]
            if curr_eval < self.pbest_eval[pidx]:
                self.pbest[pidx, :] = np.copy(self.pop[pidx, :])
                self.pbest_eval[pidx] = curr_eval
                if curr_eval < self.gbest_eval:
                    self.gbest = np.copy(self.pop[pidx, :])
                    self.gbest_eval = curr_eval
        # self.worst = np.argmax(self.pop_eval)

    def _track_values(self):
        """
        Tracks various diagnostic values over the course of the algorithm's run.
        """
        self.gbest_evals.append(self.gbest_eval)
        self.average_velocities.append(np.average(np.abs(self.velocities)))
        self.average_pop_variance.append(np.average(np.var(self.pop, axis=0)))
        self.average_pop_eval.append(np.average(self.pop_eval))
        self.fitness_list.append(self.nfitness_evals)

    def diagnostic_plots(self):
        """
        Plots the values tracked in _track_values().
        """
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
        ax1.plot(self.fitness_list, self.average_pop_variance)
        ax1.set_xlabel("# fitness evaluations", fontsize=10)
        ax1.set_ylabel("population variance", fontsize=10)
        ax1.set_title("Population Diversity", fontsize=10)

        ax2.plot(self.fitness_list, self.average_pop_eval)
        ax2.set_xlabel("# fitness evaluations", fontsize=10)
        ax2.set_ylabel("average MSE", fontsize=10)
        ax2.set_title("Average Solution Fitness", fontsize=10)

        ax3.plot(self.fitness_list, self.gbest_evals)
        ax3.set_xlabel("# fitness evaluations", fontsize=10)
        ax3.set_ylabel("gbest MSE", fontsize=10)
        ax3.set_title("Best Solution Fitness", fontsize=10)

        fig.suptitle("PSO")
        fig.tight_layout()


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
