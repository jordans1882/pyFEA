import math
from copy import deepcopy

import numpy as np

from pyfea.base_algos.bspline_specific.bspline_fea_pso import BsplineFeaPSO
from pyfea.base_algos.traditional.parallel_evaluation import parallel_eval


class ParallelBsplineFeaPSO(BsplineFeaPSO):

    def __init__(
        self,
        function,
        domain,
        generations=100,
        pop_size=20,
        phi_p=math.sqrt(2),
        phi_g=math.sqrt(2),
        omega=1 / math.sqrt(2),
        processes=4,
        chunksize=4,
        fitness_terminate=False,
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
        @param processes: the number of processes used when parallelizing fitness evaluations.
        @param chunksize: the chunksize used for the parallel fitness evaluation.
        """
        self.generations = generations
        self.pop_size = pop_size
        self.func = function
        self.domain = domain
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.omega = omega
        self.processes = processes
        self.chunksize = chunksize
        self.pop = self.init_pop()
        self.pbest = self.pop
        self.pop_eval = parallel_eval(
            self.func, self.pop, processes=self.processes, chunksize=self.chunksize
        )
        self.fitness_functions = pop_size
        self.pbest_eval = deepcopy(self.pop_eval)
        self.gbest_eval = np.min(self.pbest_eval)
        self.gbest = np.copy(self.pbest[np.argmin(self.pbest_eval), :])
        self.velocities = self.init_velocities()
        self.generations_passed = 0
        self.average_velocities = []
        self.average_pop_eval = []
        self.average_pop_variance = []
        self.gbest_evals = []
        self.fitness_list = []
        self.fitness_terminate = fitness_terminate

    def run(self):
        """
        Run the algorithm.
        """
        if self.fitness_terminate:
            while self.fitness_functions < self.generations:
                self.update_velocities()
                self.pop = self.pop + self.velocities
                self.stay_in_domain()
                self.update_bests()
                self._track_values()
                self.generations_passed += 1
        else:
            super().run()
        return self.gbest_eval

    def update_bests(self):
        """
        Update the current personal and global best values based on the new positions of the particles.
        """
        self.stay_in_domain()
        self.order_knots()
        self.pop_eval = parallel_eval(
            self.func, self.pop, processes=self.processes, chunksize=self.chunksize
        )
        self.fitness_functions += self.pop_size
        for pidx in range(self.pop_size):
            curr_eval = self.pop_eval[pidx]
            if curr_eval < self.pbest_eval[pidx]:
                self.pbest[pidx, :] = np.copy(self.pop[pidx, :])
                self.pbest_eval[pidx] = curr_eval
                if curr_eval < self.gbest_eval:
                    self.gbest = np.copy(self.pop[pidx, :])
                    self.gbest_eval = curr_eval

    def reset_fitness(self):
        """
        Reevaluate the fitness function in parallel over the entire population and update the fields accordingly.
        """
        self.pbest = self.pop
        self.pop_eval = parallel_eval(
            self.func, self.pop, processes=self.processes, chunksize=self.chunksize
        )
        self.fitness_functions += self.pop_size
        self.pbest_eval = self.pop_eval
        self.gbest_eval = np.min(self.pbest_eval)
        self.gbest = np.copy(self.pbest[np.argmin(self.pbest_eval), :])

    def order_knots(self):
        sort_idxs = self.pop.argsort()
        self.pop = np.array([p[s] for p, s in zip(self.pop, sort_idxs)])
        self.velocities = np.array([v[s] for v, s in zip(self.velocities, sort_idxs)])

    def base_reset(self):
        """
        Reset the algorithm in preparation for another run.
        """
        self.reinitialize_population()
        self.order_knots()
        self.velocities = super().init_velocities()
        self.reset_fitness()
