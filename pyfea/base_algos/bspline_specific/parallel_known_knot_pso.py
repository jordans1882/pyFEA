import math

import numpy as np

from pyfea.base_algos.bspline_specific.known_knot_bspline_fea_pso import \
    KnownKnotBsplineFeaPSO
from pyfea.base_algos.traditional.parallel_evaluation import parallel_eval


class ParallelKnownKnotPSO(KnownKnotBsplineFeaPSO):
    def __init__(
        self,
        function,
        early_stop,
        domain,
        delta,
        true_error,
        og_knot_points,
        processes=4,
        chunksize=4,
        pop_size=20,
        phi_p=math.sqrt(2),
        phi_g=math.sqrt(2),
        omega=1 / math.sqrt(2),
    ):
        super().__init__(
            function,
            early_stop,
            domain,
            delta,
            true_error,
            og_knot_points,
            pop_size,
            phi_p,
            phi_g,
            omega,
        )
        self.processes = processes
        self.chunksize = chunksize

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
