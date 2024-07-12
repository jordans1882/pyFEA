import math
import random

import numpy as np

from pyfea.base_algos import BsplineFeaGA, parallel_eval
from pyfea.base_algos.bspline_specific.known_knot_bspline_fea_ga import \
    KnownKnotBsplineFeaGA


class ParallelKnownKnotGA(KnownKnotBsplineFeaGA):
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
        mutation_rate=0.05,
        mutation_range=0.5,
        tournament_options=2,
        number_of_children=2,
    ):
        super().__init__(
            function,
            early_stop,
            domain,
            delta,
            true_error,
            og_knot_points,
            pop_size,
            mutation_rate,
            mutation_range,
            tournament_options,
            number_of_children,
        )
        self.processes = processes
        self.chunksize = chunksize

    def mutation(self, children):
        """
        Mutates children through swapping and recombines that with the parent population.
        """
        for child in children:
            for i in range(len(child)):
                if random.random() < self.mutation_rate:
                    rand_value = random.uniform(-1 * self.mutation_range, self.mutation_range)
                    child[i] += rand_value
        children = self.bounds_check(children)
        children.sort()
        child_evals = parallel_eval(
            self.func, children, processes=self.processes, chunksize=self.chunksize
        )
        self.pop_eval = np.concatenate((self.pop_eval, child_evals))
        self.fitness_functions += children.shape[0]
        self.pop = np.concatenate((self.pop, children))

    def base_reset(self):
        """
        Reset the algorithm in preparation for another run.
        """
        self.reinitialize_population()
        self.pop.sort()
        self.pop_eval = parallel_eval(
            self.func, self.pop, processes=self.processes, chunksize=self.chunksize
        )
        self.fitness_functions += self.pop_size

