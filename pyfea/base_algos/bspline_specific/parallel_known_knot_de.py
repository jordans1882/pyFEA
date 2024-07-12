import numpy as np

from pyfea.base_algos.bspline_specific.known_knot_bspline_fea_de import \
    KnownKnotBsplineFeaDE
from pyfea.base_algos.traditional.parallel_evaluation import parallel_eval


class ParallelKnownKnotDE(KnownKnotBsplineFeaDE):
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
        mutation_factor=0.5,
        crossover_rate=0.9,
    ):
        super().__init__(
            function,
            early_stop,
            domain,
            delta,
            true_error,
            og_knot_points,
            pop_size,
            mutation_factor,
            crossover_rate,
        )
        self.processes = processes
        self.chunksize = chunksize

    def selection(self):
        """
        The fitness evaluation and selection. Greedily selects whether to keep or throw out a value.
        Consider implementing and testing more sophisticated selection algorithms.
        """
        self.pop.sort()
        self.mutant_pop.sort()
        mutant_pop_eval = parallel_eval(
            self.func, self.mutant_pop, processes=self.processes, chunksize=self.chunksize
        )
        self.fitness_functions += self.pop_size
        for i in range(self.pop_size):
            fella_eval = mutant_pop_eval[i]
            if fella_eval < self.pop_eval[i]:
                self.pop_eval[i] = fella_eval
                self.pop[i, :] = np.copy(self.mutant_pop[i, :])
        self.best_solution = np.copy(self.pop[np.argmin(self.pop_eval), :])
        self.best_eval = np.min(self.pop_eval)

    def update_bests(self):
        """
        Update the evaluation of the objective function after a context vector update.
        """
        self.pop_domain_check()
        self.pop.sort()
        self.pop_eval = parallel_eval(
            self.func, self.pop, processes=self.processes, chunksize=self.chunksize
        )
        self.fitness_functions += self.pop_size
        self.best_solution = np.copy(self.pop[np.argmin(self.pop_eval), :])
        self.best_eval = np.min(self.pop_eval)
