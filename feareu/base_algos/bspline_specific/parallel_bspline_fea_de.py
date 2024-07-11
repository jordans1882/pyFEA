import numpy as np
from feareu.base_algos.bspline_specific.bspline_fea_de import BsplineFeaDE
from feareu.base_algos import parallel_eval

class ParallelBsplineFeaDE(BsplineFeaDE):

    def __init__(
        self,
        function,
        domain,
        generations=100,
        pop_size=20,
        mutation_factor = 0.5,
        crossover_rate = 0.9,
        processes = 4,
        chunksize = 4,
        fitness_terminate = False
    ):
        """
        @param function: the objective function to be minimized.
        @param domain: the domain on which we explore the function stored as a (dim,2) matrix,
        where dim is the number of dimensions we evaluate the function over.
        @param generations: the number of generations run before the algorithm terminates.
        @param pop_size: the number of individuals in the population.
        @param mutation_factor: the scalar factor used in the mutation step.
        @param crossover_rate: the probability of taking a mutated value during the crossover step.
        """
        self.fitness_terminate = fitness_terminate
        self.generations = generations
        self.pop_size = pop_size
        self.func = function
        self.domain = domain
        self.processes = processes
        self.chunksize = chunksize
        self.pop = self.init_pop()
        self.pop_eval = parallel_eval(self.func, self.pop, processes=self.processes, chunksize=self.chunksize)
        self.fitness_functions = self.pop_size
        self.best_eval = np.min(self.pop_eval)
        self.best_solution = np.copy(self.pop[np.argmin(self.pop_eval),:])
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.mutant_pop = np.zeros((self.pop_size, self.domain.shape[0]))
        self.ngenerations = 0
        self.average_pop_variance = []
        self.average_pop_eval = []
        self.fitness_list = []
        self.best_answers = []

    def run(self):
        """
        Run the minimization algorithm.
        """
        if self.fitness_terminate:
            while self.fitness_functions < self.generations:
                self.ngenerations += 1
                self.mutate()
                self.stay_in_domain()
                self.crossover()
                self.selection()
                self._track_vals()
        else:
            super().run()
        return  self.best_eval

    def selection(self):
        """
        The fitness evaluation and selection. Greedily selects whether to keep or throw out a value.
        Consider implementing and testing more sophisticated selection algorithms.
        """
        self.pop.sort()
        self.mutant_pop.sort()
        mutant_pop_eval = parallel_eval(self.func, self.mutant_pop, processes=self.processes, chunksize=self.chunksize)
        self.fitness_functions+=self.pop_size
        for i in range(self.pop_size):
            fella_eval = mutant_pop_eval[i]
            if fella_eval < self.pop_eval[i]:
                self.pop_eval[i] = fella_eval
                self.pop[i,:] = np.copy(self.mutant_pop[i,:])
        self.best_solution = np.copy(self.pop[np.argmin(self.pop_eval),:])
        self.best_eval = np.min(self.pop_eval)

    def update_bests(self):
        """
        Update the evaluation of the objective function after a context vector update.
        """
        self.pop_domain_check()
        self.pop.sort()
        self.pop_eval = parallel_eval(self.func, self.pop, processes=self.processes, chunksize=self.chunksize)
        self.fitness_functions+= self.pop_size
        self.best_solution = np.copy(self.pop[np.argmin(self.pop_eval), :])
        self.best_eval = np.min(self.pop_eval)
