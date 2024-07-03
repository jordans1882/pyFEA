from feareu.base_algos import BsplineFeaGA, parallel_eval
import math
import numpy as np
import random

class ParallelBsplineFeaGA(BsplineFeaGA):

    def __init__( self,
        function,
        domain,
        pop_size = 20,
        mutation_rate = 0.05,
        generations = 100,
        mutation_range = 0.5,
        tournament_options = 2,
        number_of_children = 2,
        processes = 4,
        chunksize = 4
    ):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.func = function
        self.domain = domain
        self.tournament_options = tournament_options
        self.number_of_children = number_of_children
        self.processes = processes
        self.chunksize = chunksize
        self.pop = self.init_pop()
        self.ngenerations = 0
        self.pop_eval = parallel_eval(self.func, self.pop, processes=self.processes, chunksize=self.chunksize)
        self.fitness_functions = pop_size
        self.update_bests()
        self.generations = generations
        self.mutation_range = mutation_range
        self.average_pop_eval = []
        self.average_pop_variance = []

    def mutation(self, children):
        """
        Mutates children through swapping and recombines that with the parent population.
        """
        for child in children:
            for i in range(len(child)):
                if random.random() < self.mutation_rate:
                    rand_value = random.uniform(-1*self.mutation_range, self.mutation_range)
                    child[i] += rand_value
        children = self.bounds_check(children)
        children.sort()
        child_evals = parallel_eval(self.func, children, processes=self.processes, chunksize=self.chunksize)
        self.pop_eval = np.concatenate((self.pop_eval, child_evals)]))
        self.fitness_functions+=children.shape[0]
        self.pop= np.concatenate((self.pop, children))

    def base_reset(self):
        """
        Reset the algorithm in preparation for another run.
        """
        self.reinitialize_population()
        self.pop.sort()
        self.pop_eval = parallel_eval(self.func, self.pop, processes=self.processes, chunksize=self.chunksize)
        self.fitness_functions+= self.pop_size
