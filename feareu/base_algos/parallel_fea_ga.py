from feareu.base_algos import FeaGA, parallel_eval
import math
import numpy as np
import random

class ParallelFeaGA(FeaGA):

    def __init__( self,
        function,
        domain,
        pop_size = 20,
        b = 0.7,
        mutation_rate = 0.05,
        generations = 100,
        mutation_range = 0.5,
        processes = 4,
        chunksize = 4
    ):
        self.pop_size = pop_size
        self.b = b
        self.mutation_rate = mutation_rate
        self.func = function
        self.domain = domain
        self.processes = processes
        self.chunksize = chunksize
        self.pop = self.init_pop()
        self.ngenerations = 0
        self.pop_eval = parallel_eval(self.func, self.pop, processes=self.processes, chunksize=self.chunksize)
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
        self.bounds_check(children)
        #print("pre: ", self.pop)
        #print("size: ", np.size(self.pop))
        child_evals = parallel_eval(self.func, children, processes=self.processes, chunksize=self.chunksize)
        for child_idx, child in enumerate(children):
            inserted = False
            for i in range(len(self.pop_eval)):
                if(self.pop_eval[i]>child_evals[child_idx]):
                    self.pop_eval = np.insert(self.pop_eval, i, child_evals[child_idx])
                    self.pop = np.insert(self.pop, i, [child], axis=0)
                    inserted=True
                    break
            if(inserted is False):
                self.pop_eval = np.concatenate((self.pop_eval, [child_evals[child_idx]]))
                self.pop= np.concatenate((self.pop, [child]))
        #print("size: ", np.size(self.pop))

    def base_reset(self):
        """
        Reset the algorithm in preparation for another run.
        """
        self.pop = self.init_pop()
        self.pop_eval = parallel_eval(self.func, self.pop, processes=self.processes, chunksize=self.chunksize)
