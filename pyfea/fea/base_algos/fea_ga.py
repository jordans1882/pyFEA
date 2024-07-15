import math

import numpy as np

from pyfea.base_algos.ga import GA
from pyfea.fea.base_algos import FeaBaseAlgo


class FeaGA(GA, FeaBaseAlgo):
    """
    The Genetic Algorithm set up for use in an FEA.
    See the base Genetic Algorithm in ga.py for the full functionality.
    """

    def get_solution_at_index(self, idx):
        """
        The method to retrieve a variable's value from the base algorithm.
        @param idx: the index of the variable to be retrieved.
        """
        return self.best_position[idx]

    def reinitialize_population(self):
        for particle in range(self.pop_size):
            for p in range(len(self.pop[0])):
                if (
                    self.pop[particle, p] < self.domain[:0].all()
                    or self.pop[particle, p] > self.domain[:1].all()
                ):
                    self.pop[particle, p] = (
                        self.domain[0, 0]
                        + (self.domain[0, 1] - self.domain[0, 0]) * np.random.random()
                    )

    def update_worst(self, context):
        """
        The second half of the share step in the FEA algorithm, where we update
        the worst individual in our population to become the context vector.
        @param context: the context vector of our FEA.
        """
        self.pop[-1, :] = context

    @classmethod
    def from_kwargs(cls, function, domain, params):
        """
        The method for inputting parameters to the DE from FEA.
        @param function: the objective function to minimize.
        @param domain: the domain over which to minimize as a numpy array of size (dim, 2)
        @param params: the remaining parameters, namely generations, pop_size, mutation_rate,
        and b. These are taken in as a dictionary here from the keyword arguments
        passed to FEA's constructor.
        """
        kwargs = {
            "generations": 100,
            "pop_size": 20,
            "mutation_rate": 0.05,
            "mutation_range": 0.5,
            "tournament_options": 2,
            "number_of_children": 2,
        }
        kwargs.update(params)
        return cls(
            function=function,
            domain=domain,
            generations=kwargs["generations"],
            pop_size=kwargs["pop_size"],
            mutation_rate=kwargs["mutation_rate"],
            mutation_range=kwargs["mutation_range"],
            tournament_options=kwargs["tournament_options"],
            number_of_children=kwargs["number_of_children"],
        )

    def base_reset(self, parallel=False, processes=4, chunksize=4):
        """
        Reset the algorithm in preparation for another run.
        """
        self.reinitialize_population()
        if not parallel:
            self.pop_eval = [self.func(self.pop[i, :]) for i in range(self.pop_size)]
        else:
            self.pop_eval = parallel_eval(self.func, self.pop, processes, chunksize)
        self.fitness_functions += self.pop_size
