import math

import numpy as np

from pyfea.base_algos.traditional.fea_base_algo import FeaBaseAlgo
from pyfea.base_algos.traditional.pso import PSO


class FeaPso(PSO, FeaBaseAlgo):
    """
    The PSO algorithm adapted to be run in an FEA.
    """

    def update_bests(self):
        """
        Calls to update_bests() in PSO during the FEA's share step.
        """
        super().update_bests()

    def run(self):
        """
        Runs the base PSO algorithm.
        """
        return super().run()

    def get_solution_at_index(self, idx):
        """
        Find the gbest value at a given variable for FEA's compete step.
        @param idx: the index for the variable.
        """
        return self.gbest[idx]

    def update_worst(self, context):
        """
        The second half of the FEA's share step. Updates the worst
        particle to be positioned at the context vector.
        @param context: the FEA's context vector.
        """
        self.pop[np.argmax(self.pop_eval), :] = context

    @classmethod
    def from_kwargs(cls, function, domain, params):
        """
        The method for constructing a PSO from input to the FEA's constructor.
        @param function: the objective function for the PSO to minimize.
        Will be of the Function class from function.py.
        @param domain: the domain over which the function is evaluated.
        A numpy array of size (dim, 2).
        @param params: other keyword arguments as a dictionary. Includes
        generations, pop_size, phi_p, phi_g, and omega.
        """
        kwargs = {
            "generations": 100,
            "pop_size": 20,
            "phi_p": math.sqrt(2),
            "phi_g": math.sqrt(2),
            "omega": 1 / math.sqrt(2),
        }
        kwargs.update(params)
        return cls(
            function=function,
            domain=domain,
            generations=kwargs["generations"],
            pop_size=kwargs["pop_size"],
            phi_p=kwargs["phi_p"],
            phi_g=kwargs["phi_g"],
            omega=kwargs["omega"],
        )

    def reset_fitness(self):
        """
        Reevaluate the fitness function over the entire population and update the fields accordingly.
        """
        self.pbest = self.pop
        self.pop_eval = [self.func(self.pop[i, :]) for i in range(self.pop_size)]
        self.fitness_functions += self.pop_size
        self.pbest_eval = self.pop_eval
        # self.worst = np.argmax(self.pop_eval)
        self.gbest_eval = np.min(self.pbest_eval)
        self.gbest = np.copy(self.pbest[np.argmin(self.pbest_eval), :])

    def reinitialize_population(self):
        self.pop = np.where(
            self.domain[:, 0] > self.pop,
            self.domain[:, 0] + (self.domain[:, 1] - self.domain[:, 0]) * np.random.random(),
            self.pop,
        )
        self.pop = np.where(
            self.domain[:, 1] < self.pop,
            self.domain[:, 0] + (self.domain[:, 1] - self.domain[:, 0]) * np.random.random(),
            self.pop,
        )
        # for particle in range(self.pop_size):
        #    for p in range(len(self.pop[0])):
        #        if self.pop[particle, p] < self.domain[: 0].all() or self.pop[particle, p] > self.domain[: 1].all():
        #            self.pop[particle, p] = self.domain[0, 0] + (self.domain[0, 1] - self.domain[0, 0]) * np.random.random()

    def base_reset(self):
        """
        Reset the algorithm in preparation for another run.
        """
        self.reinitialize_population()
        self.velocities = super().init_velocities()
        self.reset_fitness()
