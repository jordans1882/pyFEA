import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .parallel_evaluation import parallel_eval


class GA:
    def __init__(
        self,
        function,
        domain,
        pop_size=20,
        mutation_rate=0.05,
        generations=100,
        mutation_range=0.5,
        tournament_options=2,
        percent_children=0.1,
        fitness_terminate=False
    ):
        """
        @param function: the objective function to be minimized.
        @param domain: the domain on which we explore the function stored as a (dim,2) matrix,
        where dim is the number of dimensions we evaluate the function over.
        @param generations: the number of generations run before the algorithm terminates.
        @param pop_size: the number of individuals in the population.
        @param mutation_rate: the probability of mutation used in the mutation step.
        @param b: factors that survive to be parents each generation.
        """
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.func = function
        self.domain = domain
        self.pop = self._init_pop()
        self.ngenerations = 0
        self.tournament_options = tournament_options
        self.number_of_children = (1+percent_children)*pop_size
        self.generations = generations
        self.mutation_range = mutation_range
        self.average_pop_eval = []
        self.average_pop_variance = []
        self.fitness_list = []
        self.best_answers = []
        self.nfitness_evals = self.pop_size
        self.pop_eval = [sys.float_info.max] * self.pop_size
        self.fitness_terminate=fitness_terminate

    def _eval_pop(self, parallel=False, processes=4, chunksize=4):
        if not parallel:
            self.pop_eval = [self.func(self.pop[i, :]) for i in range(self.pop_size)]
            self.nfitness_evals += self.pop_size
        else:
            self.pop_eval = parallel_eval(
                self.func, self.pop, processes=processes, chunksize=chunksize
            )
            self.nfitness_evals += self.pop_size

    def _initialize(self, parallel=False, processes=4, chunksize=4):
        self._eval_pop(parallel=False, processes=4, chunksize=4)
        self.update_bests()

    def _init_pop(self):
        """
        Randomly initializes the values of the population as a numpy array of shape (pop_size, dim).
        """
        lbound = self.domain[:, 0]
        area = self.domain[:, 1] - self.domain[:, 0]
        return lbound + area * np.random.random(size=(self.pop_size, area.shape[0]))

    def run(self, progress=False, parallel=False, processes=4, chunksize=4):
        """
        Run the minimization algorithm.
        """
        self._initialize(parallel, processes, chunksize)
        if self.fitness_terminate:
            while self.nfitness_evals < self.generations:
                self.ngenerations += 1
                children = self._crossover()
                self._mutation(children, parallel, processes, chunksize)
                self.update_bests()
                self._track_vals()
        else:
            for gen in tqdm(range(self.generations), disable=(not progress)):
                self.ngenerations += 1
                children = self._crossover()
                self._mutation(children, parallel, processes, chunksize)
                self.update_bests()
                self._track_vals()

    def get_soln(self):
        return self.best_position

    def get_soln_fitness(self):
        return self.best_eval

    """def selection(self):
        
        Removes the poorest preforming (1-b)% of the population
        
        
        part_to_be_deleted = np.arange(start = self.b*self.pop_size, stop = self.pop_size, dtype=int)
        self.pop = np.delete(self.pop, part_to_be_deleted, axis=0)
        self.pop_eval = np.delete(self.pop_eval, part_to_be_deleted, axis=0)"""

    def _crossover(self):
        """
        Returns an array of new values from combinations of the existing population.
        """
        children = []
        for c in range(self.number_of_children):
            winner1 = 0
            current_winner1 = np.Infinity
            for i in range(self.tournament_options):
                rand_pop_num = int(random.random() * self.pop_size)
                if self.pop_eval[rand_pop_num] < current_winner1:
                    current_winner1 = self.pop_eval[rand_pop_num]
                    winner1 = rand_pop_num
            winner2 = 0
            current_winner2 = np.Infinity
            for i in range(self.tournament_options):
                rand_pop_num = int(random.random() * self.pop_size)
                if self.pop_eval[rand_pop_num] < current_winner2:
                    current_winner2 = self.pop_eval[rand_pop_num]
                    winner2 = rand_pop_num

            cross_rand = np.round(
                np.random.choice(
                    [0, 1],
                    size=self.domain.shape[0],
                )
            )
            crossed_guy = np.where(cross_rand == 1, self.pop[winner1], self.pop[winner2])
            children.append(crossed_guy)
        return np.array(children)

    def _mutation(self, children, parallel=False, processes=4, chunksize=4):
        """
        Mutates children through swapping and recombines that with the parent population.
        """
        for child in children:
            for i in range(len(child)):
                if random.random() < self.mutation_rate:
                    rand_value = random.uniform(-1 * self.mutation_range, self.mutation_range)
                    child[i] += rand_value
        self._bounds_check(children)
        if not parallel:
            child_evals = []
            for child in children:
                child_evals.append(self.func(child))
                self.nfitness_evals += 1
        else:
            child_evals = parallel_eval(self.func, children, processes, chunksize)
            self.nfitness_evals += children.shape[0]
        for i, child in enumerate(children):
            index = np.argmax(self.pop_eval)
            if self.pop_eval[index] > child_evals[i]:
                self.pop_eval[index] = child_evals[i]
                self.pop[index,:] = child

    def update_bests(self):
        """
        Resorts the population and updates the evaluations.
        """
        self.best_eval = np.min(self.pop_eval)
        self.best_position = np.copy(self.pop[np.argmin(self.pop_eval), :])

    def _bounds_check(self, children):
        children = np.where(self.domain[:, 0] > children, self.domain[:, 0], children)
        children = np.where(self.domain[:, 1] < children, self.domain[:, 1], children)

    def _track_vals(self):
        self.average_pop_eval.append(np.average(self.pop_eval))
        self.average_pop_variance.append(np.average(np.var(self.pop, axis=0)))
        self.fitness_list.append(self.nfitness_evals)
        self.best_answers.append(self.best_eval)

    def diagnostic_plots(self):
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

        ax1.plot(self.fitness_list, self.average_pop_variance)
        ax1.set_xlabel("fitness evaluations", fontsize=10)
        ax1.set_ylabel("population variance", fontsize=10)
        ax1.set_title("Population Diversity", fontsize=10)

        ax2.plot(self.fitness_list, self.average_pop_eval)
        ax2.set_xlabel("fitness evaluations", fontsize=10)
        ax2.set_ylabel("average MSE", fontsize=10)
        ax2.set_title("Average Solution Fitness", fontsize=10)

        ax3.plot(self.fitness_list, self.best_answers)
        ax3.set_xlabel("fitness evaluations", fontsize=10)
        ax3.set_ylabel("best MSE", fontsize=10)
        ax3.set_title("Best Solution Fitness", fontsize=10)

        fig.suptitle("GA")
        fig.tight_layout()
