import numpy as np
from feareu.base_algos.ga import FeaGA

class BsplineFeaGA(FeaGA):

    def init_pop(self):
        """
        Initialize random particles.
        """
        lbound = self.domain[:, 0]
        area = self.domain[:, 1] - self.domain[:, 0]
        pop = lbound + area * np.random.random(size=(self.pop_size, area.shape[0]))
        pop.sort()
        return pop

    def bounds_check(self, children):
        """
        Ensure that the particles don't move outside the domain of our function by projecting
        them to the domain's boundary at that point.
        """
        area = self.domain[:, 1] - self.domain[:, 0]
        children = np.where(
            self.domain[:, 0] > children,
            self.domain[:, 0] + 0.1 * area * np.random.random(),
            children,
        )
        children = np.where(
            self.domain[:, 1] < children,
            self.domain[:, 1] - 0.1 * area * np.random.random(),
            children,
        )

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
        children.sort()
        for child in children:
                self.pop_eval = np.concatenate((self.pop_eval, [self.func(child)]))
                self.fitness_functions+=1
                self.pop= np.concatenate((self.pop, [child]))

