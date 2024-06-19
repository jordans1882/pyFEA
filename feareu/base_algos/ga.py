import random
import numpy as np
import matplotlib.pyplot as plt

class GA:
    def __init__( self,
        function,
        domain,
        pop_size = 20,
        b = 0.7,
        mutation_rate = 0.05,
        generations = 100,
        mutation_range = 0.5
    ):
        self.pop_size = pop_size
        self.b = b
        self.mutation_rate = mutation_rate
        self.func = function
        self.domain = domain
        self.pop = self.init_pop()
        self.ngenerations = 0
        self.pop_eval = [self.func(self.pop[i, :]) for i in range(self.pop_size)]
        self.update_bests()
        self.generations = generations
        self.mutation_range = mutation_range
        self.average_pop_eval = []
        self.average_pop_variance = []
        
        """
        @param function: the objective function to be minimized.
        @param domain: the domain on which we explore the function stored as a (dim,2) matrix,
        where dim is the number of dimensions we evaluate the function over.
        @param generations: the number of generations run before the algorithm terminates.
        @param pop_size: the number of individuals in the population.
        @param mutation_rate: the probability of mutation used in the mutation step.
        @param b: factors that survive to be parents each generation.
        """
        
    def init_pop(self):
        """
        Randomly initializes the values of the population as a numpy array of shape (pop_size, dim).
        """
        lbound = self.domain[:, 0]
        area = self.domain[:, 1] - self.domain[:, 0]
        return lbound + area * np.random.random(size=(self.pop_size, area.shape[0]))
    
    def run(self):
        """
        Run the minimization algorithm.
        """
        for gen in range(self.generations):
            self.ngenerations +=1
            self.selection()
            children = self.crossover()
            self.mutation(children)
            self._append_varaince()
            self.update_bests()
            self._append_avg_evals()
        return self.pop[0]
            
    def selection(self):
        """
        Removes the poorest preforming (1-b)% of the population
        """
        part_to_be_deleted = np.arange(start = self.b*self.pop_size, stop = self.pop_size, dtype=int)
        self.pop = np.delete(self.pop, part_to_be_deleted, axis=0)
        self.pop_eval = np.delete(self.pop_eval, part_to_be_deleted, axis=0)
    
    def crossover(self):
        """
        Returns an array of new values from combinations of the existing population.
        """
        num_elements_to_be_added = self.pop_size - self.pop.shape[0]
        last_gen_pop = self.pop.shape[0] - 1
        children = np.zeros([num_elements_to_be_added, self.pop.shape[1]])
        for i in range(num_elements_to_be_added):
            parent1_index = random.randint(0, last_gen_pop)
            parent2_index = random.randint(0, last_gen_pop)
            new_point = (self.pop[parent1_index] + self.pop[parent2_index])/2
            children[i] = (new_point)
        return children
    
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
        for child in children:
            inserted = False
            ind_eval = self.func(child)
            for i in range(len(self.pop_eval)):
                if(self.pop_eval[i]>ind_eval):
                    self.pop_eval = np.insert(self.pop_eval, i, ind_eval)
                    self.pop = np.insert(self.pop, i, [child], axis=0)
                    inserted=True
                    break
            if(inserted is False):
                self.pop_eval = np.concatenate((self.pop_eval, [ind_eval]))
                self.pop= np.concatenate((self.pop, [child]))
        #print("size: ", np.size(self.pop))
    
    def update_bests(self):
        """
        Resorts the population and updates the evaluations.
        """
        """sorted_order = np.argsort([self.func(row) for row in self.pop])
        self.pop = self.pop[sorted_order]
        self.pop_eval = [self.func(self.pop[i, :]) for i in range(self.pop_size)]"""
        self.best_position = self.pop[0]
        self.best_eval = self.pop_eval[0]
        
    def bounds_check(self, children):
        children = np.where(self.domain[:, 0] > children, self.domain[:, 0], children)
        children = np.where(self.domain[:, 1] < children, self.domain[:, 1], children)
        
    def _append_avg_evals(self):
        self.average_pop_eval.append(np.average(self.pop_eval))
    
    def _append_varaince(self):
        self.average_pop_variance.append(np.average(np.var(self.pop, axis = 0)))
        
    def diagnostic_plots(self):
        plt.subplot(1, 2, 1)
        ret = plt.plot(range(0, self.ngenerations), self.average_pop_eval)
        plt.title("Average pop evals")

        plt.subplot(1, 2, 2)
        plt.plot(range(0, self.ngenerations), self.average_pop_variance)
        plt.title("Average Pop")
        
        plt.tight_layout()
        
        return ret
