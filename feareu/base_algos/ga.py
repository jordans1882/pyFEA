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
    ):
        self.pop_size = pop_size
        self.b = b
        self.mutation_rate = mutation_rate
        self.func = function
        self.domain = domain
        self.pop = self.init_pop()
        self.ngenerations = 0
        self.pop_eval = [self.func(self.pop[i, :]) for i in range(self.pop_size)]
        self.generations = generations
        self.average_pop_eval = []
        self.average_pop_variance = []
        
    def init_pop(self):
        lbound = self.domain[:, 0]
        area = self.domain[:, 1] - self.domain[:, 0]
        return lbound + area * np.random.random(size=(self.pop_size, area.shape[0]))
    
    def run(self):
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
        part_to_be_deleted = np.arange(start = self.b*self.pop_size, stop = self.pop_size, dtype=int)
        self.pop = np.delete(self.pop, part_to_be_deleted, axis=0)
    
    def crossover(self):
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
        for child in children:
            if random.random() < self.mutation_rate:
                index_1 = random.randint(0, child.shape[0]-1)
                c1_og = child[index_1]
                index_2 = random.randint(0, child.shape[0]-1)
                c2_og = child[index_2]
                child[index_1] = c2_og
                child[index_2] = c1_og
        self.pop = np.vstack((self.pop, children))
    
    def update_bests(self):
        #FIX SORTING
        sorted_order = np.argsort([self.func(row) for row in self.pop])
        self.pop = self.pop[sorted_order]
        self.pop_eval = [self.func(self.pop[i, :]) for i in range(self.pop_size)]

    """def _append_avg_velocities(self):
        self.average_velocities.append(np.average(np.abs(self.velocities)))
"""
    def _append_avg_evals(self):
        self.average_pop_eval.append(np.average(self.pop_eval))
        
    # FIX PLEASE
    def _append_varaince(self):
        self.average_pop_variance.append(np.average(np.var(self.pop, axis = 0)))
        
    def diagnostic_plots(self):
        plt.subplot(1, 2, 1)
        ret = plt.plot(range(0, self.ngenerations), self.average_pop_eval)
        plt.title("Average pop evals")

        plt.subplot(1, 2, 2)
        plt.plot(range(0, self.ngenerations), self.average_pop_variance)
        plt.title("Average Pop")

        """plt.subplot(1, 3, 3)
        plt.plot(range(0, self.ngenerations), self.average_velocities)
        plt.title("Average Velocities")"""
        plt.tight_layout()
        
        return ret