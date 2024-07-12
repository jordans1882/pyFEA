import random
import numpy as np
import matplotlib.pyplot as plt

class GA:
    def __init__( self,
        function,
        domain,
        pop_size = 20,
        mutation_rate = 0.05,
        generations = 100,
        mutation_range = 0.5,
        tournament_options = 2,
        number_of_children = 2
    ):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.func = function
        self.domain = domain
        self.pop = self.init_pop()
        self.fitness_functions = 0
        self.ngenerations = 0
        self.tournament_options = tournament_options
        self.number_of_children = number_of_children
        self.pop_eval = [self.func(self.pop[i, :]) for i in range(self.pop_size)]
        self.fitness_functions = self.pop_size
        self.update_bests()
        self.generations = generations
        self.mutation_range = mutation_range
        self.average_pop_eval = []
        self.average_pop_variance = []
        self.fitness_list = []
        self.best_answers = []
        
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
            #self.selection()
            children = self.crossover()
            self.mutation(children)
            self.update_bests()
            self._track_vals()
        return self.best_eval
            
    """def selection(self):
        
        Removes the poorest preforming (1-b)% of the population
        
        
        part_to_be_deleted = np.arange(start = self.b*self.pop_size, stop = self.pop_size, dtype=int)
        self.pop = np.delete(self.pop, part_to_be_deleted, axis=0)
        self.pop_eval = np.delete(self.pop_eval, part_to_be_deleted, axis=0)"""
    
    def crossover(self):
        """
        Returns an array of new values from combinations of the existing population.
        """
    
        children = []
        for c in range(self.number_of_children):
            winner1 = 0
            current_winner1 = np.Infinity
            loser1 = 0
            current_loser1 = 0
            for i in range(self.tournament_options):
                rand_pop_num = int(random.random() * self.pop_size)
                if(self.pop_eval[rand_pop_num]<current_winner1):
                    current_winner1 = self.pop_eval[rand_pop_num]
                    winner1 = rand_pop_num
                if(self.pop_eval[rand_pop_num]>current_loser1):
                    current_loser1 = self.pop_eval[rand_pop_num]
                    loser1 = rand_pop_num
            winner2 = 0
            current_winner2 = np.Infinity
            for i in range(self.tournament_options):
                rand_pop_num = int(random.random() * self.pop_size)
                if(self.pop_eval[rand_pop_num]<current_winner2):
                    current_winner2 = self.pop_eval[rand_pop_num]
                    winner2 = rand_pop_num
                if(self.pop_eval[rand_pop_num]>current_loser1):
                    current_loser1 = self.pop_eval[rand_pop_num]
                    loser1 = rand_pop_num
            np.delete(self.pop, loser1)
            np.delete(self.pop_eval, loser1)
            new_point = []
            for i in range(len(self.pop[0])):
                pick_parent = int(random.random() * 2)
                if(int(pick_parent)==0):
                    new_point.append(self.pop[winner1, i])
                else:
                    new_point.append(self.pop[winner2, i])
            children.append(new_point)
        return np.array(children)
    
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
        for child in children:
                self.pop_eval = np.concatenate((self.pop_eval, [self.func(child)]))
                self.fitness_functions+=1
                self.pop= np.concatenate((self.pop, [child]))
    
    def update_bests(self):
        """
        Resorts the population and updates the evaluations.
        """
        self.best_eval = np.min(self.pop_eval)
        self.best_position = np.copy(self.pop[np.argmin(self.pop_eval), :])
        
    def bounds_check(self, children):
        children = np.where(self.domain[:, 0] > children, self.domain[:, 0], children)
        children = np.where(self.domain[:, 1] < children, self.domain[:, 1], children)

    def _track_vals(self):
        self.average_pop_eval.append(np.average(self.pop_eval))
        self.average_pop_variance.append(np.average(np.var(self.pop, axis = 0)))
        self.fitness_list.append(self.fitness_functions)
        self.best_answers.append(self.best_eval)
        
    def diagnostic_plots(self):
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

        ax1.plot(self.fitness_list, self.average_pop_variance)
        ax1.set_xlabel('# fitness evaluations', fontsize=10)
        ax1.set_ylabel('population variance', fontsize=10)
        ax1.set_title('Population Diversity', fontsize=10)

        ax2.plot(self.fitness_list, self.average_pop_eval)
        ax2.set_xlabel('# fitness evaluations', fontsize=10)
        ax2.set_ylabel('average MSE', fontsize=10)
        ax2.set_title('Average Solution Fitness', fontsize=10)

        ax3.plot(self.fitness_list, self.best_answers)
        ax3.set_xlabel('# fitness evaluations', fontsize=10)
        ax3.set_ylabel('best MSE', fontsize=10)
        ax3.set_title('Best Solution Fitness', fontsize=10)

        fig.suptitle("GA")
        fig.tight_layout()
