import numpy as np

class DE:
    def __init__(
        self,
        function,
        domain,
        generations=100,
        pop_size=20,
        mutation_factor = 0.5,
        crossover_rate = 0.5
    ):
        self.generations = generations
        self.pop_size = pop_size
        self.func = function
        self.domain = domain
        self.pop = self.init_pop()
        self.pop_eval = [self.func(self.pop[i]) for i in range(self.pop_size)]
        self.best_eval = np.min(self.pop_eval)
        self.best_solution = np.copy(self.pop[np.argmin(self.pop_eval),:])
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate

    def init_pop(self):
        lbound = self.domain[:, 0]
        area = self.domain[:, 1] - self.domain[:, 0]
        return lbound + area * np.random.random(size=(self.pop_size, area.shape[0]))

    def run(self):
        for gen in range(self.generations):
            #print("generation: ", gen, "/", self.generations)
            mutant_pop = self.mutate()
            cross_pop = self.crossover(mutant_pop)
            self.fitness_evals(cross_pop)
        return  self.best_solution

    def mutate(self):
        mutant_pop = []
        for i in range(self.pop_size):
            idxs = np.random.choice(self.pop_size, size = 3, replace = False)
            new_pop_mem = self.pop[i,:]+self.mutation_factor*(self.pop[idxs[0],:] - self.pop[idxs[1],:] + self.pop[idxs[2],:])
            mutant_pop.append(new_pop_mem)
        return mutant_pop

    def crossover(self, mutant_pop):
        cross_pop = []
        for i in range(self.pop_size):
            check_crossing = np.random.rand()
            if check_crossing < self.crossover_rate:
                cross_idx = np.random.choice(np.delete(range(self.pop_size), i))
                cross_rand = np.round(np.random.uniform(size=self.domain.shape[0]))
                crossed_guy = np.where(cross_rand==1, self.pop[i,:], mutant_pop[cross_idx])
                cross_pop.append((i, crossed_guy))
        return cross_pop

    def fitness_evals(self, cross_pop):
        for fella in cross_pop:
            fella_eval = self.func(fella[1])
            if fella_eval < self.pop_eval[fella[0]]:
                self.pop_eval[fella[0]] = fella_eval
                self.pop[fella[0],:] = fella[1]
        self.best_solution = np.copy(self.pop[np.argmin(self.pop_eval),:])
        self.best_eval = np.min(self.pop_eval)
