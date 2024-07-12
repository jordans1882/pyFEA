from feareu.base_algos.bspline_specific.bspline_fea_de import BsplineFeaDE
import matplotlib.pyplot as plt
import math
import numpy as np
class KnownKnotBsplineFeaDE(BsplineFeaDE):
    def __init__(self, function, early_stop, domain, delta, true_error, og_knot_points, pop_size=20, mutation_factor = 0.5, crossover_rate = 0.9):
        super().__init__(function=function, domain=domain, pop_size=pop_size, mutation_factor=mutation_factor, crossover_rate=crossover_rate)
        self.stopping_point = true_error + delta
        self.early_stop = early_stop
        self.og_knot_points = og_knot_points
        self.dif_from_og = []
    def run(self):
        """
        Run the minimization algorithm.
        """
        while self.stopping_point < self.best_eval:
            #print("generation: ", gen, "/", self.generations)
            self.ngenerations += 1
            self.mutate()
            self.stay_in_domain()
            self.crossover()
            self.selection()
            self._track_vals()
            if self.ngenerations%5==0:
                print("gen: ", self.ngenerations)
                print("best eval: ", self.best_eval)
            if self.ngenerations > self.early_stop:
                print("DE early_stopped")
                break
        return  self.best_eval
    def _track_vals(self):
        self.average_pop_eval.append(np.average(self.pop_eval))
        self.average_pop_variance.append(np.average(np.var(self.pop, axis = 0)))
        self.fitness_list.append(self.fitness_functions)
        self.best_answers.append(self.best_eval)
        sum = 0
        for i in range(len(self.og_knot_points)):
            sum += abs(self.og_knot_points[i] - self.best_solution[i])
        self.dif_from_og.append(sum)
    def diagnostic_plots(self):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4)

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
        
        ax4.plot(self.fitness_list, self.dif_from_og)
        ax4.set_xlabel('# fitness evaluations', fontsize=10)
        ax4.set_ylabel('Difference from the Original Vector', fontsize=10)
        ax4.set_title('Best Solution Fitness', fontsize=10)

        fig.suptitle("DE")
        fig.tight_layout()