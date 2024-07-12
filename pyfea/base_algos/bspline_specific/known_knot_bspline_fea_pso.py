import math

import matplotlib.pyplot as plt
import numpy as np

from pyfea.base_algos.bspline_specific.bspline_fea_pso import BsplineFeaPSO


class KnownKnotBsplineFeaPSO(BsplineFeaPSO):
    def __init__(
        self,
        function,
        early_stop,
        domain,
        delta,
        true_error,
        og_knot_points,
        pop_size=20,
        phi_p=math.sqrt(2),
        phi_g=math.sqrt(2),
        omega=1 / math.sqrt(2),
    ):
        super().__init__(
            function=function,
            domain=domain,
            pop_size=pop_size,
            phi_p=phi_p,
            phi_g=phi_g,
            omega=omega,
        )
        self.stopping_point = true_error + delta
        self.early_stop = early_stop
        self.og_knot_points = og_knot_points
        self.dif_from_og = []

    def run(self):
        self._track_values()
        self.generations_passed += 1
        while self.stopping_point < self.gbest_eval:
            self.update_velocities()
            self.pop = self.pop + self.velocities
            self.stay_in_domain()
            self.update_bests()
            self._track_values()
            self.generations_passed += 1
            if self.generations_passed % 5 == 0:
                print("gen: ", self.generations_passed)
                print("best eval: ", self.gbest_eval)
            if self.generations_passed > self.early_stop:
                print("PSO early_stopped")
                break
        return self.gbest_eval

    def _track_values(self):
        """
        Tracks various diagnostic values over the course of the algorithm's run.
        """
        self.gbest_evals.append(self.gbest_eval)
        self.average_velocities.append(np.average(np.abs(self.velocities)))
        self.average_pop_eval.append(np.average(self.pop_eval))
        self.fitness_list.append(self.fitness_functions)
        sum = 0
        for i in range(len(self.og_knot_points)):
            sum += abs(self.og_knot_points[i] - self.gbest[i])
        self.dif_from_og.append(sum)

    def diagnostic_plots(self):
        """
        Plots the values tracked in _track_values().
        """
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4)

        ax1.plot(self.fitness_list, self.average_velocities)
        ax1.set_xlabel("# fitness evaluations", fontsize=10)
        ax1.set_ylabel("average velocities", fontsize=10)
        ax1.set_title("Population Diversity", fontsize=10)

        ax2.plot(self.fitness_list, self.average_pop_eval)
        ax2.set_xlabel("# fitness evaluations", fontsize=10)
        ax2.set_ylabel("average MSE", fontsize=10)
        ax2.set_title("Average Solution Fitness", fontsize=10)

        ax3.plot(self.fitness_list, self.gbest_evals)
        ax3.set_xlabel("# fitness evaluations", fontsize=10)
        ax3.set_ylabel("gbest MSE", fontsize=10)
        ax3.set_title("Best Solution Fitness", fontsize=10)

        ax4.plot(self.fitness_list, self.dif_from_og)
        ax4.set_xlabel("# fitness evaluations", fontsize=10)
        ax4.set_ylabel("Difference from the Original Vector", fontsize=10)
        ax4.set_title("Best Solution Fitness", fontsize=10)

        fig.suptitle("PSO")
        fig.tight_layout()

