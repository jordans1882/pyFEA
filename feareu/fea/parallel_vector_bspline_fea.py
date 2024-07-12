from feareu.fea import VectorComparisonBsplineFEA
import matplotlib.pyplot as plt
import numpy as np
from feareu.function import Function
from multiprocessing import Pool, Process, Queue
#from multiprocessing.sharedctypes import Array

class ParallelVectorBsplineFEA(VectorComparisonBsplineFEA):
    """Factored Evolutionary Architecture, implemented based on the 2017 paper by Strasser et al.
    Altered so that each factor has its own domain based on the context vector.
    Intended for use in BSpline knot selection problem."""
    def __init__(self, factors, early_stop, function, true_error, delta, dim, base_algo_name, domain, diagnostics_amount, og_knot_points, process_count, **kwargs):
        """
        @param factors: list of lists, contains the dimensions that each factor of the architecture optimizes over.
        @param function: the objective function that the FEA minimizes.
        @param iterations: the number of times that the FEA runs.
        @param dim: the number of dimensions our function optimizes over.
        @param base_algo_name: the base algorithm class that we optomize over. Should be a subclass of FeaBaseAlgo.
        @param domain: the minimum and maximum possible values of our domain for any variable of the context vector. It should be a tuple of size 2.
        @param **kwargs: parameters for the base algorithm.
        """
        self.process_count = process_count
        self.full_fit_func = 0
        super().__init__(factors, early_stop, function, true_error, delta, dim, base_algo_name, domain, diagnostics_amount, og_knot_points, **kwargs)
        self.subpop_domains = []
        
    def update_plots(self, subpopulations):
        self.convergences.append(self.function(self.context_variable))
        self.full_fit_func_array.append(self.full_fit_func)
        self.dif_to_og.append(np.linalg.norm(self.og_knot_points - self.context_variable))
        tot_part_fit = 0
        for i in range(len(subpopulations)):
            tot_part_fit += subpopulations[i].fitness_functions
        self.part_fit_func_array.append(tot_part_fit)
    
    def run(self):
        """
        Algorithm 3 from the Strasser et al. paper, altered to sort the context
        vector on initialization.
        """
        self.context_variable = self.init_full_global()
        self.context_variable.sort()
        self.domain_evaluation()
        parallel_i = 0
        subpopulations = {}
        while parallel_i<len(self.factors):
            processes = []
            result_queue = Queue()
            for j in range(int(self.process_count)):
                p = Process(target=self.initialize_subpop, args=(parallel_i, result_queue))
                processes.append(p)
                p.start()
                parallel_i+=1
                if(parallel_i>=(len(self.factors))):
                    break
            for p in processes:
                result = result_queue.get()
                subpopulations.update({result[0]: result[1]})
            for p in processes:
                p.join()
        while self.stopping_point < self.function(self.context_variable):
            self.niterations += 1
            parallel_i = 0
            while parallel_i<len(subpopulations):
                processes = []
                result_queue = Queue()
                for j in range(int(self.process_count)):
                    p = Process(target=self.subpop_compute, args=(parallel_i, subpopulations[parallel_i], result_queue))
                    processes.append(p)
                    p.start()
                    parallel_i+=1
                    if(parallel_i>=len(subpopulations)):
                        break
                for p in processes:
                    result = result_queue.get()
                    subpopulations[result[0]]=result[1]
                for p in processes:
                    p.join()
            self.compete(subpopulations)
            self.share(subpopulations)
            if self.niterations % self.diagnostic_amount == 0:
                self.update_plots(subpopulations)
            if self.niterations > self.early_stop:
                print("FEA ", self.base_algo, " early_stopped")
                break
            print("current func eval: ", self.function(self.context_variable))
            print("full fit func: ", self.full_fit_func)
            print("part fit func: ", self.part_fit_func_array[len(self.part_fit_func_array)-1])
            print("iters: ", self.niterations)
        return self.function(self.context_variable)
        
    def subpop_compute(self, parallel_i, subpop, result_queue):
        subpop.base_reset()
        subpop.run()
        result_queue.put([parallel_i, subpop])
        
    def domain_evaluation(self):
        """
        Ensures that each factor has its own domain in which its variables can move.
        """
        self.subpop_domains = []
        for i, factor in enumerate(self.factors):
            fact_dom = np.zeros((len(factor),2))
            if factor[0] == 0:
                fact_dom[:,0] = self.domain[0]
            else:
                fact_dom[:,0] = self.context_variable[factor[0]-1]
            if factor[-1] == len(self.context_variable)-1:
                fact_dom[:,1] = self.domain[1]
            else:
                fact_dom[:,1] = self.context_variable[factor[-1]+1]
            self.subpop_domains.append(fact_dom)

    def compete(self, subpopulations):
        """
        Algorithm 1 from the Strasser et al. paper, altered to sort the context vector
        when updated.
        @param subpopulations: the list of base algorithms, each with their own factor.
        """
        cont_var = self.context_variable
        rand_var_permutation = np.random.permutation(self.dim)
        for i in rand_var_permutation:
            overlapping_factors = self.variable_map[i]
            best_val = np.copy(cont_var[i])
            temp_cont_var = np.copy(cont_var)
            temp_cont_var.sort()
            best_fit = self.function(temp_cont_var)
            self.full_fit_func += 1
            rand_pop_permutation = np.random.permutation(len(overlapping_factors))
            for j in rand_pop_permutation:
                s_j = overlapping_factors[j]
                index = np.where(self.factors[s_j] == i)[0][0]
                cont_var[i]=np.copy(subpopulations[s_j].get_solution_at_index(index))
                temp_cont_var = np.copy(cont_var)
                temp_cont_var.sort()
                current_fit = self.function(temp_cont_var)
                self.full_fit_func +=1
                if current_fit < best_fit:
                    best_val = np.copy(subpopulations[s_j].get_solution_at_index(index))
                    best_fit = current_fit
            cont_var[i] = np.copy(best_val)
        self.context_variable = (cont_var)
        self.context_variable.sort()
    
    def share(self, subpopulations):
        """
        Algorithm 2 from the Strasser et al. paper.
        @param subpopulations: the list of subpopulations initialized in initialize_subpops. 
        """
        for i in range(len(subpopulations)):
            subpopulations[i].domain = self.subpop_domains[i]
            subpopulations[i].func.context = np.copy(self.context_variable)
            subpopulations[i].update_bests()
        
    def initialize_subpop(self, i, result_queue):
        """
        Initializes some inheritor of FeaBaseAlgo to optimize over each factor.
        Slightly altered to call domain differently.
        @param subpop_domains: the domains from domain_restriction.
        """
        fun = Function(context=self.context_variable, function=self.function, factor=self.factors[i])
        result_queue.put([i, self.base_algo.from_kwargs(fun, self.subpop_domains[i], self.base_algo_args)])
