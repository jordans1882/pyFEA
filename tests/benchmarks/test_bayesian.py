import time
import pytest
from pymoo.problems.single import Rastrigin
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from bayes_opt import BayesianOptimization
from skopt import gp_minimize
numOfKnots = 1000

rastrigin = Rastrigin()

# Comment this next line to remove benchmark from test suite
# @pytest.mark.skip(reason="Done benchmarking")
@pytest.mark.benchmark(
    group="random",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_random_python_maximize(benchmark):
    def bayInput(w, c1, c2):
        pso = PSO(w = w, c1 = c1, c2 = c2)
        psoRes = minimize(rastrigin, pso)
        minVal = psoRes.F
        return -(minVal[0])
    #pbounds = {"w": (0, 1), "c1": (0, 1), "c2": (0, 1)}
    pbounds = {"w": (0.0, 1.0), "c1": (0.0, 1.0), "c2":(0.0, 1.0)}
    obj = BayesianOptimization(bayInput, pbounds)
    def random2():
        obj.maximize()
    benchmark(random2)
    assert(0)==0
    
@pytest.mark.benchmark(
group="random",
min_time=0.1,
max_time=0.5,
min_rounds=5,
timer=time.time,
disable_gc=True,
warmup=False,
)
def test_random_python_bayesian_bulid(benchmark):
    def bayInput(w, c1, c2):
        pso = PSO(w = w, c1 = c1, c2 = c2)
        psoRes = minimize(rastrigin, pso)
        minVal = psoRes.F
        return -(minVal[0])
    #pbounds = {"w": (0, 1), "c1": (0, 1), "c2": (0, 1)}
    pbounds = {"w": (0.0, 1.0), "c1": (0.0, 1.0), "c2":(0.0, 1.0)}
    def random2():
        obj = BayesianOptimization(bayInput, pbounds)
    benchmark(random2)
    assert(0)==0
@pytest.mark.benchmark(
group="random",
min_time=0.1,
max_time=0.5,
min_rounds=5,
timer=time.time,
disable_gc=True,
warmup=False,
)
def test_random_python_pso_minimize(benchmark):
    pso = PSO()
    def random2():    
        psoRes = minimize(rastrigin, pso)
    benchmark(random2)
    assert(0)==0
@pytest.mark.benchmark(
group="random",
min_time=0.1,
max_time=0.5,
min_rounds=5,
timer=time.time,
disable_gc=True,
warmup=False,
)
def test_random_python_pso_build(benchmark):
    def random2():
        pso = PSO()
    benchmark(random2)
    assert(0)==0

@pytest.mark.benchmark(
group="random",
min_time=0.1,
max_time=0.5,
min_rounds=5,
timer=time.time,
disable_gc=True,
warmup=False,
)
def test_random_python_askandtell_bayes(benchmark):
    def ask_and_tell_PSO(algorithm, problem):
        algorithm.setup(problem, termination=('n_gen', 25), seed=1, verbose=False)
        while algorithm.has_next():
        # ask the algorithm for the next solution to be evaluated
            pop = algorithm.ask()
        # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
            algorithm.evaluator.eval(problem, pop)
        # returned the evaluated individuals which have been evaluated or even modified
            algorithm.tell(infills=pop)
            # do same more things, printing, logging, storing or even modifying the algorithm object
            #print(algorithm.n_gen, algorithm.evaluator.n_eval)
        # obtain the result objective from the algorithm
        return algorithm.result()

    # Bayesian
    def bayInput(w, c1, c2):
        pso = PSO(w = w, c1 = c1, c2 = c2)
        #psoRes = minimize(rastrigin, pso)
        psoRes = ask_and_tell_PSO(pso, rastrigin)
        minVal = psoRes.F
        return -(minVal[0])

    #pbounds = {"w": (0, 1), "c1": (0, 1), "c2": (0, 1)}
    pbounds = {"w": (0.0, 1.0), "c1": (0.0, 1.0), "c2":(0.0, 1.0)}
    def random2():
        obj = BayesianOptimization(bayInput, pbounds)
        obj.maximize()
    benchmark(random2)
    
@pytest.mark.benchmark(
group="random",
min_time=0.1,
max_time=0.5,
min_rounds=5,
timer=time.time,
disable_gc=True,
warmup=False,
)
def test_random_python_pso_minimize(benchmark):
    pso = PSO()
    def random2():    
        psoRes = minimize(rastrigin, pso)
    benchmark(random2)
    assert(0)==0
@pytest.mark.benchmark(
group="random",
min_time=0.1,
max_time=0.5,
min_rounds=5,
timer=time.time,
disable_gc=True,
warmup=False,
)
def test_random_python_pso_build(benchmark):
    def random2():
        pso = PSO()
    benchmark(random2)
    assert(0)==0

@pytest.mark.benchmark(
group="random",
min_time=0.1,
max_time=0.5,
min_rounds=5,
timer=time.time,
disable_gc=True,
warmup=False,
)
def test_random_python_askandtell_bayeInput(benchmark):
    def ask_and_tell_PSO(algorithm, problem):
        algorithm.setup(problem, termination=('n_gen', 25), seed=1, verbose=False)
        while algorithm.has_next():
        # ask the algorithm for the next solution to be evaluated
            pop = algorithm.ask()
        # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
            algorithm.evaluator.eval(problem, pop)
        # returned the evaluated individuals which have been evaluated or even modified
            algorithm.tell(infills=pop)
            # do same more things, printing, logging, storing or even modifying the algorithm object
            #print(algorithm.n_gen, algorithm.evaluator.n_eval)
        # obtain the result objective from the algorithm
        return algorithm.result()

    def bayInput():
        pso = PSO()
        #psoRes = minimize(rastrigin, pso)
        psoRes = ask_and_tell_PSO(pso, rastrigin)
        minVal = psoRes.F
        return -(minVal[0])
    benchmark(bayInput)
