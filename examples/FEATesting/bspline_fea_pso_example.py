import random
import time

import matplotlib.pyplot as plt
import numpy as np
from cmakeswig.pydatamunge import pydatamunge as dm

import pyfea
from pyfea import (BsplineFeaPSO, ParallelBsplineFeaPSO, SlowBsplineEval,
                   doppler)

random.random()
#### test out pickling ######
import pickle
import random

x = dm.Vector([1.1, 2.2, 3.3, 4.4, 5.5])
x = dm.TestVec(5)
x
x[1] = 1.1

x._get_state()


# Pickle the object to a file
with open("data.pickle", "wb") as file:
    pickle.dump(x, file)

with open("data.pickle", "rb") as file:
    y = pickle.load(file)

y

#### test out pickling ######


# dm.GSLSLM()


class FasterBsplineEval:
    def __init__(self, x, y):
        self.x = dm.Vector(x)
        self.y = dm.Vector(y)

    def __call__(self, knots):
        tx = dm.Vector(self.x)
        ty = dm.Vector(self.y)
        knotvec = dm.Vector(knots)
        start_time = time.time()
        xmat = dm.gsl_bspline_eval(tx, knotvec, 3, False)
        end_time = time.time()
        print("fill mat took", end_time - start_time)

        start_time = time.time()
        slm = dm.GSLSLM(xmat, ty)
        end_time = time.time()
        print("fitting took", end_time - start_time)
        return slm.get_mse()


sample_size = 100000
x = np.random.random(sample_size)
ytrue = doppler(x)
func_width = np.max(ytrue) - np.min(ytrue)
noise = func_width / 20
y = pyfea.make_noisy(ytrue, noise)

k = 500
kseq = np.linspace(0.0, 1.0, k)

plt.scatter(x, y)
plt.show()


benchmark = SlowBsplineEval(x, y)
start_time = time.time()
benchmark(kseq)
end_time = time.time()
print(end_time - start_time)

benchmark = FasterBsplineEval(x, y)
start_time = time.time()
benchmark(kseq)
end_time = time.time()
print(end_time - start_time)


dom = np.zeros((k, 2))
dom[:, 1] = 1.0

# Test base PSO works (DONE! it works?)
fea = ParallelBsplineFeaPSO(benchmark, dom, pop_size=40, processes=12)

fea.run()
fea.gbest


fea.diagnostic_plots()
plt.show()


plt.hist(fea.gbest, bins=10, color="skyblue", edgecolor="black")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Data")
plt.show()
