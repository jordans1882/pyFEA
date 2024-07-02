import time

import matplotlib.pyplot as plt
import numpy as np

import feareu
from feareu import BsplineFeaPSO, ParallelBsplineFeaPSO, SlowBsplineEval, doppler

sample_size = 100000
x = np.random.random(sample_size)
ytrue = doppler(x)
func_width = np.max(ytrue) - np.min(ytrue)
noise = func_width / 20
y = feareu.make_noisy(ytrue, noise)

k = 500
kseq = np.linspace(0.0, 1.0, k)

plt.scatter(x, y)
plt.show()


start_time = time.time()
benchmark = SlowBsplineEval(x, y)
end_time = time.time()
print(end_time - start_time)

benchmark(kseq)


dom = np.zeros((k, 2))
dom[:, 1] = 1.0

# Test base PSO works (DONE! it works?)
fea = BsplineFeaPSO(benchmark, dom)
fea = ParallelBsplineFeaPSO(benchmark, dom, pop_size=80, processes=12)
fea.run()
fea.gbest


ParallelBsplineFEA
