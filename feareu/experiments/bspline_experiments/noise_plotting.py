import numpy as np
import feareu
import matplotlib.pyplot as plt

sample_size = 20000

function = feareu.big_spike

x = np.random.random(sample_size)
y = function(x)
y_range = np.max(y) - np.min(y)
y = feareu.make_noisy(y, y_range/20)

plt.scatter(x,y)
plt.savefig('results/noise_test')
