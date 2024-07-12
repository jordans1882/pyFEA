import matplotlib.pyplot as plt
import numpy as np

# from pyfea.experiments.general_fea_experiments import big_spike, cliff, discontinuity, doppler, smooth_peak, second_smooth_peak
from pyfea import big_spike, cliff, discontinuity, doppler, second_smooth_peak, smooth_peak

num_points = 500
irange = np.arange(0, num_points)

"""
big_spike_ys = []
for i in irange:
    big_spike_ys.append(big_spike(i/num_points))
plt.scatter(irange, big_spike_ys)
"""
"""
cliff_ys = []
for i in irange:
    cliff_ys.append(cliff(i/num_points))
plt.scatter(irange, cliff_ys)
"""
"""
discontinuity_ys = []
for i in irange:
    discontinuity_ys.append(discontinuity(i/num_points))
plt.scatter(irange, discontinuity_ys)
"""
"""
doppler_ys = []
for i in irange:
    doppler_ys.append(doppler(i/num_points))
plt.scatter(irange, doppler_ys)
"""

peak_range = np.arange(-num_points / 2, num_points / 2)
"""
smooth_peak_ys = []
for i in peak_range:
    smooth_peak_ys.append(smooth_peak(2*i/num_points))
plt.scatter(peak_range, smooth_peak_ys)
"""
"""
second_smooth_peak_ys = []
for i in peak_range:
    second_smooth_peak_ys.append(second_smooth_peak(2*i/num_points))
plt.scatter(peak_range, second_smooth_peak_ys)
"""
plt.grid(True)
plt.show()

