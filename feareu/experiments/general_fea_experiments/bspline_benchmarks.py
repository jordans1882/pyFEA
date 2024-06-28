import numpy as np
import math

def big_spike(x):
    return 100 * np.exp(-abs(10 * x-5)) + (10 * x - 5)**5/500


def cliff(x):
    return 90/(1+np.exp(-100*(x-0.4)))

def discontinuity(x):
    return np.where(x < 0.6, 1/(0.01+(x-0.3)**2), 1/(0.015+(x-0.65)**2))

def smooth_peak(x):
    return np.sin(x) + (2 * np.e) ** (-30 * x**2)

def second_smooth_peak(x):
    return np.sin(2*x) + (2 * np.e) ** (-16 * x**2) + 2

def doppler(x):
    return np.sin(20/(x+0.15))

def recursive_spline(x, k, i, t):
    if k==0:
        return np.where(t[i] <= x < t[i+1], 1, 0)
    if t[i+k] == t[i]:
        c1 = 0
    else:
        c1 = (x-t[i])/(t[i+k]-t[i]) * recursive_spline(x, k-1, i, t)
    if t[i+k+1]==t[i+1]:
        c2 = 0
    else:
        c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * recursive_spline(x,k-1,i+1,t)
    return c1+c2

def spline_curve(x, knots, controls, degree):
    n = len(knots) - degree - 1
    ans = np.zeros(x.shape)
    for i in range(n):
        ans = ans + c[i] * recursive_spline(x, degree, i, knots)
    return ans

def generate_spline_params(num_knots, degree=None):
    knots = np.random.random(num_knots)
    knots.sort()
    if degree is None:
        degree = np.random.randint(low=0, high = num_knots/2)
    num_controls = num_knots - degree - 1
    controls = np.random.uniform(low = -10, high = 10, size=num_controls)
    return knots, controls, degree

def make_noisy(x, sigma):
    return x + np.random.normal(scale=sigma, size=len(x))
