import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.integrate import quad
from mpl_toolkits import mplot3d

def h(theta, x, y):
    return theta[2] * (x - theta[0])**2 + theta[3] * (y - theta[1])**2

xs = np.linspace(-1, 1, 20)
ys = np.linspace(-1, 1, 20)
gridx, gridy = np.meshgrid(xs, ys)
x0 = 0.1; y0 = -0.15; a = 1; b = 2; noise = 0.1
hs = h([x0, y0, a, b], gridx, gridy)
hs += noise * np.random.default_rng().random(hs.shape)

def fun(theta):
    return (h(theta, gridx, gridy) - hs).flatten()

theta0 = [0, 0, 1, 2]
res3 = least_squares(fun, theta0)
print(res3)