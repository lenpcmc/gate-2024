import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

X = np.random.random(size=(100))*10

m = 522.35453354364334153643345463543463
C = .3522353344643541463
Y = m * X ** 2 - C * X ** 3 + C * X ** 4

plt.scatter(X,Y)

plt.show()

def curve(X,m,C):
  return(m * X ** 2 - C * X ** 3 + C * X ** 4)

curve_fit(curve,X,Y,p0=[5646,1144],maxfev=100000)


p_par,p_cov = curve_fit(curve,X,Y,p0=[10,5],maxfev=10000)
p_par

Y_fit = curve(X,5.22354534e+02,3.52235334e-01)
Y_fit

plt.scatter(X,Y)
plt.scatter(X,Y_fit,color='red')

plt.show()