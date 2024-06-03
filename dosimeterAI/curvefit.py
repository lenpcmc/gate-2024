import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad

X, Y = pd.read_csv('/home/vgate/src/gate-2024/mac/l;ajsdfl;a') #incoming photon data

plt.title('eDep vs photons emitted')
plt.scatter(X,Y)

plt.show()

def landau_dist(a,b,c):
    return a*(np.exp(b/X)/(X*(c+X)))

curve_fit(landau_dist,X,Y,p0=[5646,1144],maxfev=100000)

p_par,p_cov = curve_fit(landau_dist,X,Y,p0=[10,5],maxfev=10000)

Y_fit = landau_dist(X,5.22354534e+02,3.52235334e-01)

plt.scatter(X,Y)
plt.scatter(X,Y_fit,color='red')

plt.show()