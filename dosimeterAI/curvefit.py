import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.integrate import quad
from mpl_toolkits import mplot3d

df = pd.read_csv('/home/vgate/src/gate-2024/output/data4AI.csv') #incoming photon data

#### 3D PLOT ####
# Creating figure
fig = plt.figure(figsize = (16, 9))
ax = plt.axes(projection ="3d")
   
# Add x, y gridlines 
ax.grid(b = True, color ='grey', 
        linestyle ='-.', linewidth = 0.3, 
        alpha = 0.2) 
 
# Creating color map
my_cmap = plt.get_cmap('hsv')
 
# Creating plot
sctt = ax.scatter3D(df.hi,df.there,df.bud,
                    alpha = 0.8,
                    c = (df.hi + df.there + df.bud), 
                    cmap = my_cmap, 
                    marker ='^')
 
plt.title("simple 3D scatter plot")
ax.set_xlabel('X-axis', fontweight ='bold') 
ax.set_ylabel('Y-axis', fontweight ='bold') 
ax.set_zlabel('Z-axis', fontweight ='bold')
fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
 
# show plot
plt.show()
#########

x = df.hi
y = df.there
X = x + y


def curve(a,b,c):
    return a + b + c

#print(curve_fit(curve,X,df.bud,p0=[0,0],maxfev=10000))

#Y_fit = curve(X,5.22354534e+02,3.52235334e-01)
#
#plt.scatter(X,Y)
#plt.scatter(X,Y_fit,color='red')
#
#plt.show()

########################################################################vvv can delete

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