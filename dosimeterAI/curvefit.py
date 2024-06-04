import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
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

print(curve_fit(curve,X,df.bud,p0=[0,0],maxfev=10000))

#Y_fit = curve(X,5.22354534e+02,3.52235334e-01)
#
#plt.scatter(X,Y)
#plt.scatter(X,Y_fit,color='red')
#
#plt.show()

########################################################################################vvvvv can delete
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