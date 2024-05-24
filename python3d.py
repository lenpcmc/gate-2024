#Python 3d plot
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot3d(file):
    infile = open(file)
    fig=plt.figure()
    ax=fig.add_subplot(projection='3d')
    for line in infile:
        line=line.strip()
        line = line.split()
        time = line[7]
        edep= line[9]
        posX=line[9]
        posY=line[10]
        posZ=line[11]
        ax.scatter(posX,posY,posZ,marker=1)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_zlabel('Z position')


    plt.show()
    infile.close()

plot3d("GoldFoilDataAsciiHits.dat")