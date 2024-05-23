#Python 3d plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot3d(file):
    infile = open(file)
    # posX=[]
    # posY=[]
    # posZ=[]
    fig=plt.figure()
    ax=fig.add_subplot(projection='3d')
    lino=0
    while lino<900:

        for line in infile:
            line=line.strip()
            line = line.split()
            time = line[7]
            edep= line[9]
            posX=line[10]
            posY=line[11]
            posZ=line[13]
            ax.scatter(posX,posY)#,posZ)
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        # ax.set_zlabel('Z position')
        line+=1

    plt.show()
    infile.close()

plot3d("GoldFoilDataAsciiHits.dat")