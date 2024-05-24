import matplotlib.pyplot as plt
import numpy as np

def plot3d(fileIn):
    infile = open(fileIn)
    fig=plt.figure()
    ax=fig.add_subplot(projection='3d')
    for line in infile:
        xs=float((line.strip()).split()[9])
        ys=float((line.strip()).split()[10])
        zs=float((line.strip()).split()[11])
        ax.scatter(xs,ys,zs)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_zlabel('Z position')
    plt.show()
    infile.close()

plot3d("GoldFoilDataAsciiHits.dat")