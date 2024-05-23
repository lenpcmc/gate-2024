import ROOT
import numpy as np
import matplotlib.pyplot as plt

from time import sleep

#ROOT.gApplication.Run._threaded = True

def main():
    with ROOT.TFile.Open("GoldFoilData.root") as infile:
        h = infile.Get("Hits")
        h.Draw("posX:posY:posZ")
        sleep(10)
        #ROOT.gApplication.Run()
    return


if __name__ == "__main__":
    main()
