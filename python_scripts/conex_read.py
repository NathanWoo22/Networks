import uproot
import numpy as np
import sys
import matplotlib.pyplot as plt

def readRoot(file):
    filecx=uproot.open(file)
    tshowercx=filecx["Shower"]
    theadercx=filecx["Header"]
    lgEcx=tshowercx["lgE"].array()
    zenithcx=tshowercx["zenith"].array()
    azimuthcx=tshowercx["azimuth"].array()
    Seed2cx=tshowercx["Seed2"].array()
    Seed3cx=tshowercx["Seed3"].array()
    Xfirstcx=tshowercx["Xfirst"].array()
    Hfirstcx=tshowercx["Hfirst"].array()
    XfirstIncx=tshowercx["XfirstIn"].array()
    altitudecx=tshowercx["altitude"].array()
    X0cx=tshowercx["X0"].array()
    Xmaxcx=tshowercx["Xmax"].array()
    Nmaxcx=tshowercx["Nmax"].array()
    p1cx=tshowercx["p1"].array()
    p2cx=tshowercx["p2"].array()
    p3cx=tshowercx["p3"].array()
    chi2cx=tshowercx["chi2"].array()
    Xmxcx=tshowercx["Xmx"].array()
    Nmxcx=tshowercx["Nmx"].array()
    XmxdEdXcx=tshowercx["XmxdEdX"].array()
    dEdXmxcx=tshowercx["dEdXmx"].array()
    dEdX=tshowercx["dEdX"].array()
    cpuTimecx=tshowercx["cpuTime"].array()
    nXcx=tshowercx["nX"].array()

    dEdX=np.array(tshowercx["dEdX"])
    Xcx=np.array(tshowercx["X"])
    zenith=np.array(tshowercx["zenith"])
    Xmax=np.array(tshowercx["Xmax"])
    return Xcx, dEdX, zenith, Xmax
