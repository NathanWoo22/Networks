import uproot
import numpy as np
import sys
import matplotlib.pyplot as plt
import keras
import generate_datasets.conex_read as conex_read
import convolutional_network as cn
import tensorflow as tf
import os
import glob
import re
import math
from pathlib import Path

# converts a list of zenith angles to a list of the same angle with different lengths
def expandZenithAngles(zenithAngles, newLength):
    
    zenithAnglesList = list([list([x]) for x in zenithAngles])
    extendedZenithList = []
    for i in range(len(Xcx)):
        newZenithList = []
        for j in range(len(Xcx[i])):
            newZenithList.append(float(zenithAnglesList[i][0]))    
        extendedZenithList.append(newZenithList)

    zenithAngles = extendedZenithList
    for i in range(len(zenithAngles)):
        zenithAngles[i] = np.array(zenithAngles[i]) 

    return zenithAngles

def expandXcxdEdX(Xcx, dEdX, maxLength):
    mean = 6400
    std_dev = 3200
    for i, array in enumerate(Xcx):
        for j in range(len(dEdX[i])):
            # dEdX[i][j] += (np.random.normal(mean, std_dev, 1))[0]
            dEdX[i][j] = np.log(dEdX[i][j])
            if math.isnan(dEdX[i][j]) or dEdX[i][j] < 0:
                dEdX[i][j] = 0
        while len(Xcx[i]) < maxLength:
            Xcx[i] = np.append(Xcx[i], 0)
            dEdX[i] = np.append(dEdX[i], 0)

    return Xcx, dEdX

def plotdEdX(Xcx, dEdX):
    for i in range(100):
        plt.scatter(Xcx[i], dEdX[i], s=0.5)
        plt.xlabel('X')
        plt.ylabel('dEdX')
        plt.title('Energy deposit per cm')

    plt.show()
    plt.savefig('Energy function plot', dpi = 1000)




XcxAll = []
dEdXAll = []
zenithAll = []
XmaxAll = []
massAll = []
massSingleNumberAll = []
count = 0

# Specify the directory path
directory_paths = ['/Data/Simulations/Conex_Flat_lnA/EPOS/', '/Data/Simulations/Conex_Flat_lnA/Sibyll/']

# Initialize an empty list to store folder paths
folder_list = []

# Iterate through each directory path
for directory_path in directory_paths:
    # Get a list of all folders within the directory
    folders = [str(item) for item in Path(directory_path).iterdir() if item.is_dir()]
    # Append 'showers' to each folder path
    for folder in folders:
        folder_list.append(folder + '/showers')
        
count = 0
flag = 0
for folder_path in folder_list:
    # folder_path = "/Data/Simulations/Conex_Flat_lnA/data/Conex_170-205_Prod1/showers"
    fileNames = glob.glob(os.path.join(folder_path, '*.root'))
    # Read data for network
    if flag == 1:
        break
    for fileName in fileNames:
        # if count == 10:
        #     flag = 1
        #     break
        # count += 1
        print("Reading file " + fileName)
        pattern = r'_(\d+)\.root'
        match = re.search(pattern, fileName)
        if match:
            mass = np.log(float(match.group(1))/100)
            if math.isnan(mass):
                mass = 0
            print(f"Mass is: {mass}")
        else:
            print("No match found.")

        try:
            Xcx, dEdX, zenith, Xmax = conex_read.readRoot(fileName)
        except Exception as e:
            print(f"An exception occured: {e}")
            continue
        maxLength = 700
        Xcx, dEdX = expandXcxdEdX(Xcx, dEdX, maxLength)
        zenith = expandZenithAngles(zenith, maxLength)
        Xmax = expandZenithAngles(Xmax, maxLength)
        masses = []
        for i in range(maxLength):
            masses.append(float(mass))
        

        Xcx = np.vstack(Xcx)
        dEdX = np.vstack(dEdX)
        zenith = np.vstack(zenith)
        Xmax = np.vstack(Xmax)

        for showerXcx in Xcx:
            XcxAll.append(showerXcx)
        for showerdEdX in dEdX:
            dEdXAll.append(showerdEdX)
        for showerXmax in Xmax:
            XmaxAll.append(showerXmax)
        for showerZenith in zenith:
            zenithAll.append(showerZenith)
            massAll.append(masses)
            massSingleNumberAll.append(float(mass))

        print("Finished reading file " + fileName)

X = np.stack([XcxAll, dEdXAll, zenithAll, XmaxAll, massAll], axis = -1)
np.savez("./Both.npz", showers=X)
