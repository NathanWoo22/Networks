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

plt.rcParams['axes.labelweight'] = 'bold'  # Make axis labels bold
plt.rcParams['xtick.labelsize'] = 12  # Set x-axis tick label size
plt.rcParams['ytick.labelsize'] = 12  # Set y-axis tick label size
plt.rcParams['font.weight'] = 'bold'  # Make tick labels bold
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

def expandXcxdEdX(Xcx, dEdX, maxLength, mean, std_dev, percent):
    # mean = 100
    # std_dev = 50
    global baseline_noise  
    global percent_noise
    global noisy_dEdX 
    baseline_noise = []
    percent_noise = []
    noisy_dEdX = []
    for j in range(len(dEdX[0])):
        try:
            print((np.random.normal(1.0, percent, 1))[0])
            percent_noise_piece =np.log10((np.random.normal(1, percent, 1))[0] * dEdX[0][j])
        except:
            percent_noise_piece = 0
        try:
            baseline_noise_piece = np.log10((np.random.normal(mean, std_dev, 1))[0])
        except:
            baseline_noise_piece = np.log10(mean)
        baseline_noise.append(baseline_noise_piece)
        percent_noise.append(percent_noise_piece)
        # dEdX[0][j] += (np.random.normal(mean + percent * dEdX[0][j], std_dev + percent * dEdX[0][j], 1))[0]
        dEdX[0][j] = np.log10(dEdX[0][j])
        noisy_dEdX.append(dEdX[0][j] + baseline_noise_piece + percent_noise_piece)
        if math.isnan(dEdX[0][j]) or dEdX[0][j] < 0:
            dEdX[0][j] = 0
    while len(Xcx[0]) < maxLength:
        Xcx[0] = np.append(Xcx[0], 0)
        dEdX[i] = np.append(dEdX[i], 0)

    return Xcx, dEdX

def plotdEdX(Xcx, dEdX):
    for i in range(100):
        plt.scatter(Xcx[0], baseline_noise, s=0.5)
        plt.xlabel('X')
        plt.ylabel('dEdX')
        plt.title('Energy deposit per cm')

    plt.show()
    plt.savefig('Energy function plot.pdf', dpi = 1000)





noise_baseline_mu = 25000
noise_baseline_sigma = 12500
# noise_percentage = [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
noise_percentage = 0.2
XcxAll = []
dEdXAll = []
zenithAll = []
XmaxAll = []
massAll = []
massSingleNumberAll = []
count = 0
# Specify the directory path
directory_path = '/Data/Simulations/Conex_Flat_lnA/EPOS/'

# Get a list of all folders within the directory
folder_list = [str(item) for item in Path(directory_path).iterdir() if item.is_dir()]
for i in range(len(folder_list)):
    folder_list[i] += '/showers'

flag = 0
count = 0 
noise = .40
for folder_path in folder_list:
    # folder_path = "/Data/Simulations/Conex_Flat_lnA/data/Conex_170-205_Prod1/showers"
    fileNames = glob.glob(os.path.join(folder_path, '*.root'))
    # Read data for network
    for fileName in fileNames:
        if count == 1:
            break
        count += 1
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
        Xcx, dEdX = expandXcxdEdX(Xcx, dEdX, maxLength, noise_baseline_mu, noise_baseline_sigma, noise)
        zenith = expandZenithAngles(zenith, maxLength)
        Xmax = expandZenithAngles(Xmax, maxLength)
        masses = []
        for i in range(maxLength):
            masses.append(float(mass))
        print(len(Xcx[0]))
        print(len(baseline_noise))
        for i in range(len(percent_noise)):
            if percent_noise[i] < 0:
                percent_noise[i] = 0
            if baseline_noise[i] < 0:
                baseline_noise[i] = 0
            if math.isnan(percent_noise[i]):
                percent_noise[i] = (percent_noise[i-1] + percent_noise[i+1])/2
            if math.isnan(baseline_noise[i]):
                try:
                    baseline_noise[i] = (baseline_noise[i-1] + baseline_noise[i+1])/2
                except:
                    baseline_noise[i] = baseline_noise[i-1]
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(Xcx[0][:118], baseline_noise[0:118], linestyle='-', linewidth=3,label='Baseline Noise')
        ax.plot(Xcx[0][:118], percent_noise[0:118], linestyle='-', linewidth=3,label ='Percent Noise')
        ax.plot(Xcx[0][:118], dEdX[0][:118],  linestyle='-', linewidth=3,label = 'Original Shower')
        # plt.scatter(Xcx[0][:119], noisy_dEdX[:119], s=0.5)
        ax.set_xlabel("Depth "r"$\left[\mathrm{\frac{g}{cm^2}}\right]$", fontsize=15, fontweight="bold")
        ax.set_ylabel("log[dE/dX] "r"$\left[\mathrm{\frac{eV} {(g/cm^2)}}\right]$", fontsize=15, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_title('Baseline and Percent Shower Noise', fontsize=18, fontweight='bold')
        plt.show()
        plt.savefig('Visualize_Noise.png', dpi = 1000)
        # Xcx = np.vstack(Xcx)
        # dEdX = np.vstack(dEdX)
        # zenith = np.vstack(zenith)
        # Xmax = np.vstack(Xmax)

        # for showerXcx in Xcx:
        #     XcxAll.append(showerXcx)
        # for showerdEdX in dEdX:
        #     dEdXAll.append(showerdEdX)
        # for showerXmax in Xmax:
        #     XmaxAll.append(showerXmax)
        # for showerZenith in zenith:
        #     zenithAll.append(showerZenith)
        #     massAll.append(masses)
        #     massSingleNumberAll.append(float(mass))

        # print("Finished reading file " + fileName)

# X = np.stack([XcxAll, dEdXAll, zenithAll, XmaxAll, massAll], axis = -1)
# print(X.shape)
# np.savez("epos_noise_percentage/Epos_" + str(noise * 100) + ".npz", showers=X)
