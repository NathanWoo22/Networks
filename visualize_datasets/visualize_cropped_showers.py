import numpy as np
import matplotlib.pyplot as plt
import sys
import math

import generate_plots as gen

def main():
    showers_cropped = np.load("Epos_Cropped.npz")
    showers = np.load("Epos.npz")
    # showersNoisePercent = np.load("epos_noise_percentage/Epos_40.0.npz")
    # showersNoiseFlat = np.load("epos_noise/Epos_25600_12800.npz")
    X = showers['showers']
    X_cropped = showers_cropped['showers']
    # X_noise_percent = showersNoisePercent['showers']
    # X_noise_flat = showersNoiseFlat['showers']
    # masses = X[:, :, 4]
    # X = X[:, :, 0:3]
    # print(X[0][0])
    X = X[:, :, 1]
    X_cropped = X_cropped[:, :, 1]
    # X_noise_percent = X_noise_percent[:, :, 1]
    # X_noise_flat = X_noise_flat[:, :, 1]
    # print(X.shape)
    # print(X[0].shape)
    x = range(0, 200)
    # print(X[0])
    # plt.scatter(x, X_noise_percent[20][0:144], s=2)
    # plt.scatter(x, X_noise_flat[20][0:144], s=2)
    fig, axs = plt.subplots(3)
    for i in range(3):
        # X[i] = np.array([math.exp(x) for x in X[i]])
        # X_cropped[i] = np.array([math.exp(x) for x in X_cropped[i]])
        axs[i].plot(x, X_cropped[i+3][:200])
        axs[i].plot(x, X[i+3][:200])
    plt.savefig("Visualization_Cropped_ln.pdf")

    X = showers['showers']
    X_cropped = showers_cropped['showers']
    X = X[:, :, 1]
    X_cropped = X_cropped[:, :, 1]
    x = range(0, 200)
    plt.figure()
    fig, axs = plt.subplots(3)
    for i in range(3):
        X[i] = np.array([math.exp(x) for x in X[i+3]])
        X_cropped[i] = np.array([math.exp(x) for x in X_cropped[i+3]])
        axs[i].plot(x, X_cropped[i+3][:200])
        axs[i].plot(x, X[i+3][:200])
    plt.savefig("Visualization_Cropped_linear.pdf")

if __name__ == main():
    main()