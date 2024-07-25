import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import generate_plots as gen

def main():
    showers = np.load("Epos.npz")
    showersNoisePercent = np.load("epos_noise_percentage/Epos_40.0.npz")
    showersNoiseFlat = np.load("epos_noise/Epos_25600_12800.npz")
    X = showers['showers']
    X_noise_percent = showersNoisePercent['showers']
    X_noise_flat = showersNoiseFlat['showers']
    # masses = X[:, :, 4]
    # X = X[:, :, 0:3]
    # print(X[0][0])
    X = X[:, :, 1]
    X_noise_percent = X_noise_percent[:, :, 1]
    X_noise_flat = X_noise_flat[:, :, 1]
    # print(X.shape)
    # print(X[0].shape)
    x = range(0, 144)
    # print(X[0])
    fig, ax = plt.subplots(2)
    ax[0].plot(x, X[20][0:144],  color='green', label='Line Plot', linewidth=0.5)
    ax[0].scatter(x, X_noise_percent[20][0:144], s=4, label='Line Plot')
    ax[0].scatter(x, X_noise_flat[20][0:144], s=4, label='Line Plot')
    X_noise_percent[20] = np.array([math.exp(x) for x in X_noise_percent[20]])
    X_noise_flat[20] = np.array([math.exp(x) for x in X_noise_flat[20]])
    X[20] = np.array([math.exp(x) for x in X[20]])
    ax[1].plot(x, X[20][0:144],  color='green', label='Line Plot', linewidth=0.5)
    ax[1].scatter(x, X_noise_percent[20][0:144], s=4, label='Line Plot')
    ax[1].scatter(x, X_noise_flat[20][0:144], s=4, label='Line Plot')
    plt.savefig("Noise_Visualization.pdf")
    # gen.generate_plots(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


if __name__ == main():
    main()