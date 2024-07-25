import numpy as np
import matplotlib.pyplot as plt
import sys

import generate_plots as gen

def main():
    # plotTitle = title
    # showers = np.load("Epos.npz")
    # showersNoise = np.load("epos_noise_percentage/Epos_40.0.npz")
    # X = showers['showers']
    # X_noise = showersNoise['showers']
    # print(X.shape)
    # masses = X[:, :, 4]
    # X = X[:, :, 0:3]
    gen.generate_plots(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


if __name__ == main():
    main()