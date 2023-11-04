import uproot
import numpy as np
import sys
import matplotlib.pyplot as plt
import keras
import conex_read
import convolutional_network as cn
import tensorflow as tf
import os
import glob
import re
import math
from pathlib import Path

# learning_rate = 1e-3
# batch_size = 1000
# epochs = 1000
# validation_split = 0.3
# checkpoint_path = "training_1/cp.ckpt"

# showers = np.load("showers.npz")
# X = showers['showers']

# print(X.shape)


# masses = X[:, :, 4]
# X = X[:, :, 0:3]

# print(masses.shape)
# print(X.shape)
# massSingleNumberAll = []
# for mass in masses:
#     massSingleNumberAll.append(mass[0])

def test_model(X, learning_rate, checkpoint_path, X_test, mass_test):
    model = cn.create_model(X.shape, learning_rate)
    model.load_weights(checkpoint_path)
    mass_pred = model.predict(X_test, batch_size=1000, verbose=1)[:,0]
    diff = mass_pred - mass_test
    resolution = np.std(diff)
    plt.figure()
    plt.hist(diff, bins=100)
    plt.xlabel('$E_\mathrm{rec} - E_\mathrm{true}$')
    plt.ylabel('# Events')
    plt.text(0.95, 0.95, '$\sigma = %.3f$ EeV' % resolution, ha='right', va='top', transform=plt.gca().transAxes)
    plt.text(0.95, 0.85, '$\mu = %.1f$ EeV' % diff.mean(), ha='right', va='top', transform=plt.gca().transAxes)
    plt.grid()
    plt.xlim(-10, 10)
    plt.tight_layout()
    plt.savefig("Testing_Results")