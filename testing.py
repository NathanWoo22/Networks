# import uproot
import numpy as np
import sys
import matplotlib.pyplot as plt
import keras
# import conex_read
# import convolutional_network as cn
# import tensorflow as tf
import os
import glob
import re

def plotdEdX(Xcx, dEdX):
    for i in range(1):
        plt.scatter(Xcx[i], dEdX[i], s=0.5)
        plt.xlabel('X')
        plt.ylabel('dEdX')
        plt.title('Energy deposit per cm')
    
    # plt.xlim(0, 2000)

    plt.show()
    plt.savefig('Energy function plot', dpi = 1000)

# print(np.log10(1e19))
showers = np.load("showers.npz")
X = showers['showers']

print(X.shape)
print(X[0,:,0].shape)
print(X[0,:,1].shape)
\
plt.scatter(X[0,:,0], X[0,:,1], s=0.5)
plt.xlabel('X')
plt.ylabel('dEdX')
plt.title('Interesting')
plt.savefig('Interesting', dpi = 1000)
