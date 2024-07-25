import uproot
import numpy as np
import sys
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import generate_datasets.conex_read as conex_read
import convolutional_network as cn
import tensorflow as tf
import os
import glob
import re
import math
import seaborn as sns
import generate_plots as gen
from pathlib import Path


# noise_numbers_mu = [0, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
# noise_numbers_sigma = [0, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
noise_numbers_percentage = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
MFs = []

def main():
    file = open("Merit_Factors.txt", 'w+')
    for i, filename in enumerate(noise_numbers_percentage):
        # Get the full path of the file or subdirectory
        full_path = "epos_20_percentage/" + str(noise_numbers_percentage[i])

        # Check if it's a subdirectory
        if os.path.isdir(full_path):
            print(f'Directory: {full_path}')    
            MF, correlation_coefficient = gen.generate_plots(full_path, full_path, "epos_noise_percentage/Epos_20.0.npz", "Mu = 20%, " + "Sigma = 20%")
            MFs.append(MF)
            file.write(str(MF) + ", ")
    file.close()

    # Generate MF against noise
if __name__== '__main__':
    main()
