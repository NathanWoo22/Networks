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
noise_numbers_percentage = [0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0]
MFs = []

def main():
    file = open("Merit_Factors.txt", 'w+')
    folder_path = "epos_noise_plots"
    for i, filename in enumerate(noise_numbers_percentage):
        # Get the full path of the file or subdirectory
        full_path = "epos_noise_plots_percentage/" + str(noise_numbers_percentage[i])

        # Check if it's a subdirectory
        if os.path.isdir(full_path):
            print(f'Directory: {full_path}')    
            MF, correlation_coefficient = gen.generate_plots(full_path, full_path, "epos_noise_percentage/Epos_" + str(noise_numbers_percentage[i]) + ".npz", "Mu = " + str(noise_numbers_percentage[i]) + "%, Sigma = " + str(noise_numbers_percentage[i]) + "%")
            MFs.append(MF)
            file.write(str(MF) + ", ")
    file.close()

    # Generate MF against noise
if __name__== '__main__':
    main()
