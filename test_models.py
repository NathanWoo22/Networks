import uproot
import numpy as np
import sys
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import conex_read
import convolutional_network as cn
import tensorflow as tf
import os
import glob
import re
import math
import seaborn as sns
import generate_plots as gen
from pathlib import Path

def main():
    folder_path = "optimize_models"
    
    for filename in os.listdir(folder_path):
        # Get the full path of the file or subdirectory
        full_path = os.path.join(folder_path, filename)

        # Check if it's a subdirectory
        if os.path.isdir(full_path):
            print(f'Directory: {full_path}')    
            gen.generate_plots(full_path, full_path)

main()
