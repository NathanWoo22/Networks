import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import tools
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds
from scipy.stats import pearsonr
import uproot
import numpy as np
import sys
import mpl_scatter_density
import datashader as ds
from datashader.mpl_ext import dsshow
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import keras
from keras.models import load_model
import generate_datasets.conex_read as conex_read
import convolutional_network as cn
import tensorflow as tf
import os
import glob
import re
import math
import statistics as stats
import seaborn as sns
from scipy.stats import gaussian_kde
from pathlib import Path
data_location = './current_model_data'

showers = np.load('Epos.npz')
X = showers['showers']

masses = X[:, :, 4]
X = X[:, :, 0:3]

massSingleNumberAll = []
for mass in masses:
    massSingleNumberAll.append(mass[0])

X_train, X_test = np.split(X, [-100000])
mass_train, mass_test = np.split(massSingleNumberAll, [-100000])

checkpoint_path = "./current_model_data/model_checkpoint.h5"
model = load_model(checkpoint_path)
mass_pred = model.predict(X_test, batch_size=1000, verbose=1)[:,0]
#Parses truth and predicted values into their own arrays
truth = mass_test
prediction = mass_pred


# Maps element names to their truth values and colors for plotting
element_info = {
    'hydrogen': {'value': 0.0},
    'deuterium': {'value': 0.6931471805599453},
    'tritium': {'value': 1.0986122886681098},
    'helium': {'value': 1.3862943611198906},
    'lithium': {'value': 1.9459101490553132},
    'berylium': {'value': 2.1972245773362196},
    'iron': {'value': 4.02535169073515}
}
def generate_and_save_graph(elements, file_path_prefix):
    plt.figure(figsize=(10, 6))
    colors = ['deepskyblue', 'orange', 'purple', 'green', 'brown', 'red', 'cyan', 'magenta', 'yellow', 'grey']
    bin_size = 0.1
    min_edge = prediction.min()
    max_edge = prediction.max()
    bin_edges = np.arange(start=min_edge, stop=max_edge + bin_size, step=bin_size)
    for i, element in enumerate(elements):
        element_predictions = prediction[truth == element_info[element]['value']]
        element_weights = np.ones_like(element_predictions) / len(element_predictions)
        color = colors[i % len(colors)]
        plt.hist(element_predictions, bins=bin_edges, weights=element_weights, alpha=0.5,
                 label=f'Predictions For {element.capitalize()}', color=color)
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.xlabel('Predicted Value')
    plt.ylabel('Percentage of Counts')
    plt.title('Distribution of Predicted Values Per Element')
    # plt.legend()
    # Creating a unique filename based on the selected elements
    elements_str = '_'.join(sorted(elements, key=lambda x: element_info[x]['value']))
    filename = f"{file_path_prefix}/counts_vs_prediction_{elements_str}.png"
    plt.savefig(filename)
    plt.close()
# Example usage:
elements_list = [
    # ['hydrogen'],
    # ['hydrogen', 'deuterium'],
    # ['hydrogen', 'deuterium', 'tritium'],
    # ['hydrogen', 'deuterium', 'tritium', 'helium'],
    # ['hydrogen', 'helium'],
    # ['hydrogen', 'helium', 'lithium'],
    ['hydrogen', 'helium', 'lithium', 'berylium'],
    ['hydrogen', 'helium', 'lithium', 'berylium', 'iron'],
    ['hydrogen', 'iron']
]
file_path_prefix = './plots_6'
for elements in elements_list:
    generate_and_save_graph(elements, file_path_prefix)