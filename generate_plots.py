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
from pathlib import Path

# def main():
#     # if len(sys.argv) > 2:
#     #     file = data_location
#     # else:
#     #     print("Usage: generate_plots.py <model data location> <save figure location>")
#     #     exit()
#     generate_plots(sys.argv[1], sys.argv[2])


def generate_plots(data_location, save_location):
    showers = np.load("showers.npz")
    X = showers['showers']

    masses = X[:, :, 4]
    X = X[:, :, 0:3]

    massSingleNumberAll = []
    for mass in masses:
        massSingleNumberAll.append(mass[0])

    X_train, X_test = np.split(X, [-50000])
    mass_train, mass_test = np.split(massSingleNumberAll, [-50000])

    checkpoint_path = data_location + "/model_checkpoint.h5"
    model = load_model(checkpoint_path)
    mass_pred = model.predict(X_test, batch_size=1000, verbose=1)[:,0]
    diff = mass_pred - mass_test
    correlation_coefficient = np.corrcoef(mass_pred, mass_test)[0, 1]
    print(correlation_coefficient)
    resolution = np.std(diff)
    plt.figure()
    plt.hist(diff, bins=200)
    plt.xlabel('$Mass_\mathrm{rec} - Mass_\mathrm{true} \;/\;\mathrm{Ln(a)}$')
    plt.ylabel('# Events')
    plt.text(0.95, 0.95, '$\sigma = %.3f$ Ln(a)' % resolution, ha='right', va='top', transform=plt.gca().transAxes)
    plt.text(0.95, 0.85, '$\mu = %.1f$ Ln(a)' % diff.mean(), ha='right', va='top', transform=plt.gca().transAxes)
    plt.grid()
    plt.xlim(-4, 4)
    plt.ylim(0, 3000)
    plt.tight_layout()
    plt.savefig(save_location + "/Testing_Results")

    x = [0, 1, 2, 3, 4 ]
    labels = ["0", "1", "2", "3", "4"]

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    axes[0].scatter(mass_test, mass_pred, s=1, alpha=0.60)
    axes[0].set_xlabel(r"$\mathrm{Mass_{true}}\;/\;\mathrm{Ln(a)}$")
    axes[0].set_ylabel(r"$\mathrm{Mass_{DNN}}\;/\;\mathrm{Ln(a)}$")
    axes[0].set_xlim(-0.2, 4.2)
    axes[0].set_ylim(-1, 5)
    stat_box = r"$\mu = $ %.3f" % np.mean(diff) + " / Ln(a)" + "\n" + "$\sigma = $ %.3f" % np.std(diff) + " / Ln(a)" + "\n" + "$r = %.3f$" % correlation_coefficient 
    axes[0].text(0.75, 0.2, stat_box, verticalalignment="top", horizontalalignment="left",
            transform=axes[0].transAxes, backgroundcolor="w")
    axes[0].plot([np.min(mass_test), np.max(mass_test)],
                [np.min(mass_test), np.max(mass_test)], color="red")

    epsilon = 1e-1

    sns.regplot(x=mass_test, y=diff, x_estimator=np.std, x_bins=48,
                fit_reg=False, color="royalblue", ax=axes[1])
    axes[1].tick_params(axis="both", which="major")
    # axes[1].set(xscale="log")
    plt.xticks(x, labels)

    axes[1].set_xlabel(r"$\mathrm{Mass_{true}}\;/\;\mathrm{Ln(a)}$")
    axes[1].set_ylabel(r"$\mathrm{Mass_{DNN}-Mass_{true}}\;/\;\mathrm{Ln(a)}$")
    # axes[1].set_ylim(0, 5)

    plt.savefig(save_location + "/Scatter_Plot_Results")

    # Check that training has completed before checking history
    if os.path.exists(data_location + "/history.txt"):
        with open(data_location + "/history.txt", 'r') as file:
            loss = []
            val_loss = []
            for line in file:
                data = line.strip().split()
                loss.append(float(data[0]))
                val_loss.append(float(data[1]))
                
            fig, ax = plt.subplots(1, figsize=(8,5))

            n = np.arange(len(loss))

            ax.plot(n, val_loss, label='val_loss',  color = 'magenta', linewidth=1)
            ax.plot(n, loss, ls='--',  label='loss', color = 'blue',linewidth=1)

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.semilogy()
            ax.grid()

            plt.savefig(save_location + '/Training_Curve', dpi = 1000)
    
    return np.std(diff), correlation_coefficient