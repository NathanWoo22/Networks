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

# def main():
#     # if len(sys.argv) > 2:
#     #     file = data_location
#     # else:
#     #     print("Usage: generate_plots.py <model data location> <save figure location>")
#     #     exit()
#     generate_plots(sys.argv[1], sys.argv[2])
# "Viridis-like" colormap with white background
def scatter_with_gaussian_kde(ax, x, y):
    # https://stackoverflow.com/a/20107592/3015186
    # Answer by Joel Kington

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    ax.scatter(x, y, c=z, s=1)

def generate_plots(data_location, save_location, showers, title):
    plotTitle = sys.argv[4]
    plotTitle = title
    showers = np.load(showers)
    X = showers['showers']

    masses = X[:, :, 4]
    X = X[:, :, 0:3]

    massSingleNumberAll = []
    for mass in masses:
        massSingleNumberAll.append(mass[0])

    X_train, X_test = np.split(X, [-100000])
    mass_train, mass_test = np.split(massSingleNumberAll, [-100000])

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
    plt.xlim(-2, 2)
    plt.ylim(0, 7000)
    # plt.tight_layout()
    plt.title(plotTitle)
    plt.savefig(save_location + "/Testing_Results.png")

    x = [0, 1, 2, 3, 4 ]
    labels = ["0", "1", "2", "3", "4"]

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    # axes[0].scatter(mass_test, mass_pred, s=1, alpha=0.60)

    hist = axes[0].hist2d(mass_test, mass_pred, bins=[100,100], cmin=0, norm=mpl.colors.LogNorm())
    cbar = plt.colorbar(hist[3], ax=axes[0])
    cbar.set_label("Number of Events")

    # scatter_with_gaussian_kde(axes[0], mass_test, mass_pred)
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
    fig.suptitle(plotTitle)
    plt.savefig(save_location + "/Scatter_Plot_Results.png")

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # ax.hist2d(mass_test, mass_pred, bins=[200,200], cmin=0, norm=mpl.colors.Normalize())
    scatter_with_gaussian_kde(ax, mass_test, mass_pred)
    plt.savefig("extra")


    # fig, ax1 = plt.plot()
    # binning = 50
    # counts, xedges, yedges, im = ax1.hist2d(mass_test, mass_pred, bins=[binning, binning], cmin=1,
    #                                        norm=mpl.colors.LogNorm())
    

    # Check that training has completed before checking history
    if os.path.exists(data_location + "/history.txt"):
        with open(data_location + "/history.txt", 'r') as file:
            loss = []
            val_loss = []
            for line in file:
                data = line.strip().split()
                loss.append(float(data[0]))
                val_loss.append(float(data[1]))
                

            plt.rcParams['axes.labelweight'] = 'bold'  # Make axis labels bold
            plt.rcParams['xtick.labelsize'] = 12  # Set x-axis tick label size
            plt.rcParams['ytick.labelsize'] = 12  # Set y-axis tick label size
            plt.rcParams['font.weight'] = 'bold'  # Make tick labels bold
            
            fig, ax = plt.subplots(1, figsize=(8,5))

            n = np.arange(len(loss))

            ax.plot(n, val_loss, label='val_loss',  color = 'red', linewidth=1)
            ax.plot(n, loss, ls='--',  label='loss', color = 'black',linewidth=1)
            ax.set_xlabel('Epoch', fontsize=15, fontweight='bold')
            ax.set_ylabel('Loss', fontsize=15, fontweight='bold')
            # ax.set_ylim(0.05, 1)
            ax.legend()
            ax.semilogy()
            ax.grid()
            plt.title("CNN Training and Validation Loss", fontsize=18, fontweight='bold')
            plt.savefig(save_location + '/Training_Curve.png', dpi = 1000)
            print(val_loss.index(min(val_loss)))
    # Calculate the merit factor 
    # print(len(mass_test))
    # iron_mass = 56
    # proton_mass = min(mass_test)
    # print(len(diff))
    # print(mass_test)
    # protons = []
    # irons = []
    # for i, element in enumerate(mass_test):
    #     print(element)
    #     if element == proton_mass:
    #         protons.append(mass_pred[i])
    #     if element == math.log(56):
    #         print(len(irons))
    #         irons.append(mass_pred[i])
    # # print(math.log(56))
    # # print(set(mass_test))
    # print(len(irons))
    # MF = (stats.mean(irons) - stats.mean(protons)) / math.sqrt(stats.stdev(irons)**2 + stats.stdev(protons)**2) 
    # print(MF)
    # return MF, correlation_coefficient