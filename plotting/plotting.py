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
mpl.rcParams['font.weight'] = 'bold'

plt.ioff()


def fit_e(x, p0, p1, p2):
    return np.sqrt((p0 * np.sqrt(x)**2) + (p1 * x)**2 + (p2)**2)

def fit_pol2(x, p0, p1, p2):
    return p0 * x**2 + p1 * x + p2


import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def lnA_correlation(mass_true, mass_pred, plot_dir, noise):
    diff = mass_pred - mass_true
    correlation_coefficient = np.corrcoef(mass_pred, mass_true)[0, 1]
    binning = np.arange(-0.5, 5.05, 0.05)
    _, mean_mf, std_mf, _ = tools.get_mean_and_std(mass_true, mass_pred, 48)
    merit_factor = np.abs(np.mean(mass_pred[mass_true == 0.0]) - np.mean(mass_pred[mass_true == 4.02535169073515]))/np.sqrt(np.std(mass_pred[mass_true == 0.0])**2 + np.std(mass_pred[mass_true == 4.02535169073515])**2)
    bin_centers, mean, mean_err, bin_width = tools.get_mean_and_err(mass_true, diff, 48)
    bin_centers, std, std_err, bin_width = tools.get_std_and_err(mass_true, diff, 48)

    fig = plt.figure(figsize=(20, 9))
    # Plot 1
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    counts, xedges, yedges, im = ax1.hist2d(mass_true, mass_pred, bins=[binning, binning], cmin=1,
                                               norm=mpl.colors.LogNorm())
    ax1.set_xlabel(r"$\ln{A}_\mathrm{MC}$", fontsize=20)
    ax1.set_ylabel(r"$\ln{A}_\mathrm{DNN}$", fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=15, width=2, length=6)
    ax1.set_xlim(-0.5, 4.5)
    ax1.set_ylim(-0.5, 4.5)
    ax1.set_box_aspect(1)
    stat_box = r"$\mu$ = %.3f" % np.mean(diff) + "\n" + "$\sigma$ = %.3f" % np.std(
        diff) + "\n" + "$r$ = %.3f" % correlation_coefficient
    ax1.text(0.8, 0.2, stat_box, verticalalignment="top", horizontalalignment="left",
                transform=ax1.transAxes, backgroundcolor="w", fontsize=18)
    ax1.plot([np.min(mass_true), np.max(mass_true)], [np.min(mass_true), np.max(mass_true)], color="C1")
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('# Events')

    # Plot 2
    ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
    ax2.errorbar(bin_centers, mean, xerr=bin_width, yerr=mean_err, color='grey', fmt=".", linewidth=2.5, zorder=1)
    ax2.errorbar(bin_centers[0], mean[0], xerr=bin_width[0], yerr=mean_err[0], color='red', fmt=".", linewidth=2.5,
                    label="Proton", zorder=1)
    ax2.errorbar(bin_centers[-1], mean[-1], xerr=bin_width[-1], yerr=mean_err[-1], color='blue', fmt=".", linewidth=2.5,
                    label="Iron", zorder=1)
    ax2.legend(loc='upper right', fontsize=18)
    ax2.set_ylabel(r"$\langle\Delta\ln{A}\rangle_\mathrm{(DNN-MC)}$", fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=18, width=2, length=6)

    # Plot 3
    ax3 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    ax3.errorbar(bin_centers, std, xerr=bin_width, yerr=std_err, color='grey', fmt=".", linewidth=2.5, zorder=1)
    ax3.errorbar(bin_centers[0], std[0], xerr=bin_width[0], yerr=std_err[0], color='red', fmt=".", linewidth=2.5,
                    label="Proton", zorder=1)
    ax3.errorbar(bin_centers[-1], std[-1], xerr=bin_width[-1], yerr=std_err[-1], color='blue', fmt=".", linewidth=2.5,
                    label="Iron", zorder=1)
    stat_box = r"$\mathrm{MeritFactor}=\frac{\langle\ln{A}\rangle_\mathrm{Fe} - \langle\ln{A}\rangle_\mathrm{p}}{\sigma(\ln{A})_\mathrm{Fe}\oplus\sigma(\ln{A})_\mathrm{p}}$ = %.3f" % round(merit_factor, 2)
    ax3.text(0.05, 0.3, stat_box, verticalalignment="top", horizontalalignment="left",
                transform=ax3.transAxes, backgroundcolor="w", fontsize=15)
    ax3.legend(loc='upper right', fontsize=18)
    ax3.set_xlabel(r"$\ln{A}_\mathrm{MC}$", fontsize=20)
    ax3.set_ylabel(r"$\sigma\left(\Delta\ln{A}\right)_\mathrm{(DNN-MC)}$", fontsize=20)
    ax3.tick_params(axis='both', which='major', labelsize=18, width=2, length=6)
    
    # Move the legend of Plot 1 to the bottom right
    ax1.legend(loc='lower right', fontsize=18)
    fig.tight_layout()
    print('done')
    fig.savefig(plot_dir + f"/dnn_results{noise}")

# Call the function with your data
# lnA_correlation(mass_true, mass_pred, plot_dir, noise)


    # ax.plot([np.min(binning) - 5, np.max(binning) + 5], [np.min(binning) - 5, np.max(binning) + 5], color='grey',
    #         linestyle='--')
    # ax.plot(np.NaN, np.NaN, 's', markersize=3, color='C0', label=r'$C_\mathrm{corr}$ = ' + str(corr_coeff))
    # popt, cov, chi2ndf, func = fitting.fit_chi2(fitting.fitfunc, y_true, y_pred)
    # textstr = fitting.plot_fit_stats(ax, popt, cov, chi2ndf, posx=0.98, posy=0.83, ha='right', significant_figure=True,
    #                                  color='k', parnames=[r'p_{0}', r'p_{1}'], plot_box=False)
    # xfine = binning
    # ax.plot(xfine, popt[0] * xfine + popt[1], color='C1', label=textstr)


def mf_plot(mfs, noise, plot_dir):

    fig, ax = plt.subplots(figsize=(10, 8))
    # Plot 1
    ax.plot(noise, mfs, marker='o',linestyle='None')
    #FIT 1
    popt, cov, chi2ndf, func = fit_chi2(fit_e, noise, mfs)
    print(popt)
    # textstr = plot_fit_stats(ax, popt, cov, chi2ndf, posx=0.98, posy=0.83, ha='right', significant_figure=True,
    #                                  color='k', parnames=[r'p_{0}', r'p_{1}'], plot_box=False)
    # xfine = binning
    x_new = np.linspace(0, 500, 500)
    ax.plot(x_new, fit_e(x_new, popt[0], popt[1], popt[2]), '--', color='grey')#, label=textstr)
    ax.set_xlabel(r"Noise")
    ax.set_ylabel(r"MeritFactor")
    # ax.set_xlim(-0.5, 4.5)
    # ax.set_ylim(-0.5, 4.5)
    ax.set_box_aspect(1)
    # stat_box = r"$\mu$ = %.3f" % np.mean(diff) + "\n" + "$\sigma$ = %.3f" % np.std(
    #     diff) + "\n" + "$r$ = %.3f" % correlation_coefficient
    # ax1.text(0.55, 0.2, stat_box, verticalalignment="top", horizontalalignment="left",
    #             transform=ax1.transAxes, backgroundcolor="w")
    # ax1.plot([np.min(mass_true), np.max(mass_true)], [np.min(mass_true), np.max(mass_true)], color="C1")
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes("right", size="3%", pad=0.05)
    # cbar = fig.colorbar(im, cax=cax)
    # cbar.set_label('# Events')

    fig.tight_layout()
    print('done')
    fig.savefig(plot_dir + f"/mfs")

# data_location = './best_models/convolutional'

showers = np.load('Epos_Cropped_Noise_0_0.npz')
X = showers['showers']

masses = X[:, :, 4]
X = X[:, :, 0:3]

massSingleNumberAll = []
for mass in masses:
    massSingleNumberAll.append(mass[0])

X_train, X_test = np.split(X, [-200000])
mass_train, mass_test = np.split(massSingleNumberAll, [-200000])

checkpoint_path = "./current_model_data/model_checkpoint.h5"
model = load_model(checkpoint_path)
mass_pred = model.predict(X_test, batch_size=1000, verbose=1)[:,0]

lnA_correlation(mass_test, mass_pred, f'/DataFast/nwoo/networks/', None)