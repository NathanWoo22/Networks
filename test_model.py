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
import seaborn as sns
from pathlib import Path


if len(sys.argv) > 1:
    file = sys.argv[1]
else:
    print("Usage: read_showers.py <network location>")
    exit()



showers = np.load("showers.npz")
X = showers['showers']

masses = X[:, :, 4]
X = X[:, :, 0:3]

massSingleNumberAll = []
for mass in masses:
    massSingleNumberAll.append(mass[0])

X_train, X_test = np.split(X, [-50000])
mass_train, mass_test = np.split(massSingleNumberAll, [-50000])

learning_rate = 1e-3
checkpoint_path = sys.argv[1] + "/training_1/cp.ckpt"
model = cn.create_convolutional_model(X.shape, learning_rate)
print(model.summary())

# model = cn.create_model(X.shape, learning_rate)
model.load_weights(checkpoint_path).expect_partial()
mass_pred = model.predict(X_test, batch_size=1000, verbose=1)[:,0]
diff = mass_pred - mass_test
resolution = np.std(diff)
plt.figure()
plt.hist(diff, bins=200)
plt.xlabel('$Mass_\mathrm{rec} - Mass_\mathrm{true}$')
plt.ylabel('# Events')
plt.text(0.95, 0.95, '$\sigma = %.3f$ EeV' % resolution, ha='right', va='top', transform=plt.gca().transAxes)
plt.text(0.95, 0.85, '$\mu = %.1f$ EeV' % diff.mean(), ha='right', va='top', transform=plt.gca().transAxes)
plt.grid()
plt.xlim(-3, 3)
plt.tight_layout()
plt.savefig("Plots/whoTesting_Results")

x = [0, 1, 2, 3, 4 ]
labels = ["0", "1", "2", "3", "4"]

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
axes[0].scatter(mass_test, mass_pred, s=1, alpha=0.60)
axes[0].set_xlabel(r"$Mass_{true}\;/\;\mathrm{EeV}$")
axes[0].set_ylabel(r"$Mass_{DNN}\;/\;\mathrm{EeV}$")
axes[0].set_xlim(-0.2, 4.2)
stat_box = r"$\mu = $ %.3f" % np.mean(diff) + " / EeV" + "\n" + "$\sigma = $ %.3f" % np.std(diff) + " / EeV"
axes[0].text(0.95, 0.2, stat_box, verticalalignment="top", horizontalalignment="right",
          transform=axes[0].transAxes, backgroundcolor="w")
axes[0].plot([np.min(mass_test), np.max(mass_test)],
             [np.min(mass_test), np.max(mass_test)], color="red")


sns.regplot(x=mass_test, y=diff / mass_test, x_estimator=np.std, x_bins=48,
            fit_reg=False, color="royalblue", ax=axes[1])
axes[1].tick_params(axis="both", which="major")
# axes[1].set(xscale="log")
plt.xticks(x, labels)

axes[1].set_xlabel(r"$Mass_{true}\;/\;\mathrm{EeV}$")
axes[1].set_ylabel(r"$\sigma/E$")
# axes[1].set_xlim(1, 5)
# axes[1].set_ylim(0, 4)

plt.savefig("Plots/Scatter_Plot_Results")

# print("made it here?")