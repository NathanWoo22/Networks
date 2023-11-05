# import uproot
import numpy as np
import sys
import matplotlib.pyplot as plt
import keras
# import conex_read
import convolutional_network as cn
import tensorflow as tf
import os
import glob
import re
# import test_model as tm
def runModel(model, learning_rate, batch_size, epochs, validation_split, checkpoint_path):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )


    fit = model.fit(
        X_train,
        mass_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[model_checkpoint_callback],
        validation_split=validation_split,
        workers = 10000,
        use_multiprocessing = True
    )

    return fit

showers = np.load("showers.npz")
X = showers['showers']

masses = X[:, :, 4]
X = X[:, :, 0:5]

massSingleNumberAll = []
for mass in masses:
    massSingleNumberAll.append(mass[0])

X_train, X_test = np.split(X, [-50000])
mass_train, mass_test = np.split(massSingleNumberAll, [-50000])

learning_rate = 1e-3
batch_size = 100
epochs = 100
validation_split = 0.3
checkpoint_path = "training_1/cp.ckpt"
model = cn.create_convolutional_model(X.shape, learning_rate)
print(model.summary())

fit = runModel(model, learning_rate, batch_size, epochs, validation_split, checkpoint_path)


fig, ax = plt.subplots(1, figsize=(8,5))
n = np.arange(len(fit.history['loss']))

ax.plot(n, fit.history['loss'], ls='--', c='k', label='loss')
ax.plot(n, fit.history['val_loss'], label='val_loss', c='k')

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.semilogy()
ax.grid()

plt.savefig('Training_Curve', dpi = 1000)


# tm.test_model(X, learning_rate, checkpoint_path, X_test, mass_test)
# model = cn.create_model(X.shape, learning_rate)
# model.load_weights(checkpoint_path)
# mass_pred = model.predict(X_test, batch_size=100, verbose=1)[:,0]
# diff = mass_pred - mass_test
# resolution = np.std(diff)
# plt.figure()
# plt.hist(diff.flatten(), bins=100)
# plt.xlabel('$E_\mathrm{rec} - E_\mathrm{true}$')
# plt.ylabel('# Events')
# plt.text(0.95, 0.95, '$\sigma = %.3f$ EeV' % resolution, ha='right', va='top', transform=plt.gca().transAxes)
# plt.text(0.95, 0.85, '$\mu = %.1f$ EeV' % diff.mean(), ha='right', va='top', transform=plt.gca().transAxes)
# plt.grid()
# plt.xlim(-5, 5)
# # plt.tight_layout()
# plt.savefig("Testing_Results")