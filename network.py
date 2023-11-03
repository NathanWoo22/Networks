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

def runModel(model, learning_rate, batch_size, epochs, validation_split):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./best_saved_network',
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=2
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

print(X.shape)


masses = X[:, :, 3]
# X = X[:, :, :3]
print(masses.shape)
print(X.shape)
massSingleNumberAll = []
for mass in masses:
    massSingleNumberAll.append(mass[0])

X_train, X_test = np.split(X, [20 * X.shape[0]])
mass_train, mass_test = np.split(massSingleNumberAll, [20 * X.shape[0]])

learning_rate = 1e-3
batch_size = 1000
epochs = 1000
validation_split = 0.3
# Create the model
model = cn.create_model(X.shape, learning_rate)

print(model.summary())

fit = runModel(model, learning_rate, batch_size, epochs, validation_split)


fig, ax = plt.subplots(1, figsize=(8,5))
n = np.arange(len(fit.history['loss']))

ax.plot(n, fit.history['loss'], ls='--', c='k', label='loss')
ax.plot(n, fit.history['val_loss'], label='val_loss', c='k')

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.semilogy()
ax.grid()
plt.show()

plt.savefig('Training curve', dpi = 1000)