import numpy as np
import sys
import matplotlib.pyplot as plt
import keras
import convolutional_network as cn
import tensorflow as tf
import os
import glob
import re

def runModel(model, learning_rate, batch_size, epochs, validation_split, checkpoint_path):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
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
        workers = 1000000,
        use_multiprocessing = True,
        shuffle = True
    )

    return fit

os.environ["CUDA_VISIBLE_DEVICES"]="1"

noise_numbers_mu = [0, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
noise_numbers_sigma = [0, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
for noise in noise_numbers_mu:
    file = "epos_noise/Epos_" + str(int(noise)) + "_" + str(int(noise/2)) + ".npz"
    showers = np.load(file)
    X = showers['showers']

    masses = X[:, :, 4]
    X = X[:, :, 0:3]

    print(X.shape)
    massSingleNumberAll = []
    for mass in masses:
        massSingleNumberAll.append(mass[0])

    X_train, _ = np.split(X, [-50000])
    mass_train, mass_test = np.split(massSingleNumberAll, [-50000])

    learning_rate = .0005
    batch_size = 128
    epochs = 500
    validation_split = 0.15
    checkpoint_path = "epos_noise_plots/" +str(int(noise)) + "_" + str(int(noise/2)) + "/model_checkpoint.h5"
    model = cn.create_convolutional_model(X.shape, learning_rate)
    print(model.summary())

    fit = runModel(model, learning_rate, batch_size, epochs, validation_split, checkpoint_path)

    # Save the history to a text file
    with open("epos_noise_plots/" +str(int(noise)) + "_" + str(int(noise/2)) + "/history.txt", 'w') as file:
        for loss, val_loss in zip(fit.history['loss'], fit.history['val_loss']):
            file.write(f'{loss} {val_loss}\n')
