import numpy as np
import sys
import matplotlib.pyplot as plt
import keras
import convolutional_network as cn
import tensorflow as tf
import os
import glob
import re
# import generate_plots as gen
import random as rand

"""
Auto optimizes the following parameters:

learning_rate
batch_size
validation_split

Generates a random value of each parameter for each model. 
"""

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

showers = np.load("./showers.npz")
X = showers['showers']

masses = X[:, :, 4]
X = X[:, :, 0:3]

massSingleNumberAll = []
for mass in masses:
    massSingleNumberAll.append(mass[0])

X_train, _ = np.split(X, [-50000])
mass_train, mass_test = np.split(massSingleNumberAll, [-50000])

repetitions = 100

for i in range(2, repetitions):
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    learning_rate = rand.uniform(1e-4, 1e-3)
    batch_size = rand.randint(32, 128)
    epochs = 1000
    validation_split = 0.15
    checkpoint_path = "optimize_models/trial_" + str(i) + "/model_checkpoint.h5"
    model = cn.create_convolutional_model(X.shape, learning_rate)
    print(model.summary())

    fit = runModel(model, learning_rate, batch_size, epochs, validation_split, checkpoint_path)

    # Save the history to a text file
    with open('optimize_models/trial_' + str(i) + '/history.txt', 'w') as file:
        for loss, val_loss in zip(fit.history['loss'], fit.history['val_loss']):
            file.write(f'{loss} {val_loss}\n')

    # Generate plots for the models
    # resolution, correlation = gen.generate_plots("optimize_models/trial_" + str(i) + "/", "optimize_models/trial_" + str(i) + "/")

    # Save the history to a text file
    with open('optimize_models/all_data', 'a') as file:
        file.write(f'{learning_rate} {batch_size} {validation_split}\n')
