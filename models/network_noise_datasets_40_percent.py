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

# noise_percentage = [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
for i in range(10):
    noise = 0.4
    file = "epos_noise_percentage/Epos_" + str(100 * (noise)) + ".npz"
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
    epochs = 800
    validation_split = 0.15

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)
    checkpoint_path = "epos_40_percentage/" +str(i) + "/model_checkpoint.h5"

    model = cn.create_convolutional_model(X.shape, learning_rate=lr_schedule)
    print(model.summary())

    fit = runModel(model, learning_rate, batch_size, epochs, validation_split, checkpoint_path)

    # Save the history to a text file
    with open("epos_40_percentage/" +str(i) + "/history.txt", 'w') as file:
        for loss, val_loss in zip(fit.history['loss'], fit.history['val_loss']):
            file.write(f'{loss} {val_loss}\n')
