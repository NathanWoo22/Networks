import tensorflow as tf
from tensorflow import keras
import numpy as np
# import seaborn as sns
from matplotlib import pyplot as plt
layers = keras.layers
import os
# import conex_read

plt.figure(figsize=(9,7))


# Bias: 0.2 Resolution: 0.78
# X = np.stack([T, S, newAxis, newCore, newXMax], axis=-1)

# print(X.shape)

# X_train, X_test = np.split(X, [-20000])
# energy_train, energy_test = np.split(energy, [-20000])

"""---
## Define Model
Now, we will set up a neural network to reconstruct the energy of the particle shower. In this **define step** we will set the architecture of the model.


> **Modification section**  
> Feel free to modify the model.
> For example:
> *   Change the number of nodes (but remember that the number of weights scales with n x n. Also, the final layer has to have only one node as we are reconstructing the energy, which is a scalar.).
> *   Change the activation function, e.g., use `relu, tanh, sigmoid, softplus, elu, ` (take care to not use an activation function for the final layer!).
> *   Add new layers convolutional layers.
> *   Add new pooling layers, followed by other convolutional layers.
> *   Add fully-connected layers (remember to use `Flatten()` before).
> *   Increase the Dropout fraction or place Dropout between several layers if you observe overtraining (validation loss increases).


"""



def create_fully_connected_model(shape, learning_rate):
  model = keras.models.Sequential(name="energy_regression_CNN")
  kwargs = dict(kernel_initializer="he_normal", padding="same",)
  model.add(layers.Conv1D(16, 2, activation=activationFunction, input_shape=shape[1:], **kwargs))
  # model.add(layers.Dense(100, activation=activationFunction, input_shape=shape[1:], **kwargs))
  model.add(layers.Dense(100, activation=activationFunction))
  model.add(layers.Dense(100, activation=activationFunction))
  model.add(layers.Dense(100, activation=activationFunction))
  model.add(layers.Dense(100, activation=activationFunction))
  model.add(layers.Dense(100, activation=activationFunction))
  model.add(layers.Dense(100, activation=activationFunction))
  model.add(layers.Dense(100, activation=activationFunction))
  model.add(layers.Dense(100, activation=activationFunction))
  model.add(layers.Dense(100, activation=activationFunction))
  model.add(layers.Dense(100, activation=activationFunction))
  model.add(layers.Dense(100, activation=activationFunction))
  model.add(layers.Dense(100, activation=activationFunction))
  model.add(layers.Dense(100, activation=activationFunction))
  model.add(layers.Dense(100, activation=activationFunction))
  model.add(layers.Dense(20, activation=activationFunction))
  model.add(layers.Dense(20, activation=activationFunction))
  model.add(layers.Dense(20, activation=activationFunction))
  model.add(layers.Dense(20, activation=activationFunction))
  model.add(layers.Dense(20, activation=activationFunction))
  model.add(layers.Dense(20, activation=activationFunction))
  model.add(layers.Dense(20, activation=activationFunction))
  model.add(layers.Dense(20, activation=activationFunction))
  model.add(layers.Dense(20, activation=activationFunction))


  # model.add(layers.Conv1D(16, 3, activation=activationFunction, **kwargs))
  # model.add(layers.MaxPooling1D(2))
  # model.add(layers.Conv1D(16, 3, activation=activationFunction, **kwargs))
  # model.add(layers.Conv1D(16, 3, activation=activationFunction, **kwargs))
  # model.add(layers.Conv1D(16, 3, activation=activationFunction, **kwargs))
  # model.add(layers.Conv1D(16, 3, activation=activationFunction, **kwargs))
  # model.add(layers.Conv1D(16, 3, activation=activationFunction, **kwargs))
  # model.add(layers.Conv1D(16, 3, activation=activationFunction, **kwargs))
  # model.add(layers.Conv1D(16, 3, activation=activationFunction, **kwargs))
  # model.add(layers.MaxPooling1D(2))
  # model.add(layers.Conv1D(64, 2, activation=activationFunction, **kwargs))
  # model.add(layers.Conv1D(64, 2, activation=activationFunction, **kwargs))
  # model.add(layers.Conv1D(64, 2, activation=activationFunction, **kwargs))
  # model.add(layers.Conv1D(64, 2, activation=activationFunction, **kwargs))
  # model.add(layers.Conv1D(64, 2, activation=activationFunction, **kwargs))
  # model.add(layers.Conv1D(64, 2, activation=activationFunction, **kwargs))
  # model.add(layers.MaxPooling1D(2))
  # model.add(layers.Conv1D(128, 2, activation=activationFunction, **kwargs))
  # model.add(layers.Conv1D(128, 2, activation=activationFunction, **kwargs))
  # model.add(layers.Conv1D(128, 2, activation=activationFunction, **kwargs))
  # model.add(layers.MaxPooling1D(2))
  # model.add(layers.Conv1D(256, 2, activation=activationFunction, **kwargs))
  # model.add(layers.Flatten())
  model.add(layers.Dense(1))

  model = compileModel(model, learning_rate)
  return model

activationFunction = "elu"

def create_convolutional_model(shape, learning_rate):
  model = keras.models.Sequential(name="energy_regression_CNN")
  kwargs = dict(kernel_initializer="he_normal", padding="same",)
  model.add(layers.Conv1D(16, 2, activation=activationFunction, input_shape=shape[1:], **kwargs))
  model.add(layers.Conv1D(16, 3, activation=activationFunction, **kwargs))
  model.add(layers.MaxPooling1D(2))
  model.add(layers.Conv1D(16, 3, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(16, 3, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(16, 3, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(16, 3, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(16, 3, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(16, 3, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(16, 3, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(16, 3, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(16, 3, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(16, 3, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(16, 3, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(16, 3, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(16, 3, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(16, 3, activation=activationFunction, **kwargs))
  model.add(layers.MaxPooling1D(2))
  model.add(layers.Conv1D(64, 2, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(64, 2, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(64, 2, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(64, 2, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(64, 2, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(64, 2, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(64, 2, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(64, 2, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(64, 2, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(64, 2, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(64, 2, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(64, 2, activation=activationFunction, **kwargs))
  model.add(layers.MaxPooling1D(2))
  model.add(layers.Conv1D(128, 2, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(128, 2, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(128, 2, activation=activationFunction, **kwargs))
  model.add(layers.MaxPooling1D(2))
  model.add(layers.Conv1D(256, 2, activation=activationFunction, **kwargs))
  model.add(layers.Flatten())
  model.add(layers.Dense(1))

  # model = keras.models.Sequential(name="energy_regression_CNN")
  # kwargs = dict(kernel_initializer="he_normal", padding="same",)
  # # model.add(layers.Conv2D(16, (3, 3), activation="elu", **kwargs))
  # # model.add(layers.Conv2D(16, (3, 3), activation="elu", **kwargs))
  # model.add(layers.Conv1D(16, 2, activation="elu", input_shape=shape[1:], **kwargs))
  # model.add(layers.Conv1D(16, 3, activation="elu", **kwargs))
  # model.add(layers.Conv1D(16, 3, activation="elu", **kwargs))
  # model.add(layers.Conv1D(16, 3, activation="elu", **kwargs))
  # model.add(layers.Conv1D(16, 3, activation="elu", **kwargs))
  # model.add(layers.MaxPooling1D(2))
  # model.add(layers.Conv1D(64, 2, activation="elu", **kwargs))
  # model.add(layers.Conv1D(64, 2, activation="elu", **kwargs))
  # model.add(layers.Conv1D(64, 2, activation="elu", **kwargs))
  # model.add(layers.MaxPooling1D(2))
  # model.add(layers.Conv1D(128, 2, activation="elu", **kwargs))
  # model.add(layers.Conv1D(128, 2, activation="elu", **kwargs))
  # model.add(layers.MaxPooling1D(2))
  # model.add(layers.Conv1D(256, 2, activation="elu", **kwargs))
  # model.add(layers.Flatten())
  # model.add(layers.Dense(1))

  model = compileModel(model, learning_rate)
  return model


def compileModel(model, learning_rate):
  model.compile(
    loss='mean_squared_error',
    optimizer=keras.optimizers.Adam(learning_rate=1e-4))
  return model
# Bias: -0.008 Resolution: 0.458 Batch_Size = 128 Epochs: 2000 v_split: 0.3 val_loss: 0.19568, training_rate = 0.001
  # model = keras.models.Sequential(name="energy_regression_CNN")
  # kwargs = dict(kernel_initializer="he_normal", padding="same",)
  # # model.add(layers.Conv2D(16, (3, 3), activation="elu", **kwargs))
  # # model.add(layers.Conv2D(16, (3, 3), activation="elu", **kwargs))
  # model.add(layers.Conv2D(16, (2, 2), activation="elu", input_shape=X_train.shape[1:], **kwargs))
  # model.add(layers.Conv2D(16, (3, 3), activation="elu", **kwargs))
  # model.add(layers.Conv2D(16, (3, 3), activation="elu", **kwargs))
  # model.add(layers.Conv2D(16, (3, 3), activation="elu", **kwargs))
  # model.add(layers.Conv2D(16, (3, 3), activation="elu", **kwargs))
  # model.add(layers.Conv2D(16, (3, 3), activation="elu", **kwargs))
  # model.add(layers.Conv2D(16, (3, 3), activation="elu", **kwargs))
  # model.add(layers.Conv2D(16, (3, 3), activation="elu", **kwargs))
  # model.add(layers.Conv2D(16, (3, 3), activation="elu", **kwargs))
  # model.add(layers.Conv2D(16, (3, 3), activation="elu", **kwargs))
  # model.add(layers.Conv2D(16, (3, 3), activation="elu", **kwargs))
  # model.add(layers.Conv2D(16, (3, 3), activation="elu", **kwargs))
  # model.add(layers.Conv2D(16, (3, 3), activation="elu", **kwargs))
  # model.add(layers.Conv2D(16, (3, 3), activation="elu", **kwargs))
  # model.add(layers.MaxPooling2D((2, 2)))
  # model.add(layers.Conv2D(64, (2, 2), activation="elu", **kwargs))
  # model.add(layers.Conv2D(64, (2, 2), activation="elu", **kwargs))
  # model.add(layers.Conv2D(64, (2, 2), activation="elu", **kwargs))
  # model.add(layers.MaxPooling2D((2, 2)))
  # model.add(layers.Conv2D(128, (2, 2), activation="elu", **kwargs))
  # model.add(layers.Conv2D(128, (2, 2), activation="elu", **kwargs))
  # model.add(layers.MaxPooling2D((2, 2)))
  # model.add(layers.Conv2D(256, (2, 2), activation="elu", **kwargs))
  # model.add(layers.Flatten())
  # model.add(layers.Dense(1))

# model = create_model()

"""We can have a deeper look at our designed model and inspect the number of adaptive parameters by printing the model `summary`."""

# Compile the model
# model.compile(optimizer="adam", loss="mean_squared_error")
# print(model.summary())

"""### Compile
We now compile the model to prepare it for the training. During the **compile** step, we set a loss/objective function (`mean_squared_error`, as energy reconstruction is a regression task) and set an optimizer. In this case, we use the Adam optimizer with a learning rate of 0.001.
To monitor the resolution and the bias of the model, we add them as a metric.
"""

def resolution(y_true, y_pred):
    """ Metric to control for standart deviation """
    mean, var = tf.nn.moments((y_true - y_pred), axes=[0])
    return tf.sqrt(var)


def bias(y_true, y_pred):
    """ Metric to control for standart deviation """
    mean, var = tf.nn.moments((y_true - y_pred), axes=[0])
    return mean

# model.compile(
#     loss='mean_squared_error',
#     metrics=[bias, resolution],
#     optimizer=keras.optimizers.Adam(learning_rate=1e-3))
# checkpoint_path = "training_1/cp.ckpt"

"""---
### Training
We can now run the training for 20 `epochs` (20-fold iteration over the dataset) using our training data `X_train` and our energy labels `energy_train`.
For each iteration (calculating the gradients, updating the model parameters), we use 128 samples (`batch_size`).

We furthermore can set the fraction of validation data (initially set to `0.1`) that is used to monitor overtraining.

> **Modification section**  
> Feel free to modify the training procedure, for example:
>*   Change (increase) the number of `epochs`.
>*   Modify the batch size via `batch_size`.

"""

# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path,
#     save_weights_only=True,
#     monitor='val_loss',
#     mode='min',
#     save_best_only=True,
#     verbose=1  # Add this line
# )

# fit = model.fit(
#     X_train,
#     energy_train,
#     batch_size=128,
#     epochs=2000,
#     verbose=2,
#     callbacks=[model_checkpoint_callback],
#     validation_split=0.3,
#     workers = 100,
#     use_multiprocessing = True

# )

# The model weights (that are considered the best) are loaded into the
# # model.
# model = create_model()
# model.load_weights(checkpoint_path)

# """### Plot training curves"""

# fig, ax = plt.subplots(1, figsize=(8,5))
# n = np.arange(len(fit.history['loss']))

# ax.plot(n, fit.history['loss'], ls='--', c='k', label='loss')
# ax.plot(n, fit.history['val_loss'], label='val_loss', c='k')

# ax.set_xlabel('Epoch')
# ax.set_ylabel('Loss')
# ax.legend()
# ax.semilogy()
# ax.grid()
# plt.show()

"""---
### Performance of the DNN
After training the model, we can use the test set `X_test` to evaluate the performance of the DNN.  

Particularly interesting are the resolution and the bias of the method. A bias close to 0 is desirable. A resolution below 3 EeV can be regarded as good.
"""

# energy_pred = model.predict(X_test, batch_size=128, verbose=1)[:,0]

# diff = energy_pred - energy_test
# resolution = np.std(diff)
# plt.figure()
# plt.hist(diff, bins=100)
# plt.xlabel('$E_\mathrm{rec} - E_\mathrm{true}$')
# plt.ylabel('# Events')
# plt.text(0.95, 0.95, '$\sigma = %.3f$ EeV' % resolution, ha='right', va='top', transform=plt.gca().transAxes)
# plt.text(0.95, 0.85, '$\mu = %.1f$ EeV' % diff.mean(), ha='right', va='top', transform=plt.gca().transAxes)
# plt.grid()
# plt.xlim(-10, 10)
# plt.tight_layout()

"""After estimating the bias and the resolution, we can now inspect the reconstruction via a scatter plot.  

Furthermore, we can study the energy dependence of the resolution. With increasing energy, the performance increases due to the lower sampling fluctuation of the ground-based particle detectors and the larger footprints.
"""

#Pearson Correlation coefficient for next week
# x = [3, 10, 30, 100]
# labels = ["3", "10", "30", "100"]

# diff = energy_pred - energy_test

# # Embedding plot
# fig, axes = plt.subplots(1, 2, figsize=(20, 9))
# axes[0].scatter(energy_test, energy_pred, s=3, alpha=0.60)
# axes[0].set_xlabel(r"$E_{true}\;/\;\mathrm{EeV}$")
# axes[0].set_ylabel(r"$E_{DNN}\;/\;\mathrm{EeV}$")

# stat_box = r"$\mu = $ %.3f" % np.mean(diff) + " / EeV" + "\n" + "$\sigma = $ %.3f" % np.std(diff) + " / EeV"
# axes[0].text(0.95, 0.2, stat_box, verticalalignment="top", horizontalalignment="right",
#           transform=axes[0].transAxes, backgroundcolor="w")
# axes[0].plot([np.min(energy_test), np.max(energy_test)],
#              [np.min(energy_test), np.max(energy_test)], color="red")

# sns.regplot(x=energy_test, y=diff / energy_test, x_estimator=np.std, x_bins=12,
#             fit_reg=False, color="royalblue", ax=axes[1])
# axes[1].tick_params(axis="both", which="major")
# axes[1].set(xscale="log")
# plt.xticks(x, labels)

# axes[1].set_xlabel(r"$E_{true}\;/\;\mathrm{EeV}$")
# axes[1].set_ylabel(r"$\sigma_{E}/E$")
# axes[1].set_ylim(0, 0.2)
# plt.show()

