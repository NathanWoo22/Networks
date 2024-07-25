import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
layers = keras.layers
import os


# 9.87
# def create_convolutional_model(shape, learning_rate):
#   model = keras.models.Sequential(name="energy_regression_CNN")
#   kwargs = dict(kernel_initializer="he_normal", padding="same",)
#   activationFunction = "elu"
#   model.add(layers.Conv1D(2, 2, activation=activationFunction, input_shape=shape[1:], **kwargs))
#   model.add(layers.Conv1D(2, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(2, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(2, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(2, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(2, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(2, 2, activation=activationFunction, **kwargs))
#   model.add(layers.MaxPooling1D(2))
#   model.add(layers.Conv1D(4, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(4, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(4, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(4, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(4, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(4, 2, activation=activationFunction, **kwargs))
#   model.add(layers.MaxPooling1D(2))
#   model.add(layers.Conv1D(8, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(8, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(8, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(8, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(8, 2, activation=activationFunction, **kwargs))
#   model.add(layers.MaxPooling1D(2))
#   model.add(layers.Conv1D(16, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(16, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(16, 2, activation=activationFunction, **kwargs))
#   model.add(layers.MaxPooling1D(2))
#   model.add(layers.Conv1D(32, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(32, 2, activation=activationFunction, **kwargs))
#   model.add(layers.MaxPooling1D(2))
#   model.add(layers.Conv1D(64, 2, activation=activationFunction, **kwargs))
#   model.add(layers.MaxPooling1D(2))
#   model.add(layers.Conv1D(128, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Flatten())
#   model.add(layers.Dense(1, activation="linear"))
#   model = compileModel(model, learning_rate)
#   return model

#10.56
# def create_convolutional_model(shape, learning_rate):
#   model = keras.models.Sequential(name="energy_regression_CNN")
#   kwargs = dict(kernel_initializer="he_normal", padding="same",)
#   activationFunction = "elu"
#   model.add(layers.Conv1D(2, 2, activation=activationFunction, input_shape=shape[1:], **kwargs))
#   model.add(layers.Conv1D(2, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(2, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(2, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(2, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(2, 2, activation=activationFunction, **kwargs))
#   model.add(layers.MaxPooling1D(2))
#   model.add(layers.Conv1D(4, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(4, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(4, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(4, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(4, 2, activation=activationFunction, **kwargs))
#   model.add(layers.MaxPooling1D(2))
#   model.add(layers.Conv1D(8, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(8, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(8, 2, activation=activationFunction, **kwargs))
#   model.add(layers.MaxPooling1D(2))
#   model.add(layers.Conv1D(16, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Conv1D(16, 2, activation=activationFunction, **kwargs))
#   model.add(layers.MaxPooling1D(2))
#   model.add(layers.Conv1D(32, 2, activation=activationFunction, **kwargs))
#   model.add(layers.MaxPooling1D(2))
#   model.add(layers.Conv1D(64, 2, activation=activationFunction, **kwargs))
#   # model.add(layers.MaxPooling1D(2))
#   # model.add(layers.Conv1D(256, 2, activation=activationFunction, **kwargs))
#   model.add(layers.Flatten())
  
#   model.add(layers.Dense(1, activation="linear"))
#   model = compileModel(model, learning_rate)
#   return model

# 10.84
def create_convolutional_model(shape, learning_rate):
  model = keras.models.Sequential(name="energy_regression_CNN")
  kwargs = dict(kernel_initializer="he_normal", padding="same",)
  activationFunction = "elu"
  model.add(layers.Conv1D(2, 2, activation=activationFunction, input_shape=shape[1:], **kwargs))
  model.add(layers.Conv1D(2, 2, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(2, 2, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(2, 2, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(2, 2, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(2, 2, activation=activationFunction, **kwargs))
  model.add(layers.MaxPooling1D(2))
  model.add(layers.Conv1D(4, 2, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(4, 2, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(4, 2, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(4, 2, activation=activationFunction, **kwargs))
  model.add(layers.MaxPooling1D(2))
  model.add(layers.Conv1D(8, 2, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(8, 2, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(8, 2, activation=activationFunction, **kwargs))
  model.add(layers.MaxPooling1D(2))
  model.add(layers.Conv1D(16, 2, activation=activationFunction, **kwargs))
  model.add(layers.Conv1D(16, 2, activation=activationFunction, **kwargs))
  model.add(layers.MaxPooling1D(2))
  model.add(layers.Conv1D(32, 2, activation=activationFunction, **kwargs))
  model.add(layers.MaxPooling1D(2))
  model.add(layers.Conv1D(64, 2, activation=activationFunction, **kwargs))
  model.add(layers.Flatten())
  
  model.add(layers.Dense(1, activation="linear"))
  model = compileModel(model, learning_rate)
  return model



def compileModel(model, learning_rate):
  model.compile(
    loss='mean_squared_error',
    optimizer=keras.optimizers.Adam(learning_rate))
  return model
