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

# converts a list of zenith angles to a list of the same angle with different lengths
def expandZenithAngles(zenithAngles, newLength):
    
    zenithAnglesList = list([list([x]) for x in zenithAngles])
    extendedZenithList = []
    for i in range(len(Xcx)):
        newZenithList = []
        for j in range(len(Xcx[i])):
            newZenithList.append(float(zenithAnglesList[i][0]))    
        extendedZenithList.append(newZenithList)

    zenithAngles = extendedZenithList
    for i in range(len(zenithAngles)):
        zenithAngles[i] = np.array(zenithAngles[i]) 

    return zenithAngles

def expandXcxdEdX(Xcx, dEdX, maxLength):
    for i, array in enumerate(Xcx):
        while len(Xcx[i]) < maxLength:
            Xcx[i] = np.append(Xcx[i], 0)
            dEdX[i] = np.append(dEdX[i], 0)

    for i, array in enumerate(dEdX):
        maxHeight = max(dEdX[i])
        for j, value in enumerate(dEdX[i]):
            dEdX[i][j] /= maxHeight

    return Xcx, dEdX

def runModel(model, learning_rate, batch_size, epochs, validation_split):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./best_saved_network',
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
        verbose=2,
        callbacks=[model_checkpoint_callback],
        validation_split=validation_split,
        workers = 100,
        use_multiprocessing = True
    )

    return fit

def plotdEdX(Xcx, dEdX):
    for i in range(100):
        plt.scatter(Xcx[i], dEdX[i], s=0.5)
        plt.xlabel('X')
        plt.ylabel('dEdX')
        plt.title('Energy deposit per cm')

    plt.show()
    plt.savefig('Energy function plot', dpi = 1000)


# if len(sys.argv) > 1:
#     file = sys.argv[1]
# else:
#     print("Usage: conex_read.py <rootfile.root>")
#     exit()

folder_path = "/Data/Simulations/Conex_Flat_lnA/data/Conex_170-205_Prod1/showers"
fileNames = glob.glob(os.path.join(folder_path, '*.root'))


XcxAll = []
dEdXAll = []
zenithAll = []
massAll = []
count = 0
# Read data for network
for fileName in fileNames:
    count += 1
    if count == 20:
        break
    print("Reading file " + fileName)
    pattern = r'_(\d+)\.root'
    match = re.search(pattern, fileName)
    if match:
        mass = match.group(1)
        print(f"Mass is: {mass}")
    else:
        print("No match found.")

    Xcx, dEdX, zenith = conex_read.readRoot(fileName)

    # Format all inputs to the network
    # maxLength = max(len(arr) for arr in Xcx)
    maxLength = 700
    Xcx, dEdX = expandXcxdEdX(Xcx, dEdX, maxLength)
    zenith = expandZenithAngles(zenith, maxLength)

    Xcx = np.vstack(Xcx)
    dEdX = np.vstack(dEdX)
    zenith = np.vstack(zenith)

    XcxAll.append(Xcx)
    dEdXAll.append(dEdX)
    zenithAll.append(zenith)
    massAll.append(float(mass))
    print("Finished reading file " + fileName)


X = np.stack([XcxAll, dEdXAll, zenithAll], axis = -1)
# mass = np.full(100, 20, dtype=np.float32)

X_train, X_test = np.split(X, [50 * len(fileNames)])
mass_train, mass_test = np.split(massAll, [50 * len(fileNames)])

learning_rate = 1e-3
batch_size = 50
epochs =1000
validation_split = 0.5
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