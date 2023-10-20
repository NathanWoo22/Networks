import uproot
import numpy as np
import sys
import matplotlib.pyplot as plt
import keras
import conex_read
import convolutional_network as cn

if len(sys.argv) > 1:
    file = sys.argv[1]
else:
    print("Usage: conex_read.py <rootfile.root>")
    exit()

Xcx, dEdX, zenith = conex_read.readRoot(file)

maxLength = max(len(arr) for arr in Xcx)
for i, array in enumerate(Xcx):
    while len(Xcx[i]) < maxLength:
        Xcx[i] = np.append(Xcx[i], 0)
        dEdX[i] = np.append(dEdX[i], 0)


zenith = list([list([x]) for x in zenith])
newZenithArray = []
for i in range(len(Xcx)):
    newZenith = []
    for j in range(len(Xcx[i])):
        newZenith.append(float(zenith[i][0]))    
    newZenithArray.append(newZenith)

zenith = newZenithArray
for i in range(len(zenith)):
    zenith[i] = np.array(zenith[i]) 


for i, array in enumerate(dEdX):
    maxHeight = max(dEdX[i])
    for j, value in enumerate(dEdX[i]):
        dEdX[i][j] /= maxHeight


Xcx = np.vstack(Xcx)
dEdX = np.vstack(dEdX)
zenith = np.vstack(zenith)
print(Xcx.shape)
print(dEdX.shape)
print(zenith.shape)
zenith = np.vstack(zenith)


X = np.stack([Xcx, dEdX, zenith], axis = -1)

for i in range(100):
    plt.scatter(Xcx[i], dEdX[i], s=0.5)
    plt.xlabel('X')
    plt.ylabel('dEdX')
    plt.title('Energy deposit per cm')

plt.show()
plt.savefig('Energy function plot', dpi = 1000)

# Create the model
model = cn.create_model(X.shape)

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")
print(model.summary())


mass = np.full(100, 20, dtype=np.float32)

X_train, X_test = np.split(X, [50])
mass_train, mass_test = np.split(mass, [50])

model.compile(
    loss='mean_squared_error',
    optimizer=keras.optimizers.Adam(learning_rate=1e-4))

fit = model.fit(
    X_train,
    mass_train,
    batch_size=16,
    epochs=1000,
    verbose=2,
    # callbacks=[model_checkpoint_callback],
    validation_split=0.5,
    workers = 100,
    use_multiprocessing = True

)

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