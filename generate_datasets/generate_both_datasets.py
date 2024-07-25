import numpy as np

sibyll = np.load("Sibyll.npz")
epos = np.load("Epos.npz")
sibyll = sibyll['showers']
epos = epos['showers']

sibyll_train, sibyll_test = np.split(sibyll, [-100000])
epos_train, epos_test = np.split(epos, [-100000])

print(sibyll_test.shape)
print(sibyll_train.shape)

print(epos_test.shape)
print(epos_train.shape)

training = np.concatenate((sibyll_train, epos_train), axis=0)
testing = np.concatenate((sibyll_test, epos_test), axis=0)

print(training.shape)
print(testing.shape)

np.savez("Epos_test.npz", showers=epos_test)
np.savez("Sibyll_test.npz", showers=sibyll_test)
np.savez("Both_training.npz", showers=training)
np.savez("Both_testing.npz", showers=testing)