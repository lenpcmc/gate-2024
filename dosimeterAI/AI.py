import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib as mp
from matplotlib import pyplot as plt

df = pd.read_csv('/home/uakgun/src/gate-2024/AIdataprocessing/AIdata.csv', header = None)
dupe = pd.read_csv('/home/uakgun/src/gate-2024/AIdataprocessing/AIdata.csv', header = None)

columnnames = ['Particle Energy (MeV)', 'Photon Count', 'Beam Duration (s)', 'eDep (MeV)']

df.columns = columnnames


#splitting into training and testing data
train_dataset = df.sample(frac = .99, random_state = 0)
test_dataset = df.drop(train_dataset.index)

xtrain = train_dataset.drop ('eDep (MeV)', axis = 1)
xtest = test_dataset.drop ('eDep (MeV)', axis = 1)
ytrain = train_dataset['eDep (MeV)']
ytest = test_dataset['eDep (MeV)']


#normalizing the dataset
xtr = xtrain.to_numpy()
xte = xtest.to_numpy()

normalizer1 = tf.keras.layers.Normalization(axis = 0)
normalizer1.adapt(xtr)
normalizer2 = tf.keras.layers.Normalization(axis = 0)
normalizer2.adapt(xte)


#creating our model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'linear'),
])

model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mae'])
model.summary()

#the validation split in the line below seems to make the mae more stable the lower it is... to a point. Not sure how it affects
#the actual prediction values yet. Keeping the validation split to ~0.1 to ~0.2 seems to be a good medium though.
history = model.fit(xtr, ytrain, validation_split = 0.1, epochs = 5000)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label = 'Training loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

predictions = model.predict(xte[:10])
print("Predicted values are: ", predictions)
print("Real values are: ", ytest[:10])