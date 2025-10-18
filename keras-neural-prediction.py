import numpy as np
import pandas as pd
import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler

dataInput = 'datasets/housing/cal_housing.csv'
fileDelimiter = ','
testingSampleSize = 1000
firstLayerNeurons = 64  
secondLayerNeurons = 32
thirdLayerNeurons = 16

Yscaler = 10000

firstColumn = 'median_income'
secondColumn = 'median_house_value'

trainingData = pd.read_csv(dataInput, delimiter = fileDelimiter)
testingData = trainingData.sample(testingSampleSize, ignore_index=True)
trainingData = trainingData.drop(testingData.index)

testingHere = trainingData.to_numpy()
testingHere = np.delete(testingHere, (8, 9), 1) # deleting median_house_value and ocean_proximity column

trainingArrayX = trainingData.to_numpy()
trainingArrayX = np.delete(trainingArrayX, (8, 9), 1).astype('float32')
trainingArrayY = trainingData[secondColumn].to_numpy().astype('float32')
trainingArrayY = trainingArrayY

testingArrayX = testingData.to_numpy()
testingArrayX = np.delete(testingArrayX, (8, 9), 1).astype('float32')
testingArrayY = testingData[secondColumn].to_numpy().astype('float32')
testingArrayY = testingArrayY

# Scaling data
scaler = StandardScaler()
trainingArrayX = scaler.fit_transform(trainingArrayX)
testingArrayX = scaler.fit_transform(testingArrayX)
trainingArrayY = trainingArrayY / Yscaler
testingArrayY = testingArrayY / Yscaler

model = keras.models.Sequential()
model.add(keras.layers.Dense(firstLayerNeurons, activation='relu', input_shape = (8,)))
model.add(keras.layers.Dense(secondLayerNeurons, activation='relu'))
model.add(keras.layers.Dense(thirdLayerNeurons, activation='relu'))
model.add(keras.layers.Dense(1)) # Regresion

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

model.fit(trainingArrayX, trainingArrayY, epochs = 5, batch_size=32, validation_split=0.1)

testLoss, testAcc = model.evaluate(testingArrayX, testingArrayY)
print(f"Test accuracy: {testAcc:.4f}")