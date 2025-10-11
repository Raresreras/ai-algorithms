import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib.animation import FuncAnimation

matplotlib.use('Qt5Agg')

dataset = 'datasets/housing/cal_housing.csv'
delimiter = ','
columns1 = 'median_income'
columns2 = 'median_house_value'
iterationCount = 10000
trainingRateValue = 0.02

# More info here -> https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/

class LinearRegression:
	def __init__(self):
		self.parameters = {}

	# y = m * x + c
	# calculates the prediction
	def forward_propagation(self, trainInput):
		m = self.parameters['m']
		c = self.parameters['c']
		predictions = np.multiply(m, trainInput) + c
		return predictions

	# calculates the cost_function
	def cost_function(self, predictions, trainOutput):
		cost = np.mean((trainOutput - predictions) ** 2)
		return cost

	# gradient descent alogirthm
	# Calculates the derivatives of the residual cost function
	# We use these derivatives to update the parameters in the function update_parameters
	def backward_propagation(self, trainInput, trainOutput, predictions):
		derivatives = {}
		df = (predictions - trainOutput)
		dm = 2 * np.mean(np.multiply(trainInput, df))
		dc = 2 * np.mean(df)
		derivatives['dm'] = dm
		derivatives['dc'] = dc
		return derivatives

	def update_parameters(self, derivatives, learningRate):
		self.parameters['m'] = self.parameters['m'] - learningRate * derivatives['dm']
		self.parameters['c'] = self.parameters['c'] - learningRate * derivatives['dc']

	def train(self, trainInput, trainOutput, learningRate, iters):
		self.parameters['m'] = np.random.uniform(0, 1) * (-1)
		self.parameters['c'] = np.random.uniform(0, 1) * (-1)

		self.loss = []

		for iteration in range(1, iters):
			predictions = self.forward_propagation(trainInput)
			cost = self.cost_function(predictions, trainOutput)
			derivatives = self.backward_propagation(trainInput, trainOutput, predictions)
			self.update_parameters(derivatives, learningRate)
			self.loss.append(cost)
			print("Iteration = {}, Loss = {}".format(iteration, cost))

		return self.parameters, self.loss


# Data input and conversion to numpy array
dataFrame = pd.read_csv(dataset, delimiter=delimiter)

print(dataFrame.columns)

# 100 sample testing size
testingData = dataFrame.sample(500, ignore_index = True)
dataFrame = dataFrame.drop(testingData.index)

# converting dataframes to numpy arrays
trainingInput = dataFrame[columns1].to_numpy()
trainingOutput = dataFrame[columns2].to_numpy()
testingInput = testingData[columns1].to_numpy()
testingOutput = testingData[columns2].to_numpy()

linear_reg = LinearRegression()
parameters, loss = linear_reg.train(trainingInput, trainingOutput, trainingRateValue, 10000)

print(parameters)

Residual = 0 # General errors
MSE = 0 # Penalized large errors heavily

# Testing the parameters
for input, output in zip(testingInput, testingOutput):
	predictedValue = parameters['m'] * input + parameters['c']
	Residual += abs(output - predictedValue)
	MSE += (output - predictedValue) ** 2

	# print(predictedValue, output)

print("Residual: " + str(Residual))
print("MSE: " + str( MSE / testingInput.size))
print("RMSE: " + str(np.sqrt(MSE) / testingInput.size))