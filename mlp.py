from numpy import *
import math

def sigmoid(z):
	return 1.0 / (1 + exp(-z))

def mlpfwd(inputs, weights):
	activations = dot(inputs, weights)
	activations = sigmoid(activations)
	return activations, weights
	
def train(inputs, targets, n_hidden, n_output, eta, iteration):
	change = range(shape(inputs)[0])
	inputs = concatenate((-ones((shape(inputs)[0],1)), inputs), axis = 1)

	dim = shape(inputs)[1]
	weights1 = (random.rand(dim, n_hidden) - 0.5)/10*math.sqrt(dim-1)
	weights2 = (random.rand(dim, n_output) - 0.5)/10*math.sqrt(dim-1)

	updatew1 = zeros((shape(weights1)))
	updatew2 = zeros((shape(weights2)))
	for i in range(iteration):
		# forward phase
		activations1, weights1 = mlpfwd(inputs, weights1)
		activations1 = concatenate((-ones((shape(activations1)[0],1)), activations1), axis = 1)
		activations2, weights2 = mlpfwd(activations1, weights2)
		
		# backward phase
		deltao = (targets - activations2) * activations2 * (1.0 - activations2)
		deltah = (activations1 * (1.0 - activations1)) * (dot(deltao, transpose(weights2)))

		if i%100 == 0:
			error = 0.5 * sum((targets - activations2)**2)
			print "Iteration " + str(i+1) + ": Error = " + str(error)
			if error < 0.015:
				break

		updatew1 = eta * (dot(transpose(inputs), deltah[:,:-1])) + 0.9*updatew1
		updatew2 = eta * (dot(transpose(activations1), deltao)) + 0.9*updatew2

		weights1 += updatew1
		weights2 += updatew2

		random.shuffle(change)
		inputs = inputs[change,:]
		targets = targets[change,:]

	print activations2
