from numpy import *

def pcnfwd(inputs, weights):
	activations = dot(inputs, weights)
	activations = where(activations > 0, 1, 0)
	return activations

def pcntrain(inputs, targets, eta, iters):
	n = 1 # number of neurons
	inputs = concatenate((-ones((shape(inputs)[0],1)), inputs), axis=1) # <===
	weights = initialweights(shape(inputs)[1], n)
	for i in range(iters):
		activations = pcnfwd(inputs, weights)
		weights = weights + eta * dot(transpose(inputs), targets - activations)
		print "Iteration: " + str(i+1)
		print weights
	return activations, weights
	
def initialweights(m,n):
	return random.rand(m, n) * 0.1 - 0.05


