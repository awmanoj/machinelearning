from numpy import *

def train(inputs, targets):
	inputs = concatenate((-ones((shape(inputs)[0],1)), inputs), axis = 1)
	beta = dot(dot(linalg.inv(dot(transpose(inputs), inputs)), transpose(inputs)), targets)
	return beta

def classify(inputs, beta):
	inputs = concatenate((-ones((shape(inputs)[0],1)), inputs), axis = 1)
	print dot(inputs, beta)

if __name__ == "__main__":
	inputs = array(([[0,0],[0,1],[1,0],[1,1]]))
	targets = array(([0],[1],[1],[1]))

	beta = train(inputs, targets)
	classify(inputs, beta)
