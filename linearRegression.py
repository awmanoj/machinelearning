from numpy import *

def linreg(inputs, targets):
	inputs = concatenate((-ones((shape(inputs)[0],1)), inputs), axis = 1)
	beta = dot(dot(linalg.inv(dot(transpose(inputs), inputs)), transpose(inputs)), targets)
	print dot(inputs, beta)
