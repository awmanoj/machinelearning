import mlp
from numpy import *
import sys
inputs = array(([[0,0],[0,1],[1,0],[1,1]]))
targets = array(([0],[1],[1],[0]))
targets2 = array(([0],[1],[1],[1]))
targets3 = array(([0],[0],[0],[1]))

if sys.argv[2] == "1":
	mlp.train(inputs, targets, 2, 1, 0.25, int(sys.argv[1]))
elif sys.argv[2] == "2":
	mlp.train(inputs, targets2, 2, 1, 0.25, int(sys.argv[1]))
else:
	mlp.train(inputs, targets3, 2, 1, 0.25, int(sys.argv[1]))
