from numpy import *
import sys

def loadDataset(datasetfile, sep):
  dataMat = []
  labelMat = []
  fr = open(datasetfile,"r")
  for line in fr.readlines():
    line = line.strip().split(sep)
    features = [float(x) for x in line[0:-1]]
    label = line[-1]
    dataMat.append([1.0] + features)
    labelMat.append(float(label))
  return dataMat, labelMat

def sigmoid(z):
  return 1.0 / (1 + exp(-z))

def printmatinfo(h):
  print "================"
  print "matrix: "
  print h
  print "shape: "
  print shape(h)
  print "================"

def gradDescent(dataMatIn, classLabels, alpha = 0.01, iters = 500):
  X = mat(dataMatIn)
  y = mat(classLabels).transpose();
  m,n = shape(X)
  theta = ones((n,1))
  for k in range(iters):
    h = sigmoid(X*theta)
    e = h - y
    theta = theta - (alpha/m) * (X.transpose() * e)
  return theta

def stocGradDescent(dataMatIn, classLabels, alpha = 0.01, iters = 150):
  m,n = shape(dataMatIn)
  theta = ones(n)
  for i in range(m):
    h = sigmoid(sum(dataMatIn[i]*theta))
    e = h - classLabels[i]
    theta = theta - alpha * e * dataMatIn[i]
  return theta

def stocGradDescentWithIter(dataMatIn, classLabels, iters = 150):
  m, n = shape(dataMatIn)
  theta = ones(n)
  for i in range(iters):
    dataIndex = range(m)
    for j in range(m):
      alpha = 0.01 + (4.0 / (1.0 + j + i))
      randIdx = int(random.uniform(0, len(dataIndex)))
      h = sigmoid(sum(dataMatIn[randIdx]*theta))
      e = h - classLabels[randIdx]
      theta = theta - alpha * e * dataMatIn[randIdx]
      del(dataIndex[randIndex])
  return theta

if __name__ == '__main__':
  ds, labels = loadDataset(sys.argv[1], sys.argv[2])
  theta = gradDescent(ds, labels)
  print theta
