Machine learning implementations in python
------------------------------------------

- Logistic Regression

    * gradient descent
    * stochastic gradient descent (online learning)
    * modified stochastic gradient descent implementation (random indexed records)
    
```
python> import logisticRegression as lr
python> dataset, labels = lr.loadDataset("train.csv", ",");
python> theta = lr.gradDescent(dataset, labels)
```

  You can also run the script directly as following:

```
$ python logisticRegression.py {datasetfilename} {separator}
```

- Perceptron 

   * single layer neural network

- Multi layer perceptron

   * multi layer neural network (one hidden + one output)

```
python> import mlp
python> from numpy import *
python> inputs = array(([[0,0],[0,1],[1,0],[1,1]]))
python> targets = array(([0],[1],[1],[0]))
python> mlp.train(inputs, targets, 2, 1, 0.25, 1000) # n_hidden, n_output, eta, n_iterations
```


-- Edited by http://dillinger.io/ -- 

