Machine learning implementations in python
------------------------------------------

- Logistic Regression

    * gradient descent
    * stochastic gradient descent (online learning)
    * modified stochastic gradient descent implementation (random indexed records)


    python> import logisticRegression as lr
    python> dataset, labels = lr.loadDataset("train.csv", ",");
    python> theta = lr.gradDescent(dataset, labels)

  You can also run the script directly as following:

    $ python logisticRegression.py {datasetfilename} {separator}

- *more to follow*


-- Edited by http://dillinger.io/ -- 
