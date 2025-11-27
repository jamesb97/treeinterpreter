===============================
TreeInterpreter
===============================

Package for interpreting scikit-learn's decision tree and random forest predictions.
Allows decomposing each prediction into bias and feature contribution components as described in http://blog.datadive.net/interpreting-random-forests/. For a dataset with ``n`` features, each prediction on the dataset is decomposed  as ``prediction = bias + feature_1_contribution + ... + feature_n_contribution``.

It works on scikit-learn's

* DecisionTreeRegressor
* DecisionTreeClassifier
* ExtraTreeRegressor
* ExtraTreeClassifier
* RandomForestRegressor
* RandomForestClassifier
* ExtraTreesRegressor
* ExtraTreesClassifier

Free software: BSD license

Dependencies
------------

- scikit-learn 0.17+


Installation
------------
The easiest way to install the package is via ``pip``::

    $ pip install treeinterpreter

Usage
-----
::

 from treeinterpreter import treeinterpreter as ti
 # fit a scikit-learn's regressor model

 # Load iris dataset
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

 # Split dataset into training and test sets
from sklearn.cross_validation import train_test_split
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=42)

 # fit a scikit-learn's random forest regressor model
from sklearn.ensemble import RandomForestRegressor
 rf = RandomForestRegressor()
 rf.fit(trainX, trainY)
 
 prediction, bias, contributions = ti.predict(rf, testX)
 
Prediction is the sum of bias and feature contributions::
 
 assert(numpy.allclose(prediction, bias + np.sum(contributions, axis=1)))
 assert(numpy.allclose(rf.predict(testX), bias + np.sum(contributions, axis=1)))


More usage examples at http://blog.datadive.net/random-forest-interpretation-with-scikit-learn/.

 
