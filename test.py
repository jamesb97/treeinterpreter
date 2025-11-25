import numpy as np
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
rf.fit(trainX, trainy)
 
prediction, bias, contributions = ti.predict(rf, testX)
 
# Prediction is the sum of bias and feature contributions::
 
assert(np.allclose(prediction, bias + np.sum(contributions, axis=1)))
assert(np.allclose(rf.predict(testX), bias + np.sum(contributions, axis=1)))