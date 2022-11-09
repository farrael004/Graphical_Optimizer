import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.ensemble import GradientBoostingRegressor

from hyperoptimize import GraphicalOptimizer

# Creating data

import numpy as np
from matplotlib import pyplot as plt
from time import sleep

X = np.linspace(-2,2,1000)
y = np.sin(3*X) - X**2 + 0.7*X

plt.plot(X,y)
plt.show()

# Creating model, prediction and performance functions

def modelFunction(params, X_train, y_train):

    model = np.sin(3*params['X']) - params['X']**2 + 0.7*params['X']
    #sleep(1)
    return model


def predictionFunction(model, X):
    y_pred = model
    return y_pred


def performanceFunction(y_test, y_pred):
    
    return {"score": y_pred}


# Creating hyperparameter dictionary

hyperparameters_bayesian = {'X': [-2.0, 2.0]}  # Upper and lower bounds

hyperparameters_grid_and_random = {'X': np.linspace(-2, 2, 100).tolist()}  # Upper and lower bounds

# Performing optimization

opt = GraphicalOptimizer(ModelFunction=modelFunction,
                         PredictionFunction=predictionFunction,
                         PerformanceFunction=performanceFunction,
                         performanceParameter="score",
                         hyperparameters=hyperparameters_grid_and_random,
                         optimizer="random",
                         maxNumCombinations=13,
                         crossValidation=2,
                         maxNumOfParallelProcesses=2,
                         parallelCombinations=1,
                         seed=1)



opt.fit(X, y)
