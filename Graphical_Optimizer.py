## Loading Data

import pandas as pd
import numpy as np
df1 = pd.read_csv('california_housing_test.csv')
df1 =df1.dropna()
X = df1.copy()
X.pop('median_house_value')
y = df1.median_house_value.copy()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) #Create standardization and apply to train data
X_test = sc.transform(X_test)       #Apply created standardization to new data
X_val = sc.transform(X_val)         #Apply created standardization to new data

from sklearn.decomposition import PCA
pca = PCA(n_components = 0.9, svd_solver='full')
X_train = pca.fit_transform(X_train) #Create PCA and apply to train data
X_test = pca.transform(X_test)       #Apply created PCA to new data
X_val = pca.transform(X_val)         #Apply created normalization to new data

# Creating model, predict and performance functions

from sklearn.ensemble import GradientBoostingRegressor

def ModelFunction(params, X_train, y_train):
    gbr = GradientBoostingRegressor(n_estimators      = 6000,
                                    learning_rate     = params['learning_rate'],
                                    max_depth         = params['max_depth'],
                                    max_features      = params['max_features'],
                                    min_samples_leaf  = 10,
                                    min_samples_split = 8,
                                    random_state=42)

    model = gbr.fit(X_train, y_train)
    return model

def PredictFunction(model, X):
    y_pred = model.predict(X)
    return y_pred

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def PerformanceFunction(y_test, y_pred):
    model_mae = mean_absolute_error(y_test, y_pred)
    model_mse = mean_squared_error(y_test, y_pred)
    model_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    model_r2 = r2_score(y_test, y_pred)
    
    
    model_results = {"Mean Absolute Error (MAE)": model_mae,
                     "Mean Squared Error (MSE)": model_mse,
                     "Root Mean Squared Error (RMSE)": model_rmse,
                     "Adjusted R^2 Score": model_r2}
    return model_results

# Function that will update table

def print_status(self): # Shows the parameters used and accuracy attained of the search so far.
        global app
        results = pd.DataFrame(dict(self.scores, **self.params), index=[0])
        app.table.model.df = pd.concat([app.table.model.df, results], ignore_index=True, axis=0)

        app.table.redraw()
        
# Custom Estimator wrapper

from skopt.learning.gaussian_process.kernels import Hyperparameter
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

hyperparameters = {'n_estimators':      [5000,6000], # Upper and lower bounds
                   'learning_rate':     [0.001,0.01], # Upper and lower bounds
                   'max_depth':         [2,6], # Upper and lower bounds
                   'max_features':      ['sqrt','log2','auto'], # Categorical bounds
                   'min_samples_leaf':  [1,21], # Upper and lower bounds
                   'min_samples_split': [1,16],} # Upper and lower bounds

class WrapperEstimator(BaseEstimator):
    def __init__(self, learning_rate=0.1, max_depth=4, max_features='sqrt'):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features = max_features
        self.variables = hyperparameters

    def fit(self, X, y):
        self.params = {
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'max_features': self.max_features
        }

        X, y = check_X_y(X, y, accept_sparse=True)

        self.model = ModelFunction(self.params, X, y)

        self.is_fitted_ = True
        
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        prediction = PredictFunction(self.model,X)
        return prediction

    def score(self, X, y, sample_weight=None):
        model = self.model
        y_pred = self.predict(X)
        self.scores = PerformanceFunction(y, y_pred)
        print_status(self)
        return self.scores['Adjusted R^2 Score']
    
# Optimizer types

## Bayesian

def BayesianOpt(X_train, y_train):
    global app

    from skopt.space import Real, Categorical, Integer
    from skopt import BayesSearchCV
    from sklearn.ensemble import GradientBoostingRegressor

    WrapperEstimator().set_params()
    
    bayReg = BayesSearchCV(
        WrapperEstimator(),
        {
            'learning_rate':     Real(0.001,0.01),
            'max_depth':         Integer(2,6),
            'max_features':      Categorical(['sqrt','log2','auto']),
        },
        random_state = 42 ,
        verbose = 0,
        n_iter = 100 ,
        cv = 2 ,
        n_jobs = -1 ,
        n_points = 2
    )

    model1 = bayReg.fit(X_train, y_train) # Included callback function that is called after every iteration
    
## Grid

def GridOpt(X_train, y_train):
    from skopt.space import Real, Categorical, Integer
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import GradientBoostingRegressor

    grid = GridSearchCV(
        WrapperEstimator(),
        {
            'max_features': ['sqrt','log2'] ,
            'learning_rate': np.linspace(0.1,0.3,10) ,
            'max_depth': range(3,7) ,
        },
        verbose = 0,
        cv = 2 ,
        n_jobs = -1 ,
    )
    grid.cv_results_ = {}
    model1 = grid.fit(X_train, y_train) # Included callback function that is called after every iteration

    app.table.model.df = pd.DataFrame(grid.cv_results_).drop(['params'], axis=1)
    app.table.redraw
    
## Random

def RandomOpt(X_train, y_train):    
    from skopt.space import Real, Categorical, Integer
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import GradientBoostingRegressor

    randomSearch = RandomizedSearchCV(
        WrapperEstimator(),
        {
            'max_features': ['sqrt','log2'] ,
            'learning_rate': np.linspace(0.1,0.3,10) ,
            'max_depth': range(3,7) ,
        },
        random_state = 42 ,
        verbose = 0,
        n_iter = 5 ,
        cv = 3 ,
        n_jobs = -1 ,
    )

    model1 = randomSearch.fit(X_train, y_train) # Included callback function that is called after every iteration
    
    #app.table.model.df = pd.DataFrame(randomSearch.cv_results_).drop(['params'], axis=1)
    app.table.redraw()
    
# Creating app class

from tkinter import *
import tkinter as tk
from pandastable import Table, TableModel, config
import threading

class App(Frame, threading.Thread):
        def __init__(self, parent=None):
            self.parent = parent
            threading.Thread.__init__(self)
            self.start()
            
        def callback(self):
            self.root.quit()
            
        def run(self):
            Frame.__init__(self)
            self.main = self.master
            self.main.geometry('600x400+200+100')
            self.main.title('Optimizer')
            self.f = Frame(self.main)
            self.f.pack(fill=BOTH,expand=1)
            df = pd.DataFrame()
            self.table = pt = Table(self.f, dataframe=df,
                                    showtoolbar=True, showstatusbar=True)
            pt.show() 
            options = {'colheadercolor':'green','floatprecision': 5} # set some options
            config.apply_options(options, pt)
            pt.show()

            self.mainloop()
            
# Running app (Not compatible with colab)

def bayesian():
    BayesianOpt(X_train, y_train)

def grid():
    GridOpt(X_train, y_train)

def random():
    RandomOpt(X_train, y_train)

app = App()

bayesian()