import pandas as pd
import numpy as np

from tkinter import *
import tkinter as tk
from pandastable import Table, TableModel, config
import threading

from skopt.learning.gaussian_process.kernels import Hyperparameter
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV
from sklearn.ensemble import GradientBoostingRegressor

class graphical_optimizer:

    def __init__(self,
                ModelFunction,          # Function that defines and trains models
                PredictionFunction,     # NEW Function that predicts based on the model
				PerformanceFunction,    # Fuction that calculates trained models performances
				hyperparameters,        # Dictionary of all possible parameters with keys defining the parameter name and value defining possible values
				optimizer='bayesian',        # Parameter that determines between 'grid', 'bayesian', and 'random'
                maxNumCombinations=100,      # Maximum number of combinations
                crossValidation=30):         # Splits train data and trains the same model on each split for calculating agregate performance):
        
        self.ModelFunction=ModelFunction
        self.PredictionFunction=PredictionFunction
        self.PerformanceFunction=PerformanceFunction
        self.hyperparameters=hyperparameters
        self.optimizer=optimizer
        self.maxNumCombinations=maxNumCombinations
        self.crossValidation=crossValidation
        
        if (ModelFunction == None or
            PredictionFunction == None or
            PerformanceFunction == None or
            hyperparameters == None):
            
            raise Exception('You must define the model function, prediction function, performance function and the hyperparameters to be used')

    # Function that will update table

    def print_status(self): # Shows the parameters used and accuracy attained of the search so far.
        global app
        results = pd.DataFrame(dict(self.scores, **self.params), index=[0])
        self.app.table.model.df = pd.concat([self.app.table.model.df, results], ignore_index=True, axis=0)
            
    # Custom Estimator wrapper

    class WrapperEstimator(BaseEstimator):
        def __init__(self, ModelFunction, PredictionFunction, PerformanceFunction, learning_rate=0.1, max_depth=4, max_features='sqrt'):
            
            hyperparameters = {'n_estimators':      [5000,6000], # Upper and lower bounds
                            'learning_rate':     [0.001,0.01], # Upper and lower bounds
                            'max_depth':         [2,6], # Upper and lower bounds
                            'max_features':      ['sqrt','log2'], # Categorical bounds
                            'min_samples_leaf':  [1,21], # Upper and lower bounds
                            'min_samples_split': [1,16],} # Upper and lower bounds
            
            
            self.learning_rate = learning_rate
            self.max_depth = max_depth
            self.max_features = max_features
            self.ModelFunction = ModelFunction
            self.PredictionFunction = PredictionFunction
            self.PerformanceFunction = PerformanceFunction
            self.variables = hyperparameters

        def fit(self, X, y):
            self.params = {
                'learning_rate': self.learning_rate,
                'max_depth': self.max_depth,
                'max_features': self.max_features
            }

            X, y = check_X_y(X, y, accept_sparse=True)

            self.model = self.ModelFunction(self.params, X, y)

            self.is_fitted_ = True
            
            return self

        def predict(self, X):
            X = check_array(X, accept_sparse=True)
            check_is_fitted(self, 'is_fitted_')
            prediction = self.PredictionFunction(self.model,X)
            return prediction

        def score(self, X, y, sample_weight=None):
            model = self.model
            y_pred = self.predict(X)
            self.scores = self.PerformanceFunction(y, y_pred)
            print_status(self)
            return self.scores['Adjusted R^2 Score']
        
    # Optimizer types

    ## Bayesian

    def BayesianOpt(self, X_train, y_train, ModelFunction, PredictionFunction, PerformanceFunction):
        
        bayReg = BayesSearchCV(
            self.WrapperEstimator(ModelFunction, PredictionFunction, PerformanceFunction),
            {
                'learning_rate':     Real(0.001,0.01),
                'max_depth':         Integer(2,6),
                'max_features':      Categorical(['sqrt','log2']),
            },
            random_state = 42 ,
            verbose = 0,
            n_iter = 10 ,
            cv = 2 ,
            n_jobs = -1 ,
            n_points = 2
        )

        model1 = bayReg.fit(X_train, y_train) # Included callback function that is called after every iteration
        
        self.app.table.redraw()
        
    ## Grid

    def GridOpt(self, X_train, y_train):
        from skopt.space import Real, Categorical, Integer
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import GradientBoostingRegressor

        grid = GridSearchCV(
            self.WrapperEstimator(),
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

        #app.table.model.df = pd.DataFrame(grid.cv_results_).drop(['params'], axis=1)
        self.app.table.redraw()
        
    ## Random
        
    def RandomOpt(self, X_train, y_train):    
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
        self.app.table.redraw()
        
        # Creating app class

    class App(Frame, threading.Thread):
        
            def __init__(self, parent=None):
                self.parent = parent
                self.isUpdatingTable = True
                threading.Thread.__init__(self)
                self.start()
                
            def callback(self):
                self.root.quit()
                
            def task(self):
                if not self.isUpdatingTable: return
                self.table.redraw()
                self.after(1000, self.task)
            
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
                
                self.after(1000, self.task)

                self.mainloop()
                
    # Running app (Not compatible with colab)

    def optimize(self, X_train, y_train):
        def bayesian(self):
            self.BayesianOpt(X_train, y_train, self.ModelFunction, self.PredictionFunction, self.PerformanceFunction)

        def grid(self):
            self.GridOpt(X_train, y_train)

        def random(self):
            self.RandomOpt(X_train, y_train)

        self.app = self.App()

        bayesian(self)
        self.app.isUpdatingTable = False