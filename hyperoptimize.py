# Creating model, predict and performance functions
from typing import Callable

import pandas as pd
import numpy as np

from skopt.learning.gaussian_process.kernels import Hyperparameter
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor

from tkinter import *
import tkinter as tk
from pandastable import Table, TableModel, config
import threading


# Function that will update table

def print_status(self):  # Shows the parameters used and accuracy attained of the search so far.
    global app
    results = pd.DataFrame(dict(self.scores, **self.params), index=[0])
    app.table.model.df = pd.concat([app.table.model.df, results], ignore_index=True, axis=0)


# Custom Estimator wrapper

class WrapperEstimator(BaseEstimator):
    def __init__(self, ModelFunction, PredictionFunction, PerformanceFunction, performanceParameter):
        self.ModelFunction = ModelFunction
        self.PredictionFunction = PredictionFunction
        self.PerformanceFunction = PerformanceFunction
        self.performanceParameter = performanceParameter

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)

        self.model = self.ModelFunction(self.params, X, y)

        self.is_fitted_ = True

        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        prediction = self.PredictionFunction(self.model, X)
        return prediction

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        self.scores = self.PerformanceFunction(y, y_pred)

        try:
            self.scores[self.performanceParameter]
        except:
            raise AttributeError("Could not find the chosen performance parameter in the dictionary of performance "
                                 "metrics. Check if the GraphicalOptimizer object has a valid performanceParameter.")
        print_status(self)
        return self.scores[self.performanceParameter]

    def set_params(self, **params):
        self.params = params
        return self


class _App(Frame, threading.Thread):
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
        self.f.pack(fill=BOTH, expand=1)
        df = pd.DataFrame()
        self.table = pt = Table(self.f, dataframe=df,
                                showtoolbar=True, showstatusbar=True)
        pt.show()
        options = {'colheadercolor': 'green', 'floatprecision': 5}  # set some options
        config.apply_options(options, pt)
        pt.show()

        self.after(1000, self.task)

        self.mainloop()


class GraphicalOptimizer:
    """Hyperparameter optimizer with GUI.
    
    GraphicalOptimizer creates the object that optimizes a given model using a set of hyperparameters.
    The hyperparameters along with the resulting model performance metrics are displayed
    using ``pandastable`` (a python GUI used for pandas data frame visualization).
    
    The ``fit`` method can be used to begin the optimization.
    
    Parameters
    ----------
    ModelFunction: Model training function.
    The function that implements the model that takes different hyperparameters for experimenting.
    This function is assumed to return the model so it can be used for making predictions and measuring
    its performance.
    
    PredictionFunction: Prediction function.
    The function that takes the model function and input data to make a prediction.
    This function is assumed to return the prediction so it can be used for calculating the prediction
    performance.
    
    PerformanceFunction: Performance calculation function.
    The function that takes a prediction by the model to compare its performance with labeled data.
    This function is assumed to return the scores in type dictionary.
    
    performanceParameter: Main model's performance indicator.
    A string that corresponds to which key of the ``PerformanceFunction``'s output to use as the model's
    score. This setting is important when performing Bayesian optimization as it determines which metric
    will be MAXIMIZED.
    """
    def __init__(self,
                 ModelFunction: Callable,           # Function that defines and trains models.
                 PredictionFunction: Callable,      # Function that predicts based on the model.
                 PerformanceFunction: Callable[...,dict],     # Fuction that calculates trained models performances.
                 hyperparameters: dict,   # Dictionary of all possible parameters with keys defining the parameter name and value defining value boundaries.
                 performanceParameter: str,
                 optimizer: str='bayesian',    # Parameter that determines between 'grid', 'bayesian', and 'random'
                 maxNumCombinations: int=100,  # Maximum number of combinations.
                 crossValidation: int=30,      # Splits train data and trains the same model on each split for calculating agregate performance.
                 numberInParallel: int=-1,     # Number of parallel experiments to run (-1 will set to maximum allowed).
                 parallelCombinations: int=3,  # How many simultaneous combinations should be used.
                 seed=0):

        self.ModelFunction = ModelFunction
        self.PredictionFunction = PredictionFunction
        self.PerformanceFunction = PerformanceFunction
        self.hyperparameters = hyperparameters
        self.performanceParameter = performanceParameter
        self.optimizer = optimizer
        self.maxNumCombinations = maxNumCombinations
        self.crossValidation = crossValidation
        self.numberInParallel = numberInParallel
        self.parallelCombinations = parallelCombinations
        self.seed = seed

        self.results=None

        global app
        app = _App()

    # Optimizer types

    ## Bayesian

    def BayesianOpt(self, X_train, y_train):
        hyperparameters = {}
        for k in self.hyperparameters:
            v = self.hyperparameters[k]

            if type(v) is not list:
                raise TypeError("Hyperparameters must be in the form of a python list with at least one object.")

            if type(v[0]) is not float and type(v[0]) is not int:
                hyperparameters[k] = Categorical([item for item in v])
                continue

            if len(v) != 2:
                raise Exception("Hyperparameters of float or int type must be in the form of [lower_bound, "
                                "higher_bound].")

            if type(v[0]) is float or type(v[-1]) is float:
                hyperparameters[k] = Real(v[0], v[-1])
            else:
                hyperparameters[k] = Integer(v[0], v[-1])

        bayReg = BayesSearchCV(
            WrapperEstimator(self.ModelFunction, self.PredictionFunction, self.PerformanceFunction, self.performanceParameter),
            hyperparameters,
            random_state=self.seed,
            verbose=0,
            n_iter=self.maxNumCombinations,
            cv=self.crossValidation,
            n_jobs=self.numberInParallel,
            n_points=self.parallelCombinations
        )

        self.results = bayReg.fit(X_train, y_train)

        try:
            app.table.redraw()
        except:
            pass
        app.isUpdatingTable = False

    ## Grid

    def GridOpt(self, X_train, y_train):
        grid = GridSearchCV(
            WrapperEstimator(self.ModelFunction, self.PredictionFunction, self.PerformanceFunction, self.performanceParameter),
            {
                'max_features': ['sqrt', 'log2'],
                'learning_rate': np.linspace(0.1, 0.3, 10),
                'max_depth': range(3, 7),
            },
            verbose=0,
            cv=self.crossValidation,
            n_jobs=self.numberInParallel,
        )

        self.results = grid.fit(X_train, y_train)

        try:
            app.table.redraw()
        except:
            pass
        app.isUpdatingTable = False

    ## Random

    def RandomOpt(self, X_train, y_train):
        randomSearch = RandomizedSearchCV(
            WrapperEstimator(self.ModelFunction, self.PredictionFunction, self.PerformanceFunction, self.performanceParameter),
            {
                'max_features': ['sqrt', 'log2'],
                'learning_rate': np.linspace(0.1, 0.3, 10),
                'max_depth': range(3, 7),
            },
            random_state=self.seed,
            verbose=0,
            n_iter=self.maxNumCombinations,
            cv=self.crossValidation,
            n_jobs=self.numberInParallel,
        )

        self.results = randomSearch.fit(X_train, y_train)

        try:
            app.table.redraw()
        except:
            pass
        app.isUpdatingTable = False

    # Creating app class

    def fit(self, X_train, y_train):
        """Used to start the optimization process by choosing which method to call.
        This function requires the "optimizer" attribute to be specified and can be
        substituted by the direct method that you would wish to use by doing
        ``GraphicalOptimizer.RandomOpt(X_train, y_train)`` for example.

        Args:
            X_train (Any type): Training dataset to be used by the model function.
            y_train (Any type): Training labels to be used by the model function.

        Raises:
            ValueError: Will raise and error if the ``optimizer`` setting is anything
            but one of the following strings: "bayesian", "grid", or "random".
        """
        match self.optimizer:
            case "bayesian":
                self.BayesianOpt(X_train, y_train)
            case "grid":
                self.GridOpt(X_train, y_train)
            case "random":
                self.RandomOpt(X_train, y_train)
            case _:
                raise ValueError("Choose between bayesian, grid, or random optimizers")