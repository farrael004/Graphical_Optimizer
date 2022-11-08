# Creating model, predict and performance functions
import os
import json
import shutil
import random
import string
from warnings import warn

from time import perf_counter

from typing import Callable

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from sklearn.model_selection import RandomizedSearchCV

from tkinter import *
from pandastable import Table, TableModel, config, PlotViewer
import threading


class EnhancedPlotViewer(PlotViewer):
    def replot(self, data=None):
        data = self.table.getSelectedDataFrame()

        if type(data.iloc[0].values[0]) is list:
            columns = data.columns.tolist()
            if len(columns) == 1:
                columns = columns[0]
            else:
                raise Exception("Cannot multiplot from more than one column.")

            data = pd.DataFrame(data[columns].tolist(), index=data.index.tolist())
            data = data.transpose()
            data.columns = [f'Experiment {i + 1}' for i in data.columns.tolist()]

        return super().replot(data)


class EnhancedTable(Table):

    def plotSelected(self):
        if not hasattr(self, 'pf') or self.pf == None:
            self.pf = EnhancedPlotViewer(table=self)
        else:
            if type(self.pf.main) is Toplevel:
                self.pf.main.deiconify()
        return super().plotSelected()


# Function that will update table

def print_status(self):  # Shows the parameters used and accuracy attained of the search so far.
    # results = pd.DataFrame(dict(self.scores, **self.params), index=[0])
    results = dict(self.scores, **self.params)
    json_object = json.dumps(results, indent=4)
    tempfile = ''.join(random.choice(string.ascii_letters) for i in range(10))
    filePath = os.path.join(self.tempPath, tempfile)

    with open(filePath + ".json", "w") as outfile:
        outfile.write(json_object)


# Custom Estimator wrapper

class WrapperEstimator(BaseEstimator):
    def __init__(self, ModelFunction, PredictionFunction, PerformanceFunction, performanceParameter, tempPath):
        self.ModelFunction = ModelFunction
        self.PredictionFunction = PredictionFunction
        self.PerformanceFunction = PerformanceFunction
        self.performanceParameter = performanceParameter
        self.tempPath = tempPath

        self.midTrainingPerformance = {}

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)

        modelTimerStart = perf_counter()

        modelFunctionOutput = self.ModelFunction(self.params, X, y)

        if type(modelFunctionOutput) is tuple:
            self.model, self.midTrainingPerformance = modelFunctionOutput
        else:
            self.model = modelFunctionOutput

        modelTimerEnd = perf_counter()
        self.modelTimer = modelTimerEnd - modelTimerStart

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
        self.scores["Training time"] = self.modelTimer

        for k in self.midTrainingPerformance:
            self.scores[k] = self.midTrainingPerformance[k]

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


# Creating app class
class _App(Frame, threading.Thread):
    def __init__(self, tempPath, parent=None):
        self.tempPath = tempPath
        self.parent = parent
        self.isUpdatingTable = True
        threading.Thread.__init__(self)
        self.start()

    def callback(self):
        self.root.quit()

    def task(self):
        if not self.isUpdatingTable: return

        for filename in os.listdir(self.tempPath):
            tempfile = os.path.join(self.tempPath, filename)
            with open(tempfile, 'r') as openfile:
                try:
                    json_object = json.load(openfile)
                    results = pd.DataFrame(json_object, index=[0])
                    self.table.model.df = pd.concat([self.table.model.df, results], ignore_index=True, axis=0)
                except:
                    warn("An error occurred when trying to read one of the experiment results. That is "
                         "probably my fault :(")
            try:
                os.remove(tempfile)
            except:
                warn(f'Could not remove the temporary file {filename} from {self.tempPath}')

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
        self.table = pt = EnhancedTable(self.f, dataframe=df,
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
    its performance. Optinally, a second output for displaying mid training performance can be included
    when returning the function. This second output must be a dictionary and must be able to be JSON
    serializable.
    
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
                 ModelFunction: Callable,  # Function that defines and trains models.
                 PredictionFunction: Callable,  # Function that predicts based on the model.
                 PerformanceFunction: Callable[..., dict],  # Fuction that calculates trained models performances.
                 hyperparameters: dict,
                 # Dictionary of all possible parameters with keys defining the parameter name and value defining value boundaries.
                 performanceParameter: str,
                 optimizer: str = 'bayesian',  # Parameter that determines between 'grid', 'bayesian', and 'random'
                 maxNumCombinations: int = 100,  # Maximum number of combinations.
                 crossValidation: int = 30,
                 # Splits train data and trains the same model on each split for calculating agregate performance.
                 numberInParallel: int = -1,  # Number of parallel experiments to run (-1 will set to maximum allowed).
                 parallelCombinations: int = 3,  # How many simultaneous combinations should be used.
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

        self.results = None

        self.tempPath = os.path.join(os.getcwd(), "temp")
        try:
            shutil.rmtree(self.tempPath)
        except FileNotFoundError:
            pass
        os.makedirs(self.tempPath, exist_ok=True)
        self.app = _App(self.tempPath)

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
            WrapperEstimator(self.ModelFunction, self.PredictionFunction, self.PerformanceFunction,
                             self.performanceParameter, self.tempPath),
            hyperparameters,
            random_state=self.seed,
            verbose=0,
            n_iter=self.maxNumCombinations,
            cv=self.crossValidation,
            n_jobs=self.numberInParallel,
            n_points=self.parallelCombinations
        )

        self.results = bayReg.fit(X_train, y_train)

        self._finalizeOptimization()

    ## Grid

    def GridOpt(self, X_train, y_train):
        grid = GridSearchCV(
            WrapperEstimator(self.ModelFunction, self.PredictionFunction, self.PerformanceFunction,
                             self.performanceParameter, self.tempPath),
            {
                'n_estimators': range(5000, 6000, 100),
                'max_features': ['sqrt', 'log2'],
                'learning_rate': np.linspace(0.01, 0.3, 10),
                'max_depth': range(3, 7),
                'min_samples_leaf': range(2, 22, 2),
                'min_samples_split': range(2, 17, 2)
            },
            verbose=0,
            cv=self.crossValidation,
            n_jobs=self.numberInParallel,
        )

        self.results = grid.fit(X_train, y_train)

        self._finalizeOptimization()

    ## Random

    def RandomOpt(self, X_train, y_train):
        randomSearch = RandomizedSearchCV(
            WrapperEstimator(self.ModelFunction, self.PredictionFunction, self.PerformanceFunction,
                             self.performanceParameter, self.tempPath),
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

        self._finalizeOptimization()

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

    def _finalizeOptimization(self):
        try:
            self.app.table.redraw()
        except:
            pass
        self.app.isUpdatingTable = False
        try:
            shutil.rmtree(self.tempPath)
        except FileNotFoundError:
            pass
