# Creating model, predict and performance functions
import os
import json
import shutil
import random
import string
import time
from warnings import warn

from time import perf_counter

from typing import Callable
from collections import defaultdict
from itertools import product

import pandas as pd
import numpy as np

from joblib import Parallel
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection._search import BaseSearchCV, ParameterGrid, ParameterSampler, indexable
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _fit_and_score, _warn_or_raise_about_fit_failures, _insert_error_scores
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_array, indexable, check_is_fitted, _check_fit_params
from sklearn.utils.fixes import delayed

from skopt import BayesSearchCV, Optimizer
from skopt.space import Real, Categorical, Integer, check_dimension
from skopt.utils import point_asdict, dimensions_aslist, eval_callbacks
from skopt.callbacks import check_callback
from scipy.stats import rankdata
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from tkinter import *
from pandastable import Table, TableModel, config, PlotViewer
import threading

# Modifying package classes

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


class EnhancedBaseSearchCV(BaseSearchCV):
    
        def fit(self, X, y=None, *, groups=None, **fit_params):

            estimator = self.estimator
            refit_metric = "score"

            if callable(self.scoring):
                scorers = self.scoring
            elif self.scoring is None or isinstance(self.scoring, str):
                scorers = check_scoring(self.estimator, self.scoring)
            else:
                scorers = _check_multimetric_scoring(self.estimator, self.scoring)
                self._check_refit_for_multimetric(scorers)
                refit_metric = self.refit

            X, y, groups = indexable(X, y, groups)
            fit_params = _check_fit_params(X, fit_params)

            cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
            n_splits = cv_orig.get_n_splits(X, y, groups)

            base_estimator = clone(self.estimator)

            parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

            fit_and_score_kwargs = dict(
                scorer=scorers,
                fit_params=fit_params,
                return_train_score=self.return_train_score,
                return_n_test_samples=True,
                return_times=True,
                return_parameters=False,
                error_score=self.error_score,
                verbose=self.verbose,
            )
            results = {}
            with parallel:
                all_candidate_params = []
                all_out = []
                all_more_results = defaultdict(list)

                def evaluate_candidates(candidate_params, cv=None, more_results=None):
                    cv = cv or cv_orig
                    candidate_params = list(candidate_params)
                    n_candidates = len(candidate_params)

                    if self.verbose > 0:
                        print(
                            "Fitting {0} folds for each of {1} candidates,"
                            " totalling {2} fits".format(
                                n_splits, n_candidates, n_candidates * n_splits
                            )
                        )

                    out = parallel(
                        delayed(_fit_and_score)(
                            clone(base_estimator),
                            X,
                            y,
                            train=train,
                            test=test,
                            parameters=parameters,
                            split_progress=(split_idx, n_splits),
                            candidate_progress=(cand_idx, n_candidates),
                            **fit_and_score_kwargs,
                        )
                        for (cand_idx, parameters), (split_idx, (train, test)) in product(
                            enumerate(candidate_params), enumerate(cv.split(X, y, groups))
                        )
                    )

                    if len(out) < 1:
                        raise ValueError(
                            "No fits were performed. "
                            "Was the CV iterator empty? "
                            "Were there no candidates?"
                        )
                    elif len(out) != n_candidates * n_splits:
                        raise ValueError(
                            "cv.split and cv.get_n_splits returned "
                            "inconsistent results. Expected {} "
                            "splits, got {}".format(n_splits, len(out) // n_candidates)
                        )

                    _warn_or_raise_about_fit_failures(out, self.error_score)

                    # For callable self.scoring, the return type is only know after
                    # calling. If the return type is a dictionary, the error scores
                    # can now be inserted with the correct key. The type checking
                    # of out will be done in `_insert_error_scores`.
                    if callable(self.scoring):
                        _insert_error_scores(out, self.error_score)

                    all_candidate_params.extend(candidate_params)
                    all_out.extend(out)

                    if more_results is not None:
                        for key, value in more_results.items():
                            all_more_results[key].extend(value)

                    nonlocal results
                    results = self._format_results(
                        all_candidate_params, n_splits, all_out, all_more_results
                    )

                    return results

                self._run_search(evaluate_candidates)

                # multimetric is determined here because in the case of a callable
                # self.scoring the return type is only known after calling
                first_test_score = all_out[0]["test_scores"]
                self.multimetric_ = isinstance(first_test_score, dict)

                # check refit_metric now for a callabe scorer that is multimetric
                if callable(self.scoring) and self.multimetric_:
                    self._check_refit_for_multimetric(first_test_score)
                    refit_metric = self.refit

            # For multi-metric evaluation, store the best_index_, best_params_ and
            # best_score_ iff refit is one of the scorer names
            # In single metric evaluation, refit_metric is "score"
            if self.refit or not self.multimetric_:
                self.best_index_ = self._select_best_index(
                    self.refit, refit_metric, results
                )
                if not callable(self.refit):
                    # With a non-custom callable, we can select the best score
                    # based on the best index
                    self.best_score_ = results[f"mean_test_{refit_metric}"][
                        self.best_index_
                    ]
                self.best_params_ = results["params"][self.best_index_]

            if self.refit:
                # we clone again after setting params in case some
                # of the params are estimators as well.
                self.best_estimator_ = clone(
                    clone(base_estimator).set_params(**self.best_params_)
                )
                self.best_estimator_.set_params(**self.best_params_)
                refit_start_time = time.time()
                if y is not None:
                    self.best_estimator_.fit(X, y, **fit_params)
                else:
                    self.best_estimator_.fit(X, **fit_params)
                refit_end_time = time.time()
                self.refit_time_ = refit_end_time - refit_start_time

                if hasattr(self.best_estimator_, "feature_names_in_"):
                    self.feature_names_in_ = self.best_estimator_.feature_names_in_

            # Store the only scorer not as a dict for single metric evaluation
            self.scorer_ = scorers

            self.cv_results_ = results
            self.n_splits_ = n_splits

            return self


class EnhancedBayesianSearchCV(EnhancedBaseSearchCV):
    def __init__(self, estimator, search_spaces, optimizer_kwargs=None,
                 n_iter=50, scoring=None, fit_params=None, n_jobs=1,
                 n_points=1, iid='deprecated', refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', random_state=None,
                 error_score='raise', return_train_score=False):

        self.search_spaces = search_spaces
        self.n_iter = n_iter
        self.n_points = n_points
        self.random_state = random_state
        self.optimizer_kwargs = optimizer_kwargs
        self._check_search_space(self.search_spaces)
        # Temporary fix for compatibility with sklearn 0.20 and 0.21
        # See scikit-optimize#762
        # To be consistent with sklearn 0.21+, fit_params should be deprecated
        # in the constructor and be passed in ``fit``.
        self.fit_params = fit_params

        if iid != "deprecated":
            warn("The `iid` parameter has been deprecated "
                          "and will be ignored.")
        self.iid = iid  # For sklearn repr pprint

        super(EnhancedBayesianSearchCV, self).__init__(
             estimator=estimator, scoring=scoring,
             n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
             pre_dispatch=pre_dispatch, error_score=error_score,
             return_train_score=return_train_score)

    def _check_search_space(self, search_space):
        """Checks whether the search space argument is correct"""

        if len(search_space) == 0:
            raise ValueError(
                "The search_spaces parameter should contain at least one"
                "non-empty search space, got %s" % search_space
            )

        # check if space is a single dict, convert to list if so
        if isinstance(search_space, dict):
            search_space = [search_space]

        # check if the structure of the space is proper
        if isinstance(search_space, list):
            # convert to just a list of dicts
            dicts_only = []

            # 1. check the case when a tuple of space, n_iter is provided
            for elem in search_space:
                if isinstance(elem, tuple):
                    if len(elem) != 2:
                        raise ValueError(
                            "All tuples in list of search spaces should have"
                            "length 2, and contain (dict, int), got %s" % elem
                        )
                    subspace, n_iter = elem

                    if (not isinstance(n_iter, int)) or n_iter < 0:
                        raise ValueError(
                            "Number of iterations in search space should be"
                            "positive integer, got %s in tuple %s " %
                            (n_iter, elem)
                        )

                    # save subspaces here for further checking
                    dicts_only.append(subspace)
                elif isinstance(elem, dict):
                    dicts_only.append(elem)
                else:
                    raise TypeError(
                        "A search space should be provided as a dict or"
                        "tuple (dict, int), got %s" % elem)

            # 2. check all the dicts for correctness of contents
            for subspace in dicts_only:
                for k, v in subspace.items():
                    check_dimension(v)
        else:
            raise TypeError(
                "Search space should be provided as a dict or list of dict,"
                "got %s" % search_space)

    @property
    def optimizer_results_(self):
        check_is_fitted(self, '_optim_results')
        return self._optim_results

    def _make_optimizer(self, params_space):
        """Instantiate skopt Optimizer class.

        Parameters
        ----------
        params_space : dict
            Represents parameter search space. The keys are parameter
            names (strings) and values are skopt.space.Dimension instances,
            one of Real, Integer or Categorical.

        Returns
        -------
        optimizer: Instance of the `Optimizer` class used for for search
            in some parameter space.

        """

        kwargs = self.optimizer_kwargs_.copy()
        kwargs['dimensions'] = dimensions_aslist(params_space)
        optimizer = Optimizer(**kwargs)
        for i in range(len(optimizer.space.dimensions)):
            if optimizer.space.dimensions[i].name is not None:
                continue
            optimizer.space.dimensions[i].name = list(sorted(
                params_space.keys()))[i]

        return optimizer

    def _step(self, search_space, optimizer, evaluate_candidates, n_points=1):
        """Generate n_jobs parameters and evaluate them in parallel.
        """
        # get parameter values to evaluate
        params = optimizer.ask(n_points=n_points)

        # convert parameters to python native types
        params = [[np.array(v).item() for v in p] for p in params]

        # make lists into dictionaries
        params_dict = [point_asdict(search_space, p) for p in params]

        all_results = evaluate_candidates(params_dict)
        # Feed the point and objective value back into optimizer
        # Optimizer minimizes objective, hence provide negative score
        local_results = all_results["mean_test_score"][-len(params):]
        return optimizer.tell(params, [-score for score in local_results])

    @property
    def total_iterations(self):
        """
        Count total iterations that will be taken to explore
        all subspaces with `fit` method.

        Returns
        -------
        max_iter: int, total number of iterations to explore
        """
        total_iter = 0

        for elem in self.search_spaces:

            if isinstance(elem, tuple):
                space, n_iter = elem
            else:
                n_iter = self.n_iter

            total_iter += n_iter

        return total_iter

    # TODO: Accept callbacks via the constructor?
    def fit(self, X, y=None, *, groups=None, callback=None, **fit_params):
        """Run fit on the estimator with randomly drawn parameters.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples] or [n_samples, n_output]
            Target relative to X for classification or regression (class
            labels should be integers or strings).

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        callback: [callable, list of callables, optional]
            If callable then `callback(res)` is called after each parameter
            combination tested. If list of callables, then each callable in
            the list is called.
        """
        self._callbacks = check_callback(callback)

        if self.optimizer_kwargs is None:
            self.optimizer_kwargs_ = {}
        else:
            self.optimizer_kwargs_ = dict(self.optimizer_kwargs)

        super().fit(X=X, y=y, groups=groups, **fit_params)

        # BaseSearchCV never ranked train scores,
        # but apparently we used to ship this (back-compat)
        if self.return_train_score:
            self.cv_results_["rank_train_score"] = \
                rankdata(-np.array(self.cv_results_["mean_train_score"]),
                         method='min').astype(int)
        return self

    def _run_search(self, evaluate_candidates):
        # check if space is a single dict, convert to list if so
        search_spaces = self.search_spaces
        if isinstance(search_spaces, dict):
            search_spaces = [search_spaces]

        callbacks = self._callbacks

        random_state = check_random_state(self.random_state)
        self.optimizer_kwargs_['random_state'] = random_state

        # Instantiate optimizers for all the search spaces.
        optimizers = []
        for search_space in search_spaces:
            if isinstance(search_space, tuple):
                search_space = search_space[0]
            optimizers.append(self._make_optimizer(search_space))
        self.optimizers_ = optimizers  # will save the states of the optimizers

        self._optim_results = []

        n_points = self.n_points

        for search_space, optimizer in zip(search_spaces, optimizers):
            # if not provided with search subspace, n_iter is taken as
            # self.n_iter
            if isinstance(search_space, tuple):
                search_space, n_iter = search_space
            else:
                n_iter = self.n_iter

            # do the optimization for particular search space
            while n_iter > 0:
                # when n_iter < n_points points left for evaluation
                n_points_adjusted = min(n_iter, n_points)

                optim_result = self._step(
                    search_space, optimizer,
                    evaluate_candidates, n_points=n_points_adjusted
                )
                n_iter -= n_points

                if eval_callbacks(callbacks, optim_result):
                    break
            self._optim_results.append(optim_result)


class EnhancedGridSearchCV(EnhancedBaseSearchCV):
    _required_parameters = ["estimator", "param_grid"]

    def __init__(
        self,
        estimator,
        param_grid,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.param_grid = param_grid

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""
        evaluate_candidates(ParameterGrid(self.param_grid))


class EnhancedRandomSearchCV(EnhancedBaseSearchCV):
    _required_parameters = ["estimator", "param_distributions"]

    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        n_iter=10,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score=np.nan,
        return_train_score=False,
    ):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

    def _run_search(self, evaluate_candidates):
        """Search n_iter candidates from param_distributions"""
        evaluate_candidates(
            ParameterSampler(
                self.param_distributions, self.n_iter, random_state=self.random_state
            )
        )



# Function that will write to memory experiment results

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
        modelTimerStart = perf_counter()
        
        modelFunctionOutput = self.ModelFunction(self.params, X, y)
        
        modelTimerEnd = perf_counter()
        
        if type(modelFunctionOutput) is tuple:
            self.model, self.midTrainingPerformance = modelFunctionOutput
        else:
            self.model = modelFunctionOutput

        self.modelTimer = modelTimerEnd - modelTimerStart

        self.is_fitted_ = True
        
        return self

    def predict(self, X):
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
                 maxNumOfParallelProcesses: int = -1,  # Number of parallel experiments to run (-1 will set to maximum allowed).
                 parallelCombinations: int = 3,  # How many simultaneous combinations should be used.
                 seed=0,
                 concurrentFunction: Callable=None,
                 completionFunction: Callable=None):

        self.ModelFunction = ModelFunction
        self.PredictionFunction = PredictionFunction
        self.PerformanceFunction = PerformanceFunction
        self.hyperparameters = hyperparameters
        self.performanceParameter = performanceParameter
        self.optimizer = optimizer
        self.maxNumCombinations = maxNumCombinations
        self.crossValidation = crossValidation
        self.maxNumOfParallelProcesses = maxNumOfParallelProcesses
        self.parallelCombinations = parallelCombinations
        self.seed = seed
        self.concurrentFunction = concurrentFunction
        self.completionFunction = completionFunction

        self.results = None
        self.df = None

        self.tempPath = os.path.join(os.getcwd(), "temp")
        try:
            shutil.rmtree(self.tempPath)
        except FileNotFoundError:
            pass
        os.makedirs(self.tempPath, exist_ok=True)
        self.app = App(self, concurrentFunction=self.concurrentFunction)

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

        bayReg = EnhancedBayesianSearchCV(
            WrapperEstimator(self.ModelFunction, self.PredictionFunction, self.PerformanceFunction,
                             self.performanceParameter, self.tempPath),
            hyperparameters,
            random_state=self.seed,
            verbose=0,
            n_iter=self.maxNumCombinations,
            cv=self.crossValidation,
            n_jobs=self.maxNumOfParallelProcesses,
            n_points=self.parallelCombinations
        )

        self.results = bayReg.fit(X_train, y_train)

        self._finalizeOptimization()

    ## Grid

    def GridOpt(self, X_train, y_train):
        hyperparameters = self.gridAndRandomHyperparameters()
        self.checkNumberOfCombinations(hyperparameters)
        
        grid = EnhancedGridSearchCV(
            WrapperEstimator(self.ModelFunction, self.PredictionFunction, self.PerformanceFunction,
                             self.performanceParameter, self.tempPath),
            hyperparameters,
            verbose=0,
            cv=self.crossValidation,
            n_jobs=self.maxNumOfParallelProcesses,
        )
        
        self.results = grid.fit(X_train, y_train)

        self._finalizeOptimization()

    def checkNumberOfCombinations(self, hyperparameters):
        items = 1
        for k in hyperparameters:
            items *= len(hyperparameters[k])
        
        if items > 10_000_000:
            warn(f"There is a very large number combinations of hyperparameters ({items} combinations). The more combinations, the longer it "
                 "may take for the optimization to initiate. Consider using the 'random' method instead of 'grid' or "
                 "lowering the number of combinations.")

    ## Random

    def RandomOpt(self, X_train, y_train):
        hyperparameters = self.gridAndRandomHyperparameters()
                
        randomSearch = EnhancedRandomSearchCV(
            WrapperEstimator(self.ModelFunction, self.PredictionFunction, self.PerformanceFunction,
                             self.performanceParameter, self.tempPath),
            hyperparameters,
            random_state=self.seed,
            verbose=0,
            n_iter=self.maxNumCombinations,
            cv=self.crossValidation,
            n_jobs=self.maxNumOfParallelProcesses,
        )

        self.results = randomSearch.fit(X_train, y_train)

        self._finalizeOptimization()

    def gridAndRandomHyperparameters(self):
        hyperparameters = {}
        for k in self.hyperparameters:
            v = self.hyperparameters[k]

            if type(v) is not list and type(v) is not range:
                raise TypeError("Each hyperparameter must be in the form of a python list or range with at least one "
                                "object populating it.")

            if type(v[0]) is not float and type(v[0]) is not int: # Categorical
                hyperparameters[k] = [item for item in v]
                continue

            if type(v[0]) is float: # Real
                hyperparameters[k] = [item for item in v]
            else: # Integer
                hyperparameters[k] = [item for item in v]
        return hyperparameters

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
        
        self.df = self.app.table.model.df
        
        self.completionFunction(self)

# Creating app class
class App(Frame, threading.Thread):
    def __init__(self, optimizer: GraphicalOptimizer, parent=None, concurrentFunction: Callable=None):
        self.tempPath = optimizer.tempPath
        self.optmizer = optimizer
        self.parent = parent
        self.concurrentFunction = concurrentFunction
        self.isUpdatingTable = True
        threading.Thread.__init__(self)
        self.start()

    def callback(self):
        self.root.quit()

    def retrieveExperiments(self):
        if not self.isUpdatingTable: return

        for filename in os.listdir(self.tempPath):
            tempfile = os.path.join(self.tempPath, filename)
            with open(tempfile, 'r') as openfile:
                try:
                    json_object = json.load(openfile)
                    results = pd.DataFrame(json_object, index=[0])
                    self.table.model.df = pd.concat([self.table.model.df, results], ignore_index=True, axis=0)
                except:
                    warn("An error occurred when trying to read one of the experiment results.")
            try:
                os.remove(tempfile)
            except:
                warn(f'Could not remove the temporary file {filename} from {self.tempPath}')
        self.optmizer.df = self.table.model.df
        self.table.redraw()
        self.concurrentFunction(self.optmizer)
        self.after(1000, self.retrieveExperiments)

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

        self.after(1000, self.retrieveExperiments)

        self.mainloop()