# Creating model, predict and performance functions
import os
import json
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
import matplotlib.pyplot as plt
from pandas import plotting

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
from pandastable import Table, TableModel, config, PlotViewer, util
import threading


# Modifying package classes

class _EnhancedPlotViewer(PlotViewer):
    """Class for overwritting the default PlotViewer from pandastable."""
    
    def replot(self, data=None):
        data = self.table.getSelectedDataFrame()

        if type(data.iloc[0].values[0]) is list:
            old_data = data.copy()
            columns = old_data.columns.tolist()
            data = pd.DataFrame()
            
            for column in columns:
                if type(old_data[column].values[0]) is not list: continue
                for i, values in enumerate(old_data[column].tolist()):
                    data[f'{column} {i + 1}'] = values

        return super().replot(data)

    def _doplot(self, data, ax, kind, subplots, errorbars, useindex, bw, yerr, kwargs):
        """Core plotting method where the individual plot functions are called"""
        
        kwargs = kwargs.copy()
        if self.style != None:
            keargs = self._clearArgs(kwargs)

        cols = data.columns
        if kind == 'line':
            data = data.sort_index()

        rows = int(round(np.sqrt(len(data.columns)),0))
        if len(data.columns) == 1 and kind not in ['pie']:
            kwargs['subplots'] = 0
        if 'colormap' in kwargs:
            cmap = plt.cm.get_cmap(kwargs['colormap'])
        else:
            cmap = None
        #change some things if we are plotting in b&w
        styles = []
        if bw == True and kind not in ['pie','heatmap']:
            cmap = None
            kwargs['color'] = 'k'
            kwargs['colormap'] = None
            styles = ["-","--","-.",":"]
            if 'linestyle' in kwargs:
                del kwargs['linestyle']

        if subplots == 0:
            layout = None
        else:
            layout=(rows,-1)

        if errorbars == True and yerr == None:
            yerr = data[data.columns[1::2]]
            data = data[data.columns[0::2]]
            yerr.columns = data.columns
            plt.rcParams['errorbar.capsize']=4
            #kwargs['elinewidth'] = 1

        if kind == 'bar' or kind == 'barh':
            if len(data) > 50:
                ax.get_xaxis().set_visible(False)
            if len(data) > 300:
                self.showWarning('too many bars to plot')
                return
        if kind == 'scatter':
            axs, sc = self.scatter(data, ax, **kwargs)
            if kwargs['sharey'] == 1:
                lims = self.fig.axes[0].get_ylim()
                for a in self.fig.axes:
                    a.set_ylim(lims)
        elif kind == 'boxplot':
            axs = data.boxplot(ax=ax, grid=kwargs['grid'],
                               patch_artist=True, return_type='dict')
            lw = kwargs['linewidth']
            plt.setp(axs['boxes'], color='black', lw=lw)
            plt.setp(axs['whiskers'], color='black', lw=lw)
            plt.setp(axs['fliers'], color='black', marker='+', lw=lw)
            clr = cmap(0.5)
            for patch in axs['boxes']:
                patch.set_facecolor(clr)
            if kwargs['logy'] == 1:
                ax.set_yscale('log')
        elif kind == 'violinplot':
            axs = self.violinplot(data, ax, kwargs)
        elif kind == 'dotplot':
            axs = self.dotplot(data, ax, kwargs)

        elif kind == 'histogram':
            #bins = int(kwargs['bins'])
            axs = data.plot(kind='hist',layout=layout, ax=ax, **kwargs)
        elif kind == 'heatmap':
            if len(data) > 1000:
                self.showWarning('too many rows to plot')
                return
            axs = self.heatmap(data, ax, kwargs)
        elif kind == 'bootstrap':
            axs = plotting.bootstrap_plot(data)
        elif kind == 'scatter_matrix':
            axs = pd.scatter_matrix(data, ax=ax, **kwargs)
        elif kind == 'hexbin':
            x = cols[0]
            y = cols[1]
            axs = data.plot(x,y,ax=ax,kind='hexbin',gridsize=20,**kwargs)
        elif kind == 'contour':
            xi,yi,zi = self.contourData(data)
            cs = ax.contour(xi,yi,zi,15,linewidths=.5,colors='k')
            #plt.clabel(cs,fontsize=9)
            cs = ax.contourf(xi,yi,zi,15,cmap=cmap)
            #ax.scatter(x,y,marker='o',c='b',s=5)
            self.fig.colorbar(cs,ax=ax)
            axs = ax
        elif kind == 'imshow':
            xi,yi,zi = self.contourData(data)
            im = ax.imshow(zi, interpolation="nearest",
                           cmap=cmap, alpha=kwargs['alpha'])
            self.fig.colorbar(im,ax=ax)
            axs = ax
        elif kind == 'pie':
            if useindex == False:
                x=data.columns[0]
                data.set_index(x,inplace=True)
            if kwargs['legend'] == True:
                lbls=None
            else:
                lbls = list(data.index)

            axs = data.plot(ax=ax,kind='pie', labels=lbls, layout=layout,
                            autopct='%1.1f%%', subplots=True, **kwargs)
            if lbls == None:
                axs[0].legend(labels=data.index, loc='best')
        elif kind == 'venn':
            axs = self.venn(data, ax, **kwargs)
        elif kind == 'radviz':
            if kwargs['marker'] == '':
                kwargs['marker'] = 'o'
            col = data.columns[-1]
            axs = pd.plotting.radviz(data, col, ax=ax, **kwargs)
        else:
            #line, bar and area plots
            if useindex == False:
                x=data.columns[0]
                data.set_index(x,inplace=True)
            if len(data.columns) == 0:
                msg = "Not enough data.\nIf 'use index' is off select at least 2 columns"
                self.showWarning(msg)
                return
            #adjust colormap to avoid white lines
            if cmap != None:
                cmap = util.adjustColorMap(cmap, 0.15,1.0)
                del kwargs['colormap']
            if kind == 'barh':
                kwargs['xerr']=yerr
                yerr=None
                
            kwargs.pop('subplots')
            axs = data.plot(ax=ax, layout=layout, yerr=yerr, style=styles, cmap=cmap, **kwargs)
        self._setAxisRanges()
        self._setAxisTickFormat()
        return axs


class _EnhancedTable(Table):
    """Class for overwritting the default Table from pandastable."""
    
    def plotSelected(self):
        if not hasattr(self, 'pf') or self.pf == None:
            self.pf = _EnhancedPlotViewer(table=self)
        else:
            if type(self.pf.main) is Toplevel:
                self.pf.main.deiconify()
        return super().plotSelected()


class _EnhancedBaseSearchCV(BaseSearchCV):
    """Class for overwritting the default BaseSearchCV from sklearn."""

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


class _EnhancedBayesianSearchCV(_EnhancedBaseSearchCV):
    """Class derived from the default BayesianSearchCV from sklearn."""    
    
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

        super(_EnhancedBayesianSearchCV, self).__init__(
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


class _EnhancedGridSearchCV(_EnhancedBaseSearchCV):
    """Class derived from the default GridSearchCV from sklearn."""

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


class _EnhancedRandomSearchCV(_EnhancedBaseSearchCV):
    """Class derived from the default GridSearchCV from sklearn."""
    
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


def print_status(self):
    """Publishes the parameters and performance attained in an experiment."""
    
    results = dict(self.scores, **self.params)
    json_object = json.dumps(results, indent=4)
    tempfile = self._id + ''.join(random.choice(string.ascii_letters) for i in range(10))
    filePath = os.path.join(self.tempPath, tempfile)

    with open(filePath + ".json", "w") as outfile:
        outfile.write(json_object)


# Custom Estimator wrapper

class WrapperEstimator(BaseEstimator):
    """A class to wrap to create a sklearn estimator using the model, prediction and
    performance functions defined in the GraphicalOptimizer object."""
    
    def __init__(self, ModelFunction, PredictionFunction, PerformanceFunction, performanceParameter, tempPath, _id):
        self.ModelFunction = ModelFunction
        self.PredictionFunction = PredictionFunction
        self.PerformanceFunction = PerformanceFunction
        self.performanceParameter = performanceParameter
        self.tempPath = tempPath
        self._id = _id

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
            raise AttributeError(f'Could not find the chosen performance parameter "{self.performanceParameter}" in '
                                 'the dictionary of performance metrics. Check if the GraphicalOptimizer object has a '
                                 'valid performanceParameter.')
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
    hyperparameters
    
    hyperparameters: All possible parameters.
    A dictionary where the keys are defining the parameter name. The value for this key will define value
    boundaries. Must take different forms when using bayesian search as opposed to using grid or random
    search.
    
    performanceParameter: Main model's performance indicator.
    A string that corresponds to which key of the ``PerformanceFunction``'s output to use as the model's
    score. This setting is important when performing Bayesian optimization as it determines which metric
    will be MAXIMIZED.
    
    optimizer: The optimizer search method to be used.
    A string that defines which search algorithm to use. ``bayesian`` will use bayesian search. ``grid``
    will iterate through all possible hyperparameter combinations. ``random`` will chose a random
    selection of possible combinations up to the maximum number of combinations.
    
    maxNumCombinatios: Maximum number of combinations.
    An integer that determines how many hyperparameter combinations to search for. This argument only
    affects random and bayesian search. Grid search will always try all hyperparameter combinations. The
    total number of experiments that the optimizer will run is ``maxNumCombinatios * crossValidation``.
    
    crossValidation: Number of cross validation folds.
    An integer that determines how many times the dataset will be split for performing cross validation on
    each hyperparameter combination.
    
    maxNumOfParallelProcesses: Number of experiments to run in parallel.
    This integer determines how many parallel processes will be created for training multiple experiments
    at the same time. -1 will create the maximum possible number of parallel processes.
    
    parallelCombinations: Number of simultaneous combinations to be tested.
    This setting only affects bayesian search. This integer determines how many parallel combinations can
    be tested in bayesian search. If many combinations are tested simultaneously, the bayesian algorithm
    may perform worse than if it tested sequentially each individual combination.
    
    seed: Seed for cross validation.
    An integer for determining the cross validation random state.
    
    createGUI: Determines whether GUI should be created or not.
    A boolean for allowing the App object to be created. If True, the optimizer window will be created. If
    False, the GUI will not be instantiated. The optimizer will function the same way on the background
    regardless of the presence of the GUI.
    
    concurrentFunction: A function that runs simultaneous to the optimization process.
    A function that will be called on the same thread as the GUI whenever an experiment completes. 
    
    completionFunction: A function that runs after the hyperparameter search is over.
    A function that will be called as soon as all experiments are completed. This can be used for code to run
    parallel to the GUI when hyperparameter search completes.
    
    verbose: Optimizer verbosity.
    An integer that controls how verbose the optimizer will be when queuing new experiments.
    verbose=0 will display no messages. verbose=1 will display messages about the queued experiments.
    """

    def __init__(self,
                 ModelFunction: Callable,
                 PredictionFunction: Callable,
                 PerformanceFunction: Callable[..., dict],
                 hyperparameters: dict,
                 performanceParameter: str,
                 optimizer: str = 'bayesian',
                 maxNumCombinations: int = 100,
                 crossValidation: int = 30,
                 maxNumOfParallelProcesses: int = -1,
                 parallelCombinations: int = 3,
                 seed=None,
                 createGUI=True,
                 concurrentFunction: Callable = None,
                 completionFunction: Callable = None,
                 verbose=0):

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
        self.verbose=verbose

        self.results = None
        self.df = pd.DataFrame()
        self.tempPath = os.path.join(os.getcwd(), "temp") # Folder to which the experiment results will be written to.
        
        self._isUpdatingTable = True
        self._id = ''.join(
            random.choice(string.ascii_letters)
            for i in range(10)) # Used to tag the experiment results files created by this object.

        try:
            # shutil.rmtree(self.tempPath)
            pass
        except FileNotFoundError:
            pass
        os.makedirs(self.tempPath, exist_ok=True)
        if createGUI:
            self.app = App(self, concurrentFunction=self.concurrentFunction)
        else:
            t = threading.Thread(target=self._update_results)
            t.start()

    # Optimizer types

    ## Bayesian

    def BayesianOpt(self, X_train, y_train):
        """Function used to perform bayesian search.

        Args:
            X_train (Any type): Input for training data. Must be in a form
            that can be split for cross validation.
            
            y_train (Any type): Label for training data. Must be in a form
            that can be split for cross validation.
        """
        
        hyperparameters = {}
        for k in self.hyperparameters:
            v = self.hyperparameters[k]

            if type(v) is not list:
                raise TypeError("Hyperparameters must be in the form of a python list with at least one object.")

            if type(v[0]) is not float and type(v[0]) is not int:
                hyperparameters[k] = Categorical([item for item in v])
                continue

            if len(v) != 2:
                raise Exception(
                    "Hyperparameters in bayesian search of float or int type must be in the form of [lower_bound, "
                    "higher_bound].")

            if type(v[0]) is float or type(v[1]) is float:
                hyperparameters[k] = Real(v[0], v[1])
            else:
                hyperparameters[k] = Integer(v[0], v[1])

        bayReg = _EnhancedBayesianSearchCV(
            WrapperEstimator(self.ModelFunction, self.PredictionFunction, self.PerformanceFunction,
                             self.performanceParameter, self.tempPath, self._id),
            hyperparameters,
            random_state=self.seed,
            verbose=self.verbose,
            n_iter=self.maxNumCombinations,
            cv=self.crossValidation,
            n_jobs=self.maxNumOfParallelProcesses,
            n_points=self.parallelCombinations
        )

        self.results = bayReg.fit(X_train, y_train)

        self._finalizeOptimization()

    ## Grid

    def GridOpt(self, X_train, y_train):
        """Function used to perform grid search.

        Args:
            X_train (Any type): Input for training data. Must be in a form
            that can be split for cross validation.
            
            y_train (Any type): Label for training data. Must be in a form
            that can be split for cross validation.
        """
        
        hyperparameters = self._gridAndRandomHyperparameters()
        self._checkNumberOfCombinations(hyperparameters)

        grid = _EnhancedGridSearchCV(
            WrapperEstimator(self.ModelFunction, self.PredictionFunction, self.PerformanceFunction,
                             self.performanceParameter, self.tempPath, self._id),
            hyperparameters,
            verbose=self.verbose,
            cv=self.crossValidation,
            n_jobs=self.maxNumOfParallelProcesses,
        )

        self.results = grid.fit(X_train, y_train)

        self._finalizeOptimization()

    def _checkNumberOfCombinations(self, hyperparameters):
        items = 1
        for k in hyperparameters:
            items *= len(hyperparameters[k])

        if items > 10_000_000:
            warn(
                f"There is a very large number combinations of hyperparameters ({items} combinations). The more combinations, the longer it "
                "may take for the optimization to initiate. Consider using the 'random' method instead of 'grid' or "
                "lowering the number of combinations.")

    ## Random

    def RandomOpt(self, X_train, y_train):
        """Function used to perform random search.

        Args:
            X_train (Any type): Input for training data. Must be in a form
            that can be split for cross validation.
            
            y_train (Any type): Label for training data. Must be in a form
            that can be split for cross validation.
        """
        
        hyperparameters = self._gridAndRandomHyperparameters()

        randomSearch = _EnhancedRandomSearchCV(
            WrapperEstimator(self.ModelFunction, self.PredictionFunction, self.PerformanceFunction,
                             self.performanceParameter, self.tempPath, self._id),
            hyperparameters,
            random_state=self.seed,
            verbose=self.verbose,
            n_iter=self.maxNumCombinations,
            cv=self.crossValidation,
            n_jobs=self.maxNumOfParallelProcesses,
        )

        self.results = randomSearch.fit(X_train, y_train)

        self._finalizeOptimization()

    def _gridAndRandomHyperparameters(self):
        hyperparameters = {}
        for k in self.hyperparameters:
            v = self.hyperparameters[k]

            if type(v) is not list and type(v) is not range:
                raise TypeError("Each hyperparameter must be in the form of a python list or range with at least one "
                                "object populating it.")

            if type(v[0]) is not float and type(v[0]) is not int:  # Categorical
                hyperparameters[k] = [item for item in v]
                continue

            if type(v[0]) is float:  # Real
                hyperparameters[k] = [item for item in v]
            else:  # Integer
                hyperparameters[k] = [item for item in v]
        return hyperparameters

    def _retrieve_experiments(self):
        """Function that writes to disk experiment results."""
        
        try:
            for filename in os.listdir(self.tempPath):
                tempfile = os.path.join(self.tempPath, filename)
                if filename[:10] != self._id: continue  # check if file originates from this optimization session

                with open(tempfile, 'r') as openfile:
                    try:
                        json_object = json.load(openfile)
                    except:
                        warn("An error occurred when trying to read one of the experiment results.")
                    else:
                        results = pd.DataFrame(json_object, index=[0])
                        self.df = pd.concat([self.df, results], ignore_index=True, axis=0)
                        if self.concurrentFunction: self.concurrentFunction(self)
                try:
                    os.remove(tempfile)
                except:
                    warn(f'Could not remove the temporary file {filename} from {self.tempPath}')

        except FileNotFoundError:
            if self._isUpdatingTable:
                raise FileNotFoundError(f"temp folder at {self.tempPath} not found.")
            pass

    def _update_results(self):
        while (self._isUpdatingTable):
            self._retrieve_experiments()
            time.sleep(1)

    def fit(self, X_train, y_train):
        """Used to start the optimization process by choosing which method to call.
        This function requires the "optimizer" attribute to be specified and can be
        substituted by the direct method that you would wish to use by doing
        ``GraphicalOptimizer.RandomOpt(X_train, y_train)`` for example.

        Args:
            X_train (Any type): Training dataset to be used by the model function.
            y_train (Any type): Training labels to be used by the model function.

        Raises:
            ValueError: Will raise an error if the ``optimizer`` setting is anything
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

        self._isUpdatingTable = False
        if hasattr(self, 'app'):
            self.app._isUpdatingTable = False
            self.app.update_graphical_table()
            self.df = self.app.table.model.df
        else:
            self._retrieve_experiments()

        try:
            # shutil.rmtree(self.tempPath)
            pass
        except FileNotFoundError:
            pass

        if self.completionFunction is not None:
            self.completionFunction(self)


# Creating app class
class App(Frame, threading.Thread):
    """Class for creating the pandastable app for displaying results.
    
    When an object of this class is created, a pandastable window will open with an
    empty data frame. This data frame can be changed by accessing App.table.model.df
    
    For updating the pandastable renderer so changes to the data frame are visible,
    simply call App.table.redraw().
    """
    
    def __init__(self, optimizer: GraphicalOptimizer, parent=None, concurrentFunction: Callable = None):
        self.tempPath = optimizer.tempPath
        self.optmizer = optimizer
        self._id = optimizer._id
        self.parent = parent
        self.concurrentFunction = concurrentFunction
        self._isUpdatingTable = True
        threading.Thread.__init__(self)
        self.start()

    def callback(self):
        self.root.quit()

    def update_graphical_table(self):
        self.optmizer._retrieve_experiments()
        self.table.model.df = self.optmizer.df

        self.table.redraw()
        if not self._isUpdatingTable: return
        self.after(1000, self.update_graphical_table)

    def run(self):
        Frame.__init__(self)
        self.main = self.master
        self.main.geometry('600x400+200+100')
        self.main.title('Experiment results')
        self.f = Frame(self.main)
        self.f.pack(fill=BOTH, expand=1)
        df = pd.DataFrame()
        self.table = pt = _EnhancedTable(self.f, dataframe=df,
                                        showtoolbar=True, showstatusbar=True)
        pt.show()
        options = {'colheadercolor': 'green', 'floatprecision': 5}  # set some options
        config.apply_options(options, pt)
        pt.show()

        self.after(1000, self.update_graphical_table)

        self.mainloop()
