import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.ensemble import GradientBoostingRegressor

from hyperoptimize import GraphicalOptimizer
from hyperoptimize import App

# Loading data

df1 = pd.read_csv('california_housing_test.csv')
df1 = df1.dropna()
X = df1.copy()
X.pop('median_house_value')
y = df1.median_house_value.copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,
                                                  random_state=1)  # 0.25 x 0.8 = 0.2

sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Create standardization and apply to train data
X_test = sc.transform(X_test)  # Apply created standardization to new data
X_val = sc.transform(X_val)  # Apply created standardization to new data

pca = PCA(n_components=0.9, svd_solver='full')
X_train = pca.fit_transform(X_train)  # Create PCA and apply to train data
X_test = pca.transform(X_test)  # Apply created PCA to new data
X_val = pca.transform(X_val)  # Apply created normalization to new data


def model(params, X_train, y_train):
    def model_function(params, X_train, y_train):
        model = np.sin(3*params['X'])+np.cos(2*params['X'])-2.2*np.sin(3*params['X'])-2*np.cos(3.4*params['X'])
        model += np.sin(3*params['Y'])+np.cos(2*params['Y'])-2.2*np.sin(3*params['Y'])-2*np.cos(3.4*params['Y'])
        return model

    def prediction_function(model, X):
        y_pred = model
        return y_pred

    def performance_function(y_test, y_pred):
        return {"score": y_pred}

    # Creating hyperparameter dictionary

    hyperparameters_bayesian = {'X': [-5.0, 5.0], 'Y': [-5.0, 5.0]}  # Upper and lower bounds

    hyperparameters_grid_and_random = {'X': np.linspace(-5, 5, 14).tolist(),
                                       'Y': np.linspace(-5, 5, 14).tolist()}  # Upper and lower bounds

    if params['method'] == 'bayesian':
        hyperparams = hyperparameters_bayesian
    else:
        hyperparams = hyperparameters_grid_and_random

    # Performing optimization

    opt = GraphicalOptimizer(model_function=model_function,
                             prediction_function=prediction_function,
                             performance_function=performance_function,
                             performance_parameter="score",
                             hyperparameters=hyperparams,
                             optimizer=params['method'],
                             max_num_combinations=params['total combinations'],
                             cross_validation=2,
                             max_num_of_parallel_processes=-1,
                             parallel_combinations=params['parallel combinations'],
                             seed=1,
                             create_GUI=False)

    opt.fit(X, y)

    return opt.results.best_score_


def pred(model, X):
    y_pred = model
    return y_pred


def perf(y_test, y_pred):
    return {"score": y_pred}


hyper = {'method': ['bayesian'],
         'total combinations': [15, 30, 45, 90, 125, 140, 175],
         'parallel combinations': [1, 3, 6, 9, 12]}

opt = GraphicalOptimizer(model_function=model,
                         prediction_function=pred,
                         performance_function=perf,
                         performance_parameter="score",
                         hyperparameters=hyper,
                         optimizer='grid',
                         cross_validation=2,
                         max_num_of_parallel_processes=1)

opt.fit(X_train, y_train)
