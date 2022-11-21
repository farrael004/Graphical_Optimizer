import time

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from lightgbm import LGBMRegressor

from hyperoptimize import GraphicalOptimizer

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


# Creating model, prediction and performance functions
def modelFunction(params, X_train, y_train):
    lightgbm = LGBMRegressor(objective='regression',
                             num_leaves=params['num_leaves'],
                             learning_rate=params['learning_rate'],
                             n_estimators=params['n_estimators'],
                             max_bin=params['max_bin'],
                             bagging_fraction=0.8,
                             bagging_freq=params['bagging_freq'],
                             bagging_seed=8,
                             feature_fraction=0.2,
                             feature_fraction_seed=8,
                             min_sum_hessian_in_leaf=11,
                             verbose=-1,
                             random_state=42)

    model = lightgbm.fit(X_train, y_train)

    return model


def predictionFunction(model, X):
    y_pred = model.predict(X)
    return y_pred


def performanceFunction(y_test, y_pred):
    model_mae = mean_absolute_error(y_test, y_pred)
    model_mse = mean_squared_error(y_test, y_pred)
    model_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    model_r2 = r2_score(y_test, y_pred)

    model_results = {"Mean Absolute Error (MAE)": model_mae,
                     "Mean Squared Error (MSE)": model_mse,
                     "Root Mean Squared Error (RMSE)": model_rmse,
                     "Adjusted R^2 Score": model_r2}
    return model_results


# Creating hyperparameter dictionary
hyperparameters_bayesian = {'n_estimators': [5000, 8000],  # Upper and lower bounds
                            'learning_rate': [0.001, 0.01],  # Upper and lower bounds
                            'max_bin': [100, 300],  # Upper and lower bounds
                            'num_leaves': [3, 12],  # Upper and lower bounds
                            'bagging_freq': [2, 8], }  # Upper and lower bounds

hyperparameters_grid_and_random = {'n_estimators': range(5000, 8000, 1000),  # Extensive list of possibilities
                                   'learning_rate': np.linspace(0.001, 0.01, 10),  # Extensive list of possibilities
                                   'max_bin': range(100, 300, 50),  # Extensive list of possibilities
                                   'num_leaves': range(3, 12, 3),  # Extensive list of possibilities
                                   'bagging_freq': range(2, 8, 2), }  # Extensive list of possibilities


# Creating functions that runs after and while the optimization runs.
def runMeWhileOptimizing(opt: GraphicalOptimizer):
    # print(opt.df)
    # opt.app.after(1000, opt.app.concurrentFunction(opt))
    return


def runMeAfterOptimizing(opt: GraphicalOptimizer):
    df = opt.df
    bestIndex = df["Adjusted R^2 Score"].idxmax()
    bestParams = df.iloc[bestIndex]
    print("Finished optimizing")
    print(f'Best performance: {bestParams["Adjusted R^2 Score"]}')  # or opt.results.best_score_
    print("Best combination of hyperparameters are:")
    print(bestParams[6:])  # or opt.results.best_params_

# Performing optimization
opt = GraphicalOptimizer(ModelFunction=modelFunction,
                         PredictionFunction=predictionFunction,
                         PerformanceFunction=performanceFunction,
                         performanceParameter="Adjusted R^2 Score",
                         hyperparameters=hyperparameters_bayesian,
                         optimizer="bayesian",
                         maxNumCombinations=20,
                         crossValidation=2,
                         maxNumOfParallelProcesses=-1,
                         parallelCombinations=5,
                         createGUI=True,
                         concurrentFunction=runMeWhileOptimizing,
                         completionFunction=runMeAfterOptimizing)

opt.fit(X_train, y_train)
