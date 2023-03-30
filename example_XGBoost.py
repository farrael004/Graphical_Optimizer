import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from hyperoptimize import GraphicalOptimizer

# Loading data

data = pd.read_csv('kc_house_data.csv')

features = data.iloc[:, 3:].columns.tolist()
target = data.iloc[:, 2].name

y = data.loc[:, ['sqft_living', 'grade', target]].sort_values(target, ascending=True).values
X = np.arange(y.shape[0])

new_data = data[
    ['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms', 'view', 'sqft_basement', 'lat', 'waterfront',
     'yr_built', 'bedrooms']]

X = new_data.values
y = data.price.values


def model_function(params, X_train, y_train):
    xgb = xgboost.XGBRegressor(n_estimators=180,
                               gamma=params['gamma'],
                               reg_alpha=params['reg_alpha'],
                               reg_lambda=params['reg_lambda'],
                               min_child_weight=params['min_child_weight'],
                               colsample_bytree=params['colsample_bytree'],
                               max_depth=params['max_depth'])

    xgb.fit(X_train, y_train)

    return xgb


def prediction_function(model, X):
    return model.predict(X)


def performance_function(y_test, y_pred):
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
hyperparameters_bayesian = {'gamma': [1, 9],  # Upper and lower bounds
                            'max_depth': [3, 18],  # Upper and lower bounds
                            'reg_alpha': [40, 180],  # Categorical bounds
                            'reg_lambda': [0.0, 1.0],  # Upper and lower bounds
                            'colsample_bytree': [0.5, 1.0],  # Upper and lower bounds
                            'min_child_weight': [0, 10]}  # Upper and lower bounds

hyperparameters_grid_and_random = {'gamma': range(1, 9),  # Upper and lower bounds
                                   'max_depth': range(3, 18),  # Upper and lower bounds
                                   'reg_alpha': range(40, 180, 10),  # Categorical bounds
                                   'reg_lambda': np.linspace(0.0, 1.0, 10),  # Upper and lower bounds
                                   'colsample_bytree': np.linspace(0.5, 1.0, 5),  # Upper and lower bounds
                                   'min_child_weight': range(0, 10)}  # Upper and lower bounds

# Performing optimization
opt = GraphicalOptimizer(model_function=model_function,
                         prediction_function=prediction_function,
                         performance_function=performance_function,
                         performance_parameter="Adjusted R^2 Score",
                         hyperparameters=hyperparameters_bayesian,
                         optimizer="bayesian",
                         max_num_combinations=30,
                         cross_validation=2,
                         max_num_of_parallel_processes=-1,
                         parallel_combinations=2)

opt.fit(X, y)
