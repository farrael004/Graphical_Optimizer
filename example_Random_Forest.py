import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from hyperoptimize import GraphicalOptimizer

# Loading data

data = pd.read_csv('kc_house_data.csv')

features = data.iloc[:, 3:].columns.tolist()
target = data.iloc[:, 2].name

y = data.loc[:, ['sqft_living', 'grade', target]].sort_values(target, ascending=True).values
x = np.arange(y.shape[0])

new_data = data[
    ['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms', 'view', 'sqft_basement', 'lat', 'waterfront',
     'yr_built', 'bedrooms']]

X = new_data.values
y = data.price.values


def model_function(params, X_train, y_train):
    model = RandomForestRegressor(n_estimators=params['n_estimators'],
                                  bootstrap=params['bootstrap'],
                                  max_depth=params['max_depth'],
                                  max_features=params['max_features'],
                                  min_samples_leaf=params['min_samples_leaf'],
                                  min_samples_split=params['min_samples_split'])

    model.fit(X_train, y_train)

    return model


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
hyperparameters_bayesian = {'n_estimators': [200, 2000],  # Upper and lower bounds
                            'bootstrap': [True, False],  # Categorical bounds
                            'max_depth': [10, 100],  # Upper and lower bounds
                            'max_features': ['sqrt', 'log2'],  # Categorical bounds
                            'min_samples_leaf': [1, 4],  # Upper and lower bounds
                            'min_samples_split': [2, 10], }  # Upper and lower bounds

hyperparameters_grid_and_random = {'n_estimators': range(200, 2000, 100),  # Extensive list of possibilities
                                   'bootstrap': [True, False],  # Extensive list of possibilities
                                   'max_depth': range(10, 100, 10),  # Extensive list of possibilities
                                   'max_features': ['sqrt', 'log2'],  # Extensive list of possibilities
                                   'min_samples_leaf': range(1, 4),  # Extensive list of possibilities
                                   'min_samples_split': [2, 5, 10], }  # Extensive list of possibilities

# Performing optimization
opt = GraphicalOptimizer(model_function=model_function,
                         prediction_function=prediction_function,
                         performance_function=performance_function,
                         performance_parameter="Adjusted R^2 Score",
                         hyperparameters=hyperparameters_bayesian,
                         optimizer="bayesian",
                         max_num_combinations=5,
                         cross_validation=2,
                         max_num_of_parallel_processes=-1,
                         parallel_combinations=2)

opt.fit(X, y)
