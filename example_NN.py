import time

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from keras import Sequential
from keras.layers import Dense

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
def model_function(params, X_train, y_train):
    model = Sequential()
    model.add(Dense(params['initial_neurons'], input_shape=(X_train.shape[1],)))
    for _ in range(params['layers']):
        model.add(Dense(params['neurons']))
    model.add(Dense(1))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['msle'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)
    loss = {'Training loss': [history.history['loss']]}
    return model, loss


def prediction_function(model, X):
    y_pred = model.predict(X)
    return y_pred


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
hyperparameters_bayesian = {'initial_neurons': [1, 100],
                            'layers': [0, 3],
                            'neurons': [1, 100]}

hyperparameters_grid_and_random = {'initial_neurons': range(1, 100, 10),
                                   'layers': range(0, 3),
                                   'neurons': range(1, 100, 10)}


# Creating functions that runs after and while the optimization runs.
def run_me_while_optimizing(opt: GraphicalOptimizer):
    # print(opt.df)

    # opt.app.after(1000, opt.app.concurrentFunction(opt))
    return


def run_me_after_optimizing(opt: GraphicalOptimizer):
    df = opt.df
    best_index = df["Adjusted R^2 Score"].idxmax()
    best_params = df.iloc[best_index]
    print("Finished optimizing")
    print(f'Best performance: {best_params["Adjusted R^2 Score"]}')  # or opt.results.best_score_
    print("Best combination of hyperparameters are:")
    print(best_params[6:])  # or opt.results.best_params_


# Performing optimization
opt = GraphicalOptimizer(model_function=model_function,
                         prediction_function=prediction_function,
                         performance_function=performance_function,
                         performance_parameter="Adjusted R^2 Score",
                         hyperparameters=hyperparameters_bayesian,
                         optimizer="bayesian",
                         max_num_combinations=20,
                         cross_validation=2,
                         max_num_of_parallel_processes=-1,
                         parallel_combinations=5,
                         create_GUI=True,
                         concurrent_function=run_me_while_optimizing,
                         completion_function=run_me_after_optimizing)

opt.fit(X_train, y_train)
