import pandas as pd

from hyperoptimize import GraphicalOptimizer

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.linspace(-5, 5, 1000)
y = np.linspace(-5, 5, 1000)
X, y = np.meshgrid(X, y)
Z = np.sin(3 * X) + np.cos(2 * X) - 2.2 * np.sin(3 * X) - 2 * np.cos(3.4 * X) - (
            np.sin(3 * y) + np.cos(2 * y) - 2.2 * np.sin(3 * y) - 2 * np.cos(3.4 * y))

print(f'Maximum score is: {Z.max()}')

# Plot the surface.
surf = ax.plot_surface(X, y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


# Creating model, prediction and performance functions
def model_function(params, X_train, y_train):
    model = np.sin(3 * params['X']) + np.cos(2 * params['X']) - 2.2 * np.sin(3 * params['X']) - 2 * np.cos(
        3.4 * params['X'])
    model += np.sin(3 * params['Y']) + np.cos(2 * params['Y']) - 2.2 * np.sin(3 * params['Y']) - 2 * np.cos(
        3.4 * params['Y'])
    return model


def prediction_function(model, X):
    y_pred = model
    return y_pred


def performance_function(y_test, y_pred):
    return {"score": y_pred}


# Creating hyperparameter dictionary
hyperparameters_bayesian = {'X': [-5.0, 5.0], 'Y': [-5.0, 5.0]}  # Upper and lower bounds

hyperparameters_grid_and_random = {'X': np.linspace(-2, 2, 100).tolist()}  # Upper and lower bounds

# Performing optimization
opt = GraphicalOptimizer(model_function=model_function,
                         prediction_function=prediction_function,
                         performance_function=performance_function,
                         performance_parameter="score",
                         hyperparameters=hyperparameters_bayesian,
                         optimizer="bayesian",
                         max_num_combinations=60,
                         cross_validation=2,
                         max_num_of_parallel_processes=-1,
                         parallel_combinations=8,
                         seed=1)

opt.fit(X, y)
