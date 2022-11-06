import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.ensemble import GradientBoostingRegressor

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
    gbr = GradientBoostingRegressor(n_estimators=6000,
                                    learning_rate=params['learning_rate'],
                                    max_depth=params['max_depth'],
                                    max_features=params['max_features'],
                                    min_samples_leaf=10,
                                    min_samples_split=8,
                                    random_state=42)

    model = gbr.fit(X_train, y_train)
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

hyperparameters = {'n_estimators': [5000, 6000],  # Upper and lower bounds
                   'learning_rate': [0.001, 0.01],  # Upper and lower bounds
                   'max_depth': [2, 6],  # Upper and lower bounds
                   'max_features': ['sqrt', 'log2'],  # Categorical bounds
                   'min_samples_leaf': [1, 21],  # Upper and lower bounds
                   'min_samples_split': [1, 16], }  # Upper and lower bounds

# Performing optimization

opt = GraphicalOptimizer(ModelFunction=modelFunction,
                         PerformanceFunction=performanceFunction,
                         PredictionFunction=predictionFunction,
                         hyperparameters=hyperparameters,
                         optimizer="bayesian",
                         maxNumCombinations=100,
                         crossValidation=2,
                         parallelCombinations=3)

opt.fit(X_train, y_train)