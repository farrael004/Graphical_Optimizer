import sys

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingRegressor

from hyperoptimize import GraphicalOptimizer

import wandb


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

# Creating model, prediction and performance functions
def model_function(params, X_train, y_train):
    
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="my-project",
        
        # track hyperparameters and run metadata
        config={
            "n_estimators": params['n_estimators'],
            "learning_rate": params['learning_rate'],
            "max_depth": params['max_depth'],
            "max_features": params['max_features'],
            "min_samples_leaf": params['min_samples_leaf']
        }
    )
    
    gbr = GradientBoostingRegressor(n_estimators=params['n_estimators'],
                                    learning_rate=params['learning_rate'],
                                    max_depth=params['max_depth'],
                                    max_features=params['max_features'],
                                    min_samples_leaf=params['min_samples_leaf'],
                                    min_samples_split=params['min_samples_split'],
                                    random_state=42)

    model = gbr.fit(X_train, y_train)
    
    train_score = {"Train loss": [model.train_score_.tolist()[:1000]], "Test loss": [model.train_score_.tolist()[:1000]]}
    
    df = pd.DataFrame({
        "Epochs": [i for i in range(1000)],
        "Train loss": model.train_score_.tolist()[:1000]
    })
    
    table = wandb.Table(data=df)
    wandb.log({"Train loss" : wandb.plot.line(table, "Epochs", "Train loss",
            title="Train loss")})
    return model, train_score


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

    wandb.log(model_results)
    return model_results


# Creating hyperparameter dictionary
hyperparameters_bayesian = {'n_estimators': [5000, 6000],  # Upper and lower bounds
                            'learning_rate': [0.001, 0.01],  # Upper and lower bounds
                            'max_depth': [2, 6],  # Upper and lower bounds
                            'max_features': ['sqrt', 'log2'],  # Categorical bounds
                            'min_samples_leaf': [1, 21],  # Upper and lower bounds
                            'min_samples_split': [2, 16], }  # Upper and lower bounds

hyperparameters_grid_and_random = {'n_estimators': range(5000, 6000, 100),  # Extensive list of possibilities
                                   'learning_rate': np.linspace(0.001, 0.01, 10).tolist(),  # Extensive list of possibilities
                                   'max_depth': range(2, 6),  # Extensive list of possibilities
                                   'max_features': ['sqrt', 'log2'],  # Extensive list of possibilities
                                   'min_samples_leaf': range(1, 21),  # Extensive list of possibilities
                                   'min_samples_split': range(2, 16), }  # Extensive list of possibilities


# Creating functions that runs after and while the optimization runs.
def run_me_while_optimizing(opt: GraphicalOptimizer):
    print('---------------------------')
    print('Experiment completed:')
    print(f'Adjusted R^2 Score: {opt.df.iloc[-1]["Adjusted R^2 Score"]}')


def run_me_after_optimizing(opt: GraphicalOptimizer):
    df = opt.df
    best_index = df["Adjusted R^2 Score"].idxmax()
    best_params = df.iloc[best_index]
    print("Finished optimizing")
    print(f'Best performance: {best_params["Adjusted R^2 Score"]}')
    print("Best combination of hyperparameters are:")
    print(best_params[6:])
    print('---------------------------')
    print('Best performance:')
    print(opt.results.best_score_)
    print("Best combination of hyperparameters are:")
    print(opt.results.best_params_)
    df.to_json('sample_data.pkl')
    

# Dashboard API
if len(sys.argv) > 1:
    dashboard_url = sys.argv[1]
else:
    dashboard_url = None

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
                         parallel_combinations=2,
                         create_GUI=False,
                         concurrent_function=run_me_while_optimizing,
                         completion_function=run_me_after_optimizing,
                         dashboard_url=dashboard_url,
                         verbose=1)

opt.fit(X_train, y_train)