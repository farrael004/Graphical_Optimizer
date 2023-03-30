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
    # Creating model, prediction and performance functions

    def model_function(params, X_train, y_train):
        gbr = GradientBoostingRegressor(n_estimators=params['n_estimators'],
                                        learning_rate=params['learning_rate'],
                                        max_depth=params['max_depth'],
                                        max_features=params['max_features'],
                                        min_samples_leaf=params['min_samples_leaf'],
                                        min_samples_split=params['min_samples_split'],
                                        random_state=42)

        model = gbr.fit(X_train, y_train)

        train_score = {"Train score": [model.train_score_.tolist()[:1000]]}

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
        return model_results


    # Creating hyperparameter dictionary

    hyperparameters_bayesian = {'n_estimators': [5000, 6000],  # Upper and lower bounds
                                'learning_rate': [0.001, 0.01],  # Upper and lower bounds
                                'max_depth': [2, 6],  # Upper and lower bounds
                                'max_features': ['sqrt', 'log2'],  # Categorical bounds
                                'min_samples_leaf': [1, 21],  # Upper and lower bounds
                                'min_samples_split': [2, 16], }  # Upper and lower bounds

    hyperparameters_grid_and_random = {'n_estimators': range(5000, 6000, 200),  # Upper and lower bounds
                                    'learning_rate': np.linspace(0.001, 0.01, 5).tolist(),  # Upper and lower bounds
                                    'max_depth': range(2, 6),  # Upper and lower bounds
                                    'max_features': ['sqrt', 'log2'],  # Categorical bounds
                                    'min_samples_leaf': range(1, 21, 2),  # Upper and lower bounds
                                    'min_samples_split': range(2, 16, 2), }  # Upper and lower bounds

    if params['method'] == 'bayesian':
        hyperparams = hyperparameters_bayesian
    else:
        hyperparams = hyperparameters_grid_and_random
    
    # Performing optimization

    def run_me_while_optimizing(opt: GraphicalOptimizer):
        #print(opt.df)
        return


    def run_me_after_optimizing(opt: GraphicalOptimizer):
        df = opt.df
        bestIndex = df["Adjusted R^2 Score"].idxmax()
        bestParams = df.iloc[bestIndex]
        print("Finished optimizing")
        print(f'Best performance: {bestParams["Adjusted R^2 Score"]}')
        print("Best combination of hyperparameters are:")
        print(bestParams[6:])


    opt = GraphicalOptimizer(model_function=model_function,
                            prediction_function=prediction_function,
                            performance_function=performance_function,
                            performance_parameter="Adjusted R^2 Score",
                            hyperparameters=hyperparams,
                            optimizer=params['method'],
                            max_num_combinations=params['total combinations'],
                            cross_validation=2,
                            max_num_of_parallel_processes=-1,
                            parallel_combinations=params['parallel combinations'],
                            create_GUI=False,
                            concurrent_function=run_me_while_optimizing,
                            completion_function=run_me_after_optimizing)

    opt.fit(X_train, y_train)
    
    return opt.results.best_score_

def pred(model, X):
    y_pred = model
    return y_pred

def perf(y_test, y_pred):
    return {"score": y_pred}

hyper = {'method': ['bayesian', 'grid', 'random'],
         'total combinations': [15, 30, 45, 90, 125, 140, 175],
         'parallel combinations': [1, 3, 6, 9, 12]}

opt = GraphicalOptimizer(model_function=model,
                         prediction_function=pred,
                         performance_function=perf,
                         performance_parameter="score",
                         hyperparameters=hyper,
                         optimizer='grid',
                         cross_validation=2)

opt.fit(X_train, y_train)