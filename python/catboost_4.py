from data_storage import check_and_create_and_insert, connection
from sql_commands import create_table_X_train_test, create_table_Y_train_test
import pandas as pd
from model_helpers import *
import pandas as pd
import numpy as np
from data_partitioning import train_test_split_ts, get_sample_for_cv
import os
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

#This Skript calculates the best catboost model with a random split of 0.75/0.25 and dropping the year


def Snippet_199(): 
    print()
    print(format('How to find optimal parameters for CatBoost using GridSearchCV for Regression','*^82'))    
    
    import warnings
    warnings.filterwarnings("ignore")
    
    # load libraries
    from data_storage import check_and_create_and_insert, connection
    from sklearn import datasets
    from sklearn.model_selection import GridSearchCV
    from catboost import CatBoostRegressor

    df = pd.read_sql_query('''SELECT * FROM hours_preprocessed''', connection)
    
    Y = df.cnt
    X = df.drop("cnt", axis= 1)
    
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    X_train = X_train.drop("yr", axis=1)
    X_test = X_test.drop("yr", axis=1)
    X_train = X_train.drop("datetime", axis=1)
    X_test = X_test.drop("datetime", axis=1)

    cat_var = ["season", "mnth", "hr", "holiday",
                   "weekday", "workingday", "weathersit", "rush_hour"]
    for v in cat_var:
        X_train[v] = X_train[v].astype("int64")
        X_test[v] = X_test[v].astype("int64")

    model = CatBoostRegressor(loss_function='RMSE', cat_features= ["season", "mnth", "hr", "holiday",
                   "weekday", "workingday", "weathersit", "rush_hour"])
    parameters = {'depth'         : [4, 6, 8, 10, 11],
                  'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3],
                  'iterations'    : [30, 50, 100, 200, 400, 600, 800, 1000, 1200]
                 }
    grid = GridSearchCV(estimator=model, param_grid = parameters,  cv= 2, n_jobs=-1)
    grid.fit(X_train, Y_train)    

    # Results from Grid Search
    print("\n========================================================")
    print(" Results from Grid Search " )
    print("========================================================")    
    
    print("\n The best estimator across ALL searched params:\n",
          grid.best_estimator_)
    
    print("\n The best score across ALL searched params:\n",
          grid.best_score_)
    
    print("\n The best parameters across ALL searched params:\n",
          grid.best_params_)
    
    print("\n ========================================================")

#Snippet_199()

def catboost_regressor():

    df = pd.read_sql_query('''SELECT * FROM hours_preprocessed''', connection)

    Y = df.cnt
    X = df.drop("cnt", axis= 1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    Y_train_mean = Y_train.mean()
    #Y_train_meandev = sum((Y_train-Y_train_mean)**2)
    Y_train_meandev = ((Y_train-Y_train_mean)**2).sum()
    #Y_test_meandev = sum((Y_test-Y_train_mean)**2)
    Y_test_meandev = ((Y_test-Y_train_mean)**2).sum()

    X_train = X_train.drop("yr", axis=1)
    X_test = X_test.drop("yr", axis=1)
    X_train = X_train.drop("datetime", axis=1)
    X_test = X_test.drop("datetime", axis=1)
    

    cat_var = ["season", "mnth", "hr", "holiday",
                   "weekday", "workingday", "weathersit", "rush_hour"]
    for v in cat_var:
        X_train[v] = X_train[v].astype("int64")
        X_test[v] = X_test[v].astype("int64")

    model = CatBoostRegressor(loss_function='RMSE', depth=10,
                                  learning_rate=0.05, iterations=1500, od_type='Iter', od_wait=10)

 
    if not os.path.exists("./models/catboost"):
        os.makedirs("./models/catboost")

    model.fit(
            X_train, Y_train,
            use_best_model=True,
            cat_features=["season", "mnth", "hr",
                          "holiday", "weekday", "workingday", "weathersit", "rush_hour"],
            eval_set=(X_test, Y_test),
            verbose=True,
            plot=True
        )
    model.save_model("./models/catboost/catboost_model4", format="cbm")

    Y_train_pred = model.predict(X_train)
    Y_train_dev = sum((Y_train-Y_train_pred)**2)
    r2 = 1 - Y_train_dev/Y_train_meandev

    # calculate pseudo-r-squared
    Y_test_pred = model.predict(X_test)
    Y_test_dev = sum((Y_test-Y_test_pred)**2)
    pseudor2 = 1 - Y_test_dev/Y_test_meandev
    
    print("R2: " + str(r2))
    print("PseudoR2: " + str(pseudor2))
    return r2, pseudor2

catboost_regressor()