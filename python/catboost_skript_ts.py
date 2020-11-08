from data_preprocessing import decompress_pickle, compressed_pickle
import pandas as pd
import numpy as np
import os
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split


import time
from functools import wraps


def logging(func):
    """function for logging the time

    Parameters
    ----------
    func: function
        the to measure function
    """
    @wraps(func)
    def logger(*args, **kwargs):
        """actual timer"""
        start = time.time()
        func(*args, **kwargs)
        print(f"Calling {func.__name__}: Needed {time.time() - start:.5f} seconds!")

    return logger

@logging

def catboost_regressor():
# load data
    Y_train = decompress_pickle("./data/partitioned/BikeRental_Y_train.pbz2")
    Y_test = decompress_pickle("./data/partitioned/BikeRental_Y_test.pbz2")
    X_train = decompress_pickle("./data/partitioned/BikeRental_X_train.pbz2")
    X_test = decompress_pickle("./data/partitioned/BikeRental_X_test.pbz2")


    X_train = X_train.drop('datetime', axis=1)
    X_test = X_test.drop('datetime', axis=1)



    Y_train_mean = Y_train.mean()
    Y_train_meandev = sum((Y_train-Y_train_mean)**2)
    Y_test_meandev = sum((Y_test-Y_train_mean)**2)

    cat_var = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
    for v in cat_var:
        X_train[v] = X_train[v].astype("int64")
        X_test[v] = X_test[v].astype("int64")

    if not os.path.exists("./catboost"):
        os.makedirs("./catboost")
    
    os.chdir("./catboost")

    model = CatBoostRegressor(loss_function='RMSE', depth=6, learning_rate=0.1, iterations=1000, od_type='Iter', od_wait=10) 

    model.fit(
        X_train, Y_train, 
        use_best_model=True,
        cat_features= ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"],
        eval_set=(X_test, Y_test),
        verbose=True,  
        plot=True
        
    )

    Y_train_pred = model.predict(X_train)
    Y_train_dev = sum((Y_train-Y_train_pred)**2) 
    r2 = 1 - Y_train_dev/Y_train_meandev 
    print("R2 =", r2)

    Y_test_pred = model.predict(X_test)
    Y_test_dev = sum((Y_test-Y_test_pred)**2)
    pseudor2 = 1 - Y_test_dev/Y_test_meandev
    print("Pseudo-R2 =", pseudor2)

    if not os.path.exists("./catboost"):
       pass
    
    model.save_model("catboost_model", format="cbm")

catboost_regressor()