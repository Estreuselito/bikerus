import numpy as np
import pandas as pd
import joblib
from data_preprocessing import decompress_pickle
from sklearn.model_selection import train_test_split
from data_partitioning import train_test_split_ts
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# import data
df = decompress_pickle("./data/preprocessed/BikeRental_preprocessed.pbz2")

# Drop of the datetime column, because the model cannot handle its type (object)
df = df.drop(['datetime'], axis = 1)

# Create the initial train and test samples
train_size = 0.8
X_train, Y_train, X_test, Y_test = train_test_split_ts(df, train_size)

# to evaluate the prediction quality, we will use the R2 measure
# as a benchmark, we initially calculated the mean value and the residual sum of squares of the target variable
Y_train_mean = Y_train.mean()
print("Y_train_mean =", Y_train_mean)
Y_train_meandev = sum((Y_train-Y_train_mean)**2)
print("Y_train_meandev =", Y_train_meandev)
Y_test_meandev = sum((Y_test-Y_train_mean)**2)
print("Y_test_meandev =", Y_test_meandev)

# Cross Validation
RForregCV = RandomForestRegressor(random_state=0)
param_grid = { 
    'max_depth': [6],
    'n_estimators': [1000],
    'max_features': [8],
}
CV_rfmodel = GridSearchCV(estimator=RForregCV, param_grid=param_grid, cv=5)
CV_rfmodel.fit(X_train, Y_train)
print(CV_rfmodel.best_params_)

RForregCV = RForregCV.set_params(**CV_rfmodel.best_params_)
RForregCV.fit(X_train, Y_train)
Y_train_pred = RForregCV.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)

Y_test_pred = RForregCV.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)