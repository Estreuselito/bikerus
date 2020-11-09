# import relevant modules
from data_preprocessing import decompress_pickle, compressed_pickle
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pickle
import os

# import data (based on preprocessed data from steps taken in pipelines 1-3)
X_test = decompress_pickle("./data/partitioned/BikeRental_X_test.pbz2")
X_train = decompress_pickle("./data/partitioned/BikeRental_X_train.pbz2")
Y_test = decompress_pickle("./data/partitioned/BikeRental_Y_test.pbz2")
Y_train =decompress_pickle("./data/partitioned/BikeRental_Y_train.pbz2")

# drop datetime column in order to avoid invalid type promotion error (MLPRegressor can't handle time series)
X_test = X_test.drop(['datetime'], axis = 1)
X_train = X_train.drop(['datetime'], axis = 1)

### MODEL CREATION ###
# initialize MLPRegressor (lbfgs solver used due to its efficiency)
NN_regr_CV = MLPRegressor(solver='lbfgs', max_iter=10000, random_state=0)

# set parameter grid to be searched for optimal parameters
param_grid = { 
    # tuple's ith element represents the number of neurons in the ith hidden layer. (5,) = 1 hidden layer with 5 neurons.
    "hidden_layer_sizes": [(5,), (10,), (2,2,), (5,5,)],
    # left out identity activation function due to its linearity
    "activation": ["logistic", "tanh", "relu"], 
    # L2 penalty parameter 
    "alpha": [0.01, 0.05, 0.1, 0.2],
    # learning_rate is kept at default (constant) since lbfgs solver does not use a learning rate
}

print("optimal parameters for the model are being computed")

### GRID SEARCH ###
# set up grid search with 5 fold cross validation
NN_regr_CV_model = GridSearchCV(estimator=NN_regr_CV, param_grid=param_grid, cv=5)

# execute grid search
NN_regr_CV_model.fit(X_train, Y_train)

print("the model is being trained on optimal parameters")

# set optimal paramteters
NN_regr_CV = NN_regr_CV.set_params(**NN_regr_CV_model.best_params_)

### TRAINING ###
# train model on optimal parameters
NN_regr_CV.fit(X_train, Y_train)

# store optimal parameters
if not os.path.exists("./python/NN_MLP_files"):
    os.makedirs("./python/NN_MLP_files")
optimal_parameters = pd.DataFrame(NN_regr_CV_model.best_params_)
compressed_pickle("./python/NN_MLP_files/optimal_parameters", optimal_parameters)

### EVALUATION ###
# computation of r squared
Y_train_pred = NN_regr_CV.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
Y_train_mean = Y_train.mean()
Y_train_meandev = sum((Y_train-Y_train_mean)**2)
r2 = 1 - Y_train_dev/Y_train_meandev

# computation of pseudo r squared
Y_test_pred = NN_regr_CV.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
Y_test_meandev = sum((Y_test-Y_train_mean)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev

### STORING MODEL & OUTPUT ###
# store (pseudo) r squared values
r2_df = pd.DataFrame(data=[r2], columns=["r2"])
pseudor2_df = pd.DataFrame(data=[pseudor2], columns=["pseudor2"])
r_squared_values = pd.concat([r2_df, pseudor2_df], axis=1)
compressed_pickle("./python/NN_MLP_files/r_squared_values", r_squared_values)

print(f"the model has finished training and exhibits an r squared of {r2} and a pseudo r sqaured of {pseudor2}")

# store full prediction dataframe by concatenating (unnormalized) Y_test_pred & X_test
prediction_X = pd.DataFrame.reset_index(X_test)
prediction_Y = pd.DataFrame(data=Y_test_pred, columns=["cnt"])

# unnormalize Y_test_pred
max_min_cnt = decompress_pickle("./data/preprocessed/cnt_min_max.pbz2")
max_cnt = max_min_cnt.iloc[0,0]
min_cnt = max_min_cnt.iloc[0,1]
norm_prediction_Y = prediction_Y * (max_cnt - min_cnt) + min_cnt
full_prediction_df = pd.concat([prediction_X, norm_prediction_Y], axis=1)
compressed_pickle("./python/NN_MLP_files/full_prediction_df", full_prediction_df)

print("the model's prediction has been saved, now let's save the model itself")

# save model using pickle (it can be loaded via ... pickle.load(open("./python/NN_MLP_files/NN_MLP_saved", "rb"))
pickle.dump(NN_regr_CV, open("./python/NN_MLP_files/NN_MLP_saved", "wb"))

print("the model has been saved and can be loaded again at any time")