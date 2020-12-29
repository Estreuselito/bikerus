import pandas as pd
import numpy as np
from data_partitioning import train_test_split_ts, get_sample_for_cv
from model_helpers import (import_train_test_calc,
                           r_squared_metrics)
from logger import logger
import os
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pickle
import joblib

# only activate in conda env with fastai running
# from fastai.tabular.all import *

# def fastai_neural_regression():
#     """This function will (train) and return the test dataframe with the predicted values
#     """
#    df, min_max, Y_train, Y_test, X_train, X_test, Y_train_mean, Y_train_meandev, Y_test_meandev = import_train_test_calc()
#
#     cat_names = ['season', 'yr', 'weathersit', "workingday", "holiday"]
#     cont_names = ['mnth', 'hr', 'weekday', 'temp', "windspeed", "hum"]
#     procs = [Categorify]
#     dls = TabularDataLoaders.from_df(
#         df, path, procs=procs, cat_names=cat_names, cont_names=cont_names, y_names="cnt", bs=64)
#     split = 0.8
#     splits = (list(range(0, round(len(df)*split))),
#               list(range(round(len(df)*split), len(df))))
#     to = TabularPandas(df, procs=[Categorify],
#                        cat_names=cat_names,
#                        cont_names=cont_names,
#                        y_names='cnt',
#                        splits=splits)
#     dls = to.dataloaders(bs=64)
#     learn = tabular_learner(dls,
#                             metrics=R2Score(),
#                             layers=[500, 250],
#                             n_out=1,
#                             loss_func=F.mse_loss)
#     try:
#         learn.load('./models/fastai/fastai_learner')
#         print("Model loaded!")
#     except FileNotFoundError:
#         print("This can take some time! Your model will be trained now!")
#         learn.fit_one_cycle(50)
#         if not os.path.exists("/models/fastai"):
#             os.makedirs("/models/fastai")
#         learn.save('./models/fastai/fastai_learner')
#         print("Model trained and saved!")

#     r2, pseudor2 = r_squared_metrics(X_train, X_test, Y_train, Y_train_meandev, Y_test, Y_test_meandev, learn)
#     return r2, pseudor2

###########
### SVR ###
###########

# Sklearn support vector regression trained on random split with Grid/RandomizedCV


def sklearn_support_vector_regression_rs_gridcv():
    df, min_max, Y_train, Y_test, X_train, X_test, Y_train_mean, Y_train_meandev, Y_test_meandev = import_train_test_calc(
        rs="_rs", nn="_NN_SVR")
    try:
        filename = "Model_SVR_rs_gridcv.sav"
        SVR_regr_CV_model = joblib.load(
            "./models/SVR_files/" + str(filename))
        logger.info("Model is loaded!\n")
    except:
        logger.info("Model is creating!\n")
        ### MODEL CREATION ###
        # initialize SVR
        # SVR_regr_CV = SVR(max_iter=25000)

        ## HYPERPARAMETER OPTIMIZATION ###

        # 1st RandomizedSearchCV parameters:
        # param_grid = {
        # "degree": [2, 4, 6]
        # "C": [1, 2, 4, 6],
        # "epsilon": [0.0, 0.05, 0.1],
        # "gamma": [1., 2., 3.],
        # "kernel": ["poly", "rbf"]
        # }
        # best parameters: {'kernel': 'rbf', 'gamma': 1.0, 'epsilon': 0.0, 'degree': 2, 'C': 1}

        # 2nd RandomizedSearchCV parameters:
        # param_grid = {
        # "C": [0.5, 1, 1.5],
        # "epsilon": [0.0, 0.01, 0.03],
        # "gamma": ["scale", "auto", 0., 1.],
        # "kernel": ["rbf"]
        # }
        # best parameters: {'kernel': 'rbf', 'gamma': 1.0, 'epsilon': 0.01, 'C': 1.5}

        # 3nd RandomizedSearchCV parameters:
        # param_grid = {
        # "C": [1.25, 1.5, 1.75],
        # "epsilon": [0.005, 0.01, 0.02],
        # "gamma": [0.75, 1.0, 1.25],
        # "kernel": ["rbf"]
        # }
        # best parameters: {'kernel': 'rbf', 'gamma': 1.0, 'epsilon': 0.01, 'C': 1.75}

        ## TRAINING ON OPTIMAL PARAMETERS ###

        # set optimal parameters
        SVR_regr_CV_model = SVR(C=1.75,
                                epsilon=0.01,
                                gamma=1.0,
                                kernel="rbf",
                                max_iter=25000)

        SVR_regr_CV_model.fit(X_train, Y_train.values.ravel())

        # store model
        if not os.path.exists("./models/SVR_files"):
            os.makedirs("./models/SVR_files")

        joblib.dump(SVR_regr_CV_model,
                    "./models/SVR_files/Model_SVR_rs_gridcv.sav")

    r2, pseudor2 = r_squared_metrics(
        X_train, X_test, Y_train, Y_train_meandev, Y_test, Y_test_meandev, SVR_regr_CV_model)

    return r2.values[0], pseudor2.values[0]

# Sklearn support vector regression trained on time series split with Grid/RandomizedCV


def sklearn_support_vector_regression_ts_gridcv():
    df, min_max, Y_train, Y_test, X_train, X_test, Y_train_mean, Y_train_meandev, Y_test_meandev = import_train_test_calc(
        nn="_NN_SVR")
    try:
        filename = "Model_SVR_ts_gridcv.sav"
        SVR_regr_CV_model = joblib.load(
            "./models/SVR_files/" + str(filename))
        logger.info("Model is loaded!\n")
    except:
        logger.info("Model is creating!\n")
        ### MODEL CREATION ###
        # initialize SVR
        # SVR_regr_CV = SVR(max_iter=25000)

        ## HYPERPARAMETER OPTIMIZATION ###

        # 1st RandomizedSearchCV parameters:
        # param_grid = {
        # "degree": [2, 4, 6]
        # "C": [1, 2, 4],
        # "epsilon": [0.0, 0.5, 0.1],
        # "gamma": [0.5, 1, 2],
        # "kernel": ["poly", "rbf"]
        # }
        # best parameters: {'kernel': 'rbf', 'gamma': 1, 'epsilon': 0.0, 'degree': 2, 'C': 1}

        # 2nd RandomizedSearchCV parameters:
        # param_grid = {
        # "C": [0.75, 1, 1.25, 1.5],
        # "epsilon": [0, 0.01, 0.03],
        # "gamma": [0.75, 1, 1.5],
        # "kernel": ["rbf"]
        # }
        # best parameters: {'kernel': 'rbf', 'gamma': 0.75, 'epsilon': 0.03, 'C': 0.75}

        # 3nd RandomizedSearchCV parameters:
        # param_grid = {
        # "C": [0.25, 0.5, 0.75],
        # "epsilon": [0.02, 0.025, 0.03],
        # "gamma": [0.6, 0.75, 0.8],
        # "kernel": ["rbf"]
        # }
        # best parameters: {'kernel': 'rbf', 'gamma': 0.6, 'epsilon': 0.03, 'C': 0.5}

        ## TRAINING ON OPTIMAL PARAMETERS ###

        # set optimal parameters
        SVR_regr_CV_model = SVR(C=0.5,
                                epsilon=0.03,
                                gamma=0.6,
                                kernel="rbf",
                                max_iter=25000)

        SVR_regr_CV_model.fit(X_train, Y_train.values.ravel())

        # store model
        if not os.path.exists("./models/SVR_files"):
            os.makedirs("./models/SVR_files")

        joblib.dump(SVR_regr_CV_model,
                    "./models/SVR_files/Model_SVR_ts_gridcv.sav")

    r2, pseudor2 = r_squared_metrics(
        X_train, X_test, Y_train, Y_train_meandev, Y_test, Y_test_meandev, SVR_regr_CV_model)

    return r2.values[0], pseudor2.values[0]

# Sklearn support vector regression trained on time series split with TimeSeriesCV


def sklearn_support_vector_regression_ts_tscv():
    df, min_max, Y_train, Y_test, X_train, X_test, Y_train_mean, Y_train_meandev, Y_test_meandev = import_train_test_calc(
        nn="_NN_SVR")
    try:
        filename = "Model_SVR_ts_tscv.sav"
        SVR_regr_CV_model = joblib.load(
            "./models/SVR_files/" + str(filename))
        logger.info("Model is loaded!\n")
    except:
        logger.info("Model is creating!\n")
        # ## FIND OPTIMAL PARAMETERS ###
        # 1st CV parameters:
        # param_grid = {
        # "C": [1.25, 1.5, 1.75],
        # "epsilon": [0.005, 0.01, 0.02],
        # "gamma": [0.75, 1.0, 1.25],
        # "kernel": ["rbf"]
        # }
        # best parameters: C=0.5, epsilon=0.03, gamma=0.5, kernel=”rbf”

        # see procedure below
        # Training the model incl. Cross Validation
        # df_parameters = pd.DataFrame()
        # folds = list(range(1, 6))
        # C = [1.25, 1.5, 1.75]
        # epsilon = [0.005, 0.01, 0.02]
        # gamma = [0.75, 1.0, 1.25]
        # # kernel is rbf by default (poly not taken into account based on previous performance)
        # for c in list(range(len(C))):
        #     for eps in list(range(len(epsilon))):
        #         for g in list(range(len(gamma))):
        #             # important: fold needs to be the last for-loop to be able to compute the means of Pseudo R^2 across the folds
        #             for fold in list(range(len(folds))):

        #                 X_train_cv, Y_train_cv, X_test_cv, Y_test_cv = get_sample_for_cv_NN_SVR(folds[-1],  # specify the number of total folds, last index of the list
        #                                                                                     # specifiy the current fold
        #                                                                                     folds[fold],
        #                                                                                     X_train,  # DataFrame X_train, which was created with the function train_test_split_ts
        #                                                                                     Y_train)  # DataFrame Y_train, which was created with the function train_test_split_ts

        #                 # to evaluate the prediction quality, we use the R2 measure
        #                 # as a benchmark, we initially calculated the mean value and the residual sum of squares of the target variable for the specific fold
        #                 Y_train_mean_cv = Y_train_cv.mean()

        #                 # remove-error-causing header
        #                 Y_train_cv_for_meandev = Y_train_cv.iloc[:,0]
        #                 Y_test_cv_for_meandev = Y_test_cv.iloc[:,0]

        #                 Y_train_meandev_cv = sum((Y_train_cv_for_meandev - float(Y_train_mean_cv))**2)

        #                 Y_test_meandev_cv = sum((Y_test_cv_for_meandev - float(Y_train_mean_cv))**2)

        #                 # initialize model
        #                 SVR_regr_CV_model = SVR(max_iter = 25000,
        #                                         C = C[c],
        #                                         epsilon =epsilon[eps],
        #                                         gamma = gamma[g]
        #                                         )

        #                 # train the model
        #                 SVR_regr_CV_model.fit(X_train_cv, Y_train_cv.values.ravel())

        #                 # Make predictions based on the traing set
        #                 Y_train_pred_cv = SVR_regr_CV_model.predict(X_train_cv)
        #                 Y_train_dev_cv = sum(
        #                     (Y_train_cv_for_meandev-Y_train_pred_cv)**2)
        #                 r2_cv = 1 - Y_train_dev_cv/Y_train_meandev_cv

        #                 # Evaluate the result by applying the model to the test set
        #                 Y_test_pred_cv = SVR_regr_CV_model.predict(X_test_cv)
        #                 Y_test_dev_cv = sum(
        #                     (Y_test_cv_for_meandev-Y_test_pred_cv)**2)
        #                 pseudor2_cv = 1 - Y_test_dev_cv/Y_test_meandev_cv

        #                 # Append results to dataframe
        #                 new_row = {'fold': folds[fold],
        #                             'C': C[c],
        #                             'epsilon': epsilon[eps],
        #                             'gamma': gamma[g],
        #                             'R2': r2_cv,
        #                             'PseudoR2': pseudor2_cv}

        #                 # Calculate means to find the best hyperparameters across all folds
        #                 n_folds = folds[-1]
        #                 i = 0
        #                 index = 0
        #                 mean_max = 0
        #                 while i < len(df_parameters):
        #                     if df_parameters.iloc[i:i+n_folds, 1].mean() > mean_max:
        #                         mean_max = df_parameters.iloc[i:i +
        #                                                         n_folds, 1].mean()
        #                         index = i
        #                         i += n_folds
        #                     else:
        #                         i += n_folds
        #                 df_parameters = df_parameters.append(
        #                     new_row, ignore_index=True)

        #                 # best parameters based on mean of PseudoR^2
        #                 best_parameters = pd.Series(
        #                     df_parameters.iloc[index])

        # # Initialize the model and the regressor with the best hyperparameters
        # SVR_regr_CV_model = SVR(max_iter = 25000,
        #                         C=(best_parameters['C']),
        #                         epsilon=(
        #                             best_parameters['epsilon']),
        #                         gamma=
        #                             best_parameters['gamma']
        #                         )

        # SVR_regr_CV_model.fit(X_train, Y_train.values.ravel())

        ### TRAINING ON OPTIMAL PARAMETERS ###

        # set optimal parameters
        SVR_regr_CV_model = SVR(max_iter=25000,
                                C=0.5,
                                epsilon=0.03,
                                gamma=0.5,
                                kernel='rbf'
                                )

        SVR_regr_CV_model.fit(X_train, Y_train.values.ravel())

        # store model
        if not os.path.exists("./models/SVR_files"):
            os.makedirs("./models/SVR_files")

        joblib.dump(SVR_regr_CV_model,
                    "./models/SVR_files/Model_SVR_ts_tscv.sav")

    r2, pseudor2 = r_squared_metrics(
        X_train, X_test, Y_train, Y_train_meandev, Y_test, Y_test_meandev, SVR_regr_CV_model)

    return r2.values[0], pseudor2.values[0]

##########
### NN ###
##########

# Sklearn neural net trained on random split with Grid/RandomizedCV


def sklearn_neural_net_multilayerperceptron_rs_gridcv():
    df, min_max, Y_train, Y_test, X_train, X_test, Y_train_mean, Y_train_meandev, Y_test_meandev = import_train_test_calc(
        rs="_rs", nn="_NN_SVR")
    try:
        filename = "Model_MLP_rs_gridcv.sav"
        NN_regr_CV_model = joblib.load(
            "./models/NN_MLP_files/" + str(filename))
        logger.info("Model is loaded!\n")
    except:
        logger.info("Model is creating!\n")
        ### MODEL CREATION ###

        # initialize MLPRegressor (lbfgs solver used due to its efficiency)
        # NN_regr_CV = MLPRegressor(
        #     solver='lbfgs', max_iter=10000, random_state=0)

        ### HYPERPARAMETER OPTIMIZATION ###

        # 1st RandomizedSearchCV parameters:
        # param_grid = {
        #     "hidden_layer_sizes": [(10,), (25,), (50,), (10, 10,), (25, 10,), (10, 25,), (25, 25,), (50, 50,)],
        #     "activation": ["logistic", "tanh", "relu"],
        #     "alpha": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
        # }
        # best parameters: {'hidden_layer_sizes': (50, 50,), 'alpha': 0.02, 'activation': 'tanh'}

        # 2nd RandomizedSearchCV parameters:
        # param_grid = {
        #     "hidden_layer_sizes": [(100,), (200,), (300,), (50, 50,), (75, 75,), (100, 100,)],
        #     "activation": ["logistic", "tanh", "relu"],
        #     "alpha": [0.01, 0.015, 0.02, 0.025, 0.03],
        # }
        # best parameters: {'hidden_layer_sizes': (200,), 'alpha': 0.025, 'activation': 'relu'}

        ### TRAINING ON OPTIMAL PARAMETERS ###

        # set optimal parameters
        NN_regr_CV_model = MLPRegressor(solver="lbfgs",
                                        max_iter=10000,
                                        random_state=0,
                                        hidden_layer_sizes=(200,),
                                        activation="relu",
                                        alpha=0.025
                                        )

        NN_regr_CV_model.fit(X_train, Y_train.values.ravel())

        # store model
        if not os.path.exists("./models/NN_MLP_files"):
            os.makedirs("./models/NN_MLP_files")

        joblib.dump(NN_regr_CV_model,
                    "./models/NN_MLP_files/Model_MLP_rs_gridcv.sav")

    r2, pseudor2 = r_squared_metrics(
        X_train, X_test, Y_train, Y_train_meandev, Y_test, Y_test_meandev, NN_regr_CV_model)

    return r2.values[0], pseudor2.values[0]

# Sklearn neural net trained on time series split with Grid/RandomizedCV


def sklearn_neural_net_multilayerperceptron_ts_gridcv():
    df, min_max, Y_train, Y_test, X_train, X_test, Y_train_mean, Y_train_meandev, Y_test_meandev = import_train_test_calc(
        nn="_NN_SVR")
    try:
        filename = "Model_MLP_ts_gridcv.sav"
        NN_regr_CV_model = joblib.load(
            "./models/NN_MLP_files/" + str(filename))
        logger.info("Model is loaded!\n")
    except:
        logger.info("Model is creating!\n")
        ### MODEL CREATION ###

        # initialize MLPRegressor (lbfgs solver used due to its efficiency)
        # NN_regr_CV = MLPRegressor(
        #     solver='lbfgs', max_iter=10000, random_state=0)

        ### HYPERPARAMETER OPTIMIZATION ###

        # 1st RandomizedSearchCV parameters:
        # param_grid = {
        #     "hidden_layer_sizes": [(50,), (100,), (50, 25,), (50, 50,)],
        #     "activation": ["tanh", "relu"],
        #     "alpha": [0.01, 0.02, 0.04, 0.05],
        # }
        # best parameters: {'hidden_layer_sizes': (50, 25), 'alpha': 0.02, 'activation': 'tanh'}

        # 2nd RandomizedSearchCV parameters:
        # param_grid = {
        #     "hidden_layer_sizes": [(50, 25,), (75, 25,), (75, 50,)],
        #     "activation": ["tanh", "relu"],
        #     "alpha": [0.015, 0.02, 0.025],
        # }
        # best parameters: {'hidden_layer_sizes': (50, 25), 'alpha': 0.02, 'activation': 'tanh'}

        ### TRAINING ON OPTIMAL PARAMETERS ###

        # set optimal parameters
        NN_regr_CV_model = MLPRegressor(solver="lbfgs",
                                        max_iter=10000,
                                        random_state=0,
                                        hidden_layer_sizes=(50, 25),
                                        activation='tanh',
                                        alpha=0.02
                                        )

        NN_regr_CV_model.fit(X_train, Y_train.values.ravel())

        # store model
        if not os.path.exists("./models/NN_MLP_files"):
            os.makedirs("./models/NN_MLP_files")

        joblib.dump(NN_regr_CV_model,
                    "./models/NN_MLP_files/Model_MLP_ts_gridcv.sav")

    r2, pseudor2 = r_squared_metrics(
        X_train, X_test, Y_train, Y_train_meandev, Y_test, Y_test_meandev, NN_regr_CV_model)

    return r2.values[0], pseudor2.values[0]

# Sklearn neural net trained on time series split with TimeSeriesCV


def sklearn_neural_net_multilayerperceptron_ts_tscv():
    df, min_max, Y_train, Y_test, X_train, X_test, Y_train_mean, Y_train_meandev, Y_test_meandev = import_train_test_calc(
        nn="_NN_SVR")
    try:
        filename = "Model_MLP_ts_tscv.sav"
        NN_regr_CV_model = joblib.load(
            "./models/NN_MLP_files/" + str(filename))
        logger.info("Model is loaded!\n")
    except:
        logger.info("Model is creating!\n")
        ### FIND OPTIMAL PARAMETERS ###
        # 1st CV parameters:
        # param_grid = {
        #     "hidden_layer_sizes": [(50,), (100,), (50, 25,)],
        #     "activation": ["tanh", "relu"],
        #     "alpha": [0.015, 0.02, 0.025],
        # }
        # best parameters: {hidden_layer_sizes=(50, 25,), alpha=0.015, activation=”relu”}

        # see procedure below
        # # Training the model incl. Cross Validation
        # df_parameters = pd.DataFrame()
        # folds = list(range(1, 6))
        # hidden_layer_sizes = [(50,), (100,), (50, 25,)]
        # activations = ["tanh", "relu"]
        # alphas = [0.015, 0.02, 0.025]
        # for size in list(range(len(hidden_layer_sizes))):
        #     for activation in list(range(len(activations))):
        #         for alpha in list(range(len(alphas))):
        #             # important: fold needs to be the last for-loop to be able to compute the means of Pseudo R^2 across the folds
        #             for fold in list(range(len(folds))):

        #                 X_train_cv, Y_train_cv, X_test_cv, Y_test_cv = get_sample_for_cv_NN_SVR(folds[-1],  # specify the number of total folds, last index of the list
        #                                                                                     # specifiy the current fold
        #                                                                                     folds[fold],
        #                                                                                     X_train,  # DataFrame X_train, which was created with the function train_test_split_ts
        #                                                                                     Y_train)  # DataFrame Y_train, which was created with the function train_test_split_ts

        #                 # to evaluate the prediction quality, we use the R2 measure
        #                 # as a benchmark, we initially calculated the mean value and the residual sum of squares of the target variable for the specific fold
        #                 Y_train_mean_cv = Y_train_cv.mean()

        #                 # remove-error-causing header
        #                 Y_train_cv_for_meandev = Y_train_cv.iloc[:,0]
        #                 Y_test_cv_for_meandev = Y_test_cv.iloc[:,0]

        #                 Y_train_meandev_cv = sum((Y_train_cv_for_meandev - float(Y_train_mean_cv))**2)

        #                 Y_test_meandev_cv = sum((Y_test_cv_for_meandev - float(Y_train_mean_cv))**2)

        #                 # initialize model
        #                 NN_regr_CV_model = MLPRegressor(solver= "lbfgs",
        #                                                 max_iter = 10000,
        #                                                 random_state=0,
        #                                                 hidden_layer_sizes=hidden_layer_sizes[size],
        #                                                 activation=activations[activation],
        #                                                 alpha=alphas[alpha],
        #                                                 )

        #                 # train the model
        #                 NN_regr_CV_model.fit(X_train_cv, Y_train_cv.values.ravel())

        #                 # Make predictions based on the traing set
        #                 Y_train_pred_cv = NN_regr_CV_model.predict(X_train_cv)
        #                 Y_train_dev_cv = sum(
        #                     (Y_train_cv_for_meandev-Y_train_pred_cv)**2)
        #                 r2_cv = 1 - Y_train_dev_cv/Y_train_meandev_cv

        #                 # Evaluate the result by applying the model to the test set
        #                 Y_test_pred_cv = NN_regr_CV_model.predict(X_test_cv)
        #                 Y_test_dev_cv = sum(
        #                     (Y_test_cv_for_meandev-Y_test_pred_cv)**2)
        #                 pseudor2_cv = 1 - Y_test_dev_cv/Y_test_meandev_cv

        #                 # Append results to dataframe
        #                 new_row = {'fold': folds[fold],
        #                             'hidden_layer_sizes': hidden_layer_sizes[size],
        #                             'activations': activations[activation],
        #                             'alphas': alphas[alpha],
        #                             'R2': r2_cv,
        #                             'PseudoR2': pseudor2_cv}

        #                 # Calculate means to find the best hyperparameters across all folds
        #                 n_folds = folds[-1]
        #                 i = 0
        #                 index = 0
        #                 mean_max = 0
        #                 while i < len(df_parameters):
        #                     if df_parameters.iloc[i:i+n_folds, 1].mean() > mean_max:
        #                         mean_max = df_parameters.iloc[i:i +
        #                                                         n_folds, 1].mean()
        #                         index = i
        #                         i += n_folds
        #                     else:
        #                         i += n_folds
        #                 df_parameters = df_parameters.append(
        #                     new_row, ignore_index=True)

        #                 # best parameters based on mean of PseudoR^2
        #                 # only the hyperparameters are included here, therefore the index starts at 2
        #                 best_parameters = pd.Series(
        #                     df_parameters.iloc[index, 2:])

        #                 print(df_parameters)

        # # Initialize the model and the regressor with the best hyperparameters
        # NN_regr_CV_model = MLPRegressor(solver= "lbfgs",
        #                                 max_iter = 10000,
        #                                 random_state=0,
        #                                 hidden_layer_sizes=(best_parameters['hidden_layer_sizes']),
        #                                 activation=(
        #                                     best_parameters['activations']),
        #                                 alpha=
        #                                     best_parameters['alphas']
        #                                 )

        ### TRAINING ON OPTIMAL PARAMETERS ###

        # set optimal parameters
        NN_regr_CV_model = MLPRegressor(solver="lbfgs",
                                        max_iter=10000,
                                        random_state=0,
                                        hidden_layer_sizes=(50, 25,),
                                        activation='relu',
                                        alpha=0.015
                                        )

        NN_regr_CV_model.fit(X_train, Y_train.values.ravel())

        # store model
        if not os.path.exists("./models/NN_MLP_files"):
            os.makedirs("./models/NN_MLP_files")

        joblib.dump(NN_regr_CV_model,
                    "./models/NN_MLP_files/Model_MLP_ts_tscv.sav")

    r2, pseudor2 = r_squared_metrics(
        X_train, X_test, Y_train, Y_train_meandev, Y_test, Y_test_meandev, NN_regr_CV_model)

    return r2.values[0], pseudor2.values[0]


def catboost_regressor_rs():

    df, min_max, Y_train, Y_test, X_train, X_test, Y_train_mean, Y_train_meandev, Y_test_meandev = import_train_test_calc(
        rs="_rs")

    cat_var = ["season", "yr", "mnth", "hr", "holiday",
               "weekday", "workingday", "weathersit"]
    for v in cat_var:
        X_train[v] = X_train[v].astype("int64")
        X_test[v] = X_test[v].astype("int64")

    model = CatBoostRegressor(loss_function='RMSE', depth=6,
                              learning_rate=0.1, iterations=1000, od_type='Iter', od_wait=10)

    try:
        model.load_model("./models/catboost/catboost_model_rs")
        logger.info("Model is loaded!\n")
    except:
        logger.info("Model is creating!\n")
        if not os.path.exists("./models/catboost"):
            os.makedirs("./models/catboost")

        model.fit(
            X_train, Y_train,
            use_best_model=True,
            cat_features=["season", "yr", "mnth", "hr",
                          "holiday", "weekday", "workingday", "weathersit"],
            eval_set=(X_test, Y_test),
            verbose=True,
            plot=True
        )

        model.save_model("./models/catboost/catboost_model_rs", format="cbm")

    r2, pseudor2 = r_squared_metrics(
        X_train, X_test, Y_train, Y_train_meandev, Y_test, Y_test_meandev, model)
    return r2.values[0], pseudor2.values[0]


def catboost_regressor_ts_gridcv():

    df, min_max, Y_train, Y_test, X_train, X_test, Y_train_mean, Y_train_meandev, Y_test_meandev = import_train_test_calc()

    cat_var = ["season", "yr", "mnth", "hr", "holiday",
               "weekday", "workingday", "weathersit"]
    for v in cat_var:
        X_train[v] = X_train[v].astype("int64")
        X_test[v] = X_test[v].astype("int64")

    model = CatBoostRegressor(loss_function='RMSE', depth=10,
                              learning_rate=0.05, iterations=1000, od_type='Iter', od_wait=10)

    try:
        model.load_model("./models/catboost/catboost_model_ts_gridcv")
        print("Model loaded!")
    except:
        if not os.path.exists("./models/catboost"):
            os.makedirs("./models/catboost")

        model.fit(
            X_train, Y_train,
            use_best_model=True,
            cat_features=["season", "yr", "mnth", "hr",
                          "holiday", "weekday", "workingday", "weathersit", "rush_hour"],
            eval_set=(X_test, Y_test),
            verbose=True,
            plot=True
        )

        model.save_model(
            "./models/catboost/catboost_model_ts_gridcv", format="cbm")

    r2, pseudor2 = r_squared_metrics(
        X_train, X_test, Y_train, Y_train_meandev, Y_test, Y_test_meandev, model)
    return r2.values[0], pseudor2.values[0]


def catboost_regressor_ts_tscv():
    df, min_max, Y_train, Y_test, X_train, X_test, Y_train_mean, Y_train_meandev, Y_test_meandev = import_train_test_calc()

    cat_var = ["season", "yr", "mnth", "hr", "holiday",
               "weekday", "workingday", "weathersit"]
    for v in cat_var:
        X_train[v] = X_train[v].astype("int64")
        X_test[v] = X_test[v].astype("int64")

    model = CatBoostRegressor(loss_function='RMSE', depth=6,
                              learning_rate=0.2, iterations=200, od_type='Iter', od_wait=10)

    try:
        model.load_model("./models/catboost/catboost_model_ts_tscv")
        print("Model loaded!")
    except:
        if not os.path.exists("./models/catboost"):
            os.makedirs("./models/catboost")

        df_parameters = pd.DataFrame()
        folds = list(range(1, 6))
        depths = [6, 8, 10]
        learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
        iterations = [30, 50, 100, 200, 400, 600, 800, 1000]

        for depth in list(range(len(depths))):
            for learning_rate in list(range(len(learning_rates))):
                for iteration in list(range(len(iterations))):
                    # important: fold needs to be the last for-loop to be able to compute the means of Pseudo R^2 across the folds
                    for fold in list(range(len(folds))):

                        X_train_cv, Y_train_cv, X_test_cv, Y_test_cv = get_sample_for_cv(folds[-1],  # specify the number of total folds, last index of the list
                                                                                         # specifiy the current fold
                                                                                         folds[fold],
                                                                                         X_train,  # DataFrame X_train, which was created with the function train_test_split_ts
                                                                                         Y_train)  # DataFrame Y_train, which was created with the function train_test_split_ts

                        # to evaluate the prediction quality, we use the R2 measure
                        # as a benchmark, we initially calculated the mean value and the residual sum of squares of the target variable for the specific fold
                        Y_train_mean_cv = Y_train_cv.mean()

                        # remove error-causing header
                        Y_train_cv = Y_train_cv.iloc[:, 0]
                        Y_test_cv = Y_test_cv.iloc[:, 0]
                        Y_train_meandev_cv = sum(
                            (Y_train_cv - float(Y_train_mean_cv))**2)
                        Y_test_meandev_cv = sum(
                            (Y_test_cv - float(Y_train_mean_cv))**2)

                        # initialize model

                        cat_var = ["season", "yr", "mnth", "hr", "holiday",
                                   "weekday", "workingday", "weathersit", "rush_hour"]
                        for v in cat_var:
                            X_train_cv[v] = X_train_cv[v].astype("int64")
                            X_test_cv[v] = X_test_cv[v].astype("int64")

                        model = CatBoostRegressor(loss_function='RMSE', depth=depths[depth], learning_rate=learning_rates[
                                                  learning_rate], iterations=iterations[iteration], od_type='Iter', od_wait=10)

                        # train the model
                        model.fit(
                            X_train_cv, Y_train_cv,
                            use_best_model=True,
                            cat_features=["season", "yr", "mnth", "hr",
                                          "holiday", "weekday", "workingday", "weathersit", "rush_hour"],
                            eval_set=(X_test_cv, Y_test_cv),
                            verbose=True,
                            plot=True
                        )

                        # Make predictions based on the traing set
                        Y_train_pred_cv = model.predict(X_train_cv)
                        Y_train_dev_cv = sum((Y_train_cv-Y_train_pred_cv)**2)
                        r2_cv = 1 - Y_train_dev_cv/Y_train_meandev_cv

                        # Evaluate the result by applying the model to the test set
                        Y_test_pred_cv = model.predict(X_test_cv)
                        Y_test_dev_cv = sum((Y_test_cv - Y_test_pred_cv)**2)
                        pseudor2_cv = 1 - Y_test_dev_cv/Y_test_meandev_cv

                        # Append results to dataframe
                        new_row = {'fold': folds[fold],
                                   'max_depth': depths[depth],
                                   'iterations': iterations[iteration],
                                   'learning_rate': learning_rates[learning_rate],
                                   'R2': r2_cv,
                                   'PseudoR2': pseudor2_cv}

                        # Calculate means to find the best hyperparameters across all folds
                        n_folds = folds[-1]
                        i = 0
                        index = 0
                        mean_max = 0
                        while i < len(df_parameters):
                            if df_parameters.iloc[i:i+n_folds, 0].mean() > mean_max:
                                mean_max = df_parameters.iloc[i:i +
                                                              n_folds, 0].mean()
                                index = i
                                i += n_folds
                            else:
                                i += n_folds
                        df_parameters = df_parameters.append(
                            new_row, ignore_index=True)

                        # best parameters based on mean of PseudoR^2
                        # only the hyperparameters are included here, therefore the index starts at 3
                        best_parameters = pd.Series(
                            df_parameters.iloc[index, 3:])

        model = CatBoostRegressor(loss_function='RMSE', depth=best_parameters["max_depth"],
                                  learning_rate=best_parameters["learning_rate"], iterations=best_parameters["iterations"], od_type='Iter', od_wait=10)
        model.fit(
            X_train, Y_train,
            use_best_model=True,
            cat_features=["season", "yr", "mnth", "hr",
                                                  "holiday", "weekday", "workingday", "weathersit", "rush_hour"],
            eval_set=(X_test, Y_test),
            verbose=True,
            plot=True
        )

        model.save_model(
            "./models/catboost/catboost_model_ts_tscv", format="cbm")

    r2, pseudor2 = r_squared_metrics(
        X_train, X_test, Y_train, Y_train_meandev, Y_test, Y_test_meandev, model)
    return r2.values[0], pseudor2.values[0]


def sklearn_random_forest_ts_tscv():

    df, min_max, Y_train, Y_test, X_train, X_test, Y_train_mean, Y_train_meandev, Y_test_meandev = import_train_test_calc()

    try:
        filename = 'Model_RandomForest_ts_tscv.sav'
        random_forest_ts_tscv = joblib.load(
            "./models/RandomForest_Model/" + str(filename))
        logger.info("Model is loaded!\n")
    except:
        logger.info("Model is creating!\n")
        # Training the model incl. Cross Validation
        df_parameters = pd.DataFrame()
        folds = list(range(1, 6))
        # Determine hyperparameter combinations
        # max_depth =  [8, 9, 10, 11, 12]
        # n_estimators = [100, 120, 140]
        # max_leaf_nodes = [60, 70, 80]
        # max_samples = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        # final values
        max_depth = [10]
        n_estimators = [120]
        max_leaf_nodes = [80]
        max_samples = [0.5]
        for depth in list(range(len(max_depth))):
            for number_trees in list(range(len(n_estimators))):
                for node in list(range(len(max_leaf_nodes))):
                    for sample in list(range(len(max_samples))):
                        # important: fold needs to be the last for-loop to be able to compute the means of Pseudo R^2 across the folds
                        for fold in list(range(len(folds))):

                            X_train_cv, Y_train_cv, X_test_cv, Y_test_cv = get_sample_for_cv(folds[-1],  # specify the number of total folds, last index of the list
                                                                                             # specifiy the current fold
                                                                                             folds[fold],
                                                                                             X_train,  # DataFrame X_train, which was created with the function train_test_split_ts
                                                                                             Y_train)  # DataFrame Y_train, which was created with the function train_test_split_ts

                            # to evaluate the prediction quality, we use the R2 measure
                            # as a benchmark, we initially calculated the mean value and the residual sum of squares of the target variable for the specific fold
                            Y_train_mean_cv = Y_train_cv.mean()
                            """ print("Y_train_cv type: ", type(
                                Y_train_cv), "\tValue: ", Y_train_cv, "\n",
                                "Y_train_mean_cv type: ", type(Y_train_mean_cv), "\tValue: ", Y_train_mean_cv) """
                            # print(((Y_test_cv-Y_train_mean_cv)**2).sum())
                            Y_train_meandev_cv = (
                                (Y_train_cv-Y_train_mean_cv)**2).sum()
                            Y_test_meandev_cv = (
                                (Y_test_cv-Y_train_mean_cv)**2).sum()

                            # initialize model
                            RForreg = RandomForestRegressor(max_depth=max_depth[depth],
                                                            n_estimators=n_estimators[number_trees],
                                                            max_leaf_nodes=max_leaf_nodes[node],
                                                            max_samples=max_samples[sample],
                                                            random_state=0)

                            # train the model
                            RForreg.fit(X_train_cv, Y_train_cv["cnt"])

                            # Make predictions based on the traing set
                            Y_train_pred_cv = RForreg.predict(X_train_cv)
                            Y_train_dev_cv = (
                                (Y_train_cv["cnt"]-Y_train_pred_cv)**2).sum()
                            r2_cv = 1 - Y_train_dev_cv/Y_train_meandev_cv

                            # Evaluate the result by applying the model to the test set
                            Y_test_pred_cv = RForreg.predict(X_test_cv)
                            Y_test_dev_cv = (
                                (Y_test_cv["cnt"]-Y_test_pred_cv)**2).sum()
                            pseudor2_cv = 1 - Y_test_dev_cv/Y_test_meandev_cv

                            # Append results to dataframe
                            new_row = {'fold': folds[fold],
                                       'max_depth': max_depth[depth],
                                       'n_estimators': n_estimators[number_trees],
                                       'max_leaf_nodes': max_leaf_nodes[node],
                                       'max_samples': max_samples[sample],
                                       'R2': r2_cv,
                                       'PseudoR2': pseudor2_cv}

                            # Calculate means to find the best hyperparameters across all folds
                            n_folds = folds[-1]
                            i = 0
                            index = 0
                            mean_max = 0
                            while i < len(df_parameters):
                                if df_parameters.iloc[i:i+n_folds, 0].mean() > mean_max:
                                    mean_max = df_parameters.iloc[i:i +
                                                                  n_folds, 0].mean()
                                    index = i
                                    i += n_folds
                                else:
                                    i += n_folds
                            df_parameters = df_parameters.append(
                                new_row, ignore_index=True)

                            # best parameters based on mean of PseudoR^2
                            # only the hyperparameters are included here, therefore the index starts at 3
                            best_parameters = pd.Series(
                                df_parameters.iloc[index, 3:])

        # Initialize the model and the regressor with the best hyperparameters
        random_forest_ts_tscv = RandomForestRegressor(max_depth=int(best_parameters['max_depth']),
                                                      n_estimators=int(
            best_parameters['n_estimators']),
            max_leaf_nodes=int(
            best_parameters['max_leaf_nodes']),
            max_samples=best_parameters['max_samples'],
            random_state=0)
        # train the model with the hyperparameters
        random_forest_ts_tscv.fit(X_train, Y_train.values.ravel())

        joblib.dump(random_forest_ts_tscv,
                    "./models/RandomForest_Model/Model_RandomForest_ts_tscv.sav")

    r2, pseudor2 = r_squared_metrics(
        X_train, X_test, Y_train, Y_train_meandev, Y_test, Y_test_meandev, random_forest_ts_tscv)
    return r2.values[0], pseudor2.values[0]


def sklearn_random_forest_rs_gridcv():
    df, min_max, Y_train, Y_test, X_train, X_test, Y_train_mean, Y_train_meandev, Y_test_meandev = import_train_test_calc(
        rs="_rs")

    try:
        filename = 'Model_RandomForest_rs_gridcv.sav'
        random_forest_rs_gridcv = joblib.load(
            "./models/RandomForest_Model/" + str(filename))
        logger.info("Model is loaded!\n")
    except:
        logger.info("Model is creating!\n")
        # Training the model incl. Cross Validation
        # Initialize RandomForestRegressor
        RForregCV = RandomForestRegressor(random_state=0)
        # Determine hyperparameter combinations
        # param_grid = { 'max_depth': [8, 9, 10, 11, 12, 13],
        # 'n_estimators': [80, 100, 120],
        # 'max_leaf_nodes': [60, 70, 80],
        # 'max_samples': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]}
        # final values
        param_grid = {'max_depth': [12],
                      'n_estimators': [120],
                      'max_leaf_nodes': [80],
                      'max_samples': [0.3]}

        # Cross Validation
        CV_rfmodel = GridSearchCV(
            estimator=RForregCV, param_grid=param_grid, cv=5)
        CV_rfmodel.fit(X_train, Y_train.values.ravel())

        # Final training
        random_forest_rs_gridcv = RForregCV.set_params(
            **CV_rfmodel.best_params_)
        random_forest_rs_gridcv.fit(X_train, Y_train.values.ravel())

        # Save model
        joblib.dump(random_forest_rs_gridcv,
                    "./models/RandomForest_Model/Model_RandomForest_rs_gridcv.sav")

    r2, pseudor2 = r_squared_metrics(
        X_train, X_test, Y_train, Y_train_meandev, Y_test, Y_test_meandev, random_forest_rs_gridcv)

    return r2.values[0], pseudor2.values[0]


def sklearn_random_forest_ts_gridcv():
    df, min_max, Y_train, Y_test, X_train, X_test, Y_train_mean, Y_train_meandev, Y_test_meandev = import_train_test_calc()

    try:
        filename = 'Model_RandomForest_ts_gridcv.sav'
        random_forest_model_ts_gridcv = joblib.load(
            "./models/RandomForest_Model/" + str(filename))
        logger.info("Model is loaded!\n")
    except:
        logger.info("Model is creating!\n")
        # Training the model incl. Cross Validation
        # Initialize RandomForestRegressor
        RForregCV = RandomForestRegressor(random_state=0)
        # Determine hyperparameter combinations
        # param_grid = {'max_depth': [8, 9, 10, 11, 12],
        # 'n_estimators': [80, 100, 120, 140],
        # 'max_leaf_nodes': [60, 70, 80],
        # 'max_samples': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]}

        # final values
        param_grid = {'max_depth': [10],
                      'n_estimators': [100],
                      'max_leaf_nodes': [80],
                      'max_samples': [0.2]}

        # Cross Validation
        CV_rfmodel = GridSearchCV(
            estimator=RForregCV, param_grid=param_grid, cv=5)
        CV_rfmodel.fit(X_train, Y_train.values.ravel())

        # Final training
        random_forest_model_ts_gridcv = RForregCV.set_params(
            **CV_rfmodel.best_params_)
        random_forest_model_ts_gridcv.fit(X_train, Y_train.values.ravel())

        # Save model
        joblib.dump(random_forest_model_ts_gridcv,
                    "./models/RandomForest_Model/Model_RandomForest_ts_gridcv.sav")

    r2, pseudor2 = r_squared_metrics(
        X_train, X_test, Y_train, Y_train_meandev, Y_test, Y_test_meandev, random_forest_model_ts_gridcv)

    return r2.values[0], pseudor2.values[0]
