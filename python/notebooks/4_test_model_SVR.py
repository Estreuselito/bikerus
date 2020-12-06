from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from model_helpers import import_train_test_calc_NN_SVR_ts, r_squared_metrics_NN_SVR_ts, import_train_test_calc_NN_SVR_rs, r_squared_metrics_NN_SVR_rs
import os
import joblib
import pandas as pd
from data_partitioning import train_test_split_ts, get_sample_for_cv_NN_SVR



# Sklearn support vector regression trained on random split with Grid/RandomizedCV
def sklearn_support_vector_regression_rs_gridcv():

    try:
        filename = "Model_SVR_rs_gridcv.sav"
        NN_regr_CV_model = joblib.load(
            "./models/SVR_files/" + str(filename))
    except:
        df, min_max, Y_train, Y_test, X_train, X_test, Y_train_mean, Y_train_meandev, Y_test_meandev = import_train_test_calc_NN_SVR_rs()

        ### MODEL CREATION ###
        # initialize SVR
        # SVR_regr_CV = SVR(max_iter=25000)

        ## HYPERPARAMETER OPTIMIZATION ###

        # 1st RandomizedSearchCV parameters:
            # param_grid = {
                # "degree": [1, 2, 4, 6]
                # "C": [2, 4, 6],
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
        SVR_regr_CV_model = SVR(C = 1.75, 
                                epsilon = 0.01,
                                gamma = 1.0,
                                kernel = "rbf",
                                max_iter = 25000)

        SVR_regr_CV_model.fit(X_train, Y_train.values.ravel())

        # store model
        if not os.path.exists("./models/SVR_files"):
            os.makedirs("./models/SVR_files")

        joblib.dump(SVR_regr_CV_model,
                    "./models/SVR_files/Model_SVR_rs_gridcv.sav")

        r2, pseudor2 = r_squared_metrics_NN_SVR_rs(X_train, X_test, Y_train, Y_train_meandev, Y_test, Y_test_meandev, SVR_regr_CV_model)

        return r2.values[0], pseudor2.values[0]

# Sklearn support vector regression trained on time series split with Grid/RandomizedCV
def sklearn_support_vector_regression_ts_gridcv():


    try:
        filename = "Model_SVR_ts_gridcv.sav"
        NN_regr_CV_model = joblib.load(
            "./models/SVR_files/" + str(filename))
    except:
        df, min_max, Y_train, Y_test, X_train, X_test, Y_train_mean, Y_train_meandev, Y_test_meandev = import_train_test_calc_NN_SVR_ts()

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
        SVR_regr_CV_model = SVR(C = 0.5,
                                epsilon = 0.03,
                                gamma = 0.6,
                                kernel= "rbf",
                                max_iter=25000)

        SVR_regr_CV_model.fit(X_train, Y_train.values.ravel())

        # store model
        if not os.path.exists("./models/SVR_files"):
            os.makedirs("./models/SVR_files")

        joblib.dump(SVR_regr_CV_model,
                    "./models/SVR_files/Model_SVR_ts_gridcv.sav")

        r2, pseudor2 = r_squared_metrics_NN_SVR_ts(X_train, X_test, Y_train, Y_train_meandev, Y_test, Y_test_meandev, SVR_regr_CV_model)

        return r2.values[0], pseudor2.values[0]

# Sklearn support vector regression trained on time series split with TimeSeriesCV
def sklearn_support_vector_regression_ts_tscv():

    try:
        filename = "Model_SVR_ts_tscv.sav"
        NN_regr_CV_model = joblib.load(
            "./models/SVR_files/" + str(filename))
    except:
        df, min_max, Y_train, Y_test, X_train, X_test, Y_train_mean, Y_train_meandev, Y_test_meandev = import_train_test_calc_NN_SVR_ts()

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
    SVR_regr_CV_model = SVR(max_iter = 25000,
                            C=0.5,
                            epsilon=0.03,
                            gamma=0.5,
                            kernel=”rbf”
                            )

    SVR_regr_CV_model.fit(X_train, Y_train.values.ravel())

    # store model
    if not os.path.exists("./models/SVR_files"):
        os.makedirs("./models/SVR_files")

    joblib.dump(SVR_regr_CV_model,
                "./models/SVR_files/Model_SVR_ts_tscv.sav")

    r2, pseudor2 = r_squared_metrics_NN_SVR_ts(
        X_train, X_test, Y_train, Y_train_meandev, Y_test, Y_test_meandev, SVR_regr_CV_model)

    return r2.values[0], pseudor2.values[0]

sklearn_support_vector_regression_ts_tscv()