from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from model_helpers import import_train_test_calc_NN_SVR_ts, r_squared_metrics_NN_SVR_ts, import_train_test_calc_NN_SVR_rs, r_squared_metrics_NN_SVR_rs
import os
import joblib
import pandas as pd
from data_partitioning import train_test_split_ts, get_sample_for_cv_NN_SVR
from sklearn.model_selection import train_test_split



# Sklearn neural net trained on random split with Grid/RandomizedCV
def sklearn_neural_net_multilayerperceptron_rs_gridcv():

    try:
        filename = "Model_MLP_rs_gridcv.sav"
        NN_regr_CV_model = joblib.load(
            "./models/NN_MLP_files/" + str(filename))
    except:
        df, min_max, Y_train, Y_test, X_train, X_test, Y_train_mean, Y_train_meandev, Y_test_meandev = import_train_test_calc_NN_SVR_rs()

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
        NN_regr_CV_model = MLPRegressor(solver= "lbfgs",
                                        max_iter = 10000,
                                        random_state = 0,
                                        hidden_layer_sizes = (200,),
                                        activation= "relu",
                                        alpha= 0.025
                                        )

        NN_regr_CV_model.fit(X_train, Y_train.values.ravel())

        # store model
        if not os.path.exists("./models/NN_MLP_files"):
            os.makedirs("./models/NN_MLP_files")

        joblib.dump(NN_regr_CV_model,
                    "./models/NN_MLP_files/Model_MLP_rs_gridcv.sav")

        r2, pseudor2 = r_squared_metrics_NN_SVR_rs(X_train, X_test, Y_train, Y_train_meandev, Y_test, Y_test_meandev, NN_regr_CV_model)

        return r2.values[0], pseudor2.values[0]


# Sklearn neural net trained on time series split with Grid/RandomizedCV
def sklearn_neural_net_multilayerperceptron_ts_gridcv():

    try:
        filename = "Model_MLP_ts_gridcv.sav"
        NN_regr_CV_model = joblib.load(
            "./models/NN_MLP_files/" + str(filename))
    except:
        df, min_max, Y_train, Y_test, X_train, X_test, Y_train_mean, Y_train_meandev, Y_test_meandev = import_train_test_calc_NN_SVR_ts()

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
        NN_regr_CV_model = MLPRegressor(solver= "lbfgs",
                                        max_iter = 10000,
                                        random_state = 0,
                                        hidden_layer_sizes = (50, 25),
                                        activation= 'tanh',
                                        alpha= 0.02
                                        )

        NN_regr_CV_model.fit(X_train, Y_train.values.ravel())

        # store model
        if not os.path.exists("./models/NN_MLP_files"):
            os.makedirs("./models/NN_MLP_files")

        joblib.dump(NN_regr_CV_model,
                    "./models/NN_MLP_files/Model_MLP_ts_gridcv.sav")

        r2, pseudor2 = r_squared_metrics_NN_SVR_ts(X_train, X_test, Y_train, Y_train_meandev, Y_test, Y_test_meandev, NN_regr_CV_model)

        return r2.values[0], pseudor2.values[0]

# Sklearn neural net trained on time series split with TimeSeriesCV
def sklearn_neural_net_multilayerperceptron_ts_tscv():

    try:
        filename = "Model_MLP_ts_tscv.sav"
        NN_regr_CV_model = joblib.load(
            "./models/NN_MLP_files/" + str(filename))
    except:
        df, min_max, Y_train, Y_test, X_train, X_test, Y_train_mean, Y_train_meandev, Y_test_meandev = import_train_test_calc_NN_SVR_ts()

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
        NN_regr_CV_model = MLPRegressor(solver= "lbfgs",
                                        max_iter = 10000,
                                        random_state = 0,
                                        hidden_layer_sizes = (50, 25,),
                                        activation= 'relu',
                                        alpha= 0.015
                                        )

        NN_regr_CV_model.fit(X_train, Y_train.values.ravel())

        # store model
        if not os.path.exists("./models/NN_MLP_files"):
            os.makedirs("./models/NN_MLP_files")

        joblib.dump(NN_regr_CV_model,
                    "./models/NN_MLP_files/Model_MLP_ts_tscv.sav")

        r2, pseudor2 = r_squared_metrics_NN_SVR_ts(
            X_train, X_test, Y_train, Y_train_meandev, Y_test, Y_test_meandev, NN_regr_CV_model)

        return r2.values[0], pseudor2.values[0]



