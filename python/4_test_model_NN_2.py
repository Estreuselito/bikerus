from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from model_helpers import import_train_test_calc_NN_SVR_ts, r_squared_metrics_NN_SVR_ts, import_train_test_calc_NN_SVR_rs, r_squared_metrics_NN_SVR_rs

# Sklearn neural net trained on random split with Grid/RandomizedCV
def sklearn_neural_net_multilayerperceptron_ts_gridcv():

    df, min_max, Y_train, Y_test, X_train, X_test, Y_train_mean, Y_train_meandev, Y_test_meandev = import_train_test_calc_NN_SVR_ts()

    ### MODEL CREATION ###

    # initialize MLPRegressor (lbfgs solver used due to its efficiency)
    NN_regr_CV = MLPRegressor(
        solver='lbfgs', max_iter=10000, random_state=0)

    ### HYPERPARAMETER OPTIMIZATION ###

    # 1st RandomizedSearchCV parameters:
        # param_grid = {
        #     "hidden_layer_sizes": [(10,), (25,), (50,), (10, 10,), (25, 10,), (10, 25,), (25, 25,), (50, 50,)],
        #     "activation": ["logistic", "tanh", "relu"],
        #     "alpha": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
        # }
        # best parameters: {...}

    # 2nd RandomizedSearchCV parameters:
        # param_grid = {
        #     "hidden_layer_sizes": [...],
        #     "activation": [...],
        #     "alpha": [...],
        # }
        # best parameters: {...}

    # 3rd RandomizedSearchCV parameters:
        # param_grid = {
        #     "hidden_layer_sizes": [...],
        #     "activation": ["...],
        #     "alpha": [...],
        # }
        # best parameters: {...}

    param_grid = {
        # tuple's ith element represents the number of neurons in the ith hidden layer. (5,) = 1 hidden layer with 5 neurons.
        "hidden_layer_sizes": [(10,), (25,), (50,), (10, 10,), (25, 10,), (10, 25,), (25, 25,), (50, 50,)],
        # left out identity activation function due to its linearity
        "activation": ["logistic", "tanh", "relu"],
        # L2 penalty parameter
        "alpha": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
        # learning_rate is kept at default (constant) since lbfgs solver does not use a learning rate
    }
    NN_regr_CV_model = RandomizedSearchCV(
        estimator=NN_regr_CV, param_distributions=param_grid, cv=5)

    NN_regr_CV_model.fit(X_train, Y_train.values.ravel())

    print(NN_regr_CV_model.best_params_)

sklearn_neural_net_multilayerperceptron_ts_gridcv()