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

df, min_max, Y_train, Y_test, X_train, X_test, Y_train_mean, Y_train_meandev, Y_test_meandev = import_train_test_calc()

df_parameters = pd.DataFrame()
folds = list(range(1, 6))
depths= [4, 6, 8, 10, 11]
learning_rates= [0.01, 0.05, 0.1, 0.2, 0.3]
iterations= [30, 50, 100, 200, 400, 600, 800, 1000, 1200]

df_parameters.to_excel("./images", sheet_name = "parameters.xlsx")

for depth in list(range(len(depths))):
    for learning_rate in list(range(len(learning_rates))):
        for iteration in list(range(len(iterations))):
            for fold in list(range(len(folds))): # important: fold needs to be the last for-loop to be able to compute the means of Pseudo R^2 across the folds
                    
                X_train_cv, Y_train_cv, X_test_cv, Y_test_cv = get_sample_for_cv(folds[-1], # specify the number of total folds, last index of the list
                                                                                    folds[fold], # specifiy the current fold
                                                                                    X_train, # DataFrame X_train, which was created with the function train_test_split_ts
                                                                                    Y_train) # DataFrame Y_train, which was created with the function train_test_split_ts
                        
                                # to evaluate the prediction quality, we use the R2 measure
                                # as a benchmark, we initially calculated the mean value and the residual sum of squares of the target variable for the specific fold
                Y_train_mean_cv = Y_train_cv.mean()
                Y_train_cv_sum = Y_train_cv.sum()
                Y_test_cv_sum = Y_test_cv.sum()
                Y_train_meandev_cv = sum((Y_train_cv_sum - Y_train_mean_cv)**2)
                Y_test_meandev_cv = sum((Y_test_cv_sum - Y_train_mean_cv)**2)
                        
                                # initialize model

                X_train_cv = X_train_cv.drop("yr", axis=1)
                X_test_cv = X_test_cv.drop("yr", axis=1)
                

                cat_var = ["season", "mnth", "hr", "holiday",
                                    "weekday", "workingday", "weathersit", "rush_hour"]
                for v in cat_var:
                    X_train_cv[v] = X_train_cv[v].astype("int64")
                    X_test_cv[v] = X_test_cv[v].astype("int64")

                model = CatBoostRegressor(loss_function='RMSE', depth=depths[depth], learning_rate= learning_rates[learning_rate], iterations= iterations[iteration], od_type='Iter', od_wait=10)
                
                                # train the model
                model.fit(
                            X_train_cv, Y_train_cv,
                            use_best_model=True,
                            cat_features=["season", "mnth", "hr",
                                "holiday", "weekday", "workingday", "weathersit", "rush_hour"],
                            eval_set=(X_test_cv, Y_test_cv),
                            verbose=True,
                            plot=True
                        )

                                # Make predictions based on the traing set
                Y_train_pred_cv = model.predict(X_train_cv)
                Y_train_pred_cv_sum = Y_train_pred_cv.sum()
                Y_train_dev_cv = sum((Y_train_cv_sum-Y_train_pred_cv_sum)**2)
                r2_cv = 1 - Y_train_dev_cv/Y_train_meandev_cv

                                # Evaluate the result by applying the model to the test set
                Y_test_pred_cv = model.predict(X_test_cv)
                Y_test_pred_cv_sum = Y_test_pred_cv.sum()
                Y_test_dev_cv = sum((Y_test_cv_sum - Y_test_pred_cv_sum)**2)
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
                    if df_parameters.iloc[i:i+n_folds, 1].mean() > mean_max:
                        mean_max = df_parameters.iloc[i:i+n_folds, 1].mean()
                        index = i
                        i += n_folds
                    else:
                        i += n_folds
                df_parameters = df_parameters.append(new_row, ignore_index=True)
                            
                                # best parameters based on mean of PseudoR^2
                best_parameters = pd.Series(df_parameters.iloc[index, 3:]) # only the hyperparameters are included here, therefore the index starts at 3
df_parameters.to_excel("./images", "parameters")
print(df_parameters)
print(mean_max)
print(best_parameters)
    #r2, pseudor2 = r_squared_metrics(X_train, X_test, Y_train, Y_train_meandev, Y_test, Y_test_meandev, random_forest)

    #return r2.values[0], pseudor2.values[0]