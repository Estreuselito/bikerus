from data_storage import connection
from model_helpers import predict_test_df
from catboost import CatBoostRegressor
import pandas as pd
import pickle
import joblib
import os
from logger import logger

print("  _ \                  | _)        |   _)                |\n\
 |   |  __|   _ \   _` |  |   __|  __|  |  __ \    _  |  |\n\
 ___/  |      __/  (   |  |  (     |    |  |   |  (   | _|\n\
_|    _|    \___| \__._| _| \___| \__| _| _|  _| \__. | _)\n\
                                                 |___/\n")

try:
    logger.info("\nLoading Catboost models\n")
    catboost_rs = CatBoostRegressor(loss_function='RMSE', depth=6,
                                    learning_rate=0.1, iterations=1000, od_type='Iter', od_wait=10)
    catboost_rs.load_model("./models/catboost/catboost_model_rs")

    catboost_ts_gridcv = CatBoostRegressor(loss_function='RMSE', depth=10,
                                           learning_rate=0.05, iterations=1000, od_type='Iter', od_wait=10)
    catboost_ts_gridcv.load_model("./models/catboost/catboost_model_ts_gridcv")

    catboost_model_ts_tscv = CatBoostRegressor(loss_function='RMSE', depth=6,
                                               learning_rate=0.2, iterations=200, od_type='Iter', od_wait=10)
    catboost_model_ts_tscv.load_model(
        "./models/catboost/catboost_model_ts_tscv")

    logger.info("\nLoading Random Forest models\n")
    filename = 'Model_RandomForest_ts_tscv.sav'
    random_forest_ts_tscv = joblib.load(
        "./models/RandomForest_Model/" + str(filename))

    filename = 'Model_RandomForest_rs_gridcv.sav'
    random_forest_rs_gridcv = joblib.load(
        "./models/RandomForest_Model/" + str(filename))

    filename = 'Model_RandomForest_ts_gridcv.sav'
    random_forest_model_ts_gridcv = joblib.load(
        "./models/RandomForest_Model/" + str(filename))

    logger.info("\nLoading SVR models\n")
    SVR_regr_CV_model_rs = joblib.load(
        "./models/SVR_files/Model_SVR_rs_gridcv.sav")
    SVR_regr_CV_model_ts = joblib.load(
        "./models/SVR_files/Model_SVR_ts_gridcv.sav")
    SVR_regr_CV_model_ts_tscv = joblib.load(
        "./models/SVR_files/Model_SVR_ts_tscv.sav")

    logger.info("\nLoading Neural Nets\n")
    NN_regr_CV_model_rs_gridcv = joblib.load(
        "./models/NN_MLP_files/Model_MLP_rs_gridcv.sav")
    NN_regr_CV_model_ts_gridcv = joblib.load(
        "./models/NN_MLP_files/Model_MLP_ts_gridcv.sav")
    NN_regr_CV_model_ts_tscv = joblib.load(
        "./models/NN_MLP_files/Model_MLP_ts_tscv.sav")
except:
    logger.error(
        "Something did not work! Could not load models! Execute script 4 again!")

# print("NN_regr_CV_model_ts_tscv: ", NN_regr_CV_model_ts_tscv.__module__)

# print(neural_net)
# predict_test_df(NN_regr_CV_model_ts_tscv)

predict_test_df(random_forest_ts_tscv, random_forest_rs_gridcv, random_forest_model_ts_gridcv, SVR_regr_CV_model_rs, SVR_regr_CV_model_ts, SVR_regr_CV_model_ts_tscv,
                NN_regr_CV_model_rs_gridcv, NN_regr_CV_model_ts_gridcv, NN_regr_CV_model_ts_tscv, catboost_rs, catboost_ts_gridcv, catboost_model_ts_tscv).to_sql(
    "predicted_df", connection, if_exists="replace", index=False)

print(" __ \                      |\n\
 |   |  _ \   __ \    _ \  |\n\
 |   | (   |  |   |   __/ _|\n\
____/ \___/  _|  _| \___| _)\n\
You can find the final dataframe with all the normalized and denormalized values under the table predicted_df in the database!")
