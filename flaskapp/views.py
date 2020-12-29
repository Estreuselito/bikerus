import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
from catboost import CatBoostRegressor
from datetime import datetime
import pandas as pd
import sys  # nopep8
sys.path.insert(1, './python')  # nopep8
from data_storage import connection, create_connection
from flaskapp import app
from model_helpers import predict_test_df
from logger import logger


# app = Flask(__name__)

# load train dataset
train_dataset = pd.read_sql_query(
    '''SELECT * FROM hours_preprocessed''', connection)
train_dataset["datetime"] = pd.to_datetime(train_dataset["datetime"])
min_max = pd.read_sql_query('''SELECT * FROM max_min_count''', connection)
train_dataset = train_dataset[round(len(train_dataset)*0.8):]
train_dataset["cnt_norm"] = train_dataset["cnt"].apply(
    lambda x: x * (min_max["max"][0] - min_max["min"][0]) + min_max["min"][0])
test_df = train_dataset[round(len(train_dataset)*0.8):].copy()

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


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    connection = create_connection("./database/BikeRental.db")
    date = datetime.strptime(request.form['date'], "%Y-%m-%d")
    hour = int(request.form["hour"])
    date = date.replace(hour=hour)

    # test = predict_test_df(random_forest, webapp=True, filter_=str(date))

    # nn_pred = train_dataset[(train_dataset.datetime == date)]
    # #nn_pred = nn_pred.drop('datetime', axis=1)
    # #y = test_df["cnt"]
    # nn_pred = nn_pred.drop(['cnt', "datetime", "cnt_norm"], axis=1)
    # # nn_pred_val = neural_net.predict(nn_pred)

    # rf_pred = random_forest.predict(nn_pred)

    # test_df = train_dataset[(train_dataset.datetime == date)]
    # test_df = test_df.drop('datetime', axis=1)
    # cat_var = ["season", "yr", "mnth", "hr", "holiday",
    #            "weekday", "workingday", "weathersit"]

    # for v in cat_var:
    #     test_df[v] = test_df[v].astype("int64")

    # catboost_pred = catboost.predict(test_df)
    # catboost_pred = catboost_pred.item()
    test_df = pd.read_sql_query(
        '''SELECT * FROM hours_preprocessed''', connection)
    min_max = pd.read_sql_query('''SELECT * FROM max_min_count''', connection)
    vfunc = np.vectorize(lambda x: round(
        x * (min_max["max"][0] - min_max["min"][0]) + min_max["min"][0]))
    real_cnt = vfunc(test_df[(test_df.datetime == str(date))]["cnt"])[0]

    # def normalizer(x): return x * \
    #     (min_max["max"][0] - min_max["min"][0]) + min_max["min"][0]
    # vfunc = np.vectorize(normalizer)
    # #  int_features = [int(x) for x in request.form.values()]
    # # final_features = [np.array(int_features)]
    # # prediction = model.predict(final_features)

    # # output = round(prediction[0], 2)

    # # return str(datetime.strptime(request.form['date'], "%Y-%m-%d").date())
    return render_template('index.html',
                           date=date,
                           prediction=real_cnt,
                           catboost_rs=predict_test_df(
                               catboost_rs, webapp=True, filter_=str(date))[0],
                           catboost_ts_gridcv=predict_test_df(
                               catboost_ts_gridcv, webapp=True, filter_=str(date))[0],
                           catboost_model_ts_tscv=predict_test_df(
                               catboost_model_ts_tscv, webapp=True, filter_=str(date))[0],
                           random_forest_rs_gridcv=predict_test_df(
                               random_forest_rs_gridcv, webapp=True, filter_=str(date))[0],
                           random_forest_model_ts_gridcv=predict_test_df(
                               random_forest_model_ts_gridcv, webapp=True, filter_=str(date))[0],
                           random_forest_ts_tscv=predict_test_df(
                               random_forest_ts_tscv, webapp=True, filter_=str(date))[0],
                           NN_regr_CV_model_rs_gridcv=predict_test_df(
                               NN_regr_CV_model_rs_gridcv, webapp=True, filter_=str(date))[0],
                           NN_regr_CV_model_ts_gridcv=predict_test_df(
                               NN_regr_CV_model_ts_gridcv, webapp=True, filter_=str(date))[0],
                           NN_regr_CV_model_ts_tscv=predict_test_df(
                               NN_regr_CV_model_ts_tscv, webapp=True, filter_=str(date))[0],
                           SVR_regr_CV_model_rs=predict_test_df(
                               SVR_regr_CV_model_rs, webapp=True, filter_=str(date))[0],
                           SVR_regr_CV_model_ts=predict_test_df(
                               SVR_regr_CV_model_ts, webapp=True, filter_=str(date))[0],
                           SVR_regr_CV_model_ts_tscv=predict_test_df(SVR_regr_CV_model_ts_tscv, webapp=True, filter_=str(date))[0])


# if __name__ == "__main__":
#     app.run(debug=True)
