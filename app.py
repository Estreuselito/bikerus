import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
from catboost import CatBoostRegressor
from datetime import datetime
import pandas as pd
import sys  # nopep8
sys.path.insert(1, './python')  # nopep8
from data_storage import connection


app = Flask(__name__)

# load train dataset
train_dataset = pd.read_sql_query(
    '''SELECT * FROM hours_preprocessed''', connection)
train_dataset["datetime"] = pd.to_datetime(train_dataset["datetime"])
min_max = pd.read_sql_query('''SELECT * FROM max_min_count''', connection)
train_dataset = train_dataset[round(len(train_dataset)*0.8):]
train_dataset["cnt_norm"] = train_dataset["cnt"].apply(
    lambda x: x * (min_max["max"][0] - min_max["min"][0]) + min_max["min"][0])
test_df = train_dataset[round(len(train_dataset)*0.8):].copy()

# load catboost model
catboost = CatBoostRegressor(loss_function='RMSE', depth=6,
                             learning_rate=0.1, iterations=1000, od_type='Iter', od_wait=10)
catboost.load_model("./models/catboost/catboost_model")

neural_net = joblib.load(
    "./models/NN_MLP_files/Model_MLP_rs_gridcv.sav")

random_forest = joblib.load(
    "./models/RandomForest_Model/Model_RandomForest.sav")
# model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    date = datetime.strptime(request.form['date'], "%Y-%m-%d")
    hour = int(request.form["hour"])
    date = date.replace(hour=hour)

    nn_pred = train_dataset[(train_dataset.datetime == date)]
    #nn_pred = nn_pred.drop('datetime', axis=1)
    #y = test_df["cnt"]
    nn_pred = nn_pred.drop(['cnt', "datetime", "cnt_norm"], axis=1)
    # nn_pred_val = neural_net.predict(nn_pred)

    rf_pred = random_forest.predict(nn_pred)

    test_df = train_dataset[(train_dataset.datetime == date)]
    test_df = test_df.drop('datetime', axis=1)
    cat_var = ["season", "yr", "mnth", "hr", "holiday",
               "weekday", "workingday", "weathersit"]

    for v in cat_var:
        test_df[v] = test_df[v].astype("int64")

    catboost_pred = catboost.predict(test_df)
    catboost_pred = catboost_pred.item()
    real_cnt = train_dataset[(train_dataset.datetime == date)]["cnt_norm"]

    def normalizer(x): return x * \
        (min_max["max"][0] - min_max["min"][0]) + min_max["min"][0]
    vfunc = np.vectorize(normalizer)
    #  int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)

    # output = round(prediction[0], 2)

    # return str(datetime.strptime(request.form['date'], "%Y-%m-%d").date())
    return render_template('index.html',
                           date=date,
                           prediction=round(real_cnt.item()),
                           Catboost=int(vfunc(catboost_pred).round()),
                           #    Neural_net=int(vfunc(nn_pred_val).round().item()),
                           random_forest=int(vfunc(rf_pred).round().item()))


if __name__ == "__main__":
    app.run(debug=True)
