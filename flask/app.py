import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from catboost import CatBoostRegressor
from datetime import datetime
import pandas as pd

app = Flask(__name__)

#### load train dataset
train_dataset = pd.read_csv("./data/preprocessed/BikeRental_preprocessed.csv", index_col=[0], parse_dates=["datetime"])
train_dataset = train_dataset[round(len(train_dataset)*0.8):]
#train_dataset["day"] = train_dataset["datetime"].day
min_max = pd.read_csv("./data/preprocessed/cnt_min_max.csv", index_col=[0])
train_dataset["cnt_norm"] = train_dataset["cnt"].apply(
        lambda x: x * (min_max["max"][0] - min_max["min"][0]) + min_max["min"][0])


test_df = train_dataset[round(len(train_dataset)*0.8):].copy()



#### load catboost model
catboost = CatBoostRegressor(loss_function='RMSE', depth=6,
                                  learning_rate=0.1, iterations=1000, od_type='Iter', od_wait=10)
catboost.load_model("./models/catboost/catboost_model")

neural_net = pickle.load(open("./models/NN_MLP_files/NN_MLP_saved", "rb"))

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
    nn_pred_val = neural_net.predict(nn_pred)



    test_df = train_dataset[(train_dataset.datetime == date)]
    test_df = test_df.drop('datetime', axis=1)
    cat_var = ["season", "yr", "mnth", "hr", "holiday",
                "weekday", "workingday", "weathersit"]

    for v in cat_var:
        test_df[v] = test_df[v].astype("int64")

    catboost_pred = catboost.predict(test_df)
    catboost_pred = catboost_pred.item()
    real_cnt = train_dataset[(train_dataset.datetime == date)]["cnt_norm"]
    normalizer = lambda x: x * (min_max["max"][0] - min_max["min"][0]) + min_max["min"][0]
    vfunc = np.vectorize(normalizer)
    #  int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)

    # output = round(prediction[0], 2)

    # return str(datetime.strptime(request.form['date'], "%Y-%m-%d").date())
    return render_template('index.html',
     date = date,
     prediction=round(real_cnt.item()),
     Catboost = int(vfunc(catboost_pred).round()),
     Neural_net = int(vfunc(nn_pred_val).round().item()))


""" app.route('/results', methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output) """


if __name__ == "__main__":
    app.run(debug=True)
