from model_helpers import predict_test_df
from catboost import CatBoostRegressor
import pickle
import joblib
import os

print("  _ \                  | _)        |   _)                |\n\
 |   |  __|   _ \   _` |  |   __|  __|  |  __ \    _  |  |\n\
 ___/  |      __/  (   |  |  (     |    |  |   |  (   | _|\n\
_|    _|    \___| \__._| _| \___| \__| _| _|  _| \__. | _)\n\
                                                 |___/\n")

try:
    catboost = CatBoostRegressor(loss_function='RMSE', depth=6,
                                 learning_rate=0.1, iterations=1000, od_type='Iter', od_wait=10)
    catboost.load_model("./models/catboost/catboost_model")
    neural_net = pickle.load(open("./models/NN_MLP_files/NN_MLP_saved", "rb"))
    random_forest = joblib.load("./models/RandomForest_Model/Model_RandomForest.sav")
except:
    print("Something did not work! Could not load models! Execute script 4 again!")

final_df = predict_test_df(neural_net, random_forest, catboost)

if not os.path.exists("./data/predictions"):
    os.makedirs("./data/predictions")

final_df.to_csv("./data/predictions/final_df.csv")

print(" __ \                      |\n\
 |   |  _ \   __ \    _ \  |\n\
 |   | (   |  |   |   __/ _|\n\
____/ \___/  _|  _| \___| _)\n\
You can find the final dataframe with all the normalized and denormalized values under data/predictions/final_df.csv!")
