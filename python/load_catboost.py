import os
from catboost import CatBoostRegressor
import pandas as pd

model = CatBoostRegressor()

os.chdir("./catboost")
model.load_model("catboost_model") 

#create dataframe
lst = [[1, 2, 1, 0, 0, 1, 0, 1, 0.255, 0.6, 0.09]]

test = pd.DataFrame(lst, columns=['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'hum', 'windspeed'])


cat_var = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
for v in cat_var:
    test[v] = test[v].astype("int64")

print(model.predict(test))