import os
from catboost import CatBoostRegressor
import pandas as pd

model = CatBoostRegressor()
model.load_model("model") 

#create dataframe
lst = [[1, 2, 1, 0, 0, 1, 0, 1, 0.255, 0.6, 0.09]]
lst3 = [[4, 2, 12, 16, 0, 1, 0, 1, 0.5, 0.1, 0.09]]
test = pd.DataFrame(lst, columns=['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'hum', 'windspeed'])


cat_var = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
for v in cat_var:
    test[v] = test[v].astype("int64")

print(model.predict(test))