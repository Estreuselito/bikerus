import os
from catboost import CatBoostRegressor

model = CatBoostRegressor
model.load_model("C:/Users/janfa/OneDrive/Dokumente/git_repos/bikerus/catboost_model/model.cpm") 
