from model_creation import catboost_regressor
#from model_creation import fastai_neural_regression
import pandas as pd


# creating a report
report = pd.DataFrame(columns=['Modelname', 'R-Squared', 'Pseudo R-Squared'])
