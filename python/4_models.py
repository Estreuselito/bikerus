from model_creation import catboost_regressor
from model_creation import fastai_neural_regression

# creating a report
report = pd.DataFrame(columns=['Modelname', 'R-Squared', 'Pseudo R-Squared'])
