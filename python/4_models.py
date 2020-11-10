from model_creation import catboost_regressor
from model_creation import sklearn_neural_net_multilayerperceptron
import pandas as pd

# only activate in an environment with fastai running
# from model_creation import fastai_neural_regression


# creating a report
report = pd.DataFrame(columns=['Modelname', 'R-Squared', 'Pseudo R-Squared'])

# Fastais neural net
# Only uncomment in an enviorment where fastai is running
# print("\nFastais neural net\n")
# r2, pseudor2 = fastai_neural_regression()
# report = report.append({"Modelname": "fastai neural regression",
#                         "R-Squared": r2,
#                         "Pseudo R-Squared": pseudor2},
#                         ignore_index=True)

# catboost regression
print("\nCatboost regressor\n")
r2, pseudor2 = catboost_regressor()
report = report.append({"Modelname": "Catboost regression",
                        "R-Squared": r2,
                        "Pseudo R-Squared": pseudor2},
                        ignore_index=True)

# Sklearn neural net
print("\nsklearn nn MLP\n")
r2, pseudor2 = sklearn_neural_net_multilayerperceptron()
report = report.append({"Modelname": "Sklearn NN MLP",
                        "R-Squared": r2,
                        "Pseudo R-Squared": pseudor2},
                        ignore_index=True)

# print report
print(report)
