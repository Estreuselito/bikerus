# TODO SVR
# TODO from model_creation import sklearn_neural_net_multilayerperceptron
from model_creation import (sklearn_random_forest_ts_tscv,
                            sklearn_random_forest_ts_gridcv,
                            sklearn_random_forest_rs_gridcv,
                            catboost_regressor_rs,
                            catboost_regressor_ts_gridcv,
                            catboost_regressor_ts_tscv,
                            sklearn_support_vector_regression_rs_gridcv,
                            sklearn_support_vector_regression_ts_gridcv,
                            sklearn_support_vector_regression_ts_tscv,
                            sklearn_neural_net_multilayerperceptron_rs_gridcv,
                            sklearn_neural_net_multilayerperceptron_ts_gridcv,
                            sklearn_neural_net_multilayerperceptron_ts_tscv)
import pandas as pd
from logger import logger

# only activate in an environment with fastai running
# from model_creation import fastai_neural_regression

print("  _ \                                                  |        \  |        |         _)              |\n\
 |   |       __|   _` |  |   |   _` |   __|   _ \   _` |       |\/ |   _ \  __|   __|  |   __|   __|  |\n\
 __ <      \__ \  (   |  |   |  (   |  |      __/  (   |       |   |   __/  |    |     |  (    \__ \ _|\n\
_| \_\     ____/ \__. | \__._| \__._| _|    \___| \__._|      _|  _| \___| \__| _|    _| \___| ____/ _)\n\
                     _|\n")

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

# Sklearn support vector regression
logger.info("\nSupport Vector Regression random split and grid search")
r2, pseudor2 = sklearn_support_vector_regression_rs_gridcv()
report = report.append({"Modelname": "SVR random split",
                        "R-Squared": r2,
                        "Pseudo R-Squared": pseudor2},
                       ignore_index=True)


logger.info("\nSupport Vector Regression time series split and grid search")
r2, pseudor2 = sklearn_support_vector_regression_ts_gridcv()
report = report.append({"Modelname": "SVR time series split",
                        "R-Squared": r2,
                        "Pseudo R-Squared": pseudor2},
                       ignore_index=True)

logger.info("\nSupport Vector Regression time series split and randomized CV")
r2, pseudor2 = sklearn_support_vector_regression_ts_tscv()
report = report.append({"Modelname": "SVR time series with times series CV",
                        "R-Squared": r2,
                        "Pseudo R-Squared": pseudor2},
                       ignore_index=True)


# Sklearn neural net
logger.info("\nNeural Net MLP with random split")
r2, pseudor2 = sklearn_neural_net_multilayerperceptron_rs_gridcv()
report = report.append({"Modelname": "Neural Net MLP with random split",
                        "R-Squared": r2,
                        "Pseudo R-Squared": pseudor2},
                       ignore_index=True)

logger.info("\nNeural Net MLP with time series split")
r2, pseudor2 = sklearn_neural_net_multilayerperceptron_ts_gridcv()
report = report.append({"Modelname": "Neural Net MLP time series split",
                        "R-Squared": r2,
                        "Pseudo R-Squared": pseudor2},
                       ignore_index=True)

logger.info("\nNeural Net MLP with time series split and random CV")
r2, pseudor2 = sklearn_neural_net_multilayerperceptron_ts_tscv()
report = report.append({"Modelname": "Neural Net MLP with time series split and time series CV",
                        "R-Squared": r2,
                        "Pseudo R-Squared": pseudor2},
                       ignore_index=True)

# catboost regression
logger.info("\nCatboost regressor random split")
r2, pseudor2 = catboost_regressor_rs()
report = report.append({"Modelname": "Catboost regression rs",
                        "R-Squared": r2,
                        "Pseudo R-Squared": pseudor2},
                       ignore_index=True)

logger.info("\nCatboost regressor time series split and gridCV")
r2, pseudor2 = catboost_regressor_ts_gridcv()
report = report.append({"Modelname": "Catboost regression ts gridcv",
                        "R-Squared": r2,
                        "Pseudo R-Squared": pseudor2},
                       ignore_index=True)

logger.info("\nCatboost regressor time series split and CV")
r2, pseudor2 = catboost_regressor_ts_tscv()
report = report.append({"Modelname": "Catboost regression ts cv",
                        "R-Squared": r2,
                        "Pseudo R-Squared": pseudor2},
                       ignore_index=True)


# # Sklearn random forest
logger.info("\nsklearn RandomForest with time series split")
r2, pseudor2 = sklearn_random_forest_ts_tscv()
report = report.append({"Modelname": "Sklearn RF time series split",
                        "R-Squared": r2,
                        "Pseudo R-Squared": pseudor2},
                       ignore_index=True)

logger.info("\nsklearn RandomForest with time series split and random CV")
r2, pseudor2 = sklearn_random_forest_ts_gridcv()
report = report.append({"Modelname": "Sklearn RF with time series split and random CV",
                        "R-Squared": r2,
                        "Pseudo R-Squared": pseudor2},
                       ignore_index=True)

logger.info("\nsklearn RandomForest with random split")
r2, pseudor2 = sklearn_random_forest_rs_gridcv()
report = report.append({"Modelname": "Sklearn RF with random split",
                        "R-Squared": r2,
                        "Pseudo R-Squared": pseudor2},
                       ignore_index=True)

# print report
print(" __ \                      |\n\
 |   |  _ \   __ \    _ \  |\n\
 |   | (   |  |   |   __/ _|\n\
____/ \___/  _|  _| \___| _)\n\
This is the final report: \n", report)
