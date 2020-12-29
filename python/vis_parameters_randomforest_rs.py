import pandas as pd
import numpy as np
from data_partitioning import train_test_split_ts, get_sample_for_cv
from model_helpers import (import_train_test_calc,
                           r_squared_metrics)
from logger import logger
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pickle
import joblib
import matplotlib.pyplot as plt

# Create initial train_test_split
df, min_max, Y_train, Y_test, X_train, X_test, Y_train_mean, Y_train_meandev, Y_test_meandev = import_train_test_calc(rs="_rs")

# Split Trainig set once more for an initial visualization of the impact of the hyperparameters to avoid overfitting
X_train_2, X_test_2, Y_train_2, Y_test_2 = train_test_split(X_train, Y_train, test_size = 0.2, random_state=0)

# to evaluate the prediction quality, we will use the R2 measure
# as a benchmark, we initially calculated the mean value and the residual sum of squares of the target variable
Y_train_2_mean = Y_train_2.values.ravel().mean()
print("Y_train_2_mean =", Y_train_2_mean)
Y_train_2_meandev = ((Y_train_2.values.ravel()-Y_train_2_mean)**2).sum()
print("Y_train_2_meandev =", Y_train_2_meandev)
Y_test_2_meandev = ((Y_test_2.values.ravel()-Y_train_2_mean)**2).sum()
print("Y_test_meandev =", Y_test_2_meandev)

# Visualization n_estimators
# R2_report = np.zeros((2,31), float)
# ntrees = [5]
# extension = (np.arange(30)+1) * 20
# ntrees = np.append([ntrees], [extension])
# for k in range(0, 31):
#     # Initialize model
#     RForreg = RandomForestRegressor(n_estimators=ntrees[k], 
#                                     random_state=0)
#     RForreg.fit(X_train_2, Y_train_2.values.ravel())
    
#     # training: R2
#     Y_train_2_pred = RForreg.predict(X_train_2)
#     Y_train_2_dev = ((Y_train_2.values.ravel()-Y_train_2_pred)**2).sum()
#     r2 = 1 - Y_train_2_dev/Y_train_2_meandev
#     R2_report[0,k] = r2
    
#     # testing: Pseudo-R2
#     Y_test_2_pred = RForreg.predict(X_test_2)
#     Y_test_2_dev = ((Y_test_2.values.ravel()-Y_test_2_pred)**2).sum()
#     pseudor2 = 1 - Y_test_2_dev/Y_test_2_meandev
#     R2_report[1,k] = pseudor2

# # plot
# plt.plot(ntrees, R2_report[0,:], label = 'R2')
# plt.plot(ntrees, R2_report[1,:], label = 'Pseudo-R2')
# plt.xticks(ntrees, rotation=90)
# plt.xlabel('n_estimators')
# plt.ylabel('R2 / Pseudo-R2')
# plt.title('Random Forest - Random Sampling Approach - n_estimators')
# plt.legend()
# plt.show()
# plt.savefig("./images/n_estimators_rs", dpi=300)


# max_depth
R2_report = np.zeros((2, 30), float)
max_depth = np.arange(1, 31)
for k in range(0, 30):
    # Initialize model
    RForreg = RandomForestRegressor(max_depth = max_depth[k], 
                                    random_state=0)
    RForreg.fit(X_train_2, Y_train_2.values.ravel())
    
    # training: R2
    Y_train_2_pred = RForreg.predict(X_train_2)
    Y_train_2_dev = ((Y_train_2.values.ravel()-Y_train_2_pred)**2).sum()
    r2 = 1 - Y_train_2_dev/Y_train_2_meandev
    R2_report[0,k] = r2
    
    # testing: Pseudo-R2
    Y_test_2_pred = RForreg.predict(X_test_2)
    Y_test_2_dev = ((Y_test_2.values.ravel()-Y_test_2_pred)**2).sum()
    pseudor2 = 1 - Y_test_2_dev/Y_test_2_meandev
    R2_report[1,k] = pseudor2

# plot
plt.plot(max_depth, R2_report[0,:], label = 'R2')
plt.plot(max_depth, R2_report[1,:], label = 'Pseudo-R2')
plt.xticks(max_depth, rotation=90)
plt.xlabel('max_depth')
plt.ylabel('R2 / Pseudo-R2')
plt.title('Random Forest - Random Sampling Approach - max_depth')
plt.show()
plt.legend
plt.savefig("./images/max_depth_rs", dpi=300)


# max_leaf_nodes
# R2_report = np.zeros((2,20), float)
# max_leaf_nodes = (np.arange(20)+1)*40
# for k in range(0, 20):
#     # Initialize model
#     RForreg = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes[k], 
#                                     random_state=0)
#     RForreg.fit(X_train_2, Y_train_2.values.ravel())
    
#     # training: R2
#     Y_train_2_pred = RForreg.predict(X_train_2)
#     Y_train_2_dev = ((Y_train_2.values.ravel()-Y_train_2_pred)**2).sum()
#     r2 = 1 - Y_train_2_dev/Y_train_2_meandev
#     R2_report[0,k] = r2
    
#     # testing: Pseudo-R2
#     Y_test_2_pred = RForreg.predict(X_test_2)
#     Y_test_2_dev = ((Y_test_2.values.ravel()-Y_test_2_pred)**2).sum()
#     pseudor2 = 1 - Y_test_2_dev/Y_test_2_meandev
#     R2_report[1,k] = pseudor2

# # plot
# plt.plot(max_leaf_nodes, R2_report[0,:], label = 'R2')
# plt.plot(max_leaf_nodes, R2_report[1,:], label = 'Pseudo-R2')
# plt.xticks(max_leaf_nodes, rotation=90)
# plt.xlabel('max_leaf_nodes')
# plt.ylabel('R2 / Pseudo-R2')
# plt.title('Random Forest - Random Sampling Approach - max_leaf_nodes')
# plt.show()
# plt.legend()
# plt.savefig("./images/max_leaf_nodes_rs", dpi=300)