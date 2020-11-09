import numpy as np
import pandas as pd
import joblib
from data_preprocessing import decompress_pickle
from data_partitioning import train_test_split_ts
from data_partitioning import get_sample_for_cv
from sklearn.ensemble import RandomForestRegressor

# import data
df = decompress_pickle("./data/preprocessed/BikeRental_preprocessed.pbz2")

# Drop of the datetime column, because the model cannot handle its type (object)
df = df.drop(['datetime'], axis = 1)

# Create the initial train and test samples
train_size = 0.8
X_train, Y_train, X_test, Y_test = train_test_split_ts(df, train_size)

# Training the model incl. Cross Validation
df_parameters = pd.DataFrame()
folds = list(range(1, 6))
max_depth = [11]
n_estimators = [250]
max_features = [9]
min_samples_leaf = [1]
max_leaf_nodes = [80]
for depth in list(range(len(max_depth))):
    for number_trees in list(range(len(n_estimators))):
        for feature in list(range(len(max_features))):
            for leaf in list(range(len(min_samples_leaf))):
                for node in list(range(len(max_leaf_nodes))):
                    for fold in list(range(len(folds))): # important fold as the last category so that the dataframe is ordered to compute the means
            
                        X_train_cv, Y_train_cv, X_test_cv, Y_test_cv = get_sample_for_cv(folds[-1], # specify the number of total folds, last index of the list
                                                                                        folds[fold], # specifiy the current fold
                                                                                        X_train, # DataFrame X_train, which was created with the function train_test_split_ts
                                                                                        Y_train) # DataFrame Y_train, which was created with the function train_test_split_ts
                
                        # to evaluate the prediction quality, we will use the R2 measure
                        # as a benchmark, we initially calculated the mean value and the residual sum of squares of the target variable for the specific fold
                        Y_train_mean_cv = Y_train_cv.mean()
                        Y_train_meandev_cv = sum((Y_train_cv-Y_train_mean_cv)**2)
                        Y_test_meandev_cv = sum((Y_test_cv-Y_train_mean_cv)**2)
                
                        # initialize model
                        RForreg = RandomForestRegressor(max_depth=max_depth[depth], 
                                                        n_estimators=n_estimators[number_trees], 
                                                        max_features=max_features[feature],
                                                        min_samples_leaf=min_samples_leaf[leaf],
                                                        max_leaf_nodes=max_leaf_nodes[node],
                                                        random_state=0)
        
                        # train the model
                        RForreg.fit(X_train_cv, Y_train_cv)

                        # Make predictions based on the traing set
                        Y_train_pred_cv = RForreg.predict(X_train_cv)
                        Y_train_dev_cv = sum((Y_train_cv-Y_train_pred_cv)**2)
                        r2_cv = 1 - Y_train_dev_cv/Y_train_meandev_cv

                        # Evaluate the result by applying the model to the test set
                        Y_test_pred_cv = RForreg.predict(X_test_cv)
                        Y_test_dev_cv = sum((Y_test_cv-Y_test_pred_cv)**2)
                        pseudor2_cv = 1 - Y_test_dev_cv/Y_test_meandev_cv
        
                        # Append results to dataframe
                        new_row = {'fold': folds[fold],
                                    'max_depth': max_depth[depth], 
                                    'n_estimators': n_estimators[number_trees],
                                    'max_features': max_features[feature],
                                    'min_samples_leaf': min_samples_leaf[leaf],
                                    'max_leaf_nodes': max_leaf_nodes[node],
                                    'R2': r2_cv, 
                                    'PseudoR2': pseudor2_cv,
                                    'Diff R2-PseudoRS': abs(r2_cv - pseudor2_cv)}

                        # Calculate means to find the best hyperparameters across all folds
                        n_folds = folds[-1]
                        i = 0
                        index = 0
                        mean_max = 0
                        while i < len(df_parameters):
                            if df_parameters.iloc[i:i+n_folds, 1].mean() > mean_max:
                                mean_max = df_parameters.iloc[i:i+n_folds, 1].mean()
                                index = i
                                i += n_folds
                            else:
                                i += n_folds
                        df_parameters = df_parameters.append(new_row, ignore_index=True)
                    
                        # best parameters based on mean of PseudoR^2
                        best_parameters = pd.Series(df_parameters.iloc[index, 4:]) # only the hyperparameters are included here, therefore the index starts at 4

print(mean_max, best_parameters)                    
# print(df_parameters)

# Apply the model with the found hyperparameters to the test set
# to evaluate the prediction quality, we will use the R2 measure
# as a benchmark, we initially calculated the mean value and the residual sum of squares of the target variable
Y_train_mean = Y_train.mean()
Y_train_meandev = sum((Y_train-Y_train_mean)**2)
Y_test_meandev = sum((Y_test-Y_train_mean)**2)

# Initialize the model and the regressor with the best hyperparameters
RForreg = RandomForestRegressor(max_depth=int(best_parameters['max_depth']), 
                                n_estimators=int(best_parameters['n_estimators']), 
                                max_features=int(best_parameters['max_features']),
                                min_samples_leaf = int(best_parameters['min_samples_leaf']),
                                max_leaf_nodes=int(best_parameters['max_leaf_nodes']),
                                random_state=0)

# train the model with the hyperparameters
RForreg.fit(X_train, Y_train)
Y_train_pred = RForreg.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)

# Evaluate the result by applying the model to the test set
Y_test_pred = RForreg.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)

# Save the model
filename = 'Model_RandomForest_bad.sav'
joblib.dump(RForreg, filename)