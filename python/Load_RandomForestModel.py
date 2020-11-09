import joblib
import pandas as pd
from data_preprocessing import decompress_pickle
from data_partitioning import train_test_split_ts

# import data
df = decompress_pickle("./data/preprocessed/BikeRental_preprocessed.pbz2")

# Create DataFrame to store all results to make comparisons
df_results_test = df.iloc[round(len(df)*0.8):]

# Drop of the datetime column, because the model cannot handle its type (object)
df = df.drop(['datetime'], axis = 1)

# Create the initial train and test samples
train_size = 0.8
X_train, Y_train, X_test, Y_test = train_test_split_ts(df, train_size)

# load the model from disk
filename = 'Model_RandomForest.sav'
loaded_model = joblib.load(filename)

# Create Prediction
Y_pred = pd.Series(loaded_model.predict(X_test), index = (list(range(round(len(df)*0.8), len(df)))))

# Ddd the predicted count of bike rentals (normalized) to the DataFrame
df_results_test['cnt_pred'] = Y_pred 

# Add columns for the de-normalized count of rental bikes and its de-normalized prediction
df_results_test['cnt_norm'] = df_results_test["cnt"].apply(lambda x: x * (977 - 1) + 1)
df_results_test['cnt_pred_norm'] = df_results_test["cnt_pred"].apply(lambda x: x * (977 - 1) + 1).round(0) 

print(df_results_test)