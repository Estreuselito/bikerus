import joblib
import pandas as pd
from data_preprocessing import decompress_pickle
from data_partitioning import train_test_split_ts

# import data
df = decompress_pickle("./data/preprocessed/BikeRental_preprocessed.pbz2")

# Drop of the datetime column, because the model cannot handle its type (object)
df = df.drop(['datetime'], axis = 1)

# Inputdata for prediction
start = 15500
end = 15511
input = df.iloc[start:end, :]

# Create X, which includes all features except for the target variable and Y, which only includes the target variable
X = input.drop('cnt', axis = 1)
Y = input['cnt']

# load the model from disk
filename = 'Model_RandomForest.sav'
loaded_model = joblib.load(filename)

# Create Prediction
Y_pred = pd.Series(loaded_model.predict(X), index = (list(range(start,end))))
df_comparison = pd.DataFrame(Y)
df_comparison['cnt_pred'] = Y_pred
df_comparison['cnt_norm'] = df_comparison["cnt"].apply(lambda x: x * (977 - 1) + 1)
df_comparison['cnt_pred_norm'] = df_comparison["cnt_pred"].apply(lambda x: x * (977 - 1) + 1).round(0)
print(df_comparison)