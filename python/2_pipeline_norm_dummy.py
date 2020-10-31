from data_preprocessing import decompress_pickle, compressed_pickle
import pandas as pd
from sklearn import preprocessing
import os

print("Loading data")

# load data
df = decompress_pickle("./data/preprocessed/BikeRental_complete.pbz2")


# drop leakage variables
leak_var = ["casual", "registered"]
df = df.drop(leak_var, axis=1)


# drop highly correlated variables
high_corr_var = ["atemp"]
df = df.drop(high_corr_var, axis=1)


# drop redundant dteday variable
red_var = ["dteday"]
df = df.drop(red_var, axis=1)


# coerce correct data types for categorical data
cat_var = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
for v in cat_var:
    df[v] = df[v].astype("category")


# normaliza continous variables
cont_var = ["temp", "windspeed", "cnt"]
# store true min & max to be able to revert normalization
count_var = ["cnt"]
max_count = pd.DataFrame(df[count_var].max())
min_count = pd.DataFrame(df[count_var].min())
max_min_count = pd.concat([max_count, min_count], axis=1)
max_min_count.columns = ["max", "min"]
# store in pbz2 file
compressed_pickle("./data/preprocessed/cnt_min_max", max_min_count)
# normalize data
mm_scaler = preprocessing.MinMaxScaler()
df[cont_var] = mm_scaler.fit_transform(df[cont_var])


# storage of preprocessed file
compressed_pickle("./data/preprocessed/BikeRental_preprocessed", df)
print("done (wo)man")
