from data_preprocessing import decompress_pickle, compressed_pickle
import pandas as pd
from sklearn import preprocessing
import os

print("Loading data")

# load data
df = decompress_pickle("./data/preprocessed/BikeRental_complete.pbz2")


# NORMALIZATION
cont_var = ["temp", "windspeed", "casual", "registered", "cnt"]

# store true min & max to be able to revert normalization
count_var = ["casual","registered","cnt"]
max_count = pd.DataFrame(df[count_var].max())
min_count = pd.DataFrame(df[count_var].min())
max_min_count = pd.concat([max_count, min_count], axis=1)
max_min_count.columns = ["max", "min"]

# store in csv
max_min_count.to_csv("./data/preprocessed/true_max_mins.csv")

# normalize data
mm_scaler = preprocessing.MinMaxScaler()
df[cont_var] = mm_scaler.fit_transform(df[cont_var])


# CREATION OF DUMMY VARIABLES 
df.weathersit = df.weathersit.astype(int)

dummy_var = ["season",
            "yr",
            "mnth",
            "hr",
            "holiday",
            "weekday",
            "workingday",
            "weathersit"
            ]

for v in dummy_var:
    dummies = pd.get_dummies(df[v], prefix=v, drop_first=False)
    df = pd.concat([df, dummies], axis=1)

# drop redundant variables
df = df.drop(dummy_var, axis=1)

# storage
compressed_pickle("./data/preprocessed/BikeRental_preprocessed", df)
print('DONE Dikka')
