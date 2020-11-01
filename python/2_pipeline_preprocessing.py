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


# normalize continous variables
conti_var = ["temp", "windspeed", "cnt"]
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
df[conti_var] = mm_scaler.fit_transform(df[conti_var])


# storage of preprocessed file
compressed_pickle("./data/preprocessed/BikeRental_preprocessed", df)


print("                              _       _           _                 \n\
                             ( )     ( )_        ( )                \n\
 _   _   _   _   _ _ __     _| |  _ _|  _)  _ _  | |__    _ _  ___  \n\
( ) ( )/ _ \( ) ( )  __)  / _  |/ _  ) |  / _  ) |  _  \/ _  )  __) \n\
| (_) | (_) ) (_) | |    ( (_| | (_| | |_( (_| | | | | | (_| |__  \ \n\
 \__  |\___/ \___/(_)     \__ _)\__ _)\__)\__ _) (_) (_)\__ _)____/ \n\
( )_| |                                                             \n\
 \___/                                                              \n\
 _                                                                                           _ \n\
( )                                                                                         ( )\n\
| |_     __    __   ___    _ _   _ __   __  _ _   _ __   _     ___   __   ___  ___   __    _| |\n\
|  _ \ / __ \/ __ \  _  \ (  _ \(  __)/ __ \  _ \(  __)/ _ \ / ___)/ __ \  __)  __)/ __ \/ _  |\n\
| |_) )  ___/  ___/ ( ) | | (_) ) |  (  ___/ (_) ) |  ( (_) ) (___(  ___/__  \__  \  ___/ (_| |\n\
(_ __/ \____)\____)_) (_) |  __/(_)   \____)  __/(_)   \___/ \____)\____)____/____/\____)\__ _)\n\
                          | |              | |                                                 \n\
                          (_)              (_)                                                 \n\
\n")