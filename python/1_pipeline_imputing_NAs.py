# imports
from data_preprocessing import decompress_pickle, compressed_pickle
import pandas as pd
import os

print("Loading data")

# load data
df = decompress_pickle("./data/interim/BikeRental.pbz2")

#
df = df.set_index(pd.to_datetime(df["dteday"] + " " + pd.to_datetime(df["hr"], format = "%H").dt.strftime('%H')))
df = df.asfreq("H")
df = df.reset_index()

df["dteday"] = df["index"].dt.date
df["hr"] = df["index"].dt.hour
# df["yr"] = df["index"].dt.year
df["mnth"] = df["index"].dt.month
df["weekday"] = df["index"].dt.weekday
# working day can be interfered from weekday!
# map month to season

columns = ["holiday","yr", "season", "workingday"]
df[columns] = df[columns].ffill()

columns = ["temp", "weathersit", "atemp", "hum", "windspeed", "casual", "registered", "cnt"] # please review interpolate methods, since the filling with interpolate of registered and cnt is maybe not the best way
df[columns] = df[columns].interpolate()

df = df.drop("instant", axis=1).rename(columns={"index":"datetime"})

if not os.path.exists("./data/preprocessed"):
    os.makedirs("./data/preprocessed")

compressed_pickle("./data/preprocessed/BikeRental_complete", df)
print('DONE Dikka')