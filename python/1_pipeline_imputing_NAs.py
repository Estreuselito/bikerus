# imports
from data_preprocessing import decompress_pickle, compressed_pickle
import pandas as pd
import os
from pandas_profiling import ProfileReport

print("_)                           |   _)                      \  |     \ \n\
 |  __ `__ \   __ \   |   |  __|  |  __ \    _  |         \ |    _ \     __|\n\
 |  |   |   |  |   |  |   |  |    |  |   |  (   |       |\  |   ___ \  \__ \ \n\
_| _|  _|  _|  .__/  \__._| \__| _| _|  _| \__. |      _| \_| _/    _\ ____/\n\
              _|                           |___/\n")

# load data
df = decompress_pickle("./data/interim/BikeRental.pbz2")
df1 = decompress_pickle("./data/interim/BikeRental.pbz2")
#profpath = os.path.join("./images", profilename)
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

os.chdir("./")
if not os.path.exists("./data/preprocessed"):
    os.makedirs("./data/preprocessed")

compressed_pickle("./data/preprocessed/BikeRental_complete", df)

os.chdir("./images")
prof = ProfileReport(df)
prof.to_file(output_file='imputed_profile.html')

prof1 = ProfileReport(df1)
prof1.to_file(output_file='starting_profile.html')

# print statement
print(" __ \                      |\n\
 |   |  _ \   __ \    _ \  |\n\
 |   | (   |  |   |   __/ _|\n\
____/ \___/  _|  _| \___| _)\n\
You can find the data with imputed NAs under data/preprocessed! \
Moreover, you can find some nice Panda Profiling reports under images. Those are \
.html files, where you can look at the different correlations and missings.")