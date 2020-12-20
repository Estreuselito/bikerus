from data_storage import connection
import pandas as pd
import numpy as np
from sklearn import preprocessing
from logger import logger

print("                                                               _)\n\
 __ \    __|   _ \  __ \    __|   _ \    __|   _ \   __|   __|  |  __ \    _  |\n\
 |   |  |      __/  |   |  |     (   |  (      __/ \__ \ \__ \  |  |   |  (   |\n\
 .__/  _|    \___|  .__/  _|    \___/  \___| \___| ____/ ____/ _| _|  _| \__. |\n\
_|                 _|                                                    |___/\n")

logger.info("Preprocessing for ML Models (Regression)")
# load data
df = pd.read_sql_query('''SELECT * FROM hours_complete''', connection)

# add new rush-hour column on working days
df["rush_hour"] = 0

rush_hours = [7, 8, 16, 17, 18]

for rush_hour in rush_hours:
    df.loc[df.hr == rush_hour, "rush_hour"] = 1
df.loc[df.workingday == 0, 'rush_hour'] = 0

# drop leakage variables
leak_var = ["casual", "registered"]
df = df.drop(leak_var, axis=1)


# drop highly correlated variables
high_corr_var = ["temp"]
df = df.drop(high_corr_var, axis=1)


# drop redundant dteday variable
red_var = ["dteday"]
df = df.drop(red_var, axis=1)


# coerce correct data types for categorical data
# cat_var = ["season", "yr", "mnth", "hr", "holiday",
#            "weekday", "workingday", "weathersit"]
# for v in cat_var:
#     df[v] = df[v].astype("category")


# normalize continous variables
conti_var = ["atemp", "windspeed", "cnt"]
# store true min & max to be able to revert normalization
count_var = ["cnt"]
max_count = pd.DataFrame(df[count_var].max())
min_count = pd.DataFrame(df[count_var].min())
max_min_count = pd.concat([max_count, min_count], axis=1)
max_min_count.columns = ["max", "min"]

# store in database
max_min_count.to_sql("max_min_count", connection,
                     if_exists="replace", index=False)

# normalize data
mm_scaler = preprocessing.MinMaxScaler()
df[conti_var] = mm_scaler.fit_transform(df[conti_var])

# storage of preprocessed file
df.to_sql("hours_preprocessed", connection, if_exists="replace", index=False)

logger.info("Preprocessing for Neural Net and Support Vector Regression")
# load data
df = pd.read_sql_query('''SELECT * FROM hours_preprocessed''', connection)

# apply cos & sin transformation for cyclical features
cycl_var = ["season", "mnth", "hr", "weekday"]
mm_scaler = preprocessing.MinMaxScaler()
for i in cycl_var:
    df[i] = df[i].astype("int32")
    df[f"{i}_sin"] = np.sin(2 * np.pi * df[i]/df[i].nunique())
    df[f"{i}_cos"] = np.cos(2 * np.pi * df[i]/df[i].nunique())
    sin_cos = [f"{i}_sin", f"{i}_cos"]
    df[sin_cos] = mm_scaler.fit_transform(df[sin_cos])
    df = df.drop(i, axis=1)

# create dummy variables for categorial weathersit while avoiding dummy variable trap
df_dummies = pd.get_dummies(
    df["weathersit"], drop_first=True, prefix="weathersit")
df = df.drop(["weathersit"], axis=1)
df = df.join(df_dummies)

# storage of NN & SVR specific preprocessed file
df.to_sql("hours_preprocessed_NN_SVR", connection,
          if_exists="replace", index=False)

# print statement
print(" __ \                      |\n\
 |   |  _ \   __ \    _ \  |\n\
 |   | (   |  |   |   __/ _|\n\
____/ \___/  _|  _| \___| _)\n\
The data is now saved in the database!")
