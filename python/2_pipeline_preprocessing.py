from data_storage import connection, check_and_create_and_insert
from sql_commands import create_table_max_min_count, create_table_hours_preprocessed
import pandas as pd
from sklearn import preprocessing

print("                                                               _)\n\
 __ \    __|   _ \  __ \    __|   _ \    __|   _ \   __|   __|  |  __ \    _  |\n\
 |   |  |      __/  |   |  |     (   |  (      __/ \__ \ \__ \  |  |   |  (   |\n\
 .__/  _|    \___|  .__/  _|    \___/  \___| \___| ____/ ____/ _| _|  _| \__. |\n\
_|                 _|                                                    |___/\n")

# load data
df = pd.read_sql_query('''SELECT * FROM hours_complete''', connection)


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
cat_var = ["season", "yr", "mnth", "hr", "holiday",
           "weekday", "workingday", "weathersit"]
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

# store in database
check_and_create_and_insert(connection, "max_min_count", max_min_count, create_table_max_min_count)

# normalize data
mm_scaler = preprocessing.MinMaxScaler()
df[conti_var] = mm_scaler.fit_transform(df[conti_var])


# storage of preprocessed file
check_and_create_and_insert(connection, "hours_preprocessed", df, create_table_hours_preprocessed)

# print statement
print(" __ \                      |\n\
 |   |  _ \   __ \    _ \  |\n\
 |   | (   |  |   |   __/ _|\n\
____/ \___/  _|  _| \___| _)\n\
The data is now saved in the database!")
