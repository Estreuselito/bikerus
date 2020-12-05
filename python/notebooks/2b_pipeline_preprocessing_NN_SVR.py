from data_storage import connection, check_and_create_and_insert
import pandas as pd
import numpy as np
from sql_commands import create_table_hours_preprocessed_NN_SVR
from sklearn import preprocessing

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
df_dummies = pd.get_dummies(df["weathersit"], drop_first=True, prefix ="weathersit")
df = df.drop(["weathersit"],axis=1)
df = df.join(df_dummies)

# storage of NN & SVR specific preprocessed file
check_and_create_and_insert(
    connection, "hours_preprocessed_NN_SVR", df, create_table_hours_preprocessed_NN_SVR)

# print statement
print(" __ \                      |\n\
 |   |  _ \   __ \    _ \  |\n\
 |   | (   |  |   |   __/ _|\n\
____/ \___/  _|  _| \___| _)\n\
The data is now saved in the database!")