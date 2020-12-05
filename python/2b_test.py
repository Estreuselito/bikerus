from data_storage import connection, check_and_create_and_insert
import pandas as pd
import numpy as np
from sql_commands import create_table_hours_preprocessed_NN_SVR
from sklearn import preprocessing

# df = pd.read_sql_query('''SELECT * FROM hours_preprocessed_NN_SVR''', connection)
# print(df)

# a = pd.read_sql_query('''SELECT * FROM raw''', connection)
# print(a.head())

# b = pd.read_sql_query('''SELECT * FROM hours_complete''', connection)
# print(b)

# c = pd.read_sql_query('''SELECT * FROM hours_preprocessed''', connection)
# print(c)

d = pd.read_sql_query('''SELECT * FROM hours_preprocessed_NN_SVR''', connection)
print(d)
