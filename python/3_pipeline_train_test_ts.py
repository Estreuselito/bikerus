from data_partitioning import train_test_split_ts
from data_storage import check_and_create_and_insert, connection
from sql_commands import create_table_X_train_test, create_table_Y_train_test
import pandas as pd

print(" |                _)              |                |                      | _)  |\n\
 __|   __|   _` |  |  __ \        __|   _ \   __|  __|        __|  __ \   |  |  __|\n\
 |    |     (   |  |  |   |       |     __/ \__ \  |        \__ \  |   |  |  |  |\n\
\__| _|    \__._| _| _|  _|      \__| \___| ____/ \__|      ____/  .__/  _| _| \__|\n\
                                                                  _|\n")

# load data
df = pd.read_sql_query('''SELECT * FROM hours_preprocessed''', connection)

# define training size
train_size = 0.8

print(f'Train-Test-Split\nTraining Size: {train_size}')

# create train test split samples
X_train, Y_train, X_test, Y_test = train_test_split_ts(df, train_size)

# save in database
check_and_create_and_insert(connection, "X_train", X_train, create_table_X_train_test.format("X_train"))
check_and_create_and_insert(connection, "X_test", X_test, create_table_X_train_test.format("X_test"))
check_and_create_and_insert(connection, "Y_train", Y_train, create_table_Y_train_test.format("Y_train"))
check_and_create_and_insert(connection, "Y_test", Y_test, create_table_Y_train_test.format("Y_test"))

# print statement
print(" __ \                      |\n\
 |   |  _ \   __ \    _ \  |\n\
 |   | (   |  |   |   __/ _|\n\
____/ \___/  _|  _| \___| _)\n\
You can find the splited data in the database!")