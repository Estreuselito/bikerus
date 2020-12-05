from data_partitioning import train_test_split_ts, train_test_split_rs
from data_storage import check_and_create_and_insert, connection
from sql_commands import create_table_X_train_test, create_table_Y_train_test, create_table_X_train_test_NN_SVR, create_table_Y_train_test_NN_SVR
import pandas as pd

print(" |                _)              |                |                      | _)  |\n\
 __|   __|   _` |  |  __ \        __|   _ \   __|  __|        __|  __ \   |  |  __|\n\
 |    |     (   |  |  |   |       |     __/ \__ \  |        \__ \  |   |  |  |  |\n\
\__| _|    \__._| _| _|  _|      \__| \___| ____/ \__|      ____/  .__/  _| _| \__|\n\
                                                                  _|\n")

#####################################
### for general preprocessed data ###
#####################################

# load data
df = pd.read_sql_query('''SELECT * FROM hours_preprocessed''', connection)

print("Let's start with the time-series-split!")

# train-test-split based on time
# define training size
train_size = 0.8

print(f'Train-Test-Split\nTraining Size: {train_size}')

# create train test split samples
X_train, Y_train, X_test, Y_test = train_test_split_ts(df, train_size)

# save in database
check_and_create_and_insert(
    connection, "X_train", X_train, create_table_X_train_test.format("X_train"))
check_and_create_and_insert(
    connection, "X_test", X_test, create_table_X_train_test.format("X_test"))
check_and_create_and_insert(
    connection, "Y_train", Y_train, create_table_Y_train_test.format("Y_train"))
check_and_create_and_insert(
    connection, "Y_test", Y_test, create_table_Y_train_test.format("Y_test"))

print('Now the random sample!')

# random train-test-split
# define training size
train_size = 0.8

print(f'Train-Test-Split\nTraining Size: {train_size}')

# create train test split samples
X_train_rs, Y_train_rs, X_test_rs, Y_test_rs = train_test_split_rs(
    df, train_size)

# save in database
check_and_create_and_insert(
    connection, "X_train_rs", X_train_rs, create_table_X_train_test.format("X_train_rs"))
check_and_create_and_insert(
    connection, "X_test_rs", X_test_rs, create_table_X_train_test.format("X_test_rs"))
check_and_create_and_insert(
    connection, "Y_train_rs", Y_train_rs, create_table_Y_train_test.format("Y_train_rs"))
check_and_create_and_insert(
    connection, "Y_test_rs", Y_test_rs, create_table_Y_train_test.format("Y_test_rs"))


######################################
### for NN & SVR preprocessed data ###
######################################

print("Let's do the same things for the preprocessed data for the NN & SVR models!")

# load data
df_NN_SVR = pd.read_sql_query('''SELECT * FROM hours_preprocessed_NN_SVR''', connection)

print("Let's start with the time-series-split!")

# train-test-split based on time
# define training size
train_size_NN_SVR = 0.8

print(f'Train-Test-Split\nTraining Size: {train_size}')

# create train test split samples
X_train_NN_SVR, Y_train_NN_SVR, X_test_NN_SVR, Y_test_NN_SVR = train_test_split_ts(df_NN_SVR, train_size_NN_SVR)

# save in database
check_and_create_and_insert(
    connection, "X_train_NN_SVR", X_train_NN_SVR, create_table_X_train_test_NN_SVR.format("X_train_NN_SVR"))
check_and_create_and_insert(
    connection, "X_test_NN_SVR", X_test_NN_SVR, create_table_X_train_test_NN_SVR.format("X_test_NN_SVR"))
check_and_create_and_insert(
    connection, "Y_train_NN_SVR", Y_train_NN_SVR, create_table_Y_train_test_NN_SVR.format("Y_train_NN_SVR"))
check_and_create_and_insert(
    connection, "Y_test_NN_SVR", Y_test_NN_SVR, create_table_Y_train_test_NN_SVR.format("Y_test_NN_SVR"))

print('Now the random sample!')

# random train-test-split
# define training size
train_size_NN_SVR = 0.8

print(f'Train-Test-Split\nTraining Size: {train_size_NN_SVR}')

# create train test split samples
X_train_rs_NN_SVR, Y_train_rs_NN_SVR, X_test_rs_NN_SVR, Y_test_rs_NN_SVR = train_test_split_rs(
    df_NN_SVR, train_size_NN_SVR)

# save in database
check_and_create_and_insert(
    connection, "X_train_rs_NN_SVR", X_train_rs_NN_SVR, create_table_X_train_test_NN_SVR.format("X_train_rs_NN_SVR"))
check_and_create_and_insert(
    connection, "X_test_rs_NN_SVR", X_test_rs_NN_SVR, create_table_X_train_test_NN_SVR.format("X_test_rs_NN_SVR"))
check_and_create_and_insert(
    connection, "Y_train_rs_NN_SVR", Y_train_rs_NN_SVR, create_table_Y_train_test_NN_SVR.format("Y_train_rs_NN_SVR"))
check_and_create_and_insert(
    connection, "Y_test_rs_NN_SVR", Y_test_rs_NN_SVR, create_table_Y_train_test_NN_SVR.format("Y_test_rs_NN_SVR"))


# print statement
print(" __ \                      |\n\
 |   |  _ \   __ \    _ \  |\n\
 |   | (   |  |   |   __/ _|\n\
____/ \___/  _|  _| \___| _)\n\
You can find the splitted data in the database!")
