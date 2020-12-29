from data_partitioning import train_test_split_ts, train_test_split_rs
from data_storage import connection
import pandas as pd
from logger import logger

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

logger.info("Let's start with the time-series-split!")

# train-test-split based on time
# define training size
train_size = 0.8

logger.info(f'Train-Test-Split\nTraining Size: {train_size}')

# create train test split samples
X_train, Y_train, X_test, Y_test = train_test_split_ts(df, train_size)

# save in database
X_train.to_sql("X_train", connection, if_exists="replace", index=False)
Y_train.to_sql("Y_train", connection, if_exists="replace", index=False)
X_test.to_sql("X_test", connection, if_exists="replace", index=False)
Y_test.to_sql("Y_test", connection, if_exists="replace", index=False)

logger.info('Now the random sample!')

# random train-test-split
# define training size
train_size = 0.8

logger.info(f'Train-Test-Split\nTraining Size: {train_size}')

# create train test split samples
X_train_rs, Y_train_rs, X_test_rs, Y_test_rs = train_test_split_rs(
    df, train_size)

# save in database
X_train_rs.to_sql("X_train_rs", connection, if_exists="replace", index=False)
Y_train_rs.to_sql("Y_train_rs", connection, if_exists="replace", index=False)
X_test_rs.to_sql("X_test_rs", connection, if_exists="replace", index=False)
Y_test_rs.to_sql("Y_test_rs", connection, if_exists="replace", index=False)


######################################
### for NN & SVR preprocessed data ###
######################################

logger.info(
    "Let's do the same things for the preprocessed data for the NN & SVR models!")

# load data
df_NN_SVR = pd.read_sql_query(
    '''SELECT * FROM hours_preprocessed_NN_SVR''', connection)

logger.info("Let's start with the time-series-split!")

# train-test-split based on time
# define training size
train_size_NN_SVR = 0.8

logger.info(f'Train-Test-Split\nTraining Size: {train_size_NN_SVR}')

# create train test split samples
X_train, Y_train, X_test, Y_test = train_test_split_ts(
    df_NN_SVR, train_size_NN_SVR)

# save in database
X_train.to_sql("X_train_NN_SVR", connection, if_exists="replace", index=False)
Y_train.to_sql("Y_train_NN_SVR", connection, if_exists="replace", index=False)
X_test.to_sql("X_test_NN_SVR", connection, if_exists="replace", index=False)
Y_test.to_sql("Y_test_NN_SVR", connection, if_exists="replace", index=False)

logger.info('Now the random sample!')

# random train-test-split
# define training size
train_size_NN_SVR = 0.8

logger.info(f'Train-Test-Split\nTraining Size: {train_size_NN_SVR}')

# create train test split samples
X_train_rs, Y_train_rs, X_test_rs, Y_test_rs = train_test_split_rs(
    df_NN_SVR, train_size_NN_SVR)

# create train test split samples
# X_train_rs, Y_train_rs, X_test_rs, Y_test_rs = train_test_split_rs(
    # df, train_size)

# save in database
X_train_rs.to_sql("X_train_rs_NN_SVR", connection,
               if_exists="replace", index=False)
Y_train_rs.to_sql("Y_train_rs_NN_SVR", connection,
               if_exists="replace", index=False)
X_test_rs.to_sql("X_test_rs_NN_SVR", connection,
               if_exists="replace", index=False)
Y_test_rs.to_sql("Y_test_rs_NN_SVR", connection,
               if_exists="replace", index=False)


# print statement
print(" __ \                      |\n\
 |   |  _ \   __ \    _ \  |\n\
 |   | (   |  |   |   __/ _|\n\
____/ \___/  _|  _| \___| _)\n\
You can find the splitted data in the database!")
