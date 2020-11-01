from data_partitioning import train_test_split_ts
from data_preprocessing import decompress_pickle
import os
print('Loading Data')
df = decompress_pickle("./data/preprocessed/BikeRental_preprocessed.pbz2")
train_size = 0.8
print(f'Train-Test-Split\nTraining Size: {train_size}')
if not os.path.exists("./data/partitioned"):
    os.makedirs("./data/partitioned")
train_test_split_ts(df, train_size)
print('Done!')