from data_partitioning import train_test_split_ts
from data_preprocessing import decompress_pickle
import os

print(" |                _)              |                |                      | _)  |\n\
 __|   __|   _` |  |  __ \        __|   _ \   __|  __|        __|  __ \   |  |  __|\n\
 |    |     (   |  |  |   |       |     __/ \__ \  |        \__ \  |   |  |  |  |\n\
\__| _|    \__._| _| _|  _|      \__| \___| ____/ \__|      ____/  .__/  _| _| \__|\n\
                                                                  _|\n")

df = decompress_pickle("./data/preprocessed/BikeRental_preprocessed.pbz2")
train_size = 0.8
print(f'Train-Test-Split\nTraining Size: {train_size}')
if not os.path.exists("./data/partitioned"):
    os.makedirs("./data/partitioned")
train_test_split_ts(df, train_size)

# print statement
print(" __ \                      |\n\
 |   |  _ \   __ \    _ \  |\n\
 |   | (   |  |   |   __/ _|\n\
____/ \___/  _|  _| \___| _)\n\
You can find the splited data under data/partioned!")