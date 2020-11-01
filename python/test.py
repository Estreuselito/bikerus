from data_preprocessing import decompress_pickle, compressed_pickle

df_0 = decompress_pickle("./data/preprocessed/BikeRental_complete.pbz2")
# df_1 = decompress_pickle("./data/partitioned/cross_validation/BikeRental_X_train_current.pbz2")
# df_2 = decompress_pickle("./data/partitioned/cross_validation/BikeRental_X_test_cv_current.pbz2")
# df_3 = decompress_pickle("./data/partitioned/BikeRental_X_train.pbz2")
# df_4 = decompress_pickle("./data/partitioned/BikeRental_X_test.pbz2")

# print(len(df_0), len(df_1), len(df_2), len(df_3), len(df_4))
df_100 = decompress_pickle("./data/partitioned/cross_validation/BikeRental_Y_test_cv_current.pbz2")
print(df_100)