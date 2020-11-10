import pandas as pd
import numpy as np
from data_preprocessing import compressed_pickle, decompress_pickle
import os
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

# only activate in conda env with fastai running
# from fastai.tabular.all import *

# def fastai_neural_regression():
#     """This function will (train) and return the test dataframe with the predicted values
#     """
#     # df = decompress_pickle("../data/preprocessed/BikeRental_preprocessed.pbz2")
#     # min_max = decompress_pickle("../data/preprocessed/cnt_min_max.pbz2")

#     path = "../data/preprocessed/BikeRental_preprocessed.csv"
#     df = pd.read_csv(
#         "../data/preprocessed/BikeRental_preprocessed.csv", index_col=[0])
#     min_max = pd.read_csv("../data/preprocessed/cnt_min_max.csv")

#     cat_names = ['season', 'yr', 'weathersit', "workingday", "holiday"]
#     cont_names = ['mnth', 'hr', 'weekday', 'temp', "windspeed", "hum"]
#     procs = [Categorify]
#     dls = TabularDataLoaders.from_df(
#         df, path, procs=procs, cat_names=cat_names, cont_names=cont_names, y_names="cnt", bs=64)
#     split = 0.8
#     splits = (list(range(0, round(len(df)*split))),
#               list(range(round(len(df)*split), len(df))))
#     to = TabularPandas(df, procs=[Categorify],
#                        cat_names=cat_names,
#                        cont_names=cont_names,
#                        y_names='cnt',
#                        splits=splits)
#     dls = to.dataloaders(bs=64)
#     learn = tabular_learner(dls,
#                             metrics=R2Score(),
#                             layers=[500, 250],
#                             n_out=1,
#                             loss_func=F.mse_loss)
#     try:
#         learn.load('/fastai/fastai_learner')
#         print("Model loaded!")
#     except FileNotFoundError:
#         print("This can take some time! Your model will be trained now!")
#         learn.fit_one_cycle(50)
#         if not os.path.exists("/fastai"):
#             os.makedirs("/fastai")
#         learn.save('/fastai/fastai_learner')
#         print("Model trained and saved!")

#     test_df = df[round(len(df)*0.8):].copy()
#     test_df.rename(columns={"cnt": "cnt_real"})
#     dl = learn.dls.test_dl(test_df)
#     preds = learn.get_preds(dl=dl)
#     test_df["cnt_pred"] = [preds[0][i].item() for i in range(0, len(preds[0]))]
#     test_df["cnt_norm"] = test_df["cnt"].apply(
#         lambda x: x * (min_max["max"][0] - min_max["min"][0]) + min_max["min"][0])
#     test_df["cnt_pred_norm"] = test_df["cnt_pred"].apply(lambda x: round(
#         x * (min_max["max"][0] - min_max["min"][0]) + min_max["min"][0]))

#     if not os.path.exists("../data/predictions"):
#         os.makedirs("../data/predictions")

#     train_df = df[:round(len(df)*0.8)].copy()
#     train_df.rename(columns={"cnt": "cnt_real"})
#     dl = learn.dls.test_dl(train_df)
#     y_train_preds = learn.get_preds(dl=dl)
#     y_train_preds = [y_train_preds[0][i].item()
#                      for i in range(0, len(y_train_preds[0]))]
#     Y_train_mean = train_df["cnt"].values.mean()
#     Y_train_meandev = sum((train_df["cnt"].values-Y_train_mean)**2)
#     Y_train_dev = sum((train_df["cnt"].values-y_train_preds)**2)
#     r2 = 1 - Y_train_dev/Y_train_meandev
#     print("R2 =", r2)

#     test_df = df[round(len(df)*0.8):].copy()
#     test_df.rename(columns={"cnt": "cnt_real"})
#     dl = learn.dls.test_dl(test_df)
#     y_test_preds = learn.get_preds(dl=dl)
#     y_test_preds = [y_test_preds[0][i].item()
#                     for i in range(0, len(y_test_preds[0]))]
#     Y_test_dev = sum((test_df["cnt"].values-y_test_preds)**2)
#     Y_test_mean = test_df["cnt"].values.mean()
#     Y_test_meandev = sum((test_df["cnt"].values-Y_test_mean)**2)
#     pseudor2 = 1 - Y_test_dev/Y_test_meandev
#     print("Pseudo-R2 =", pseudor2)
#     compressed_pickle("../data/predictions/fastai", test_df)

#     return r2, pseudor2

# FIXME: Be aware that if you are running this in a anaconda enviroment you have to change the dot to double dots!!!

def catboost_regressor():
    # Y_train = decompress_pickle(
    #     "./data/partitioned/BikeRental_Y_train.pbz2")
    # Y_test = decompress_pickle("./data/partitioned/BikeRental_Y_test.pbz2")
    # X_train = decompress_pickle(
    #     "./data/partitioned/BikeRental_X_train.pbz2")
    # X_test = decompress_pickle("./data/partitioned/BikeRental_X_test.pbz2")

    Y_train = pd.read_csv(
        "./data/partitioned/BikeRental_Y_train.csv", index_col=[0])
    Y_test = pd.read_csv(
        "./data/partitioned/BikeRental_Y_test.csv", index_col=[0])
    X_train = pd.read_csv(
        "./data/partitioned/BikeRental_X_train.csv", index_col=[0])
    X_test = pd.read_csv(
        "./data/partitioned/BikeRental_X_test.csv", index_col=[0])

    X_train = X_train.drop('datetime', axis=1)
    X_test = X_test.drop('datetime', axis=1)

    Y_train_mean = Y_train.mean()
    #Y_train_meandev = sum((Y_train-Y_train_mean)**2)
    Y_train_meandev = ((Y_train-Y_train_mean)**2).sum()
    #Y_test_meandev = sum((Y_test-Y_train_mean)**2)
    Y_test_meandev = ((Y_test-Y_train_mean)**2).sum()

    try:
        # df = decompress_pickle(
        #     "./data/preprocessed/BikeRental_preprocessed.pbz2")
        df = pd.read_csv(
            "./data/preprocessed/BikeRental_preprocessed.csv", index_col=[0])
        min_max = pd.read_csv("./data/preprocessed/cnt_min_max.csv")
        #min_max = decompress_pickle("./data/preprocessed/cnt_min_max.pbz2")
        
        model.load_model("./catboost/catboost_model")
    except:
        cat_var = ["season", "yr", "mnth", "hr", "holiday",
                   "weekday", "workingday", "weathersit"]
        for v in cat_var:
            X_train[v] = X_train[v].astype("int64")
            X_test[v] = X_test[v].astype("int64")

        if not os.path.exists("./catboost"):
            os.makedirs("./catboost")

        os.chdir("./catboost/")

        model = CatBoostRegressor(loss_function='RMSE', depth=6,
                                  learning_rate=0.1, iterations=1000, od_type='Iter', od_wait=10)

        model.fit(
            X_train, Y_train,
            use_best_model=True,
            cat_features=["season", "yr", "mnth", "hr",
                          "holiday", "weekday", "workingday", "weathersit"],
            eval_set=(X_test, Y_test),
            verbose=True,
            plot=True
        )

        model.save_model("catboost_model", format="cbm")

    test_df = df[round(len(df)*0.8):].copy()
    test_df = test_df.drop('datetime', axis=1)
    test_df.rename(columns={"cnt": "cnt_real"})
    cat_var = ["season", "yr", "mnth", "hr", "holiday",
               "weekday", "workingday", "weathersit"]

    for v in cat_var:
        test_df[v] = test_df[v].astype("int64")

    test_df["cnt_pred"] = model.predict(test_df)
    test_df["cnt_norm"] = test_df["cnt"].apply(
        lambda x: x * (min_max["max"][0] - min_max["min"][0]) + min_max["min"][0])
    test_df["cnt_pred_norm"] = test_df["cnt_pred"].apply(lambda x: round(
        x * (min_max["max"][0] - min_max["min"][0]) + min_max["min"][0]))

    if not os.path.exists("./data/predictions"):
        os.makedirs("./data/predictions")

    compressed_pickle("./data/predictions/catboost", test_df)
    test_df.to_csv("./data/predictions/catboost.csv")
    print(test_df)

    Y_train_pred = model.predict(X_train)
    Y_train_dev = sum((Y_train["cnt"].array-Y_train_pred)**2)
    # Y_train_dev = ((Y_train-Y_train_pred)**2).sum()
    r2 = 1 - Y_train_dev/Y_train_meandev
    print("R2 =", r2)

    Y_test_pred = model.predict(X_test)
    Y_test_dev = sum((Y_test["cnt"].array-Y_test_pred)**2)
    # Y_test_dev = ((Y_test-Y_test_pred)**2).sum()
    pseudor2 = 1 - Y_test_dev/Y_test_meandev
    print("Pseudo-R2 =", pseudor2)
    return r2.values[0], pseudor2.values[0]
