import pandas as pd
from tqdm import tqdm


def import_train_test_calc():
    df = pd.read_csv(
        "./data/preprocessed/BikeRental_preprocessed.csv", index_col=[0])
    min_max = pd.read_csv("./data/preprocessed/cnt_min_max.csv")

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
    return df, min_max, Y_train, Y_test, X_train, X_test, Y_train_mean, Y_train_meandev, Y_test_meandev


def r_squared_metrics(X_train, X_test, Y_train, Y_train_meandev, Y_test, Y_test_meandev, model, print=False):

    # calculate r-squared
    Y_train_pred = model.predict(X_train)
    Y_train_dev = sum((Y_train["cnt"].array-Y_train_pred)**2)
    r2 = 1 - Y_train_dev/Y_train_meandev

    # calculate pseudo-r-squared
    Y_test_pred = model.predict(X_test)
    Y_test_dev = sum((Y_test["cnt"].array-Y_test_pred)**2)
    pseudor2 = 1 - Y_test_dev/Y_test_meandev

    if print == True:
        print(f"R2 = {r2}\nPseudo-R2 = {pseudor2}")
    return r2, pseudor2


def predict_test_df(*models):
    df = pd.read_csv(
        "./data/preprocessed/BikeRental_preprocessed.csv", index_col=[0])
    min_max = pd.read_csv("./data/preprocessed/cnt_min_max.csv")

    test_df = df[round(len(df)*0.8):].copy()
    final_df = df[round(len(df)*0.8):].copy()
    final_df["cnt_norm"] = test_df["cnt"].apply(
        lambda x: x * (min_max["max"][0] - min_max["min"][0]) + min_max["min"][0])
    test_df = test_df.drop(["datetime", "cnt"], axis=1)
    for i in tqdm(models):
        if i.__module__ == 'catboost.core':
            # test_df = test_df.drop(["cnt"], axis = 1)
            cat_var = ["season", "yr", "mnth", "hr", "holiday",
                       "weekday", "workingday", "weathersit"]

            for v in cat_var:
                test_df[v] = test_df[v].astype("int64")

            final_df["cnt_pred_" + i.__module__] = i.predict(test_df)
            final_df["cnt_pred_norm_" + i.__module__] = final_df["cnt_pred_" + i.__module__].apply(
                lambda x: round(x * (min_max["max"][0] - min_max["min"][0]) + min_max["min"][0]))
        else:
            final_df["cnt_pred_" + i.__module__] = i.predict(test_df)
            final_df["cnt_pred_norm_" + i.__module__] = final_df["cnt_pred_" + i.__module__].apply(
                lambda x: round(x * (min_max["max"][0] - min_max["min"][0]) + min_max["min"][0]))
    return final_df
