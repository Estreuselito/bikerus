from data_storage import connection
import pandas as pd
from tqdm import tqdm

########################
### for general data ###
########################


def import_train_test_calc(rs=None, nn=None):
    """Returns various metrics regarding the train and test splits

    set nn = "_NN_SVR" for using this parameter
    """

    df = pd.read_sql_query(
        '''SELECT * FROM hours_preprocessed''' + str(nn or ""), connection)
    min_max = pd.read_sql_query(
        '''SELECT * FROM max_min_count''', connection)

    Y_train = pd.read_sql_query(
        '''SELECT * FROM Y_train''' + str(rs or "") + str(nn or ""), connection)
    X_train = pd.read_sql_query(
        '''SELECT * FROM X_train''' + str(rs or "") + str(nn or ""), connection)
    Y_test = pd.read_sql_query(
        '''SELECT * FROM Y_test''' + str(rs or "") + str(nn or ""), connection)
    X_test = pd.read_sql_query(
        '''SELECT * FROM X_test''' + str(rs or "") + str(nn or ""), connection)

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


def predict_test_df(*models, webapp=False, filter=None):

    df = pd.read_sql_query('''SELECT * FROM hours_preprocessed''', connection)
    min_max = pd.read_sql_query('''SELECT * FROM max_min_count''', connection)

    test_df = df[round(len(df)*0.8):].copy()

    if filter is not None:
        test_df = test_df[(test_df.datetime)]

    final_df = df[round(len(df)*0.8):].copy()
    final_df["cnt_norm"] = test_df["cnt"].apply(
        lambda x: x * (min_max["max"][0] - min_max["min"][0]) + min_max["min"][0])
    test_df = test_df.drop(["datetime", "cnt"], axis=1)

    for i in tqdm(models):
        if i.__module__ == 'catboost.core':
            df = pd.read_sql_query(
                '''SELECT * FROM hours_preprocessed''', connection)
            test_df = df[round(len(df)*0.8):].copy()
            test_df = test_df.drop(["datetime", "cnt"], axis=1)
            cat_var = ["season", "yr", "mnth", "hr", "holiday",
                       "weekday", "workingday", "weathersit"]

            for v in cat_var:
                test_df[v] = test_df[v].astype("int64")

            final_df["cnt_pred_" + f'{i}'] = i.predict(test_df)
            final_df["cnt_pred_norm_" + f'{i}'] = final_df["cnt_pred_" + f'{i}'].apply(
                lambda x: round(x * (min_max["max"][0] - min_max["min"][0]) + min_max["min"][0]))

            if webapp == True:
                return i.predict(test_df).apply(
                    lambda x: round(x * (min_max["max"][0] - min_max["min"][0]) + min_max["min"][0]))

        elif i.__module__ in ["sklearn.svm._classes", "sklearn.neural_network._multilayer_perceptron"]:
            df = pd.read_sql_query(
                '''SELECT * FROM hours_preprocessed_NN_SVR''', connection)
            test_df = df[round(len(df)*0.8):].copy()
            test_df = test_df.drop(["datetime", "cnt"], axis=1)
            final_df["cnt_pred_" + f'{i}'] = i.predict(test_df)
            final_df["cnt_pred_norm_" + f'{i}'] = final_df["cnt_pred_" + f'{i}'].apply(
                lambda x: round(x * (min_max["max"][0] - min_max["min"][0]) + min_max["min"][0]))

            if webapp == True:
                return i.predict(test_df).apply(
                    lambda x: round(x * (min_max["max"][0] - min_max["min"][0]) + min_max["min"][0]))
        else:
            final_df["cnt_pred_" + f'{i}'] = i.predict(test_df)
            final_df["cnt_pred_norm_" + f'{i}'] = final_df["cnt_pred_" + f'{i}'].apply(
                lambda x: round(x * (min_max["max"][0] - min_max["min"][0]) + min_max["min"][0]))

            if webapp == True:
                return i.predict(test_df).apply(
                    lambda x: round(x * (min_max["max"][0] - min_max["min"][0]) + min_max["min"][0]))
    return final_df


# only for grid search for Catboost model

def GridSearch_for_Catboost():
    print(format('How to find optimal parameters for CatBoost using GridSearchCV for Regression', '*^82'))

    import warnings
    warnings.filterwarnings("ignore")

    # load libraries
    from data_storage import check_and_create_and_insert, connection
    from sklearn import datasets
    from sklearn.model_selection import GridSearchCV
    from catboost import CatBoostRegressor

    df = pd.read_sql_query('''SELECT * FROM hours_preprocessed''', connection)

    Y = df.cnt
    X = df.drop("cnt", axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0)

    X_train = X_train.drop("datetime", axis=1)
    X_test = X_test.drop("datetime", axis=1)

    cat_var = ["season", "mnth", "hr", "holiday",
               "weekday", "workingday", "weathersit", "rush_hour"]
    for v in cat_var:
        X_train[v] = X_train[v].astype("int64")
        X_test[v] = X_test[v].astype("int64")

    model = CatBoostRegressor(loss_function='RMSE', cat_features=["season", "yr", "mnth", "hr", "holiday",
                                                                  "weekday", "workingday", "weathersit", "rush_hour"])
    parameters = {'depth': [6, 8, 10],
                  'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                  'iterations': [30, 50, 100, 200, 400, 600, 800, 1000]
                  }
    grid = GridSearchCV(
        estimator=model, param_grid=parameters,  cv=5, n_jobs=-1)
    grid.fit(X_train, Y_train)

    # Results from Grid Search
    print("\n========================================================")
    print(" Results from Grid Search ")
    print("========================================================")

    print("\n The best estimator across ALL searched params:\n",
          grid.best_estimator_)

    print("\n The best score across ALL searched params:\n",
          grid.best_score_)

    print("\n The best parameters across ALL searched params:\n",
          grid.best_params_)

    print("\n ========================================================")

# GridSearch_for_Catboost()
