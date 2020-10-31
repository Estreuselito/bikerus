import pandas as pd
from data_preprocessing import decompress_pickle, compressed_pickle
from sklearn.model_selection import TimeSeriesSplit

def test_train_split_ts(data, train_size):
    """This function creates the train and test set for time-series data.
    
    Parameters
    ----------
    data: DataFrame
        This is the data, which will be splitted into the train and test set.
    
    train_size: float
        This determines the size of the train and test set.
        
    Returns
    -------
    This functions returns the sets for X_train, Y_train, X_test and Y_test.
    X_train and X_test include all columns except for the target column.
    Y_train and Y_test only include the target column (field to predict).
    X_train and Y_train are used for training including determining the samples for cross validation.
    X_test and Y_test are only used for the testing.
    """

    if train_size <= 0:
        return 'The size of the train set has to be greater than 0.'
    if train_size >= 1:
        return 'The size of the train set has to be smaller than 1.'
    else:
        X = df.drop(['cnt'], axis = 1)
        Y = df['cnt']
        index = round(len(X) * train_size)
        X_train = X.iloc[:index]
        Y_train = Y.iloc[:index]
        X_test = X.iloc[index:]
        Y_test = Y.iloc[index:]
    return X_train, Y_train, X_test, Y_test

def get_sample_for_cv(n_splits, fold, X_train, Y_train):
    """This function creates the train and test sets for cross validation for time-series data.
    The following import is necessary: from sklearn.model_selection import TimeSeriesSplit
    
    Parameters
    ----------
    n_splits: int
        This determines the number of splits used for cross-validation.
        It must be greater than 1.

    fold: int
        This determines the current fold of the train and test set for cross validation.
        It must be greater than 0 and not greter than the number of splits.
    
    X_train: DataFrame
        Data used for training including determining the samples for cross validation.
        X_train includes all columns except for the target column.
    
    Y_train: DataFrame
        Data are used for training including determining the samples for cross validation.
        Y_train onlys include the target column (field to predict).

    Returns
    -------
    This functions returns the four sets: 
    X_train_current and Y_train_current for training.
    X_cv_current and Y_cv_current for cross validation.
    """
    if n_splits < 2:
        return 'Number of splits must be at least 2.'
    if fold == 0:
        return 'Fold must be greater than 0.'
    if fold > n_splits:
        return 'Fold cannot be greater than number of splits.'

    tscv = TimeSeriesSplit(n_splits=n_splits)
    list_tscv = []
    for train, test in tscv.split(X_train):
        list_tscv.append([test[0], test[-1]])

    if n_splits == fold:
        X_train_current = X_train.iloc[:list_tscv[fold-1][0]]
        Y_train_current = Y_train.iloc[:list_tscv[fold-1][0]]
        X_cv_current = X_train.iloc[list_tscv[fold-1][0]:list_tscv[fold-1][1]+1]    # +1 to include the last element X_train
        Y_cv_current = Y_train.iloc[list_tscv[fold-1][0]:list_tscv[fold-1][1]+1]    # +1 to include the last element of Y_train
    else: 
        X_train_current = X_train.iloc[:list_tscv[fold-1][0]]
        Y_train_current = Y_train.iloc[:list_tscv[fold-1][0]]
        X_cv_current = X_train.iloc[list_tscv[fold-1][0]:list_tscv[fold-1][1]]
        Y_cv_current = Y_train.iloc[list_tscv[fold-1][0]:list_tscv[fold-1][1]]
    return X_train_current, Y_train_current, X_cv_current, Y_cv_current
