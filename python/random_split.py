def train_test_split_random(data, train_size):
    """This function creates the train and test set based on a random sample.

    Parameters
    ----------
    data: DataFrame
        This is the data, which will be splitted into the train and test set.

    train_size: float
        This determines the size of the train and test set.
        It must be greater than 0 and smaller than 1.

    Returns
    -------
    This functions returns the sets for X_train, Y_train, X_test and Y_test.
    X_train and X_test include all columns except for the target column.
    Y_train and Y_test only include the target column (field to predict).
    X_train and Y_train are used for training including determining the samples for cross validation.
    X_test and Y_test are only used for the testing.
    """

    # Necessary Imports
    from sklearn.model_selection import train_test_split

    # Sanity Check
    if train_size <= 0:
        print('The size of the train set has to be greater than 0.')
        return None
    if train_size >= 1:
        print('The size of the train set has to be smaller than 1.')
        return None
    # Split
    else:
        X = data.drop(['cnt'], axis=1)
        Y = data['cnt']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = train_size, random_state=0)
    
    return X_train, Y_train, X_test, Y_test





