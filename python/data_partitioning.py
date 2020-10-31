# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from data_preprocessing import decompress_pickle, compressed_pickle
# from sklearn.model_selection import TimeSeriesSplit

def train_test_split_ts(data, train_size):
    """This function creates the train and test set for time-series data.
    
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
    from data_preprocessing import decompress_pickle, compressed_pickle

    # Sanity Check
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

    compressed_pickle("./data/partitioned/BikeRental_X_train", X_train)
    compressed_pickle("./data/partitioned/BikeRental_Y_train", Y_train)
    compressed_pickle("./data/partitioned/BikeRental_X_test", X_test)
    compressed_pickle("./data/partitioned/BikeRental_Y_test", Y_test)

def get_sample_for_cv(n_splits, fold, X_train, Y_train, X_test = None, vis = None):
    """This function creates the train and test sets for cross validation for time-series data.
    Furthermore, it creates the horizontal bardiagramm to visualiuze cross-validation/testing iterations 
    including the final testing after all cross-validations have been performed.
    
    Parameters
    ----------
    n_splits: int
        This determines the number of splits used for cross-validation.
        It must be greater than 1.

    fold: int
        This determines the current fold of the train and test set for cross validation.
        It must be greater than 0 and not greater than the number of splits.
    
    X_train: DataFrame
        Data used for training including determining the samples for cross validation.
        X_train includes all columns except for the target column.
    
    Y_train: DataFrame
        Data are used for training including determining the samples for cross validation.
        Y_train onlys include the target column (field to predict).
    
    X_test: DataFrame
        Here, X_test is only used to create the horizontal bardiagramm to visualize the testing
        iterations. It is therefore initialized as 'None'. To create the horizontal bardiagramm, 
        X_test has to be added to the function.
    
    vis: str
        Vis is used to create the diagramm. It is initalized as 'None'. Therefore, the horizontal bardiagramm
        will not be created. To create the horizontal bardiagramm, add 'yes' as the last parameter, when calling
        the function. The figure is saved in the path './data/partitioned/'.

    Returns
    -------
    This functions returns the four sets: 
    X_train_current and Y_train_current for training.
    X_cv_current and Y_cv_current for cross validation.
    Optionally, it returns the horizontal bardiagramm to visualize the testing iterations.
    """

    # Necessary Imports
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from data_preprocessing import decompress_pickle, compressed_pickle
    from sklearn.model_selection import TimeSeriesSplit

    # Sanity Check
    if type(n_splits) != int:
        return 'Number of splits must be an integer.'
    if type(fold) != int:
        print('Number of folds must be an integer.')
    if n_splits < 2:
        return 'Number of splits must be at least 2.'
    if fold == 0:
        return 'Fold must be greater than 0.'
    if fold > n_splits:
        return 'Fold cannot be greater than number of splits.'

    # Creation of train and test sets for cross validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    list_tscv = []
    for train, test in tscv.split(X_train):
        list_tscv.append([test[0], test[-1]])
    
    if n_splits == fold:
        X_train_current = X_train.iloc[:list_tscv[fold-1][0]]
        Y_train_current = Y_train.iloc[:list_tscv[fold-1][0]]
        X_test_cv_current = X_train.iloc[list_tscv[fold-1][0]:list_tscv[fold-1][1]+1]    # +1 to include the last element X_train
        Y_test_cv_current = Y_train.iloc[list_tscv[fold-1][0]:list_tscv[fold-1][1]+1]    # +1 to include the last element of Y_train
    else: 
        X_train_current = X_train.iloc[:list_tscv[fold-1][0]]
        Y_train_current = Y_train.iloc[:list_tscv[fold-1][0]]
        X_test_cv_current = X_train.iloc[list_tscv[fold-1][0]:list_tscv[fold-1][1]]
        Y_test_cv_current = Y_train.iloc[list_tscv[fold-1][0]:list_tscv[fold-1][1]]

    compressed_pickle("./data/partitioned/cross_validation/BikeRental_X_train_current", X_train_current)
    compressed_pickle("./data/partitioned/cross_validation/BikeRental_Y_train_current", Y_train_current)
    compressed_pickle("./data/partitioned/cross_validation/BikeRental_X_test_cv_current", X_test_cv_current)
    compressed_pickle("./data/partitioned/cross_validation/BikeRental_Y_test_cv_current", Y_test_cv_current)
    
    # Visualization (Optional, only executed if vis is set to 'yes' when calling the function)
    if vis == 'yes':
        list_vs_train = []
        list_vs_test = []
        for fold in list_tscv:
            list_vs_train.insert(0, fold[0])
            list_vs_test.append(fold[1]-fold[0])
        list_vs_train.insert(0, X_test.index[0]) #insert elements for final testing (not part of cross validation)
        list_vs_test.insert(0, len(X_test)) #insert elements for final testing (not part of cross validation)

        # Plot
        y = np.arange(len(list_vs_train))
        plt.barh(y, list_vs_train, color='blue')
        plt.barh(y, list_vs_test, color='red', left = list_vs_train)

        # Labels
        plt.xticks(np.arange(0, list_vs_train[0]+5000, 2000))
        folds = list(reversed(range(1, len(list_vs_train)+1)))
        folds[0] = 'Final'
        y_labels = folds 
        plt.yticks(y, y_labels)
        plt.title('Visualization of Train and Test Set for Cross Validation')
        plt.xlabel('Sample Index')
        plt.ylabel('CV Iteration / Iteration of Testing')
    
        # Legends
        train_patch = mpatches.Patch(color='blue', label='train set')
        test_patch = mpatches.Patch(color='red', label='test set')
        plt.legend(handles=[train_patch, test_patch])
    
        # Save Diagramm
        plt.savefig("./data/partitioned/Train_Test_Split_Visualization", dpi = 300)

# Example for Execution
# Create train and test set
from data_preprocessing import decompress_pickle, compressed_pickle
df = decompress_pickle("./data/preprocessed/BikeRental_complete.pbz2")
train_test_split_ts(df, 0.8)

# Create train and test sets for validation
X_train = decompress_pickle("./data/partitioned/BikeRental_X_train.pbz2")
Y_train = decompress_pickle("./data/partitioned/BikeRental_Y_train.pbz2")
X_test = decompress_pickle("./data/partitioned/BikeRental_X_test.pbz2")
get_sample_for_cv(9, 9, X_train, Y_train)
# inlcuding the bar diagram
# get_sample_for_cv(9, 9, X_train, Y_train, X_test = X_test, vis = 'yes')