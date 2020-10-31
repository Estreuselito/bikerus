Authors: *[Yannik Suhre](https://github.com/yanniksuhre), [Jan Faulstich](https://github.com/TazTornadoo), [Skyler MacGowan](https://github.com/Schuyler-lab), [Sebastian Sydow](https://gitlab.com/sydow), [Jacob Umland](https://gitlab.com/jacobumland)*

# Bikerus

![language](https://img.shields.io/badge/language-Python%20%7C%20Docker-blue)
![version](https://img.shields.io/badge/version-v0.0.1-yellow)
![last-edited](https://img.shields.io/badge/last%20edited-28.10.2020-green)
![licence](https://img.shields.io/badge/licence-GPLv3-red)

> 🚴 This repository shows how to predict the demand of bikes needed for a bike rental service.

- [Bikerus](#bikerus)
- [Data acquisition](#data-acquisition)
- [Imputing NAs](#imputing-nas)
- [Data visualization](#data-visualization)
  - [Bike Rental Station Map](#bike-rental-station-map)

# Data acquisition

> 💾 This paragraph will explain how you can obtain the used data

In order to obtain the data, which is used within this project please clone this repository and execute then the file `0_pipeline_data_getting_compression.py` file. This file will:
- Download the files from the web
- Extract them into a folder within the parent directory called `data/raw`
- Loads these raw datasets and converts them into a compressed file in `data/interim` (for the sake of convenience we left the raw data there, in order you want to change things).

# Imputing NAs

> 🥋 This paragraph will show how NAs are imputed

In order to impute your own missing values please execute the script named `1_pipeline_impute_NAs.py`. This will create a file in the folder `data` which is named `preprocessed`. In this folder you can find the final version of the Bike Rental data. 

# Data visualization

> 🗺️ Here will be shown how the data visualizations can be created

## Bike Rental Station Map

In order to reproduce the map with the bike share rental stations, you have to execute the file `0.1_pipeline_bike_station_viz.py` within the `python` folder. This will create a folder `images` within the parent directory. Once you enter this folder there should be an `.html` file, which contains this map.

# Data Partitioning

> This paragraph will explain how you can partition the used data into a train and test set and how the
train set can be partitioned into a train set and a cross-validation-set (for additional testing).

Steps for creating a train and test set:
1. Import the data. Use df = decompress_pickle(<path>.pbz2) for importing.
2. Call the function train_test_split_ts. 
    The function takes two arguments: The first one is the data (type: DataFrame). The second is the size of the training set (type: float). The size of the training set must be greater than 0 and smaller than 1.
    The functions returns the sets for X_train, Y_train, X_test and Y_test. X_train and X_test include all columns except for the target column. Y_train and Y_test only include the target column (field to predict). X_train and Y_train are used for training including determining the samples for cross validation. X_test and Y_test are only used for the (final) testing. The files are exported to: './data/partitioned/'.

Steps for creating a train and test set for cross validation:
1. Import the date: X_train, Y_train. Use df = decompress_pickle(<path>.pbz2) for importing.
2. Call the function get_sample_for_cv.
    The function takes six arguments. Two arguments are optional (refer to the steps for creating a horizontal bar diagram to visualize the train-test-splits). 
    - n_splits: This determines the number of splits used for cross-validation. It must be an integer and greater than 1.
    - fold: This determines the current fold (subsample) of the train and test set for cross validation.
      It must be an integer and greater than 0 and not greater than the number of splits.
    - X_train and Y_train: Data used for training including determining the samples for cross validation. X_train includes all columns except for the target column. Y_train onlys include the target column (field to predict).
    The functions returns the sets for X_train_current and Y_train_current as the current fold/sub-sample. Additionally it returns X_test_cv_current and Y_test_cv_current for cross-validation.

Steps for creating a horizontal bar diagramm to visualizse the train-test-splits:
1. The function get_sample_for_cv can create a horizontal bar diagram for the visualization of the
the train-test-splits. The function only creates the bardiagram if X_test is added as a parameter and if the parameter vis == 'yes'.
- X_test: X_test is needed to visualize the final round of testing with X_test and Y_test, which we created at the beginning with the function train_test_split_ts. To create the horizontal bardiagramm, X_test has to be added to the function.
- vis: Vis is used as decision variable for the creation of the diagram. It is initalized as 'None'. Therefore, the horizontal bardiagramm will not be created. To create the horizontal bardiagramm, add 'yes' as the last parameter, when calling the function. The figure is saved in the path './data/partitioned/'.
