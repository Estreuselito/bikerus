Authors: *[Yannik Suhre](https://github.com/Estreuselito), [Jan Faulstich](https://github.com/TazTornadoo), [Skyler MacGowan](https://github.com/Schuyler-lab), [Sebastian Sydow](https://gitlab.com/sydow), [Jacob Umland](https://gitlab.com/jacobumland)*

# Bikerus

![language](https://img.shields.io/badge/language-Python%20%7C%20Docker-blue)
![version](https://img.shields.io/badge/version-v0.0.1-yellow)
![last-edited](https://img.shields.io/badge/last%20edited-08.11.2020-green)
![licence](https://img.shields.io/badge/licence-GPLv3-red)

> 🚴 This repository shows how to predict the demand of bikes needed for a bike rental service.

- [Bikerus](#bikerus)
- [Data acquisition](#data-acquisition)
- [Data visualization](#data-visualization)
  - [Bike Rental Station Map](#bike-rental-station-map)
- [Data Preprocessing](#data-preprocessing)
  - [Imputing NAs](#imputing-nas)
  - [Further preprocessing](#further-preprocessing)
  - [Data Partitioning](#data-partitioning)
- [Data modelling](#data-modelling)
  - [CatBoost - Gradient Boosting on Decision Trees](#catboost---gradient-boosting-on-decision-trees)

# Data acquisition

> 💾 This paragraph will explain how you can obtain the used data

In order to obtain the data, which is used within this project please clone this repository and execute then the file `0_pipeline_data_getting_compression.py` file. This file will:
- Download the files from the web
- Extract them into a folder within the parent directory called `data/raw`
- Loads these raw datasets and converts them into a compressed file in `data/interim` (for the sake of convenience we left the raw data there, in order you want to change things).

# Data visualization

> 🗺️ Here will be shown how the data visualizations can be created

## Bike Rental Station Map

In order to reproduce the map with the bike share rental stations, you have to execute the file `0.1_pipeline_bike_station_viz.py` within the `python` folder. This will create a folder `images` within the parent directory. Once you enter this folder there should be an `.html` file, which contains this map.

# Data Preprocessing

## Imputing NAs

> 🥋 This paragraph will show how NAs are imputed

In order to impute your own missing values please execute the script named `1_pipeline_impute_NAs.py`. This will create a file in the folder `data` which is named `preprocessed`. In this folder you can find the final version of the Bike Rental data.

## Further preprocessing

> 🌾 This paragraph describes how the further preprocessing works

Based on the data resulting from imputing NAs, further preprocessing is done by executing the script `2_pipeline_preprocessing`: ❌ unnecessary data features are dropped, ➡️ data is transformed to correct data types, 📊 and the continous variables are normalized. This script will create a file for the preprocessed data in the folder `data` as well as another file for the storing the actual (non-normalized) minimum and maximum values for the variable to be predicted.

## Data Partitioning

> 🗂️ This paragraph will explain how you can partition the used data into a train and test set. Additionally it explains how the train set can be partitioned into a train set and a cross-validation-set (for additional testing), if `GridSearchCV` is not used. `GridSearchCV` uses [(Stratified)KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) as a cross validation splitting strategy (see parameter `cv`). The following explaination describes a variation of KFold which returns first k folds as train set and the (k + 1)th fold as test set.

**Steps for creating a train and test set:**
1. Import the data. Use `df = decompress_pickle(<path>.pbz2)` for importing.
2. Call the function `train_test_split_ts`. 

    The function takes two arguments: The first one is the `data (type: DataFrame)`. The second is the size of the `training set (type: float)`. The size of the training set must be greater than `0` and smaller than `1`.

    The function returns the sets for `X_train`, `Y_train`, `X_test` and `Y_test`. `X_train` and `X_test` include all columns except for the target column. `Y_train` and `Y_test` only include the target column (field to predict). `X_train` and `Y_train` are used for training including determining the samples for cross validation. `X_test` and `Y_test` are only used for the (final) testing. The files are exported to: `'./data/partitioned/'`.

**Steps for creating a train and test set for cross validation (if GridSearchCV is not used):**
1. Import the date: `X_train`, `Y_train`. Use `df = decompress_pickle(<path>.pbz2)` for importing.
2. Call the function `get_sample_for_cv`.

    The function takes six arguments. Two arguments are optional (refer to the steps for creating a horizontal bar diagram to visualize the train-test-splits). 
    - `n_splits`: This determines the number of splits used for cross-validation. It must be an integer and greater than `1`.
    - `fold`: This determines the current fold (subsample) of the `train` and `test` set for cross validation. It must be an integer and greater than `0` and not greater than the `number of splits`.
    - `X_train` and `Y_train`: Data used for training including determining the samples for cross validation. `X_train` includes all columns except for the target column. `Y_train` only includes the target column (field to predict).

    The functions returns the sets for `X_train_current` and `Y_train_current` as the current fold/sub-sample. Additionally it returns `X_test_cv_current` and `Y_test_cv_current` for cross-validation.

**Steps for creating a horizontal bar diagramm to visualizse the train-test-splits:**

The function `get_sample_for_cv` can create a horizontal bar diagram for the visualization of the the train-test-splits. The function only creates the bardiagram if `X_test` is added as a parameter and if the parameter `vis == True`.
- `X_test`: `X_test` is needed to visualize the final round of testing with `X_test` and `Y_test`, which we created at the beginning with the function `train_test_split_ts`. To create the horizontal bardiagramm, `X_test` has to be added to the function.
- `vis`: `Vis` is used as decision variable for the creation of the diagram. It is initalized as `False`. Therefore, the horizontal bardiagramm will not be created. To create the horizontal bardiagramm, add `True` as the last parameter, when calling the function. The figure is saved in the path `./data/partitioned/`.

# Data modelling

## CatBoost - Gradient Boosting on Decision Trees

> 🐈 This paragraph explains how the catboost regressor is used

1. Run the `Grid_Search_Catboost-param.ipynb` to comprehend my Catboost settings. The best parameters of the CatBoostRegressor for this dataset are `depth = 6`, `learning_rate = 0.1` and `iterations = 1000`. 
2. Open the `catboost_skript_ts.py` script. Proof the calculated parameters with the parameters in the `CatBoostRegressor`. Afterwards run the `catboost_skript_ts.py` script to create the CatBoost model based on the parameters and the BikeRental dataset. Additionally the script also saves the state of the CatBoost model in a file in the bikerus folder. The file is named `Catboost_model`.
3. Last but not least open the `load_catboost.py` script. This script loads the previous saved CatBoost model. Additionally, there is also a test dataset of of `1. January 2013 0pm`. If you run the script, the model will predict the Bike Rentals for this specific hour based on the testdata set. Since we fed the model with normalized data, it returns a normalized count value.

## Scikit-learn - RandomForestRegressor

> 🌲 This paragraph explains how the RandomForestRegressor is used

1. Open the `random_forstest.py` script and run the script.
2. The following steps are performed within the script:
    1. The script loads the preprocessed data using `decompress_pickle("./data/preprocessed/BikeRental_preprocessed.pbz2")`. 
    2. The column `'datetime'` needs to be dropped, because the RandomForestRegressor cannot handle its type.
    3. The train and test samples are created using the function `train_test_split_ts`.
    4. Here, `GridSearchCV` is not used. Following from the explaination about cross validation iterators in [scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html) (chapter 3.1.2.), if one knows that the samples have been generated using a time-dependent process, it is safer to use a time-series aware cross-validation scheme. Therefore, cross validation is performed by applying the function [get_sample_for_cv](#Data-Partitioning) to also consider the time series character for cross validation. Here, 5 folds are created. The different hyperparameters are applied to the folds through cascaded for loops. The `Pseudo-R^2` is calculated for each fold and the respective hyperparameter combination. At the end, a mean of each hyperparameter combination across the five folds is calculated. The `hyperparameter combination` with the highest mean is returned. Under consideration of the trade-off between a high `Pseudo-R^2` and the models robustness, the hyperparameters `max_depth = 13`, `n_estimators = 300`, `max_features = 10` and `max_leaf_nodes = 80` have been chosen.
    5. The RandomForestRegressor is trained with the best hyperparameters and the `R^2` and `Pseudo-R^2` are calculated.
    6. The Model is saved using `joblib.dump(RForreg, "./RandomForest_Model/" + str(filename))`.
