Authors: *[Yannik Suhre](https://github.com/Estreuselito), [Jan Faulstich](https://github.com/TazTornadoo), [Skyler MacGowan](https://github.com/Schuyler-lab), [Sebastian Sydow](https://gitlab.com/sydow), [Jacob Umland](https://gitlab.com/jacobumland)*

# Bikerus

![language](https://img.shields.io/badge/language-Python%20%7C%20Docker-blue)
![version](https://img.shields.io/badge/version-v1.0.0-yellow)
![last-edited](https://img.shields.io/badge/last%20edited-12.11.2020-green)
![licence](https://img.shields.io/badge/licence-GPLv3-red)

> 🚴 This repository shows how to predict the demand of bikes needed for a bike rental service.

- [Bikerus](#bikerus)
- [Introduction for reproduciability](#introduction-for-reproduciability)
  - [Docker](#docker)
  - [Anaconda](#anaconda)
- [TL;DR](#tldr)
- [Data acquisition](#data-acquisition)
- [Data visualization](#data-visualization)
  - [Bike Rental Station Map](#bike-rental-station-map)
- [Data Preprocessing](#data-preprocessing)
  - [Imputing NAs](#imputing-nas)
  - [Further preprocessing](#further-preprocessing)
  - [Data Partitioning](#data-partitioning)
- [Data modelling](#data-modelling)
  - [CatBoost - Gradient Boosting on Decision Trees](#catboost---gradient-boosting-on-decision-trees)
  - [Fastai - Neural Net Regressor](#fastai---neural-net-regressor)
  - [Scikit-learn - RandomForestRegressor](#scikit-learn---randomforestregressor)
- [Deployment and live predictions](#deployment-and-live-predictions)

# Introduction for reproduciability

> 💡 Hereafter will be explained how to completely reproduce all the findings within this repo

To reproduce our findings you must first install and then run this repository. To do so we recommend that you use [docker](https://www.docker.com/products/docker-desktop) and [Visual Studio Code](https://code.visualstudio.com/) as this corresponds with our methodology. Alternatively, you can also use [Anaconda](https://www.anaconda.com/).

## Docker

> 🐋 This will explain how to use docker for an easy install

To use Visual Studio Code and Docker, please follow the steps outlined below.
1. Download the repository and open the folder in [Visual Studio Code (VS Code)](https://code.visualstudio.com/). If you are new to VS Code please install the [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) Add-On. 
2. On the bottom left corner of your open window on VS Code, two signs should appear. Click on those. Doing so should cause a list to open.
3. On this list, click the `Remote-Containers: Reopen in Container` entry. Note: If this is the first time you are doing this, it can take some time as Docker will create your image with all the necessary requirements. 
4. Following the completion of Step 3, you have your file editor on the left side and can click through the files. If you want to execute a file, just click the Play button on the top right corner; doing so will execute the Python script.

## Anaconda

> 🐍 This will explain how to use Anaconda to use this repo

Should you use [`Anaconda`](https://www.anaconda.com/) (*`miniconda` was not tested*) and you want to reproduce our findings download the repo. The open you `Anaconda prompt`. Navigate to the downloaded git-repository `Bikerus` (*one can navigate within the `Anaconda prompt` using normal command line commands. In that case `cd <your-path-to-bikerus>`. Should you have any spaces within your path use quotation marks around your path. Also, should you have to change your Harddrive use `\<your-harddrive-letter>`. In total that would look like: `cd \\<your-harddrive-letter> "<your-path-to-bikerus>"`*). Once you navigated there with your prompt, create a new python environment:

`conda create --name bikerus python=3.8`

Next activate this environment:

`conda activate bikerus`

Now, we use `pip` to install the necessary packages from the `requirements.txt`.

`pip install -r requirements.txt`

This install all the necessary packages within your `Anaconda` environment. Now you can start and execute every script by itself without worrying about packages and versions.

# TL;DR

> 🐳 This paragraph is only useful, if you are using Docker with VSCode

Once your Docker Container is running inside your VSCode, you can just enter the following:

`./execute_all_scripts.sh`

This will execute all scripts in the correct order and your don't have to run them indivdually. Should you use `Anaconda` you have to run them individually, since the `Anaconda prompt` cannot execute shell scripts.

# Data acquisition

> 💾 This paragraph will explain how you can obtain the data used

In order to obtain the data used by this project, please clone this repository and execute then the file `0_pipeline_data_getting_compression.py` file. This file will:

- Download the files from the web
- Extract them into a folder within the parent directory called `data/raw`
- Load these raw datasets and convert them into a compressed file in `data/interim` (for the sake of convenience we left the raw data there, should you want to change things).

# Data visualization

> 🗺️ This section shows how the data visualizations can be created

## Bike Rental Station Map

In order to reproduce the map with the bike share rental stations, you have to execute the file `0.1_pipeline_bike_station_viz.py` within the `python` folder. This will create a folder `images` within the parent directory. Once you enter this folder there should be a `.html` file, which contains this map.

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

In order to load or train all models (with exception of fastai - see [here](#fastai---neural-net-regressor) why) with the given train and test split, execute the script `4_models.py`. This will:
- Load the given models or train them
- Save them to the local drive
- Creates two dataframe:
  - One dataframe with all given prediction (normalized and unnormalized)
  - Another with the given $R^2$ values
- Saves the aforementioned dataframes


## CatBoost - Gradient Boosting on Decision Trees

> 🐈 This paragraph explains how the catboost regressor is used

1. Run the `Grid_Search_Catboost-param.ipynb` to comprehend my Catboost settings. The best parameters of the CatBoostRegressor for this dataset are `depth = 6`, `learning_rate = 0.1` and `iterations = 1000`. 
2. Open the `catboost_skript_ts.py` script. Proof the calculated parameters with the parameters in the `CatBoostRegressor`. Afterwards run the `catboost_skript_ts.py` script to create the CatBoost model based on the parameters and the BikeRental dataset. Additionally the script also saves the state of the CatBoost model in a file in the bikerus folder. The file is named `Catboost_model`.
3. Last but not least open the `load_catboost.py` script. This script loads the previous saved CatBoost model. Additionally, there is also a test dataset of of `1. January 2013 0pm`. If you run the script, the model will predict the Bike Rentals for this specific hour based on the testdata set. Since we fed the model with normalized data, it returns a normalized count value.

## Fastai - Neural Net Regressor

> ⚠️In order to try this one, one has to install [fastai](https://docs.fast.ai/#Installing) within an anaconda environment, since the pip version is really hard to install. Thus it cannot be installed within a container. Please follow the fastai link above to get more detailed explanation of how to setup fastai in your local environment. You have to uncomment the function `fastai_neural_net_regression` within the script `model_creation` as well as the line where you import fastai. If you want to run the script `4_models.py` with the `fastai_neural_net_regression` make sure also to uncomment that, when you are in a anaconda enviroment with fastai.

> 🌠 In the following will be explained how to use FastAI for a regression task

FastAI is a framework developed for fast and accessible artificial intelligence. Since its second version it can deal with structure tabular data, using neural nets as a regressor.


## Scikit-learn - RandomForestRegressor

> 🌲 This paragraph explains how the RandomForestRegressor is used.

1. Open the `random_forstest.py` script and run it.
2. The following steps are performed within the script:
    1. The script loads the preprocessed data using `decompress_pickle("./data/preprocessed/BikeRental_preprocessed.pbz2")`. 
    2. The column `'datetime'` needs to be dropped, because the RandomForestRegressor cannot handle its type.
    3. The train and test samples are created using the function `train_test_split_ts`.
    4. Here, `GridSearchCV` is not used. Following from the explanation about cross validation iterators in [scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html) (chapter 3.1.2.), if one knows that the samples have been generated using a time-dependent process, it is safer to use a time-series aware cross-validation scheme. Therefore, cross validation is performed by applying the function [get_sample_for_cv](#Data-Partitioning) to also consider the time series character for cross validation. Here, 5 folds are created. The different hyperparameters are applied to the folds through cascaded for loops. The `Pseudo-R^2` is calculated for each fold and the respective hyperparameter combination. At the end, a mean of each hyperparameter combination across the five folds is calculated. The `hyperparameter combination` with the highest mean is returned. Under consideration of the trade-off between a high `Pseudo-R^2` and the model's robustness, the hyperparameters `max_depth = 11`, `n_estimators = 300`, `max_features = 10` and `max_leaf_nodes = 80` were chosen.
    5. The RandomForestRegressor is trained with the best hyperparameters and the `R^2` and `Pseudo-R^2` are calculated.
    6. The Model is saved using `joblib.dump(RForreg, "./RandomForest_Model/" + str(filename))`.

## Scikit-learn - MLPRegressor

> 🕸️ This paragraph explains how the MLPRegressor is used.

1. after having finished all preprocessing steps, run `4_models.py` in order to run the models
2. after execution you can find the saved multilayer perceptron model, its optimal hyperparamters and r squared values as well as the predicted dataframe in `NN_MLP_files`in the `models`folder

# Deployment and live predictions

> 🚀 This paragraph explains how to start up a flask app, which deploys the models and makes live predictions

Bascially all you have to do is to run the `app.py`. In VSCode with Docker backend just click the play button in the top right corners. In Anaconda run the `app.py` from the top level of the bikerus folder:

`python flask/app.py`

Go to the given webpage (*most likely 127.0.0.1:5000*) and enjoy predicting!
