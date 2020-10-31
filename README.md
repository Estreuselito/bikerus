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

# Further preprocessing

Based on the data resulting from imputing NAs, further preprocessing is done by executing the script `2_pipeline_preprocessing`: ❌ unnecessary data features are dropped, ➡️ data is transformed to correct data types, 📊 and the continous variables are normalized. This script will create a file for the preprocessed data in the folder `data` as well as another file for the storing the actual (non-normalized) minimum and maximum values for the variable to be predicted.

# Data visualization

> 🗺️ Here will be shown how the data visualizations can be created

## Bike Rental Station Map

In order to reproduce the map with the bike share rental stations, you have to execute the file `0.1_pipeline_bike_station_viz.py` within the `python` folder. This will create a folder `images` within the parent directory. Once you enter this folder there should be an `.html` file, which contains this map.
