# This file compresses all the data to smaller pieces
import pandas as pd
import os
import shutil
from data_preprocessing import import_data
from data_storage import connection
from download import download

print(" _       _       _                                 _           ___     _                           _\n\
( )  _  ( )     (_ )                              ( )_        (  _ \ _( )                         ( )\n\
| | ( ) | |  __  | |   ___   _    ___ ___    __   |  _)  _    | (_) )_) |/ )   __  _ __ _   _  ___| |\n\
| | | | | |/ __ \| | / ___)/ _ \/  _   _  \/ __ \ | |  / _ \  |  _ (| |   (  / __ \  __) ) ( )  __) |\n\
| (_/ \_) |  ___/| |( (___( (_) ) ( ) ( ) |  ___/ | |_( (_) ) | (_) ) | |\ \(  ___/ |  | (_) |__  \_)\n\
 \__/\___/ \____)___)\____)\___/(_) (_) (_)\____)  \__)\___/  (____/(_)_) (_)\____)_)   \___/(____/\n\
                                                                                                   (_)\n\n\
                                             ___         _ \n\
                                            (  _ \      ( )_\n\
                                            | (_(_)  __ |  _)_   _ _ _ \n\
                                            \__ \ / __  \ | ( ) ( )  _ \ \n\
                                            ( )_) |  ___/ |_| (_) | (_) ) \n\
                                            \____)\____)\__)\____/|  __/ \n\
                                                                  | | \n\
                                                                  (_) \n\n\
Let's get you started by downloading you data and inputting this in a database!\n")
# load data

list_of_zip_path = ["http://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip",
                    "https://s3.amazonaws.com/capitalbikeshare-data/2011-capitalbikeshare-tripdata.zip",
                    "https://s3.amazonaws.com/capitalbikeshare-data/2012-capitalbikeshare-tripdata.zip"]


for file_path in list_of_zip_path:
    path = download(file_path, "./data/raw", kind="zip", replace=True)

station_locs = pd.read_csv(
    "https://opendata.arcgis.com/datasets/a1f7acf65795451d89f0a38565a975b3_5.csv")
print("\n\nYour data has been downloaded! \n\nNow your data is inputed into a database!\n")

raw = pd.concat([import_data("./data/raw/2011-capitalbikeshare-tripdata.csv",
                             parse_dates=["Start date", "End date"]),
                 import_data("./data/raw/2012Q1-capitalbikeshare-tripdata.csv",
                             parse_dates=["Start date", "End date"]),
                 import_data("./data/raw/2012Q3-capitalbikeshare-tripdata.csv",
                             parse_dates=["Start date", "End date"]),
                 import_data("./data/raw/2012Q4-capitalbikeshare-tripdata.csv",
                             parse_dates=["Start date", "End date"]),
                 import_data("./data/raw/2012Q4-capitalbikeshare-tripdata.csv", parse_dates=["Start date", "End date"])])

pd.read_csv(
    "./data/raw/hour.csv", index_col=[0]).to_sql("hours", connection, if_exists="replace")

raw.to_sql("raw", connection, if_exists="replace")

station_locs.to_sql("station_locs", connection, if_exists="replace")


dir_path = './data'
try:
    shutil.rmtree(dir_path)
except OSError as e:
    print("Error: %s : %s" % (dir_path, e.strerror))

# print statement
print(" __ \                      |\n\
 |   |  _ \   __ \    _ \  |\n\
 |   | (   |  |   |   __/ _|\n\
____/ \___/  _|  _| \___| _)\n\
Your data is now stored in a database within the folder database. You can access this database using DBeaver or other tools!")
