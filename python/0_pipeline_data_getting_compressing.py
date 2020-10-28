# This file compresses all the data to smaller pieces
import pandas as pd
import os
from data_preprocessing import compressed_pickle, decompress_pickle, import_data
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
Let's get you started by downloading you data!\n")
# load data

list_of_zip_path = ["http://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip",
                    "https://s3.amazonaws.com/capitalbikeshare-data/2011-capitalbikeshare-tripdata.zip",
                    "https://s3.amazonaws.com/capitalbikeshare-data/2012-capitalbikeshare-tripdata.zip"]

if not os.path.exists("./data/raw"):
    os.makedirs("./data/raw")

for file_path in list_of_zip_path:
    path = download(file_path, "./data/raw", kind="zip", replace=True)

station_locs = pd.read_csv(
    "https://opendata.arcgis.com/datasets/a1f7acf65795451d89f0a38565a975b3_5.csv")
print("\n\nYour data has been downloaded! \n\nNow your data is going to be compressed!\n")

ori_br = import_data("./data/raw/hour.csv")
art11_br = import_data("./data/raw/2011-capitalbikeshare-tripdata.csv",
                       parse_dates=["Start date", "End date"])
art12_br = pd.concat([import_data("./data/raw/2012Q1-capitalbikeshare-tripdata.csv", parse_dates=["Start date", "End date"]),
                      import_data("./data/raw/2012Q2-capitalbikeshare-tripdata.csv",
                                  parse_dates=["Start date", "End date"]),
                      import_data("./data/raw/2012Q3-capitalbikeshare-tripdata.csv",
                                  parse_dates=["Start date", "End date"]),
                      import_data("./data/raw/2012Q4-capitalbikeshare-tripdata.csv", parse_dates=["Start date", "End date"])])

if not os.path.exists("./data/interim"):
    os.makedirs("./data/interim")
# compress them
compressed_pickle("./data/interim/BikeRental", ori_br)
compressed_pickle("./data/interim/ArtificalRentals11", art11_br)
compressed_pickle("./data/interim/ArtificalRentals12", art12_br)
compressed_pickle("./data/interim/Stations", station_locs)

<<<<<<< HEAD
print("Done Men! \n\nYou can find you data in you parent folder under data/raw the raw files and under interim the compressed files!")
=======
print("\n\nDone Men! \n\nYou can find you data in you parent folder under data/raw the raw files and under interim the compressed files!")
>>>>>>> ace2dfe8404524e87c3b5eae5c3d82cad096831a
