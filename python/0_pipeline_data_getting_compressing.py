# This file compresses all the data to smaller pieces
import pandas as pd
from data_preprocessing import compressed_pickle, decompress_pickle, import_data
from download import download

# load data
print("Start downloading your data!")
# The capital bike share data can be downloaded here: https://www.capitalbikeshare.com/system-data

list_of_zip_path = ["http://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip",
                    "https://s3.amazonaws.com/capitalbikeshare-data/2011-capitalbikeshare-tripdata.zip",
                    "https://s3.amazonaws.com/capitalbikeshare-data/2012-capitalbikeshare-tripdata.zip"]

for file_path in list_of_zip_path:
    path = download(file_path, "./data/raw", kind = "zip", replace = True)

station_locs = pd.read_csv("https://opendata.arcgis.com/datasets/a1f7acf65795451d89f0a38565a975b3_5.csv")
print("Your data has been downloaded! \n\nNow your data is going to be compressed!")

ori_br = import_data("./data/raw/hour.csv")
art11_br = import_data("./data/raw/2011-capitalbikeshare-tripdata.csv", parse_dates=["Start date", "End date"])
art12_br = pd.concat([import_data("./data/raw/2012Q1-capitalbikeshare-tripdata.csv", parse_dates=["Start date", "End date"]),
                      import_data("./data/raw/2012Q2-capitalbikeshare-tripdata.csv", parse_dates=["Start date", "End date"]),
                      import_data("./data/raw/2012Q3-capitalbikeshare-tripdata.csv", parse_dates=["Start date", "End date"]),
                      import_data("./data/raw/2012Q4-capitalbikeshare-tripdata.csv", parse_dates=["Start date", "End date"])])


# compress them
compressed_pickle("./data/interim/BikeRental", ori_br)
compressed_pickle("./data/interim/ArtificalRentals11", art11_br)
compressed_pickle("./data/interim/ArtificalRentals12", art12_br)
compressed_pickle("./data/interim/Stations", station_locs)

print("Done Men! \n\nYou can find you data in you parent folder under data/raw the raw files and under interim the compressed files!")
