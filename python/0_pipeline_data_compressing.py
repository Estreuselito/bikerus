# This file compresses all the data to smaller pieces

from data_preprocessing import compressed_pickle, decompress_pickle, import_data

# load data
# The capital bike share data can be downloaded here: https://www.capitalbikeshare.com/system-data
ori_br = import_data("./data/raw/BikeRental.csv")
art11_br = import_data("./data/raw/2011-capitalbikeshare-tripdata.csv", parse_dates=["Start date", "End date"])
art12_br = pd.concat([import_data("./data/raw/2012Q1-capitalbikeshare-tripdata.csv", parse_dates=["Start date", "End date"]),
                      import_data("./data/raw/2012Q2-capitalbikeshare-tripdata.csv", parse_dates=["Start date", "End date"]),
                      import_data("./data/raw/2012Q3-capitalbikeshare-tripdata.csv", parse_dates=["Start date", "End date"]),
                      import_data("./data/raw/2012Q4-capitalbikeshare-tripdata.csv", parse_dates=["Start date", "End date"])])
station_locs = import_data("./data/raw/Capital_Bike_Share_Locations.csv")

# compress them
compressed_pickle("./data/interim/BikeRental", ori_br)
compressed_pickle("./data/interim/ArtificalRentals11", art11_br)
compressed_pickle("./data/interim/ArtificalRentals12", art12_br)
compressed_pickle("./data/interim/Stations", station_locs)